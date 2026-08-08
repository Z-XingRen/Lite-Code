"""Session-scoped worker lifecycle for subagents."""

import json
import queue
import threading
from dataclasses import dataclass, field

from .worker_cancellation import WorkerCancellationMixin
from .worker_execution import run_background_worker, run_worker
from .worker_inputs import clean_worker_type, clean_write_scope
from .worker_runtime import build_child_runtime
from .workspace import now


@dataclass
class WorkerTask:
    id: str
    description: str
    subagent_type: str
    write_scope: tuple[str, ...]
    runtime: object | None
    thread: threading.Thread | None = None
    stop_requested: bool = False
    cancel_unsubscribe: object | None = None
    state: dict = field(default_factory=dict)


class WorkerManager(WorkerCancellationMixin):
    def __init__(self, runtime):
        self.runtime = runtime
        self.runtime.session.setdefault("workers", {"next_id": 1, "items": []})
        self._tasks = {}
        self._lock = threading.Lock()
        self._notifications = queue.Queue()

    @property
    def state(self):
        return self.runtime.session.setdefault("workers", {"next_id": 1, "items": []})

    def spawn(self, description, prompt, subagent_type="worker", write_scope=None):
        subagent_type = clean_worker_type(subagent_type)
        if self.runtime.runtime_mode == "plan" and subagent_type != "Explore":
            raise ValueError("plan mode only allows Explore agents")
        background = self._can_run_background()
        task = self._new_task(
            description,
            subagent_type,
            write_scope,
            defer_runtime=background,
        )
        self._tasks[task.id] = task
        self._prepare_run(task)
        if background:
            self._start_background(task, prompt, action="spawn")
            return self._public_payload(task, status="started")
        run_worker(self, task, prompt, action="spawn")
        return self._public_payload(task)

    def continue_task(self, task_id, message):
        task = self._get_active_task(task_id)
        item = self._get_item(task_id)
        if item.get("status") in {"running", "stopping"}:
            raise ValueError(f"worker is running: {task_id}")
        if self.runtime.runtime_mode == "plan" and task.subagent_type != "Explore":
            raise ValueError("plan mode only allows Explore agents")
        self._prepare_run(task)
        if self._can_run_background():
            self._start_background(task, message, action="continue")
            return self._public_payload(task, status="started")
        run_worker(self, task, message, action="continue")
        return self._public_payload(task)

    def stop_task(self, task_id):
        item = self._get_item(task_id)
        if item["status"] == "running":
            task = self._tasks.get(str(task_id))
            if task is not None:
                self._request_stop(task)
            item["status"] = "stopping"
            item["updated_at"] = now()
            self.runtime.session_event_bus.emit(
                "worker_stop_requested", {"worker_id": item["id"], "status": "stopping"}
            )
            self._save()
        return {
            "task_id": item["id"],
            "status": item["status"],
            "description": item["description"],
        }

    def to_dict(self):
        return {
            "next_id": int(self.state.get("next_id", 1)),
            "items": [dict(item) for item in self.state.get("items", [])],
        }

    def _new_task(self, description, subagent_type, write_scope, defer_runtime=False):
        with self._lock:
            worker_id = f"agent_{int(self.state.get('next_id', 1))}"
            self.state["next_id"] = int(self.state.get("next_id", 1)) + 1
        scope = tuple(clean_write_scope(write_scope))
        child = (
            None
            if defer_runtime
            else build_child_runtime(self.runtime, subagent_type, scope)
        )
        item = {
            "id": worker_id,
            "description": str(description or "").strip() or "Worker task",
            "subagent_type": subagent_type,
            "write_scope": list(scope),
            "status": "idle",
            "result": "",
            "tool_steps": 0,
            "attempts": 0,
            "duration_ms": 0,
            "notification_drained": False,
            "created_at": now(),
            "updated_at": now(),
        }
        with self._lock:
            self.state.setdefault("items", []).append(item)
            self._save()
        return WorkerTask(worker_id, item["description"], subagent_type, scope, child)

    def _can_run_background(self):
        return getattr(self.runtime, "model_client_factory", None) is not None

    def _start_background(self, task, prompt, action):
        thread = threading.Thread(
            target=run_background_worker,
            args=(self, task, prompt, action),
            daemon=True,
            name=f"lite-worker-{task.id}",
        )
        task.thread = thread
        thread.start()

    def drain_notifications(self):
        drained = []
        while True:
            try:
                task_id, notification = self._notifications.get_nowait()
            except queue.Empty:
                break
            item = self._get_item(task_id)
            with self._lock:
                if item.get("notification_drained"):
                    continue
                item["notification_drained"] = True
                item["updated_at"] = now()
            drained.append(notification)
        if drained:
            self._save()
        return drained

    def _get_active_task(self, task_id):
        task = self._tasks.get(str(task_id))
        if task is None:
            raise ValueError(f"unknown or inactive worker: {task_id}")
        return task

    def _get_item(self, task_id):
        for item in self.state.setdefault("items", []):
            if item.get("id") == str(task_id):
                return item
        raise ValueError(f"unknown worker: {task_id}")

    def _public_payload(self, task, status=None):
        item = self._get_item(task.id)
        return {
            "task_id": task.id,
            "status": status or item["status"],
            "description": task.description,
        }

    def _save(self):
        self.runtime.session_path = self.runtime.session_store.save(
            self.runtime.session
        )


def dumps_payload(payload):
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)
