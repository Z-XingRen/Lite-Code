"""Worker cancellation binding and bounded thread cleanup."""

import threading
import time

from .workspace import now


class WorkerCancellationMixin:
    def shutdown(self, timeout=2.0):
        tasks = list(self._tasks.values())
        changed = False
        for task in tasks:
            item = self._get_item(task.id)
            if item.get("status") not in {"running", "stopping"}:
                continue
            self._request_stop(task)
            with self._lock:
                item["status"] = "stopping"
                item["updated_at"] = now()
            changed = True
            self.runtime.session_event_bus.emit(
                "worker_stop_requested",
                {"worker_id": item["id"], "status": "stopping"},
            )
        if changed:
            self._save()
        deadline = time.monotonic() + float(timeout)
        for task in tasks:
            thread = task.thread
            if (
                thread is None
                or not thread.is_alive()
                or thread is threading.current_thread()
            ):
                continue
            remaining = max(0.0, deadline - time.monotonic())
            if remaining:
                thread.join(remaining)
        live_task_ids = [
            task.id
            for task in tasks
            if task.thread is not None and task.thread.is_alive()
        ]
        if live_task_ids:
            self.runtime.session_event_bus.emit(
                "worker_cleanup_timeout",
                {"worker_ids": live_task_ids, "timeout_ms": int(timeout * 1000)},
            )
        return {
            "stopped": sum(1 for task in tasks if task.stop_requested),
            "live_task_ids": live_task_ids,
        }

    def _request_stop(self, task):
        with self._lock:
            if task.stop_requested:
                return False
            task.stop_requested = True
        abort = getattr(task.runtime, "abort_current_turn", None)
        if callable(abort):
            abort()
        return True

    def _prepare_run(self, task):
        self._detach_run_cancellation(task)
        with self._lock:
            task.stop_requested = False
        token = getattr(self.runtime, "current_cancellation_token", None)
        add_callback = getattr(token, "add_callback", None)
        if callable(add_callback):
            task.cancel_unsubscribe = add_callback(lambda: self._request_stop(task))

    def _detach_run_cancellation(self, task):
        remove_callback = task.cancel_unsubscribe
        task.cancel_unsubscribe = None
        if callable(remove_callback):
            remove_callback()
