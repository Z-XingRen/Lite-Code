"""Runtime workspace snapshot and checkpoint helpers."""

import hashlib
import time
import uuid

from ..features import memory as memorylib
from .workspace import IGNORED_PATH_NAMES, clip, now

CHECKPOINT_SCHEMA_VERSION = "phase1-v1"


class RuntimeCheckpointsMixin:
    def begin_workspace_change(self, tool, args):
        return self.workspace_change_tracker.begin(tool, args)

    def finish_workspace_change(self, token, result=None):
        return self.workspace_change_tracker.finish(token, result)

    def prepare_workspace_change(self, tool, args):
        self._workspace_tracking_error = ""
        self._last_workspace_tracking_metadata = {}
        token = self.begin_workspace_change(tool, args) if tool.risky else None
        before = {}
        if tool.risky and (token is None or token.mode == "opaque"):
            try:
                before = self.capture_workspace_snapshot()
            except Exception as exc:
                self._workspace_tracking_error = str(exc)
                self._last_workspace_tracking_metadata = {
                    "workspace_tracker_mode": (
                        "opaque" if token is not None and token.mode == "opaque" else "legacy"
                    ),
                    "workspace_tracker_fallback": token is None or token.mode == "opaque",
                    "workspace_tracker_error": str(exc),
                }
                raise
        return token, before

    def complete_workspace_change(self, tool, token, before, result=None):
        if self._workspace_tracking_error or before is None:
            return [], []
        try:
            if token is not None:
                if token.mode == "opaque":
                    started = time.perf_counter()
                    after = self.capture_workspace_snapshot()
                    outcome = self.workspace_change_tracker.finish(
                        token,
                        result,
                        legacy_before=before,
                        legacy_after=after,
                        fallback_duration_ms=(time.perf_counter() - started) * 1000,
                    )
                else:
                    outcome = self.finish_workspace_change(token, result)
            else:
                after = self.capture_workspace_snapshot() if tool.risky else before
                outcome = self.diff_workspace_snapshots(before, after)
        except Exception as exc:
            self._workspace_tracking_error = str(exc)
            self._last_workspace_tracking_metadata = dict(
                self.workspace_change_tracker.last_observation
            )
            self._last_workspace_tracking_metadata["workspace_tracker_error"] = str(exc)
            raise
        self._last_workspace_tracking_metadata = dict(
            self.workspace_change_tracker.last_observation
        )
        return outcome

    def capture_workspace_snapshot(self):
        snapshot = {}
        for path in self.root.rglob("*"):
            try:
                relative_parts = path.relative_to(self.root).parts
            except ValueError:
                continue
            if any(part in IGNORED_PATH_NAMES for part in relative_parts) or not path.is_file():
                continue
            try:
                snapshot[path.relative_to(self.root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
            except Exception:
                continue
        return snapshot

    @staticmethod
    def diff_workspace_snapshots(before, after):
        changed_paths = []
        summaries = []
        for path in sorted(set(before) | set(after)):
            if before.get(path) == after.get(path):
                continue
            changed_paths.append(path)
            if path not in before:
                summaries.append(f"created:{path}")
            elif path not in after:
                summaries.append(f"deleted:{path}")
            else:
                summaries.append(f"modified:{path}")
        return changed_paths, summaries

    def create_checkpoint(self, task_state, user_message, trigger):
        state = self.checkpoint_state()
        current = self.current_checkpoint()
        checkpoint_id = "ckpt_" + uuid.uuid4().hex[:8]
        key_files = []
        freshness = {}
        for path in self.memory.to_dict()["working"]["recent_files"]:
            file_freshness = memorylib.file_freshness(path, self.root)
            freshness[path] = file_freshness
            key_files.append({"path": path, "freshness": file_freshness})
        checkpoint = {
            "checkpoint_id": checkpoint_id,
            "parent_checkpoint_id": current.get("checkpoint_id", "") if current else "",
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "created_at": now(),
            "current_goal": str(user_message),
            "completed": [task_state.final_answer] if task_state.final_answer else [],
            "excluded": [],
            "current_blocker": "" if str(task_state.stop_reason or "") in ("", "final_answer_returned") else str(task_state.stop_reason),
            "next_step": self.infer_next_step(task_state),
            "key_files": key_files,
            "freshness": freshness,
            "summary": f"{trigger}: {clip(str(user_message), 120)}",
            "runtime_identity": self.current_runtime_identity(),
        }
        state["items"][checkpoint_id] = checkpoint
        state["current_id"] = checkpoint_id
        task_state.checkpoint_id = checkpoint_id
        self.session["runtime_identity"] = checkpoint["runtime_identity"]
        self.session_path = self.session_store.save(self.session)
        return checkpoint
