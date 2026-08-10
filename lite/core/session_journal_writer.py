"""Exclusive append writer for one versioned session journal."""

import hashlib
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from ..cancellation import CancellationRequested
from .session_journal_reducer import (
    JournalState,
    apply_prepared_journal_record,
    prepare_journal_record,
)
from .session_journal_recovery import (
    recovery_actions_from_state,
    restore_session_journal,
    snapshot_path_for,
    write_atomic_snapshot,
)
from .session_journal_schema import (
    EFFECT_INTENT,
    EFFECT_RESULT,
    HISTORY_APPENDED,
    HISTORY_REPLACED,
    HEAD_MOVED,
    JOURNAL_SCHEMA_VERSION,
    SESSION_CREATED,
    SESSION_UPDATED,
    TREE_ENTRY_APPENDED,
    TREE_LABEL_UPDATED,
    JournalRecord,
    canonical_json,
)
from .session_tree import SessionTreeEntry, tree_rows


_ACTIVE_HEAD = object()


class JournalWriterError(RuntimeError):
    """The journal cannot safely accept a write from this writer."""


class SessionJournalWriter:
    def __init__(self, path, *, sync=True):
        self.path = Path(path)
        self.lock_path = self.path.with_name(f"{self.path.name}.lock")
        self.snapshot_path = snapshot_path_for(self.path)
        self.sync = bool(sync)
        self.state = JournalState.empty()
        self.discarded_tail = b""
        self.recovery_actions = ()
        self._state_lock = threading.RLock()
        self._operation_lock = threading.Lock()
        self._closed = False
        self._owner_token = f"{os.getpid()}:{threading.get_ident()}:{uuid.uuid4().hex}"

    @classmethod
    def create(cls, path, session, *, sync=True):
        writer = cls(path, sync=sync)
        writer.path.parent.mkdir(parents=True, exist_ok=True)
        writer._acquire_file_lock()
        try:
            if writer.path.exists() and writer.path.stat().st_size:
                raise JournalWriterError(
                    f"session journal already contains records: {writer.path}"
                )
            writer._append_record(
                SESSION_CREATED,
                {"session": session},
                operation_id=str(session.get("id", "")),
            )
        except BaseException:
            writer._release_file_lock()
            raise
        return writer

    @classmethod
    def open(cls, path, *, sync=True):
        writer = cls(path, sync=sync)
        writer._acquire_file_lock()
        try:
            restored = restore_session_journal(
                writer.path, snapshot_path=writer.snapshot_path
            )
            writer.state = restored.state
            writer.discarded_tail = restored.discarded_tail
            if restored.discarded_tail:
                writer._truncate_to(restored.complete_size)
            writer._recover_open_operation()
            writer.recovery_actions = recovery_actions_from_state(writer.state)
        except BaseException:
            writer._release_file_lock()
            raise
        return writer

    def append_history(self, item, *, operation_id=None, record_id=None):
        return self._append_mutation(
            HISTORY_APPENDED,
            {"item": item},
            operation_id=operation_id,
            record_id=record_id,
        )

    def replace_history(self, items, *, operation_id=None, record_id=None):
        return self._append_mutation(
            HISTORY_REPLACED,
            {"items": items},
            operation_id=operation_id,
            record_id=record_id,
        )

    def update_session(self, updates, *, operation_id=None, record_id=None):
        return self._append_mutation(
            SESSION_UPDATED,
            {"updates": updates},
            operation_id=operation_id,
            record_id=record_id,
        )

    def append_tree_entry(
        self,
        entry_type,
        data,
        *,
        parent_id=_ACTIVE_HEAD,
        entry_id=None,
        turn_id="",
        run_id="",
        created_at=None,
        operation_id=None,
        record_id=None,
    ):
        """Append one node to the active branch and advance its head."""

        if parent_id is _ACTIVE_HEAD:
            parent_id = self.state.tree.active_head
        entry = SessionTreeEntry.from_dict(
            {
                "entry_id": entry_id or _new_id("entry"),
                "parent_id": parent_id,
                "entry_type": entry_type,
                "turn_id": str(turn_id or ""),
                "run_id": str(run_id or ""),
                "created_at": created_at or datetime.now(timezone.utc).isoformat(),
                "data": dict(data),
            }
        )
        self._append_mutation(
            TREE_ENTRY_APPENDED,
            {"entry": entry.to_dict()},
            operation_id=operation_id,
            record_id=record_id,
        )
        return entry

    def append_message(self, item, **kwargs):
        value = dict(item)
        return self.append_tree_entry(
            "message",
            {"message": value},
            turn_id=value.get("turn_id", ""),
            run_id=value.get("run_id", ""),
            created_at=value.get("created_at") or None,
            **kwargs,
        )

    def append_compaction(self, history, *, metadata=None, **kwargs):
        return self.append_tree_entry(
            "compaction",
            {"history": list(history), "metadata": dict(metadata or {})},
            **kwargs,
        )

    def move_head(self, target_entry_id, *, reason="branch", operation_id=None, record_id=None):
        self._append_mutation(
            HEAD_MOVED,
            {"target_entry_id": target_entry_id, "reason": str(reason or "")},
            operation_id=operation_id,
            record_id=record_id,
        )
        return self.state.tree.active_head

    def label_head(self, label, *, entry_id=None, operation_id=None, record_id=None):
        target = entry_id or self.state.tree.active_head
        self._append_mutation(
            TREE_LABEL_UPDATED,
            {"label": label, "entry_id": target},
            operation_id=operation_id,
            record_id=record_id,
        )
        return target

    def tree_rows(self):
        return tree_rows(self.state.tree)

    def write_snapshot(self):
        with self.effect(
            "snapshot",
            request={"snapshot_file": self.snapshot_path.name},
            replay_policy="interrupt",
        ) as effect:
            snapshot_sequence = self.state.last_sequence
            write_atomic_snapshot(
                self.snapshot_path,
                self.path,
                self.state,
                sync=self.sync,
            )
            effect.complete(
                "ok",
                {
                    "snapshot_file": self.snapshot_path.name,
                    "snapshot_sequence": snapshot_sequence,
                },
            )
        return self.snapshot_path

    def effect(
        self,
        effect_type,
        *,
        call_id=None,
        request=None,
        replay_policy="interrupt",
        operation_id=None,
    ):
        return JournalEffect(
            self,
            effect_type=effect_type,
            call_id=call_id or _new_id("call"),
            request=dict(request or {}),
            replay_policy=replay_policy,
            operation_id=operation_id or _new_id("operation"),
        )

    def begin_effect(
        self,
        effect_type,
        *,
        call_id,
        request,
        replay_policy,
        operation_id,
    ):
        self._operation_lock.acquire()
        try:
            return self._append_record(
                EFFECT_INTENT,
                {
                    "effect_type": effect_type,
                    "call_id": call_id,
                    "replay_policy": replay_policy,
                    "request": request,
                },
                operation_id=operation_id,
            )
        except Exception:
            self._operation_lock.release()
            raise

    def finish_effect(self, intent, *, outcome, result, tree_delta=None):
        try:
            payload = {
                "effect_type": intent.payload["effect_type"],
                "call_id": intent.payload["call_id"],
                "outcome": outcome,
                "result": result,
            }
            if tree_delta is not None:
                payload["tree_delta"] = tree_delta
            return self._append_record(
                EFFECT_RESULT,
                payload,
                operation_id=intent.operation_id,
            )
        finally:
            self._operation_lock.release()

    def close(self):
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
            self._release_file_lock()

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _traceback):
        self.close()
        return False

    def _append_mutation(self, kind, payload, *, operation_id, record_id):
        with self._operation_lock:
            return self._append_record(
                kind,
                payload,
                operation_id=operation_id or _new_id("operation"),
                record_id=record_id,
            )

    def _append_record(self, kind, payload, *, operation_id, record_id=None):
        with self._state_lock:
            self._ensure_open()
            record = JournalRecord(
                schema_version=JOURNAL_SCHEMA_VERSION,
                sequence=self.state.last_sequence + 1,
                record_id=record_id or _new_id("record"),
                operation_id=operation_id,
                kind=kind,
                payload=dict(payload),
            )
            transition = prepare_journal_record(self.state, record)
            self._write_line(record)
            apply_prepared_journal_record(self.state, transition)
            return record

    def _write_line(self, record):
        payload = (canonical_json(record.to_dict()) + "\n").encode("utf-8")
        with self.path.open("ab") as journal:
            journal.write(payload)
            journal.flush()
            if self.sync:
                os.fsync(journal.fileno())

    def _truncate_to(self, size):
        with self.path.open("r+b") as journal:
            journal.truncate(size)
            journal.flush()
            if self.sync:
                os.fsync(journal.fileno())

    def _recover_open_operation(self):
        operation = self.state.open_operation
        if operation is None:
            return
        action = "retry" if operation.replay_policy == "replay_safe" else "interrupt"
        record_id = "recovery_" + hashlib.sha256(
            operation.operation_id.encode("utf-8")
        ).hexdigest()
        with self._operation_lock:
            self._append_record(
                EFFECT_RESULT,
                {
                    "effect_type": operation.effect_type,
                    "call_id": operation.call_id,
                    "outcome": "interrupted",
                    "result": {
                        "reason": "process_interrupted",
                        "recovery_action": action,
                        "synthetic": True,
                    },
                },
                operation_id=operation.operation_id,
                record_id=record_id,
            )

    def _acquire_file_lock(self):
        descriptor = None
        for attempt in range(2):
            try:
                descriptor = os.open(
                    self.lock_path,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                    0o600,
                )
                break
            except FileExistsError as exc:
                if attempt == 0 and self._remove_stale_file_lock():
                    continue
                raise JournalWriterError(
                    f"session journal writer already active: {self.path}"
                ) from exc
        if descriptor is None:  # pragma: no cover - loop either opens or raises
            raise JournalWriterError(f"could not lock session journal: {self.path}")
        try:
            os.write(descriptor, self._owner_token.encode("ascii"))
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _remove_stale_file_lock(self):
        try:
            owner = self.lock_path.read_text(encoding="ascii")
            process_id = int(owner.split(":", 1)[0])
        except (OSError, UnicodeDecodeError, ValueError):
            return False
        if _process_is_alive(process_id):
            return False
        try:
            if self.lock_path.read_text(encoding="ascii") != owner:
                return False
            self.lock_path.unlink()
        except (FileNotFoundError, OSError):
            return False
        return True

    def _release_file_lock(self):
        try:
            owner = self.lock_path.read_text(encoding="ascii")
        except OSError:
            return
        if owner != self._owner_token:
            return
        try:
            self.lock_path.unlink()
        except FileNotFoundError:
            pass

    def _ensure_open(self):
        if self._closed:
            raise JournalWriterError("session journal writer is closed")


class JournalEffect:
    def __init__(
        self,
        writer,
        *,
        effect_type,
        call_id,
        request,
        replay_policy,
        operation_id,
    ):
        self.writer = writer
        self.intent = writer.begin_effect(
            effect_type,
            call_id=call_id,
            request=request,
            replay_policy=replay_policy,
            operation_id=operation_id,
        )
        self._completed = False

    def complete(self, outcome, result=None, *, tree_delta=None):
        if self._completed:
            raise JournalWriterError("journal effect is already complete")
        self._completed = True
        kwargs = {
            "outcome": outcome,
            "result": dict(result or {}),
        }
        if tree_delta is not None:
            kwargs["tree_delta"] = tree_delta
        return self.writer.finish_effect(self.intent, **kwargs)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, _exc, _traceback):
        if not self._completed:
            if exc_type is None:
                self.complete("ok", {})
            else:
                outcome = (
                    "interrupted"
                    if issubclass(
                        exc_type,
                        (CancellationRequested, GeneratorExit, KeyboardInterrupt, SystemExit),
                    )
                    else "error"
                )
                self.complete(outcome, {"error_type": exc_type.__name__})
        return False


def _new_id(prefix):
    return f"{prefix}_{uuid.uuid4().hex}"


def _process_is_alive(process_id):
    if process_id <= 0:
        return False
    if process_id == os.getpid():
        return True
    if os.name == "nt":
        return _windows_process_is_alive(process_id)
    try:
        os.kill(process_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _windows_process_is_alive(process_id):
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.GetExitCodeProcess.argtypes = (
            wintypes.HANDLE,
            ctypes.POINTER(wintypes.DWORD),
        )
        kernel32.GetExitCodeProcess.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        kernel32.CloseHandle.restype = wintypes.BOOL
        handle = kernel32.OpenProcess(0x1000, False, process_id)
        if not handle:
            return ctypes.get_last_error() == 5
        try:
            exit_code = wintypes.DWORD()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return True
            return exit_code.value == 259
        finally:
            kernel32.CloseHandle(handle)
    except (AttributeError, OSError):
        return True
