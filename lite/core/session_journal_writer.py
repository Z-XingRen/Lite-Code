"""Exclusive append writer for one versioned session journal."""

import os
import threading
import uuid
from pathlib import Path

from ..cancellation import CancellationRequested
from .session_journal_reducer import JournalState, reduce_journal_record
from .session_journal_schema import (
    EFFECT_INTENT,
    EFFECT_RESULT,
    HISTORY_APPENDED,
    HISTORY_REPLACED,
    JOURNAL_SCHEMA_VERSION,
    SESSION_CREATED,
    SESSION_UPDATED,
    JournalRecord,
    canonical_json,
)


class JournalWriterError(RuntimeError):
    """The journal cannot safely accept a write from this writer."""


class SessionJournalWriter:
    def __init__(self, path, *, sync=True):
        self.path = Path(path)
        self.lock_path = self.path.with_name(f"{self.path.name}.lock")
        self.sync = bool(sync)
        self.state = JournalState.empty()
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
        except Exception:
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

    def finish_effect(self, intent, *, outcome, result):
        try:
            return self._append_record(
                EFFECT_RESULT,
                {
                    "effect_type": intent.payload["effect_type"],
                    "call_id": intent.payload["call_id"],
                    "outcome": outcome,
                    "result": result,
                },
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
            next_state = reduce_journal_record(self.state, record)
            self._write_line(record)
            self.state = next_state
            return record

    def _write_line(self, record):
        payload = (canonical_json(record.to_dict()) + "\n").encode("utf-8")
        with self.path.open("ab") as journal:
            journal.write(payload)
            journal.flush()
            if self.sync:
                os.fsync(journal.fileno())

    def _acquire_file_lock(self):
        try:
            descriptor = os.open(
                self.lock_path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o600,
            )
        except FileExistsError as exc:
            raise JournalWriterError(
                f"session journal writer already active: {self.path}"
            ) from exc
        try:
            os.write(descriptor, self._owner_token.encode("ascii"))
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

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

    def complete(self, outcome, result=None):
        if self._completed:
            raise JournalWriterError("journal effect is already complete")
        self._completed = True
        return self.writer.finish_effect(
            self.intent,
            outcome=outcome,
            result=dict(result or {}),
        )

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
