"""Runtime integration for journal-backed effects and session state."""

import json
import weakref

from .session_journal_writer import JournalWriterError, SessionJournalWriter


def open_runtime_journal(runtime, *, migrate_legacy=False):
    """Open the journal authority for a runtime without rewriting legacy JSON."""

    writer = getattr(runtime, "session_journal_writer", None)
    if writer is not None:
        return writer
    session_id = runtime.session["id"]
    if runtime.session_store.session_authority(session_id) != "journal":
        if not migrate_legacy:
            return None
        runtime.session_store.migrate_session(session_id)
    with runtime.session_store._lock:
        _claim_runtime_owner(runtime, session_id)
        writer = SessionJournalWriter.open(
            runtime.session_store.journal_path(session_id)
        )
        runtime.session_journal_writer = writer
        runtime.session_path = writer.path
        runtime.session_store._runtime_journal_owners[session_id] = weakref.ref(runtime)
    return writer


def _claim_runtime_owner(runtime, session_id):
    owners = getattr(runtime.session_store, "_runtime_journal_owners", None)
    if owners is None:
        owners = {}
        runtime.session_store._runtime_journal_owners = owners
    owner_ref = owners.get(session_id)
    owner = owner_ref() if owner_ref is not None else None
    if owner is None or owner is runtime:
        return
    writer = getattr(owner, "session_journal_writer", None)
    if writer is not None:
        task_state = getattr(owner, "current_task_state", None)
        if (
            writer.state.open_operation is not None
            or getattr(task_state, "status", "") == "running"
        ):
            raise JournalWriterError(
                f"runtime session is still active: {session_id}"
            )
        writer.close()
        owner.session_journal_writer = None


def synchronize_runtime_session(runtime, *, replace_history=False):
    """Append the runtime's current state to its already-open journal."""

    writer = open_runtime_journal(runtime, migrate_legacy=True)
    if writer is None:
        raise JournalWriterError("journal writer could not be opened")
    if writer.state.open_operation is not None:
        runtime._session_journal_dirty = True
        return writer.path

    journal_session = writer.state.session or {}
    current_session = _json_compatible(runtime.session)
    current_history = list(current_session.get("history", []))
    journal_history = list(journal_session.get("history", []))
    if current_history[: len(journal_history)] == journal_history:
        for item in current_history[len(journal_history) :]:
            writer.append_history(item)
    elif current_history != journal_history and replace_history:
        writer.replace_history(current_history)
    elif current_history != journal_history:
        raise ValueError("runtime history diverged from journal authority")

    protected = {"id", "history"}
    updates = {
        key: _json_compatible(value)
        for key, value in current_session.items()
        if key not in protected and journal_session.get(key) != value
    }
    if updates:
        writer.update_session(updates)
    runtime._session_journal_dirty = False
    runtime.session_path = writer.path
    return writer.path


def attach_runtime_journal(runtime, writer):
    if not isinstance(writer, SessionJournalWriter):
        raise TypeError("writer must be a SessionJournalWriter")
    journal_session = writer.state.session or {}
    current_session = _json_compatible(runtime.session)
    if str(journal_session.get("id", "")) != str(runtime.session.get("id", "")):
        raise ValueError("journal session id does not match runtime session")
    if journal_session.get("history", []) != current_session.get("history", []):
        raise ValueError("journal history does not match runtime session")
    updates = {
        key: _json_compatible(value)
        for key, value in current_session.items()
        if key not in {"id", "history"} and journal_session.get(key) != value
    }
    if updates:
        writer.update_session(updates)
    runtime.session_journal_writer = writer
    runtime.session_path = writer.path
    return writer


def _json_compatible(value):
    return json.loads(
        json.dumps(value, ensure_ascii=False, allow_nan=False)
    )


class NullJournalEffect:
    def complete(self, _outcome, _result=None):
        return None

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _traceback):
        return False


def runtime_journal_effect(
    runtime,
    effect_type,
    *,
    request,
    call_id=None,
    replay_policy="interrupt",
):
    writer = getattr(runtime, "session_journal_writer", None)
    if writer is None:
        writer = open_runtime_journal(runtime, migrate_legacy=True)
    if writer is None:
        return NullJournalEffect()
    return writer.effect(
        effect_type,
        call_id=call_id,
        request=runtime.redact_artifact(request),
        replay_policy=replay_policy,
    )


def run_permission_effect(runtime, tool, args, decide, *, call_id=None):
    with runtime_journal_effect(
        runtime,
        "permission",
        request={"tool_name": tool.name, "args": args or {}},
        call_id=call_id,
    ) as effect:
        decision = decide()
        effect.complete(
            "ok",
            runtime.redact_artifact(
                {
                    "decision": decision.decision,
                    "reason": decision.reason,
                    "security_event_type": decision.security_event_type,
                }
            ),
        )
        return decision


def run_tool_effect(runtime, tool, args, *, call_id=None):
    try:
        with runtime_journal_effect(
            runtime,
            "tool",
            request={"name": tool.name, "args": args or {}},
            call_id=call_id,
        ) as effect:
            result = tool.execute(args)
            effect.complete(
                "error" if result.is_error else "ok",
                runtime.redact_artifact(
                    {"content": result.content, "is_error": result.is_error}
                ),
            )
    finally:
        writer = getattr(runtime, "session_journal_writer", None)
        if (
            getattr(runtime, "_session_journal_dirty", False)
            and writer is not None
            and writer.state.open_operation is None
        ):
            synchronize_runtime_session(runtime)
    return result


def model_result_payload(runtime, result):
    return runtime.redact_artifact(
        {
            "text": result.text,
            "stop_reason": result.stop_reason,
            "tool_calls": [
                {
                    "call_id": call.call_id,
                    "name": call.name,
                    "arguments": call.arguments,
                }
                for call in result.tool_calls
            ],
            "continuation": list(result.continuation or ()),
            "metadata": dict(result.metadata or {}),
        }
    )
