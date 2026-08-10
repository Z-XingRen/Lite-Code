"""Runtime integration for journal-backed effects and session state."""

import json
import uuid
import weakref

from .session_journal_writer import JournalWriterError, SessionJournalWriter
from .session_tree import project_branch_state


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
            writer.append_message(item)
    elif current_history != journal_history and replace_history:
        writer.append_compaction(
            current_history,
            metadata={"source": "runtime_history_replacement"},
            turn_id=str(getattr(runtime, "current_turn_id", "") or ""),
            run_id=str(getattr(runtime, "current_run_id", "") or ""),
        )
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


def runtime_tree_rows(runtime):
    writer = open_runtime_journal(runtime, migrate_legacy=True)
    return writer.tree_rows()


def move_runtime_tree_head(runtime, target, *, reason="branch"):
    writer = open_runtime_journal(runtime, migrate_legacy=True)
    entry_id = _resolve_tree_target(writer, target)
    writer.move_head(entry_id, reason=reason)
    runtime.session["history"] = _json_compatible(writer.state.session["history"])
    runtime.session_path = writer.path
    runtime._last_session_tree_warning = _workspace_drift_warning(runtime, writer)
    return entry_id


def rewind_runtime_tree(runtime, steps=1):
    writer = open_runtime_journal(runtime, migrate_legacy=True)
    steps = int(steps)
    if steps <= 0:
        raise ValueError("rewind steps must be positive")
    target = writer.state.tree.active_head
    for _ in range(steps):
        if target is None:
            break
        target = writer.state.tree.entries[target].parent_id
    writer.move_head(target, reason=f"rewind:{steps}")
    runtime.session["history"] = _json_compatible(writer.state.session["history"])
    runtime.session_path = writer.path
    runtime._last_session_tree_warning = _workspace_drift_warning(runtime, writer)
    return target


def label_runtime_tree_head(runtime, label):
    writer = open_runtime_journal(runtime, migrate_legacy=True)
    if writer.state.tree.active_head is None:
        raise ValueError("cannot label an empty session tree")
    return writer.label_head(label)


def _resolve_tree_target(writer, target):
    value = str(target or "").strip()
    if not value:
        raise ValueError("tree target is required")
    if value in writer.state.tree.labels:
        return writer.state.tree.labels[value]
    if value in writer.state.tree.entries:
        return value
    matches = [entry_id for entry_id in writer.state.tree.entries if entry_id.startswith(value)]
    if not matches:
        raise ValueError(f"tree entry not found: {value}")
    if len(matches) > 1:
        raise ValueError(f"tree entry prefix is ambiguous: {value}")
    return matches[0]


def _workspace_drift_warning(runtime, writer):
    checkpoint = project_branch_state(writer.state.tree).get("checkpoint")
    if not isinstance(checkpoint, dict):
        return ""
    identity = checkpoint.get("runtime_identity", {})
    expected = str(identity.get("workspace_fingerprint", "")) if isinstance(identity, dict) else ""
    if not expected:
        expected = str(checkpoint.get("workspace_fingerprint", ""))
    current = str(runtime.workspace.__class__.build(runtime.root).fingerprint())
    if expected and expected != current:
        return (
            "warning: session context moved, but workspace files were not restored "
            f"(checkpoint {expected[:12]}, current {current[:12]})"
        )
    return ""


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


def commit_tool_exchange(runtime, effect, history_item, metadata):
    """Commit effect completion and its model-visible tool observation together."""

    return commit_tool_batch_exchange(
        runtime,
        effect,
        [history_item],
        [metadata],
    )[0]


def commit_tool_batch_exchange(
    runtime, effect, history_items, metadata_items, *, outcome=None
):
    """Atomically commit one assistant tool-call batch and all ordered results."""

    history_items = list(history_items)
    metadata_items = list(metadata_items)
    if not history_items or len(history_items) != len(metadata_items):
        raise ValueError("tool exchange requires one metadata item per result")

    writer = getattr(runtime, "session_journal_writer", None)
    if writer is None or not hasattr(effect, "intent"):
        effect.complete(
            outcome
            or (
                "error"
                if any(item.get("tool_status") != "ok" for item in metadata_items)
                else "ok"
            ),
            runtime.redact_artifact(
                {
                    "results": [
                        {"content": item.get("content", ""), "metadata": metadata}
                        for item, metadata in zip(history_items, metadata_items)
                    ]
                }
            ),
        )
        for item in history_items:
            runtime.record(item)
        return history_items

    items = [runtime.turn_history.enrich(item) for item in history_items]
    fallback_call_id = str(effect.intent.payload["call_id"])
    for index, item in enumerate(items):
        call_id = str(item.get("call_id", "") or "")
        if not call_id and len(items) == 1:
            call_id = fallback_call_id
        if not call_id:
            raise ValueError(f"tool exchange result {index} is missing call_id")
        item["call_id"] = call_id
    persisted_items = [runtime.redact_artifact(item) for item in items]
    entry_id = f"entry_{uuid.uuid4().hex}"
    assistant = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "call_id": item["call_id"],
                "name": item.get("name", ""),
                "arguments": runtime.redact_artifact(item.get("args", {})),
            }
            for item in items
        ],
        "turn_id": items[0].get("turn_id", ""),
        "run_id": items[0].get("run_id", ""),
        "created_at": items[0].get("created_at", ""),
    }
    tree_delta = {
        "expected_head": writer.state.tree.active_head,
        "entries": [
            {
                "entry_id": entry_id,
                "parent_id": writer.state.tree.active_head,
                "entry_type": "tool_exchange",
                "turn_id": items[0].get("turn_id", ""),
                "run_id": items[0].get("run_id", ""),
                "created_at": items[0].get("created_at", ""),
                "data": {"assistant": assistant, "results": persisted_items},
            }
        ],
    }
    effect.complete(
        outcome
        or (
            "error"
            if any(item.get("tool_status") != "ok" for item in metadata_items)
            else "ok"
        ),
        runtime.redact_artifact(
            {
                "results": [
                    {"content": item.get("content", ""), "metadata": metadata}
                    for item, metadata in zip(items, metadata_items)
                ]
            }
        ),
        tree_delta=tree_delta,
    )
    runtime.session["history"] = _json_compatible(writer.state.session["history"])
    runtime.session_path = writer.path
    runtime._session_journal_dirty = False
    return items


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
