"""Optional Runtime integration for journal-backed effects and history."""

from .session_journal_writer import SessionJournalWriter


def attach_runtime_journal(runtime, writer):
    if not isinstance(writer, SessionJournalWriter):
        raise TypeError("writer must be a SessionJournalWriter")
    journal_session = writer.state.session or {}
    if str(journal_session.get("id", "")) != str(runtime.session.get("id", "")):
        raise ValueError("journal session id does not match runtime session")
    if journal_session.get("history", []) != runtime.session.get("history", []):
        raise ValueError("journal history does not match runtime session")
    updates = {
        key: value
        for key, value in runtime.session.items()
        if key not in {"id", "history"} and journal_session.get(key) != value
    }
    if updates:
        writer.update_session(updates)
    runtime.session_journal_writer = writer
    runtime.session_path = writer.path
    return writer


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
