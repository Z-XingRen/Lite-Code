"""Tool-call validation, authorization, execution, and evidence recording."""

from .runtime_journal import commit_tool_exchange, runtime_journal_effect
from .tool_execution import finish_tool_call, prepare_tool_call
from .tool_history import build_tool_history_item
from .workspace import now


def run_tool(agent, name, args, *, call_id=""):
    prepared = prepare_tool_call(agent, name, args, call_id=call_id)
    if not prepared.ready:
        agent._last_tool_result_metadata = prepared.metadata
        return prepared.result
    with runtime_journal_effect(
        agent,
        "tool",
        request={"name": prepared.tool.name, "args": prepared.args or {}},
        call_id=call_id or None,
    ) as effect:
        try:
            execution_result = prepared.tool.execute(prepared.args)
        except Exception as exc:
            result, metadata = finish_tool_call(agent, prepared, error=exc)
        else:
            result, metadata = finish_tool_call(
                agent, prepared, execution_result=execution_result
            )
        history_item = build_tool_history_item(
            prepared.name,
            prepared.args,
            result,
            str(call_id or ""),
            metadata,
            created_at=now(),
        )
        commit_tool_exchange(agent, effect, history_item, metadata)
    metadata = {
        **metadata,
        "journal_history_committed": True,
    }
    agent._last_tool_result_metadata = metadata
    return result
