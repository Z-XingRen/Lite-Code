"""Tool-call validation, authorization, execution, and evidence recording."""

from .runtime_journal import run_tool_effect
from .tool_execution import finish_tool_call, prepare_tool_call


def run_tool(agent, name, args, *, call_id=""):
    prepared = prepare_tool_call(agent, name, args, call_id=call_id)
    if not prepared.ready:
        agent._last_tool_result_metadata = prepared.metadata
        return prepared.result
    try:
        execution_result = run_tool_effect(
            agent, prepared.tool, prepared.args, call_id=call_id or None
        )
    except Exception as exc:
        result, metadata = finish_tool_call(agent, prepared, error=exc)
    else:
        result, metadata = finish_tool_call(
            agent, prepared, execution_result=execution_result
        )
    agent._last_tool_result_metadata = metadata
    return result
