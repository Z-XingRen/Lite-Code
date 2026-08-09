"""Conservative parallel execution for explicitly safe read-only tool batches."""

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from ..cancellation import CancellationRequested
from .runtime_journal import runtime_journal_effect, synchronize_runtime_session
from .tool_execution import finish_tool_call, prepare_tool_call


@dataclass(frozen=True)
class ToolBatchOutcome:
    result: str
    metadata: dict


def parallel_batch_eligible(agent, calls):
    calls = tuple(calls)
    if len(calls) < 2:
        return False
    identities = []
    for call in calls:
        tool = agent.tools.get(call.name)
        if (
            tool is None
            or tool.risky
            or tool.execution_mode != "parallel"
            or tool.effect_class != "read_only"
        ):
            return False
        identities.append(
            (
                call.name,
                json.dumps(
                    call.arguments,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    default=str,
                ),
            )
        )
    return len(identities) == len(set(identities))


def execute_parallel_tool_batch(agent, calls):
    """Execute effects concurrently and return finalized outcomes in source order."""

    calls = tuple(calls)
    prepared = [
        prepare_tool_call(
            agent,
            call.name,
            call.arguments,
            call_id=call.call_id,
        )
        for call in calls
    ]
    runnable = [item for item in prepared if item.ready]
    executions = {}
    agent._pending_tool_result_metadata = {}
    if runnable:
        with runtime_journal_effect(
            agent,
            "tool",
            request={
                "calls": [
                    {
                        "call_id": item.call_id,
                        "name": item.name,
                        "args": item.args,
                    }
                    for item in runnable
                ]
            },
            call_id=f"batch:{runnable[0].call_id}",
        ) as effect:
            executions = _run_concurrently(agent, runnable)
            outcome = _batch_outcome(agent, executions)
            effect.complete(
                outcome,
                agent.redact_artifact(
                    {
                        "results": [
                            _journal_result(item, executions[item.call_id])
                            for item in runnable
                        ]
                    }
                ),
            )
        _synchronize_dirty_session(agent)

    outcomes = []
    for item in prepared:
        if not item.ready:
            outcomes.append(ToolBatchOutcome(item.result, item.metadata))
            continue
        execution_result, error = executions[item.call_id]
        result, metadata = finish_tool_call(
            agent,
            item,
            execution_result=execution_result,
            error=error,
            consume_pending=False,
        )
        outcomes.append(ToolBatchOutcome(result, metadata))
    agent._pending_tool_result_metadata = {}
    return tuple(outcomes)


def _run_concurrently(agent, prepared):
    worker_count = min(len(prepared), 8)
    with ThreadPoolExecutor(
        max_workers=worker_count,
        thread_name_prefix="lite-tool-",
    ) as executor:
        futures = {
            item.call_id: executor.submit(_invoke, agent, item) for item in prepared
        }
        return {call_id: future.result() for call_id, future in futures.items()}


def _invoke(agent, prepared):
    try:
        agent.current_cancellation_token.raise_if_cancelled()
        return prepared.tool.execute(prepared.args), None
    except Exception as exc:
        return None, exc


def _batch_outcome(agent, executions):
    if agent.abort_requested or any(
        isinstance(error, CancellationRequested) for _, error in executions.values()
    ):
        return "interrupted"
    if any(
        error is not None or result.is_error for result, error in executions.values()
    ):
        return "error"
    return "ok"


def _journal_result(prepared, execution):
    result, error = execution
    return {
        "call_id": prepared.call_id,
        "name": prepared.name,
        "content": result.content if result is not None else "",
        "is_error": bool(error is not None or (result and result.is_error)),
        "error_type": type(error).__name__ if error is not None else "",
    }


def _synchronize_dirty_session(agent):
    writer = getattr(agent, "session_journal_writer", None)
    if (
        getattr(agent, "_session_journal_dirty", False)
        and writer is not None
        and writer.state.open_operation is None
    ):
        synchronize_runtime_session(agent)
