"""Conservative parallel execution for explicitly safe read-only tool batches."""

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from ..cancellation import CancellationRequested
from .runtime_journal import commit_tool_batch_exchange, runtime_journal_effect
from .tool_execution import finish_tool_call, prepare_tool_call
from .tool_history import build_tool_history_item
from .workspace import now


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
    finalized = {}
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
            history_items = []
            metadata_items = []
            for item in runnable:
                execution_result, error = executions[item.call_id]
                result, metadata = finish_tool_call(
                    agent,
                    item,
                    execution_result=execution_result,
                    error=error,
                    consume_pending=False,
                )
                history_items.append(
                    build_tool_history_item(
                        item.name,
                        item.args,
                        result,
                        item.call_id,
                        metadata,
                        created_at=now(),
                    )
                )
                metadata_items.append(metadata)
                finalized[item.call_id] = ToolBatchOutcome(
                    result,
                    {**metadata, "journal_history_committed": True},
                )
            commit_tool_batch_exchange(
                agent,
                effect,
                history_items,
                metadata_items,
                outcome=outcome,
            )

    outcomes = []
    for item in prepared:
        if not item.ready:
            outcomes.append(ToolBatchOutcome(item.result, item.metadata))
            continue
        outcomes.append(finalized[item.call_id])
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
