"""Control-loop tool and retry helpers shared by Engine.

These helpers execute tool payloads and handle retry summaries while Engine
keeps the turn loop shape visible. Terminal-state policy lives in
completion_governance.
"""

import time

from ..providers.base import (
    ToolOutput,
    complete_model,
    is_truncated_stop_reason,
    normalize_stop_reason,
)
from ..providers.errors import ProviderError
from ..providers.streaming import ModelStreamProtocolError
from .turn_transitions import (
    CONTINUE_EMPTY_RESPONSE_RETRY,
    emit_continue_transition,
)
from .tool_batch_scheduler import (
    execute_parallel_tool_batch,
    parallel_batch_eligible,
)
from .tool_history import build_tool_history_item
from .workspace import clip, now


def model_result_kind(result):
    if result.tool_calls:
        return "tool" if len(result.tool_calls) == 1 else "tools"
    return "final" if (result.text or "").strip() else "retry"


def execute_tool_payload(engine, task_state, user_message, payload):
    agent = engine.runtime
    name = payload.get("name", "")
    args = payload.get("args", {})
    call_id = str(payload.get("call_id", "") or "")
    tool_started_at, event = start_tool_payload(agent, task_state, name, args, call_id)
    yield event
    tool_result = agent.run_tool(name, args, call_id=call_id)
    tool_metadata = dict(agent._last_tool_result_metadata or {})
    return (
        yield from commit_tool_payload(
            engine,
            task_state,
            user_message,
            name,
            args,
            call_id,
            tool_started_at,
            tool_result,
            tool_metadata,
        )
    )


def start_tool_payload(agent, task_state, name, args, call_id):
    task_state.record_tool(name)
    tool_started_at = time.monotonic()
    tool = getattr(agent, "tools", {}).get(name)
    defer_projection = bool(
        agent.feature_enabled("journal_checkpoint_policy")
        and getattr(tool, "read_only", False)
    )
    if not defer_projection:
        agent.session_event_bus.emit(
            "tool_started", {"run_id": task_state.run_id, "tool_name": name, "args": args}
        )
    starts = getattr(agent, "_tool_persistence_starts", None)
    if starts is None:
        starts = agent._tool_persistence_starts = {}
    starts[str(call_id or name)] = agent.persistence_write_count()
    return tool_started_at, {
        "type": "tool_call",
        "run_id": task_state.run_id,
        "call_id": call_id,
        "name": name,
        "args": args,
    }


def commit_tool_payload(
    engine,
    task_state,
    user_message,
    name,
    args,
    call_id,
    tool_started_at,
    tool_result,
    tool_metadata,
):
    agent = engine.runtime
    tool_duration_ms = int((time.monotonic() - tool_started_at) * 1000)
    if not (
        agent.feature_enabled("journal_checkpoint_policy")
        and tool_metadata.get("read_only")
        and tool_metadata.get("tool_status") == "ok"
    ):
        agent.session_event_bus.emit(
            "tool_finished",
            {
                "run_id": task_state.run_id,
                "tool_name": name,
                "status": tool_metadata.get("tool_status", ""),
                "tool_error_code": tool_metadata.get("tool_error_code", ""),
                "workspace_changed": bool(tool_metadata.get("workspace_changed", False)),
                "affected_paths": list(tool_metadata.get("affected_paths", [])),
                "duration_ms": tool_duration_ms,
            },
        )
    history_item = build_tool_history_item(
        name, args, tool_result, call_id, tool_metadata, created_at=now()
    )
    if not tool_metadata.get("journal_history_committed"):
        agent.record(history_item)
    notifications = engine.drain_worker_notifications()
    for notification in notifications:
        yield {
            "type": "worker_notification",
            "run_id": getattr(agent, "current_run_id", ""),
            "content": notification,
        }
    deferred_governance = agent.current_task_state.evidence_summaries.pop(
        "deferred_governance_decisions", []
    )
    if deferred_governance:
        if tool_metadata.get("tool_status") == "ok":
            agent.emit_trace(
                task_state,
                "governance_batch",
                {"decisions": deferred_governance, "tool_name": name},
                persist_state=False,
            )
        else:
            for decision in deferred_governance:
                agent.emit_trace(
                    task_state,
                    "governance_decision",
                    {**decision, "tool_name": name},
                    persist_state=False,
                )
    checkpoint = None
    workspace_changed = bool(tool_metadata.get("workspace_changed"))
    successful_change = tool_metadata.get("tool_status") == "ok"
    if (
        not agent.feature_enabled("journal_checkpoint_policy")
        or (workspace_changed and successful_change)
    ):
        checkpoint = agent.create_checkpoint(
            task_state, user_message, trigger="tool_executed"
        )
    persistence_start = getattr(agent, "_tool_persistence_starts", {}).pop(
        str(call_id or name), agent.persistence_write_count()
    )
    trace_write_cost = 1 if (
        agent.feature_enabled("journal_checkpoint_policy")
        and tool_metadata.get("read_only")
    ) else 2
    tool_metadata["persistence_write_count"] = max(
        0, agent.persistence_write_count() - persistence_start + trace_write_cost
    )
    agent.emit_trace(
        task_state,
        "tool_executed",
        {
            "name": name,
            "args": args,
            "result": clip(tool_result, 500),
            "duration_ms": tool_duration_ms,
            **tool_metadata,
        },
        persist_state=not (
            agent.feature_enabled("journal_checkpoint_policy")
            and tool_metadata.get("read_only")
        ),
    )
    if checkpoint is not None:
        agent.emit_trace(
            task_state,
            "checkpoint_created",
            {"checkpoint_id": checkpoint["checkpoint_id"], "trigger": "tool_executed"},
        )
    yield {
        "type": "tool_result",
        "run_id": task_state.run_id,
        "call_id": call_id,
        "name": name,
        "content": tool_result,
        "metadata": tool_metadata,
    }
    return (
        ToolOutput(
            call_id=call_id,
            name=name,
            content=str(tool_result),
            is_error=str(tool_metadata.get("tool_status", "")) != "ok",
        ),
        tuple(notifications),
        completion_contract_feedback(tool_metadata),
    )


def completion_contract_feedback(tool_metadata):
    if (
        tool_metadata.get("tool_status") == "ok"
        and tool_metadata.get("workspace_changed") is True
        and not tool_metadata.get("verification_receipt")
    ):
        return (
            "Completion contract: the workspace changed. Run focused verification "
            "before returning the final answer."
        )
    return ""


def execute_native_tool_calls(
    engine, task_state, user_message, conversation, result, tool_steps
):
    """Execute one Provider-native call batch and attach its outputs."""
    agent = engine.runtime
    calls = list(result.tool_calls)
    if is_truncated_stop_reason(result.stop_reason):
        rejected = yield from reject_truncated_tool_calls(
            engine, task_state, conversation, result
        )
        return tool_steps, 0, rejected
    if len(calls) <= agent.max_steps - tool_steps and parallel_batch_eligible(
        agent, calls
    ):
        return (
            yield from execute_parallel_native_tool_calls(
                engine,
                task_state,
                user_message,
                conversation,
                result,
                calls,
                tool_steps,
            )
        )
    outputs = []
    feedback = []
    executed = 0
    for call in calls:
        if tool_steps >= agent.max_steps:
            outputs.append(
                ToolOutput(
                    call_id=call.call_id,
                    name=call.name,
                    content=(
                        "error: tool call not executed because the per-turn "
                        "tool budget was exhausted"
                    ),
                    is_error=True,
                )
            )
            continue
        output, notifications, contract_feedback = yield from execute_tool_payload(
            engine,
            task_state,
            user_message,
            {"call_id": call.call_id, "name": call.name, "args": call.arguments},
        )
        outputs.append(output)
        feedback.extend(notifications)
        if contract_feedback:
            feedback.append(contract_feedback)
        tool_steps += 1
        executed += 1
        if agent.abort_requested:
            break
    if not agent.abort_requested:
        conversation.append_result(result, tool_outputs=outputs, feedback=feedback)
    return tool_steps, executed, len(calls)


def execute_parallel_native_tool_calls(
    engine,
    task_state,
    user_message,
    conversation,
    result,
    calls,
    tool_steps,
):
    agent = engine.runtime
    started = []
    for call in calls:
        tool_started_at, event = start_tool_payload(
            agent,
            task_state,
            call.name,
            call.arguments,
            call.call_id,
        )
        started.append(tool_started_at)
        yield event
    outcomes = execute_parallel_tool_batch(agent, calls)
    outputs = []
    feedback = []
    for call, tool_started_at, outcome in zip(calls, started, outcomes):
        agent._last_tool_result_metadata = outcome.metadata
        output, notifications, contract_feedback = yield from commit_tool_payload(
            engine,
            task_state,
            user_message,
            call.name,
            call.arguments,
            call.call_id,
            tool_started_at,
            outcome.result,
            outcome.metadata,
        )
        outputs.append(output)
        feedback.extend(notifications)
        if contract_feedback:
            feedback.append(contract_feedback)
    tool_steps += len(calls)
    if not agent.abort_requested:
        conversation.append_result(result, tool_outputs=outputs, feedback=feedback)
    return tool_steps, len(calls), len(calls)


def reject_truncated_tool_calls(engine, task_state, conversation, result):
    """Pair truncated calls with ordered errors without entering the executor."""
    agent = engine.runtime
    stop_reason = normalize_stop_reason(result.stop_reason)
    outputs = []
    for call in result.tool_calls:
        content = (
            "error: tool call not executed because the model response was "
            f"truncated (stop reason: {stop_reason}); resend the complete tool "
            "call with all arguments"
        )
        metadata = {
            "tool_status": "rejected",
            "tool_error_code": "truncated_tool_call",
            "workspace_changed": False,
            "affected_paths": [],
            "synthetic": True,
            "stop_reason": stop_reason,
        }
        agent.record(
            {
                "role": "tool",
                "name": call.name,
                "args": call.arguments,
                "content": content,
                "call_id": call.call_id,
                "created_at": now(),
                **metadata,
            }
        )
        outputs.append(
            ToolOutput(
                call_id=call.call_id,
                name=call.name,
                content=content,
                is_error=True,
            )
        )
        yield {
            "type": "tool_result",
            "run_id": task_state.run_id,
            "call_id": call.call_id,
            "name": call.name,
            "content": content,
            "metadata": metadata,
        }
    conversation.append_result(result, tool_outputs=outputs)
    return len(outputs)


def emit_empty_result_retry(engine, task_state):
    agent = engine.runtime
    payload = "Model returned an empty response: neither text nor native tool calls."
    agent.record({"role": "assistant", "content": payload, "created_at": now()})
    agent.session_event_bus.emit(
        "assistant_message",
        {
            "run_id": task_state.run_id,
            "kind": "retry",
            "content": clip(payload, 500),
        },
    )
    agent.run_store.write_task_state(task_state)
    yield {"type": "retry", "run_id": task_state.run_id, "content": payload}
    emit_continue_transition(agent, task_state, CONTINUE_EMPTY_RESPONSE_RETRY)


def should_retry_model_error(exc, provider_retries):
    if isinstance(exc, ModelStreamProtocolError):
        code = type(exc).__name__
        return provider_retries.get(code, 0) < 1
    if not isinstance(exc, ProviderError):
        return False
    code = str(getattr(exc, "code", "") or "")
    if code not in {"empty_response"}:
        return False
    return provider_retries.get(code, 0) < 1


_STEP_LIMIT_SUMMARY_NOTICE = (
    "You have hit the per-turn tool budget (max_steps). Do not call any more tools. "
    "Return a concise answer in the user's language that "
    "briefly covers: (1) what you accomplished this turn, (2) what remains undone, "
    "(3) how the user can continue (e.g., `/resume` then `继续`). Keep it concise."
)


def request_step_limit_summary(engine, task_state, user_message):
    """Ask the model to write a graceful step-limit summary.

    Returns the final text, or None if the model fails or refuses to comply.
    Side effects: emits a trace event but does NOT mutate session history —
    the caller decides whether to record the resulting final.
    """
    agent = engine.runtime
    started_at = time.monotonic()
    try:
        prompt, _ = agent._build_prompt_and_metadata(_STEP_LIMIT_SUMMARY_NOTICE)
        result = complete_model(
            agent.model_client, prompt, agent.max_new_tokens
        )
    except Exception as exc:
        agent.emit_trace(
            task_state,
            "step_limit_summary_failed",
            {"error": clip(str(exc), 200)},
        )
        return None
    summary = (result.text or "").strip() if result else ""
    duration_ms = int((time.monotonic() - started_at) * 1000)
    agent.emit_trace(
        task_state,
        "step_limit_summary",
        {
            "kind": "final" if summary else "empty",
            "duration_ms": duration_ms,
            "produced": bool(summary),
        },
    )
    return summary or None
