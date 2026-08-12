"""Turn-level control loop for model calls, tools, and user-visible events."""

import time

from ..features import memory as memorylib
from .completion_governance import (
    final_readiness_action,
    finish_limited_run,
    finish_stopped_run,
    finish_successful_run,
)
from .context_replacements import commit_proposed_replacements
from .model_errors import finish_model_error
from .model_streaming import consume_model_stream
from .request_context import finish_request_context_error, rebuild_model_request
from .engine_helpers import (
    emit_empty_result_retry,
    execute_native_tool_calls,
    model_result_kind,
    request_step_limit_summary,
    should_retry_model_error,
)
from .task_state import STOP_REASON_FINAL_GATE_BLOCKED, TaskState
from .turn_transitions import (
    CONTINUE_FINAL_READINESS_NOTICE,
    CONTINUE_PLAN_NOTICE,
    CONTINUE_PROVIDER_RETRY,
    CONTINUE_TOOL_BATCH_EXECUTED,
    emit_continue_transition,
)
from .workspace import clip, now

CHECKPOINT_NONE_STATUS = "no-checkpoint"
CHECKPOINT_PARTIAL_STALE_STATUS = "partial-stale"
CHECKPOINT_WORKSPACE_MISMATCH_STATUS = "workspace-mismatch"
FINAL_ONLY_NOTICE = (
    "The per-turn tool budget is exhausted. No tools are available for this request. "
    "Return the best accurate final answer now, including verification limitations "
    "or remaining work when applicable."
)


class Engine:
    def __init__(self, runtime):
        self.runtime = runtime

    def ask(self, user_message):
        final_answer = ""
        for event in self.run_turn(user_message):
            if event["type"] in {"final", "stop"}:
                final_answer = event["content"]
        return final_answer

    def drain_worker_notifications(self):
        agent = self.runtime
        notifications = agent.worker_manager.drain_notifications()
        for notification in notifications:
            agent.record({"role": "user", "content": notification, "created_at": now()})
            agent.session_event_bus.emit(
                "worker_notification_drained",
                {
                    "run_id": getattr(agent, "current_run_id", ""),
                    "content": clip(notification, 500),
                },
            )
        return notifications

    def _drain_worker_notification_events(self):
        notifications = self.drain_worker_notifications()
        for notification in notifications:
            yield {
                "type": "worker_notification",
                "run_id": getattr(self.runtime, "current_run_id", ""),
                "content": notification,
            }
        return tuple(notifications)

    def run_turn(self, user_message):
        agent = self.runtime
        run_started_at = time.monotonic()
        task_state = TaskState.create(
            run_id=agent.new_run_id(),
            task_id=agent.new_task_id(),
            user_request=user_message,
        )
        task_state.resume_status = agent.resume_state.get(
            "status", CHECKPOINT_NONE_STATUS
        )
        agent.current_task_state = task_state
        agent.current_turn_id = task_state.task_id
        agent.current_run_id = task_state.run_id
        agent.start_cancellation_scope()
        agent.current_run_dir = agent.run_store.start_run(task_state)
        agent.session_event_bus.emit(
            "turn_started",
            {
                "run_id": task_state.run_id,
                "task_id": task_state.task_id,
                "runtime_mode": agent.runtime_mode,
            },
        )
        yield {
            "type": "turn_started",
            "run_id": task_state.run_id,
            "task_id": task_state.task_id,
        }

        agent.memory.set_task_summary(user_message)
        turn_base_history = list(agent.session.get("history", []))
        agent.record({"role": "user", "content": user_message, "created_at": now()})
        agent.session_event_bus.emit(
            "user_message",
            {"run_id": task_state.run_id, "content": clip(user_message, 300)},
        )
        agent.emit_trace(
            task_state,
            "run_started",
            {
                "task_id": task_state.task_id,
                "user_request": clip(user_message, 300),
            },
        )
        agent.start_turn_context(user_message, turn_base_history)

        tool_steps = 0
        attempts = 0
        provider_retries = {}
        conversation = None
        final_only_attempted = False
        # 不放大 attempts，避免出现"看不见的隐形重试"——失败必须被用户察觉。
        max_attempts = agent.max_steps + 2

        while attempts < max_attempts and (
            tool_steps < agent.max_steps or not final_only_attempted
        ):
            final_only = tool_steps >= agent.max_steps
            if final_only:
                final_only_attempted = True
            if agent.abort_requested:
                yield from finish_stopped_run(
                    self,
                    task_state,
                    user_message,
                    "Stopped after abort request.",
                    "aborted",
                    run_started_at,
                )
                return
            notifications = yield from self._drain_worker_notification_events()
            if conversation is not None and notifications:
                conversation.add_feedback(*notifications)
            if final_only and conversation is not None:
                conversation.add_feedback(FINAL_ONLY_NOTICE)
            attempts += 1
            task_state.record_attempt()
            agent.run_store.write_task_state(task_state)
            prompt_started_at = time.monotonic()
            prompt, prompt_metadata = agent._build_prompt_and_metadata(user_message)
            try:
                conversation, request_context = rebuild_model_request(
                    agent,
                    prompt,
                    conversation,
                    tools=() if final_only else None,
                )
            except Exception as exc:
                yield from finish_request_context_error(
                    self, task_state, user_message, prompt_metadata, exc,
                    prompt_started_at, run_started_at,
                )
                return
            prompt_metadata["request_context"] = request_context
            prompt_metadata.update(
                {
                    key: request_context[key]
                    for key in (
                        "base_context_hash",
                        "session_projection_event_count",
                        "current_turn_delta_count",
                        "provider_turn_count",
                        "duplicate_tool_result_count",
                        "assembled_input_chars",
                        "estimated_input_tokens",
                        "context_source",
                    )
                    if key in request_context
                }
            )
            prompt_metadata["final_only"] = final_only
            if commit_proposed_replacements(agent.session, prompt_metadata):
                agent.persist_session(replace_history=True)
            agent.emit_trace(
                task_state,
                "prompt_built",
                {
                    "prompt_metadata": prompt_metadata,
                    "duration_ms": int((time.monotonic() - prompt_started_at) * 1000),
                },
            )
            structured_memory = getattr(getattr(agent, "memory", None), "last_retrieval", None)
            if structured_memory is not None:
                agent.emit_trace(
                    task_state,
                    "memory.retrieval",
                    {
                        "query_hash": structured_memory.get("query_hash", ""),
                        "selected": list(structured_memory.get("selected", [])),
                        "rejected": list(structured_memory.get("rejected", [])),
                        "workspace_fingerprint": memorylib.workspace_fingerprint(agent.root),
                    },
                )
            for file_read in memorylib.memory_file_read_payloads(agent.memory_dir, agent.root, reason="retrieval"):
                agent.emit_trace(task_state, "memory.file_read", file_read)
            if prompt_metadata.get("resume_status") == CHECKPOINT_PARTIAL_STALE_STATUS:
                checkpoint = agent.create_checkpoint(
                    task_state, user_message, trigger="freshness_mismatch"
                )
                agent.run_store.write_task_state(task_state)
                agent.emit_trace(
                    task_state,
                    "checkpoint_created",
                    {
                        "checkpoint_id": checkpoint["checkpoint_id"],
                        "trigger": "freshness_mismatch",
                    },
                )
            elif (
                prompt_metadata.get("resume_status")
                == CHECKPOINT_WORKSPACE_MISMATCH_STATUS
            ):
                agent.emit_trace(
                    task_state,
                    "runtime_identity_mismatch",
                    {
                        "fields": list(
                            prompt_metadata.get("runtime_identity_mismatch_fields", [])
                        ),
                    },
                )
                checkpoint = agent.create_checkpoint(
                    task_state, user_message, trigger="workspace_mismatch"
                )
                agent.run_store.write_task_state(task_state)
                agent.emit_trace(
                    task_state,
                    "checkpoint_created",
                    {
                        "checkpoint_id": checkpoint["checkpoint_id"],
                        "trigger": "workspace_mismatch",
                    },
                )
            if prompt_metadata.get("budget_reductions") or (
                prompt_metadata.get("pressure", {}).get("tier", "tier0_observe") != "tier0_observe"
            ):
                checkpoint = agent.create_checkpoint(
                    task_state, user_message, trigger="context_reduction"
                )
                agent.run_store.write_task_state(task_state)
                agent.emit_trace(
                    task_state,
                    "checkpoint_created",
                    {
                        "checkpoint_id": checkpoint["checkpoint_id"],
                        "trigger": "context_reduction",
                    },
                )
            agent.emit_trace(
                task_state,
                "model_requested",
                {
                    "attempts": task_state.attempts,
                    "tool_steps": task_state.tool_steps,
                    "final_only": final_only,
                    "prompt_cache_key": prompt_metadata.get("prompt_cache_key"),
                },
            )
            agent.session_event_bus.emit(
                "model_requested",
                {
                    "run_id": task_state.run_id,
                    "attempts": task_state.attempts,
                    "tool_steps": task_state.tool_steps,
                    "final_only": final_only,
                },
            )
            yield {
                "type": "model_requested",
                "run_id": task_state.run_id,
                "attempts": task_state.attempts,
                "tool_steps": task_state.tool_steps,
                "final_only": final_only,
            }

            prompt_cache_key = None
            prompt_cache_retention = None
            prompt_cache_prefix_chars = None
            if getattr(agent.model_client, "supports_prompt_cache", False):
                prompt_cache_key = prompt_metadata.get("prompt_cache_key")
                prompt_cache_prefix_chars = (
                    prompt_metadata.get("sections", {})
                    .get("prefix", {})
                    .get("rendered_chars")
                )

            model_started_at = time.monotonic()
            try:
                result = yield from consume_model_stream(
                    self, task_state, conversation, agent.max_new_tokens,
                    prompt_cache_key=prompt_cache_key,
                    prompt_cache_retention=prompt_cache_retention,
                    prompt_cache_prefix_chars=prompt_cache_prefix_chars,
                )
            except Exception as exc:
                if agent.abort_requested:
                    yield from finish_stopped_run(
                        self,
                        task_state,
                        user_message,
                        "Stopped after abort request.",
                        "aborted",
                        run_started_at,
                    )
                    return
                if should_retry_model_error(exc, provider_retries):
                    code = getattr(exc, "code", type(exc).__name__)
                    provider_retries[code] = provider_retries.get(code, 0) + 1
                    agent.session_event_bus.emit(
                        "model_retry_scheduled",
                        {
                            "run_id": task_state.run_id,
                            "code": code,
                            "attempts": task_state.attempts,
                            "retry_count": provider_retries[code],
                        },
                    )
                    agent.emit_trace(
                        task_state,
                        "model_retry_scheduled",
                        {
                            "code": code,
                            "duration_ms": int(
                                (time.monotonic() - model_started_at) * 1000
                            ),
                            "retry_count": provider_retries[code],
                        },
                    )
                    emit_continue_transition(agent, task_state, CONTINUE_PROVIDER_RETRY)
                    continue
                yield from finish_model_error(
                    self,
                    task_state,
                    user_message,
                    prompt_metadata,
                    exc,
                    int((time.monotonic() - model_started_at) * 1000),
                    int((time.monotonic() - run_started_at) * 1000),
                )
                return
            if agent.abort_requested:
                yield from finish_stopped_run(
                    self,
                    task_state,
                    user_message,
                    "Stopped after abort request.",
                    "aborted",
                    run_started_at,
                )
                return
            raw = result.text or ""
            completion_metadata = dict(
                result.metadata
                or getattr(agent.model_client, "last_completion_metadata", {})
                or {}
            )
            if completion_metadata:
                prompt_metadata.update(completion_metadata)
            agent.last_completion_metadata = completion_metadata
            agent.last_prompt_metadata = prompt_metadata
            kind = model_result_kind(result)
            duration_ms = int((time.monotonic() - model_started_at) * 1000)
            agent.emit_trace(
                task_state,
                "model_parsed",
                {
                    "kind": kind,
                    "completion_metadata": completion_metadata,
                    "duration_ms": duration_ms,
                },
            )
            agent.session_event_bus.emit(
                "model_parsed",
                {"run_id": task_state.run_id, "kind": kind, "duration_ms": duration_ms},
            )
            yield {
                "type": "model_parsed",
                "run_id": task_state.run_id,
                "kind": kind,
                "duration_ms": duration_ms,
            }

            if kind in {"tool", "tools"}:
                tool_steps, executed_tools, requested_tools = yield from (
                    execute_native_tool_calls(
                        self,
                        task_state,
                        user_message,
                        conversation,
                        result,
                        tool_steps,
                    )
                )
                if agent.abort_requested:
                    yield from finish_stopped_run(
                        self,
                        task_state,
                        user_message,
                        "Stopped after abort request.",
                        "aborted",
                        run_started_at,
                    )
                    return
                emit_continue_transition(
                    agent, task_state, CONTINUE_TOOL_BATCH_EXECUTED,
                    tool_call_count=executed_tools,
                    tool_requested_count=requested_tools,
                    tool_executed_count=executed_tools,
                )
                continue

            if kind == "retry":
                yield from emit_empty_result_retry(self, task_state)
                continue

            final = raw.strip()
            notifications = yield from self._drain_worker_notification_events()
            if agent.runtime_mode == "plan" and not agent.plan_mode.can_finish():
                notice = agent.plan_mode.final_notice()
                agent.record(
                    {"role": "assistant", "content": notice, "created_at": now()}
                )
                agent.session_event_bus.emit(
                    "assistant_message",
                    {
                        "run_id": task_state.run_id,
                        "kind": "runtime_notice",
                        "content": notice,
                    },
                )
                agent.run_store.write_task_state(task_state)
                yield {
                    "type": "runtime_notice",
                    "run_id": task_state.run_id,
                    "content": notice,
                }
                conversation.append_result(
                    result, feedback=(*notifications, notice)
                )
                emit_continue_transition(agent, task_state, CONTINUE_PLAN_NOTICE)
                continue

            readiness_action, notice = final_readiness_action(self, task_state, final)
            if readiness_action == "runtime_notice":
                yield {
                    "type": "runtime_notice",
                    "run_id": task_state.run_id,
                    "content": notice,
                }
                conversation.append_result(
                    result, feedback=(*notifications, notice)
                )
                emit_continue_transition(
                    agent,
                    task_state,
                    CONTINUE_FINAL_READINESS_NOTICE,
                )
                continue
            if readiness_action == "block":
                yield from finish_stopped_run(
                    self,
                    task_state,
                    user_message,
                    notice,
                    STOP_REASON_FINAL_GATE_BLOCKED,
                    run_started_at,
                )
                return

            yield from finish_successful_run(
                self, task_state, user_message, final, run_started_at
            )
            return

        if attempts >= max_attempts and tool_steps < agent.max_steps:
            final = "Stopped after too many empty model responses without a native tool call or final answer."
            task_state.stop_retry_limit(final)
        else:
            summary = None
            if tool_steps > 0:
                summary = request_step_limit_summary(self, task_state, user_message)
            if summary:
                final = (
                    summary
                    + "\n\n— 已达本轮 step 预算上限（max_steps）。以上是当前进展总结。"
                    "继续工作：在 REPL 输入 /resume 续接本会话，或直接说"
                    "「继续」让我接着干。"
                )
            else:
                final = "Stopped after reaching the step limit without a final answer."
            task_state.stop_step_limit(final)
        yield from finish_limited_run(
            self, task_state, user_message, final, run_started_at
        )
