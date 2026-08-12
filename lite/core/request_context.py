"""Per-request context transformation and provider-boundary hardening."""

import copy
import json
import time

from ..cancellation import CancellationRequested
from ..providers.base import ModelConversation, ToolCall
from ..providers.errors import ProviderError
from .completion_governance import finish_stopped_run
from .model_errors import finish_model_error

SYNTHETIC_RESULT_TEXT = (
    "error: synthetic tool result inserted because the recorded assistant "
    "tool call is missing its matching result"
)


class HistoryHardeningError(ProviderError):
    """An unrecoverable provider-history invariant violation."""

    def __init__(self, message):
        super().__init__(
            str(message),
            code="history_hardening_error",
            retryable=False,
        )


def finish_request_context_error(
    engine,
    task_state,
    user_message,
    prompt_metadata,
    exc,
    prompt_started_at,
    run_started_at,
):
    """Finish a transform/hardening failure before any provider effect begins."""

    agent = engine.runtime
    if isinstance(exc, CancellationRequested) or agent.current_cancellation_token.cancelled:
        yield from finish_stopped_run(
            engine,
            task_state,
            user_message,
            "Stopped after abort request.",
            "aborted",
            run_started_at,
        )
        return
    yield from finish_model_error(
        engine,
        task_state,
        user_message,
        prompt_metadata,
        exc,
        int((time.monotonic() - prompt_started_at) * 1000),
        int((time.monotonic() - run_started_at) * 1000),
    )


def rebuild_model_request(agent, prompt, previous=None, *, tools=None):
    """Build, transform, and harden a fresh request view for one provider call."""

    token = agent.current_cancellation_token
    token.raise_if_cancelled()
    selected_tools = tuple(agent.model_tools()) if tools is None else tuple(tools)
    conversation = ModelConversation(
        initial_input=copy.deepcopy(prompt),
        tools=selected_tools,
        turns=copy.deepcopy(list(getattr(previous, "turns", ()) or ())),
    )
    messages = conversation_messages(conversation)
    transform = getattr(agent, "context_transform", None)
    if callable(transform):
        transformed = transform(copy.deepcopy(messages), token)
        if transformed is None:
            raise HistoryHardeningError("context transform returned no messages")
        messages = list(transformed)
    token.raise_if_cancelled()
    hardened, report = harden_context_messages(messages)
    report["transform_applied"] = callable(transform)
    conversation.request_messages = tuple(hardened)
    report.update(_request_telemetry(agent, conversation, hardened))
    conversation.context_metadata = report
    return conversation, report


def _request_telemetry(agent, conversation, messages):
    source = (
        "session_projection_plus_turn_delta"
        if getattr(agent, "_frozen_turn_context", None) is not None
        else "session_transcript_plus_turn_delta"
    )
    serialized = json.dumps(messages, ensure_ascii=False, sort_keys=True)
    return {
        "base_context_hash": str(
            getattr(agent, "_frozen_turn_context", {})
            and agent._frozen_turn_context.get("base_context_hash", "")
            or ""
        ),
        "session_projection_event_count": int(
            getattr(agent, "_turn_context_projection_event_count", 0) or 0
        ),
        "current_turn_delta_count": len(conversation.turns),
        "provider_turn_count": len(conversation.turns),
        "duplicate_tool_result_count": _duplicate_tool_result_count(
            conversation.initial_input, messages
        ),
        "assembled_input_chars": len(serialized),
        "estimated_input_tokens": max(1, len(serialized) // 4),
        "context_source": source,
    }


def _duplicate_tool_result_count(initial_input, messages):
    duplicates = 0
    seen_calls = set()
    for message in messages:
        if message.get("role") != "tool":
            continue
        call_key = (str(message.get("call_id", "")), str(message.get("name", "")))
        if call_key in seen_calls:
            duplicates += 1
        seen_calls.add(call_key)
        content = str(message.get("content", ""))
        if len(content) >= 16:
            duplicates += max(0, str(initial_input).count(content) - 1)
    return duplicates


def conversation_messages(conversation):
    """Project a conversation into provider-neutral messages without mutation."""

    messages = [{"role": "user", "content": copy.deepcopy(conversation.initial_input)}]
    for turn in conversation.turns:
        calls = tuple(getattr(turn, "tool_calls", ()) or ())
        if turn.continuation or calls or getattr(turn, "text", ""):
            messages.append(
                {
                    "role": "assistant",
                    "content": str(getattr(turn, "text", "") or ""),
                    "continuation": copy.deepcopy(tuple(turn.continuation or ())),
                    "tool_calls": [_tool_call_dict(call) for call in calls],
                }
            )
        messages.extend(
            {
                "role": "tool",
                "call_id": output.call_id,
                "name": output.name,
                "content": output.content,
                "is_error": bool(output.is_error),
            }
            for output in turn.tool_outputs
        )
        if turn.feedback:
            messages.append(
                {"role": "user", "content": "\n\n".join(turn.feedback)}
            )
    return messages


def harden_context_messages(messages):
    """Return a deterministic valid request history and a secret-free report."""

    normalized = [_normalize_message(message) for message in list(messages)]
    if not normalized:
        raise HistoryHardeningError("history is empty")
    if normalized[0]["role"] == "tool":
        raise HistoryHardeningError("history begins with an orphan tool result")
    if normalized[0]["role"] != "user":
        raise HistoryHardeningError("history must begin with a user message")

    merged = _merge_same_role_messages(normalized)
    hardened, pair_count, synthetic_count = _pair_tool_messages(merged)
    _validate_role_order(hardened)
    report = {
        "version": "local-v1",
        "input_message_count": len(normalized),
        "output_message_count": len(hardened),
        "merged_same_role": len(normalized) - len(merged),
        "validated_tool_pairs": pair_count,
        "synthetic_tool_results": synthetic_count,
        "effective_first_role": _effective_role(hardened[0]),
        "effective_last_role": _effective_role(hardened[-1]),
        "actions": [
            *(["merge_same_role"] if len(normalized) != len(merged) else []),
            *(["insert_missing_tool_result"] if synthetic_count else []),
        ],
    }
    return hardened, report


def _normalize_message(message):
    if not isinstance(message, dict):
        raise HistoryHardeningError("history message must be an object")
    value = copy.deepcopy(message)
    role = str(value.get("role", "")).strip().lower()
    if role not in {"user", "assistant", "tool"}:
        raise HistoryHardeningError(f"unsupported history role: {role or '(empty)'}")
    value["role"] = role
    value.setdefault("content", "")
    if role == "assistant":
        value["tool_calls"] = [
            _normalize_tool_call(call) for call in value.get("tool_calls", ()) or ()
        ]
        value["continuation"] = copy.deepcopy(
            tuple(value.get("continuation", ()) or ())
        )
    elif role == "tool":
        value["call_id"] = str(value.get("call_id", "")).strip()
        value["name"] = str(value.get("name", "")).strip()
        value["is_error"] = bool(value.get("is_error", False))
    return value


def _normalize_tool_call(call):
    value = _tool_call_dict(call)
    value["call_id"] = str(value.get("call_id", "")).strip()
    value["name"] = str(value.get("name", "")).strip()
    value["arguments"] = copy.deepcopy(dict(value.get("arguments", {}) or {}))
    if not value["call_id"] or not value["name"]:
        raise HistoryHardeningError("tool call requires a call id and name")
    return value


def _tool_call_dict(call):
    if isinstance(call, ToolCall):
        return {
            "call_id": call.call_id,
            "name": call.name,
            "arguments": copy.deepcopy(call.arguments),
        }
    if not isinstance(call, dict):
        raise HistoryHardeningError("tool call must be an object")
    return copy.deepcopy(call)


def _merge_same_role_messages(messages):
    merged = []
    for message in messages:
        if (
            merged
            and message["role"] in {"user", "assistant"}
            and merged[-1]["role"] == message["role"]
        ):
            merged[-1] = _merge_messages(merged[-1], message)
        else:
            merged.append(copy.deepcopy(message))
    return merged


def _merge_messages(left, right):
    merged = copy.deepcopy(left)
    merged["content"] = _merge_content(left.get("content"), right.get("content"))
    if left["role"] == "assistant":
        merged["continuation"] = tuple(left.get("continuation", ())) + tuple(
            right.get("continuation", ())
        )
        merged["tool_calls"] = list(left.get("tool_calls", ())) + list(
            right.get("tool_calls", ())
        )
    return merged


def _merge_content(left, right):
    if isinstance(left, str) and isinstance(right, str):
        return "\n\n".join(value for value in (left, right) if value)
    if not left:
        return copy.deepcopy(right)
    if not right:
        return copy.deepcopy(left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return type(left)(list(left) + list(right))
    raise HistoryHardeningError("same-role message content cannot be merged safely")


def _pair_tool_messages(messages):
    hardened = []
    pair_count = 0
    synthetic_count = 0
    index = 0
    while index < len(messages):
        message = copy.deepcopy(messages[index])
        if message["role"] == "tool":
            raise HistoryHardeningError("orphan tool result has no assistant tool call")
        hardened.append(message)
        index += 1
        if message["role"] != "assistant":
            continue
        calls = list(message.get("tool_calls", ()))
        if not calls:
            continue
        pending = _pending_calls(calls)
        while index < len(messages) and messages[index]["role"] == "tool":
            output = copy.deepcopy(messages[index])
            call_id = output.get("call_id", "")
            name = output.get("name", "")
            if call_id not in pending:
                if any(call["name"] == name for call in pending.values()):
                    raise HistoryHardeningError("tool result call id mismatch")
                raise HistoryHardeningError("orphan tool result has no matching call id")
            call = pending.pop(call_id)
            if name != call["name"]:
                raise HistoryHardeningError("tool result name mismatch")
            hardened.append(output)
            pair_count += 1
            index += 1
        for call in pending.values():
            hardened.append(_synthetic_result(call))
            pair_count += 1
            synthetic_count += 1
    return hardened, pair_count, synthetic_count


def _pending_calls(calls):
    pending = {}
    for call in calls:
        call_id = call["call_id"]
        if call_id in pending:
            raise HistoryHardeningError("duplicate tool call id")
        pending[call_id] = call
    return pending


def _synthetic_result(call):
    return {
        "role": "tool",
        "call_id": call["call_id"],
        "name": call["name"],
        "content": SYNTHETIC_RESULT_TEXT,
        "is_error": True,
        "synthetic": True,
        "reason": "missing_tool_result",
    }


def _validate_role_order(messages):
    previous = None
    for message in messages:
        role = _effective_role(message)
        if role != previous:
            previous = role
            continue
        if role == "user" and message["role"] == "tool":
            continue
        if role == "user" and previous == "user":
            continue
        raise HistoryHardeningError(f"consecutive {role} history messages")
    if _effective_role(messages[-1]) != "user":
        raise HistoryHardeningError("history must end with user input or a tool result")


def _effective_role(message):
    return "user" if message["role"] == "tool" else message["role"]
