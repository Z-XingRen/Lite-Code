"""Incremental decoding for OpenAI Responses-compatible SSE streams."""

import copy
import json

from .base import normalize_stop_reason
from .streaming import ModelStreamEvent


def iter_sse_events(response):
    """Yield decoded SSE frames without buffering the provider response."""

    data_lines = []
    event_name = ""
    for raw_line in _response_lines(response):
        if isinstance(raw_line, bytes):
            line = raw_line.decode("utf-8", errors="replace")
        else:
            line = str(raw_line)
        line = line.rstrip("\r\n")
        if not line:
            if data_lines:
                yield event_name, _decode_sse_data(data_lines)
            data_lines = []
            event_name = ""
            continue
        if line.startswith(":"):
            continue
        if line.startswith("event:"):
            event_name = line[6:].strip()
        elif line.startswith("data:"):
            data_lines.append(line[5:].lstrip(" "))
    if data_lines:
        yield event_name, _decode_sse_data(data_lines)


def decode_openai_stream(
    response,
    *,
    metadata,
    build_result,
    stop_reason_from_response,
    usage_metadata,
    provider_error,
    cancellation_token=None,
):
    """Translate Responses and compatible chat chunks into native events."""

    text_seen = False
    tool_seen = False
    done_seen = False
    stop_reason = ""
    tool_states = {}
    item_indexes = {}
    continuation_items = {}

    def continuation():
        values = []
        for index in sorted(set(continuation_items) | set(tool_states)):
            item = copy.deepcopy(continuation_items.get(index, {}))
            state = tool_states.get(index)
            if state:
                item.update(
                    {
                        "type": "function_call",
                        "id": state.get("id", ""),
                        "call_id": state.get("call_id", ""),
                        "name": state.get("name", ""),
                        "arguments": state.get("arguments", ""),
                    }
                )
            if item:
                values.append(item)
        return tuple(values)

    def usage_event(usage):
        details = usage_metadata(usage or {})
        metadata.update(details)
        return ModelStreamEvent(kind="usage", metadata=details)

    yield ModelStreamEvent(kind="message_start", metadata=dict(metadata))
    for event_name, event in iter_sse_events(response):
        if cancellation_token is not None:
            cancellation_token.raise_if_cancelled()
        if event is None:
            if not done_seen:
                done_seen = True
                yield ModelStreamEvent(
                    kind="done",
                    stop_reason=normalize_stop_reason(stop_reason or "stop"),
                    continuation=continuation(),
                    metadata=dict(metadata),
                )
            break
        if not isinstance(event, dict):
            raise provider_error("OpenAI stream returned a non-object SSE event")
        event_type = str(event.get("type") or event_name or "")

        if event_type in {"response.created", "response.in_progress"}:
            response_data = event.get("response") or {}
            if isinstance(response_data, dict) and response_data.get("id"):
                metadata["provider_response_id"] = str(response_data["id"])
            continue

        if event_type in {"response.output_text.delta", "response.output_text.done"}:
            value = event.get("delta")
            if event_type.endswith(".done") and not text_seen:
                value = event.get("text")
            if isinstance(value, str) and value:
                text_seen = True
                yield ModelStreamEvent(kind="text_delta", text_delta=value)
            continue

        if event_type == "response.output_item.added":
            item = event.get("item") or {}
            if item.get("type") != "function_call":
                continue
            index = _stream_index(event)
            state = tool_states.setdefault(index, _empty_tool_state())
            state.update(
                {
                    "id": str(item.get("id") or ""),
                    "call_id": str(item.get("call_id") or ""),
                    "name": str(item.get("name") or ""),
                    "arguments": str(item.get("arguments") or ""),
                }
            )
            item_indexes[state["id"]] = index
            continuation_items[index] = copy.deepcopy(item)
            tool_seen = True
            yield ModelStreamEvent(
                kind="tool_call_delta",
                tool_call_index=index,
                call_id_delta=state["call_id"],
                name_delta=state["name"],
                arguments_delta=state["arguments"],
            )
            continue

        if event_type in {
            "response.function_call_arguments.delta",
            "response.function_call_arguments.done",
        }:
            item_id = str(event.get("item_id") or "")
            index = _stream_index(event, item_indexes.get(item_id, len(tool_states)))
            state = tool_states.setdefault(index, _empty_tool_state(item_id))
            value = _argument_delta(
                state,
                event,
                final=event_type.endswith(".done"),
                provider_error=provider_error,
            )
            if value:
                state["arguments"] += value
                tool_seen = True
                yield ModelStreamEvent(
                    kind="tool_call_delta",
                    tool_call_index=index,
                    arguments_delta=value,
                )
            continue

        if event_type == "response.output_item.done":
            item = event.get("item") or {}
            if item.get("type") != "function_call":
                continue
            index = _stream_index(event)
            state = tool_states.setdefault(index, _empty_tool_state())
            state.update(
                {
                    "id": str(item.get("id") or state.get("id") or ""),
                    "call_id": str(item.get("call_id") or state.get("call_id") or ""),
                    "name": str(item.get("name") or state.get("name") or ""),
                }
            )
            missing = _missing_final_arguments(
                state["arguments"],
                str(item.get("arguments") or ""),
                provider_error,
            )
            if missing:
                state["arguments"] += missing
                yield ModelStreamEvent(
                    kind="tool_call_delta",
                    tool_call_index=index,
                    arguments_delta=missing,
                )
            continuation_items[index] = copy.deepcopy(item)
            continue

        if event_type in {"response.usage", "response.completed", "response.incomplete"}:
            response_data = event.get("response") or {}
            if not isinstance(response_data, dict):
                response_data = {}
            usage = response_data.get("usage") or event.get("usage")
            if usage:
                yield usage_event(usage)
            if event_type == "response.usage":
                continue
            stop_reason = stop_reason_from_response(response_data)
            result = build_result(response_data, metadata) if response_data else None
            if result is not None and not text_seen and not tool_seen:
                done_seen = True
                yield ModelStreamEvent(
                    kind="done",
                    stop_reason=normalize_stop_reason(
                        stop_reason or result.stop_reason or "stop"
                    ),
                    continuation=tuple(result.continuation or ()),
                    metadata=dict(metadata),
                    result=result,
                )
                continue
            if result is not None and result.text and not text_seen:
                text_seen = True
                yield ModelStreamEvent(kind="text_delta", text_delta=result.text)
            done_seen = True
            yield ModelStreamEvent(
                kind="done",
                stop_reason=normalize_stop_reason(
                    stop_reason or (result.stop_reason if result else "stop")
                ),
                continuation=tuple(
                    result.continuation
                    if result is not None and result.continuation
                    else continuation()
                ),
                metadata=dict(metadata),
            )
            continue

        if event_type in {"error", "response.failed"}:
            error_value = event.get("error") or event.get("response") or event
            raise provider_error(f"OpenAI-compatible error: {error_value}")

        choices = event.get("choices")
        if isinstance(choices, list):
            for choice in choices:
                if not isinstance(choice, dict):
                    continue
                delta = choice.get("delta") or {}
                if isinstance(delta.get("content"), str) and delta["content"]:
                    text_seen = True
                    yield ModelStreamEvent(
                        kind="text_delta", text_delta=delta["content"]
                    )
                for tool_call in delta.get("tool_calls", []) or []:
                    call_index = _stream_index(tool_call, len(tool_states))
                    state = tool_states.setdefault(call_index, _empty_tool_state())
                    function = tool_call.get("function") or {}
                    call_id = tool_call.get("id")
                    name = function.get("name")
                    arguments = function.get("arguments")
                    if call_id:
                        state["call_id"] = str(call_id)
                    if name:
                        state["name"] = str(name)
                    if arguments:
                        state["arguments"] += str(arguments)
                    tool_seen = True
                    yield ModelStreamEvent(
                        kind="tool_call_delta",
                        tool_call_index=call_index,
                        call_id_delta=str(call_id or ""),
                        name_delta=str(name or ""),
                        arguments_delta=str(arguments or ""),
                    )
                if choice.get("finish_reason"):
                    stop_reason = str(choice["finish_reason"])
            if event.get("usage"):
                yield usage_event(event["usage"])


def _response_lines(response):
    readline = getattr(response, "readline", None)
    if callable(readline):
        while True:
            line = readline()
            if line in (b"", ""):
                return
            yield line
    elif hasattr(response, "__iter__"):
        buffer = b""
        for chunk in response:
            if isinstance(chunk, str):
                chunk = chunk.encode("utf-8")
            buffer += bytes(chunk)
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                yield line + b"\n"
        if buffer:
            yield buffer
    else:
        body = response.read()
        yield from body.splitlines(keepends=True)


def _decode_sse_data(data_lines):
    payload = "\n".join(data_lines)
    if payload.strip() == "[DONE]":
        return None
    try:
        return json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError("OpenAI stream returned invalid SSE JSON") from exc


def _stream_index(payload, fallback=0):
    value = payload.get("output_index", payload.get("index", fallback))
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(fallback)


def _empty_tool_state(item_id=""):
    return {"id": item_id, "call_id": "", "name": "", "arguments": ""}


def _argument_delta(state, event, *, final, provider_error):
    value = str(event.get("arguments") or "") if final else event.get("delta")
    if not isinstance(value, str):
        return ""
    existing = state["arguments"]
    if existing and value.startswith(existing):
        return value[len(existing) :]
    if final and existing and value != existing:
        raise provider_error("OpenAI stream returned inconsistent function arguments")
    return value


def _missing_final_arguments(existing, final, provider_error):
    if final.startswith(existing):
        return final[len(existing) :]
    if final == existing:
        return ""
    raise provider_error("OpenAI stream returned inconsistent function arguments")
