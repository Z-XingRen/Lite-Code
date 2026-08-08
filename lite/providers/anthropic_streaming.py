"""Incremental decoding for Anthropic Messages-compatible SSE streams."""

import copy
import json

from .base import normalize_stop_reason
from .openai_streaming import iter_sse_events
from .streaming import ModelStreamEvent


def decode_anthropic_stream(
    response,
    *,
    metadata,
    provider_error,
    cancellation_token=None,
):
    """Translate Anthropic message/content block events to native events."""

    blocks = {}
    remote_started = False
    done_seen = False
    stop_reason = ""

    # Emit the local start before reading the first network frame.
    yield ModelStreamEvent(kind="message_start", metadata=dict(metadata))
    for event_name, event in iter_sse_events(response):
        if cancellation_token is not None:
            cancellation_token.raise_if_cancelled()
        if event is None:
            if not done_seen:
                done_seen = True
                yield _done_event(blocks, stop_reason or "stop", metadata, provider_error)
            break
        if not isinstance(event, dict):
            raise provider_error("Anthropic stream returned a non-object SSE event")
        event_type = str(event.get("type") or event_name or "")

        if event_type == "message_start":
            remote_started = True
            message = event.get("message") or {}
            if isinstance(message, dict):
                if message.get("id"):
                    metadata["provider_response_id"] = str(message["id"])
                if message.get("model"):
                    metadata["provider_response_model"] = str(message["model"])
                usage = message.get("usage")
                if isinstance(usage, dict) and usage:
                    details = _usage_details(usage)
                    metadata.update(details)
                    yield ModelStreamEvent(kind="usage", metadata=details)
            continue

        if event_type == "ping":
            continue

        if event_type == "content_block_start":
            _require_message_start(remote_started, provider_error)
            index = _block_index(event)
            content_block = event.get("content_block") or {}
            if not isinstance(content_block, dict):
                raise provider_error("Anthropic stream content block is not an object")
            block_type = str(content_block.get("type") or "")
            if block_type not in {"text", "tool_use"}:
                continue
            block = {
                "type": block_type,
                "id": str(content_block.get("id") or ""),
                "name": str(content_block.get("name") or ""),
                "text": str(content_block.get("text") or ""),
                "arguments": "",
            }
            initial_input = content_block.get("input")
            if block_type == "tool_use" and isinstance(initial_input, dict) and initial_input:
                block["arguments"] = json.dumps(initial_input, separators=(",", ":"))
            blocks[index] = block
            if block_type == "tool_use":
                yield ModelStreamEvent(
                    kind="tool_call_delta",
                    tool_call_index=index,
                    call_id_delta=block["id"],
                    name_delta=block["name"],
                    arguments_delta=block["arguments"],
                )
            elif block["text"]:
                yield ModelStreamEvent(kind="text_delta", text_delta=block["text"])
            continue

        if event_type == "content_block_delta":
            _require_message_start(remote_started, provider_error)
            index = _block_index(event)
            delta = event.get("delta") or {}
            if not isinstance(delta, dict):
                raise provider_error("Anthropic stream content delta is not an object")
            delta_type = str(delta.get("type") or "")
            if delta_type == "text_delta":
                block = blocks.setdefault(index, _empty_block())
                value = delta.get("text")
                if isinstance(value, str) and value:
                    block["type"] = "text"
                    block["text"] += value
                    yield ModelStreamEvent(kind="text_delta", text_delta=value)
            elif delta_type == "input_json_delta":
                block = blocks.setdefault(index, _empty_block())
                value = delta.get("partial_json")
                if isinstance(value, str) and value:
                    block["type"] = "tool_use"
                    block["arguments"] += value
                    yield ModelStreamEvent(
                        kind="tool_call_delta",
                        tool_call_index=index,
                        arguments_delta=value,
                    )
            continue

        if event_type == "message_delta":
            _require_message_start(remote_started, provider_error)
            delta = event.get("delta") or {}
            if isinstance(delta, dict) and delta.get("stop_reason"):
                stop_reason = str(delta["stop_reason"])
            usage = event.get("usage")
            if isinstance(usage, dict) and usage:
                details = _usage_details(usage)
                metadata.update(details)
                yield ModelStreamEvent(kind="usage", metadata=details)
            continue

        if event_type == "message_stop":
            _require_message_start(remote_started, provider_error)
            if event.get("stop_reason"):
                stop_reason = str(event["stop_reason"])
            done_seen = True
            yield _done_event(blocks, stop_reason or "stop", metadata, provider_error)
            break

        if event_type == "error":
            error_value = event.get("error") or event
            raise provider_error(f"Anthropic-compatible error: {error_value}")

        # Unknown extension events are intentionally ignored. Standard Anthropic
        # streams include events such as ``content_block_stop`` and ``citation``
        # that do not carry model content needed by the provider-neutral contract.


def _done_event(blocks, stop_reason, metadata, provider_error):
    continuation = []
    for index in sorted(blocks):
        block = blocks[index]
        if block["type"] == "text":
            if block["text"]:
                continuation.append({"type": "text", "text": block["text"]})
            continue
        raw_arguments = block["arguments"] or "{}"
        try:
            arguments = json.loads(raw_arguments)
        except json.JSONDecodeError as exc:
            raise provider_error(
                "Anthropic stream returned invalid tool input JSON", cause=exc
            ) from exc
        if not isinstance(arguments, dict):
            raise provider_error("Anthropic stream tool input must be an object")
        continuation.append(
            {
                "type": "tool_use",
                "id": block["id"],
                "name": block["name"],
                "input": copy.deepcopy(arguments),
            }
        )
    return ModelStreamEvent(
        kind="done",
        stop_reason=normalize_stop_reason(stop_reason),
        continuation=tuple(continuation),
        metadata=dict(metadata),
    )


def _usage_details(usage):
    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    details = {}
    if input_tokens is not None:
        details["input_tokens"] = input_tokens
    if input_tokens is not None or "cache_read_input_tokens" in usage:
        cached_tokens = int(usage.get("cache_read_input_tokens") or 0)
        details["cached_tokens"] = cached_tokens
        details["cache_hit"] = cached_tokens > 0
    if output_tokens is not None:
        details["output_tokens"] = output_tokens
    if usage.get("total_tokens") is not None:
        details["total_tokens"] = usage["total_tokens"]
    return details


def _block_index(event):
    try:
        return int(event.get("index", 0))
    except (TypeError, ValueError):
        return 0


def _empty_block():
    return {"type": "", "id": "", "name": "", "text": "", "arguments": ""}


def _require_message_start(started, provider_error):
    if not started:
        raise provider_error("Anthropic stream event arrived before message_start")
