"""OpenAI Responses and Anthropic Messages protocol adapters."""

import copy
import json
import socket
import time
from contextlib import nullcontext
import urllib.error
import urllib.request

from ..cancellation import CancellationRequested
from ..core.content_blocks import ensure_model_input
from .base import ModelResult, ToolCall, ensure_conversation, normalize_stop_reason
from .errors import ProviderError, sanitize_url
from .anthropic_streaming import decode_anthropic_stream
from .http_cancellation import HttpResponseController
from .openai_streaming import decode_openai_stream
from .retry import (
    DEFAULT_RETRY_POLICY,
    RETRYABLE_HTTP_STATUS,
    RetryExhausted,
    RetryPolicy,
    calculate_retry_delay,
    classify_retry,
    retry_after_seconds,
    run_with_retries,
)
from .streaming import ModelStreamEvent

OPENAI_COMPATIBLE_USER_AGENT = "lite/0.1"


def _normalize_versioned_base_url(base_url):
    base = str(base_url).rstrip("/")
    if not base.endswith("/v1"):
        base += "/v1"
    return base


def _extract_openai_response_from_sse(body_text):
    last_response = None
    text_deltas = []
    output_items = []
    for line in body_text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            event = json.loads(payload)
        except json.JSONDecodeError:
            continue
        response = event.get("response")
        if isinstance(response, dict):
            last_response = response
            if event.get("type") == "response.completed":
                return response
        event_type = event.get("type", "")
        if event_type == "response.output_text.delta":
            delta = event.get("delta")
            if isinstance(delta, str):
                text_deltas.append(delta)
        elif event_type == "response.output_text.done":
            text = event.get("text")
            if isinstance(text, str) and text:
                return {
                    "output": [
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": text}],
                        }
                    ]
                }
        elif event_type in {"response.output_item.done", "response.output_item.added"}:
            item = event.get("item")
            if isinstance(item, dict):
                if event_type == "response.output_item.done":
                    output_items = [
                        existing
                        for existing in output_items
                        if existing.get("id") != item.get("id")
                    ]
                output_items.append(item)
    if isinstance(last_response, dict):
        if output_items and not last_response.get("output"):
            last_response["output"] = output_items
        return last_response
    if output_items:
        return {"output": output_items}
    if text_deltas:
        return {
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {"type": "output_text", "text": "".join(text_deltas)}
                    ],
                }
            ]
        }
    return {}


def _openai_result(data, metadata, provider, model, base_url):
    output = data.get("output") if isinstance(data.get("output"), list) else []
    continuation = [copy.deepcopy(item) for item in output if isinstance(item, dict)]
    texts = []
    calls = []
    for item in continuation:
        if item.get("type") == "function_call":
            calls.append(
                ToolCall(
                    call_id=str(item.get("call_id") or item.get("id") or ""),
                    name=str(item.get("name") or ""),
                    arguments=_tool_arguments(
                        item.get("arguments", {}), provider, model, base_url, metadata
                    ),
                )
            )
            continue
        for content in item.get("content", []) or []:
            if not isinstance(content, dict):
                continue
            # A few Responses-compatible gateways omit the content block's
            # ``type`` even though the shape otherwise matches output_text.
            if content.get("type") in {None, "", "output_text", "text"}:
                text = content.get("text")
                if isinstance(text, str) and text:
                    texts.append(text)

    # Some compatible Responses endpoints expose the convenience field only.
    if not texts and isinstance(data.get("output_text"), str):
        texts.append(data["output_text"])

    # Preserve the previous text-only compatibility path for non-tool responses.
    choices = data.get("choices") or []
    if choices and not continuation:
        message = choices[0].get("message", {}) or {}
        content = message.get("content")
        if isinstance(content, str) and content:
            texts.append(content)
        elif isinstance(content, list):
            texts.extend(
                str(part.get("text"))
                for part in content
                if isinstance(part, dict) and part.get("text")
            )
        for tool_call in message.get("tool_calls", []) or []:
            function = tool_call.get("function", {}) or {}
            call = ToolCall(
                call_id=str(tool_call.get("id") or ""),
                name=str(function.get("name") or ""),
                arguments=_tool_arguments(
                    function.get("arguments", {}), provider, model, base_url, metadata
                ),
            )
            calls.append(call)
            continuation.append(
                {
                    "type": "function_call",
                    "call_id": call.call_id,
                    "name": call.name,
                    "arguments": json.dumps(call.arguments),
                }
            )

    return ModelResult(
        text="\n".join(texts).strip(),
        tool_calls=tuple(calls),
        stop_reason=_openai_stop_reason(data),
        continuation=tuple(continuation),
        metadata=metadata,
    )


def _openai_stop_reason(data):
    raw = data.get("status") or data.get("stop_reason")
    if str(raw or "").lower() in {"incomplete", "in_progress"}:
        details = data.get("incomplete_details") or {}
        raw = details.get("reason") or raw
    if not raw:
        choices = data.get("choices") or []
        if choices and isinstance(choices[0], dict):
            raw = choices[0].get("finish_reason")
    return normalize_stop_reason(raw)


def _tool_arguments(value, provider, model, base_url, request_metadata):
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise _provider_failure(
                provider,
                model,
                base_url,
                "invalid_tool_arguments",
                f"{provider} provider returned invalid JSON tool arguments",
                request_metadata,
                cause=exc,
            ) from exc
        if isinstance(parsed, dict):
            return parsed
    raise _provider_failure(
        provider,
        model,
        base_url,
        "invalid_tool_arguments",
        f"{provider} provider returned non-object tool arguments",
        request_metadata,
    )


def _openai_input_content(prompt):
    model_input = ensure_model_input(prompt)
    content = [{"type": "input_text", "text": model_input.text}]
    for image in model_input.images:
        content.append(
            {
                "type": "input_image",
                "image_url": image.data_url(),
            }
        )
    return content, model_input.image_count


def _anthropic_input_content(prompt):
    model_input = ensure_model_input(prompt)
    content = []
    for image in model_input.images:
        content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": image.mime_type,
                    "data": image.base64_data(),
                },
            }
        )
    content.append({"type": "text", "text": model_input.text})
    return content, model_input.image_count


def _openai_conversation_input(conversation):
    if conversation.request_messages:
        return _openai_request_messages(conversation.request_messages)
    content, image_input_count = _openai_input_content(conversation.initial_input)
    items = [{"role": "user", "content": content}]
    for turn in conversation.turns:
        items.extend(copy.deepcopy(list(turn.continuation)))
        items.extend(
            {
                "type": "function_call_output",
                "call_id": output.call_id,
                "output": output.content,
            }
            for output in turn.tool_outputs
        )
        if turn.feedback:
            items.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "\n\n".join(turn.feedback)}
                    ],
                }
            )
    return items, image_input_count


def _add_openai_prompt_cache_breakpoint(input_items, prefix_chars):
    """Mark stable leading text without changing the model-visible prompt."""

    remaining = max(0, int(prefix_chars or 0))
    if not remaining:
        return
    for item in input_items:
        if item.get("role") != "user" or not isinstance(item.get("content"), list):
            continue
        for index, block in enumerate(item["content"]):
            if block.get("type") != "input_text":
                continue
            text = str(block.get("text", ""))
            if remaining > len(text):
                remaining -= len(text)
                continue
            stable = text[:remaining]
            dynamic = text[remaining:]
            if not stable:
                return
            replacement = [
                {
                    "type": "input_text",
                    "text": stable,
                    "prompt_cache_breakpoint": {"mode": "explicit"},
                }
            ]
            if dynamic:
                replacement.append({"type": "input_text", "text": dynamic})
            item["content"][index : index + 1] = replacement
            return
        return


def _openai_request_messages(messages):
    items = []
    image_input_count = 0
    for message in messages:
        role = message["role"]
        if role == "user":
            content, image_count = _openai_input_content(message.get("content", ""))
            items.append({"role": "user", "content": content})
            image_input_count += image_count
        elif role == "tool":
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": message["call_id"],
                    "output": str(message.get("content", "")),
                }
            )
        else:
            continuation = copy.deepcopy(list(message.get("continuation", ())))
            if continuation and not any(
                item.get("type") == "tool_use" for item in continuation
            ):
                items.extend(continuation)
                continue
            text = str(message.get("content", "") or "")
            if text:
                items.append(
                    {
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": text}],
                    }
                )
            items.extend(
                {
                    "type": "function_call",
                    "call_id": call["call_id"],
                    "name": call["name"],
                    "arguments": json.dumps(call.get("arguments", {})),
                }
                for call in message.get("tool_calls", ())
            )
    return items, image_input_count


def _anthropic_conversation_messages(conversation):
    if conversation.request_messages:
        return _anthropic_request_messages(conversation.request_messages)
    content, image_input_count = _anthropic_input_content(conversation.initial_input)
    messages = [{"role": "user", "content": content}]
    for turn in conversation.turns:
        if turn.continuation:
            messages.append(
                {"role": "assistant", "content": copy.deepcopy(list(turn.continuation))}
            )
        user_content = [
            {
                "type": "tool_result",
                "tool_use_id": output.call_id,
                "content": output.content,
                **({"is_error": True} if output.is_error else {}),
            }
            for output in turn.tool_outputs
        ]
        if turn.feedback:
            user_content.append({"type": "text", "text": "\n\n".join(turn.feedback)})
        if user_content:
            messages.append({"role": "user", "content": user_content})
    return messages, image_input_count


def _add_anthropic_prompt_cache_breakpoint(messages, prefix_chars):
    """Mark stable leading text for Anthropic-compatible prompt caching."""

    remaining = max(0, int(prefix_chars or 0))
    if not remaining:
        return
    for message in messages:
        if message.get("role") != "user" or not isinstance(message.get("content"), list):
            continue
        for index, block in enumerate(message["content"]):
            if block.get("type") != "text":
                continue
            text = str(block.get("text", ""))
            if remaining > len(text):
                remaining -= len(text)
                continue
            stable = text[:remaining]
            dynamic = text[remaining:]
            if not stable:
                return
            replacement = [
                {
                    "type": "text",
                    "text": stable,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
            if dynamic:
                replacement.append({"type": "text", "text": dynamic})
            message["content"][index : index + 1] = replacement
            return
        return


def _anthropic_request_messages(request_messages):
    messages = []
    image_input_count = 0

    def append_user(content):
        if messages and messages[-1]["role"] == "user":
            messages[-1]["content"].extend(content)
        else:
            messages.append({"role": "user", "content": list(content)})

    for message in request_messages:
        role = message["role"]
        if role == "user":
            content, image_count = _anthropic_input_content(message.get("content", ""))
            append_user(content)
            image_input_count += image_count
        elif role == "tool":
            append_user(
                [
                    {
                        "type": "tool_result",
                        "tool_use_id": message["call_id"],
                        "content": str(message.get("content", "")),
                        **({"is_error": True} if message.get("is_error") else {}),
                    }
                ]
            )
        else:
            continuation = copy.deepcopy(list(message.get("continuation", ())))
            if continuation and not any(
                item.get("type") in {"function_call", "message"}
                for item in continuation
            ):
                content = continuation
            else:
                content = []
                text = str(message.get("content", "") or "")
                if text:
                    content.append({"type": "text", "text": text})
                content.extend(
                    {
                        "type": "tool_use",
                        "id": call["call_id"],
                        "name": call["name"],
                        "input": copy.deepcopy(call.get("arguments", {})),
                    }
                    for call in message.get("tool_calls", ())
                )
            messages.append({"role": "assistant", "content": content})
    return messages, image_input_count


def _openai_tool_definitions(conversation):
    tools = []
    for tool in conversation.tools:
        schema = (
            _strict_openai_schema(tool.input_schema)
            if tool.strict
            else copy.deepcopy(tool.input_schema)
        )
        tools.append(
            {
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": schema,
                "strict": bool(tool.strict),
            }
        )
    return tools


def _anthropic_tool_definitions(conversation):
    tools = []
    for tool in conversation.tools:
        definition = {
            "name": tool.name,
            "description": tool.description,
            "input_schema": copy.deepcopy(tool.input_schema),
        }
        if tool.strict:
            definition["strict"] = True
        tools.append(definition)
    return tools


def _anthropic_result(data, metadata, model, base_url, request_metadata):
    content = tuple(
        copy.deepcopy(item)
        for item in (data.get("content") or [])
        if isinstance(item, dict)
    )
    texts = []
    calls = []
    for item in content:
        if item.get("type") == "text" and isinstance(item.get("text"), str):
            texts.append(item["text"])
        elif item.get("type") == "tool_use":
            calls.append(
                ToolCall(
                    call_id=str(item.get("id") or ""),
                    name=str(item.get("name") or ""),
                    arguments=_tool_arguments(
                        item.get("input", {}),
                        "anthropic",
                        model,
                        base_url,
                        request_metadata,
                    ),
                )
            )
    return ModelResult(
        text="\n".join(texts).strip(),
        tool_calls=tuple(calls),
        stop_reason=normalize_stop_reason(data.get("stop_reason")),
        continuation=content,
        metadata=metadata,
    )


def _strict_openai_schema(schema):
    schema = copy.deepcopy(schema)

    def normalize(value):
        if isinstance(value, dict):
            value.pop("default", None)
            if value.get("type") == "object" or "properties" in value:
                properties = value.get("properties", {}) or {}
                value["additionalProperties"] = False
                value["required"] = list(properties)
            for child in value.values():
                normalize(child)
        elif isinstance(value, list):
            for child in value:
                normalize(child)

    normalize(schema)
    return schema


def _extract_usage_cache_details(data):
    # 把不同 OpenAI-compatible 返回里的 usage 字段整理成统一结构，
    # 让 runtime/trace/report 不需要关心 provider 细节。
    usage = data.get("usage") or {}
    input_tokens = usage.get("input_tokens", usage.get("prompt_tokens"))
    output_tokens = usage.get("output_tokens", usage.get("completion_tokens"))
    input_details = usage.get("input_tokens_details") or usage.get("prompt_tokens_details") or {}
    cached_tokens = int(
        usage.get("cache_read_input_tokens")
        or usage.get("prompt_cache_hit_tokens")
        or input_details.get("cached_tokens")
        or 0
    )
    cache_write_tokens = int(
        usage.get("cache_write_input_tokens")
        or usage.get("cache_creation_input_tokens")
        or usage.get("prompt_cache_miss_tokens")
        or usage.get("cache_write_tokens")
        or input_details.get("cache_write_tokens")
        or 0
    )
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": usage.get("total_tokens"),
        "cached_tokens": cached_tokens,
        "cache_write_tokens": cache_write_tokens,
        "cache_hit": cached_tokens > 0,
    }


def _request_with_retries(
    provider,
    model,
    base_url,
    request,
    timeout,
    retry_budget=2,
    *,
    cancellation_token=None,
    response_controller=None,
):
    policy = RetryPolicy(max_retries=int(retry_budget))

    def request_body(_attempt):
        response = urllib.request.urlopen(request, timeout=timeout)
        registration = (
            response_controller.register(response, cancellation_token)
            if response_controller is not None
            else None
        )
        managed_response = callable(getattr(response, "__enter__", None))
        response_context = response if managed_response else nullcontext(response)
        try:
            with response_context as body:
                body_text = body.read().decode("utf-8")
                headers = getattr(body, "headers", {}) or {}
                content_type = headers.get("Content-Type", "")
            if registration is not None:
                registration.raise_if_cancelled()
            if cancellation_token is not None:
                cancellation_token.raise_if_cancelled()
        except Exception:
            if registration is not None:
                registration.raise_if_cancelled()
            if cancellation_token is not None:
                cancellation_token.raise_if_cancelled()
            raise
        finally:
            if registration is not None:
                registration.release()
        payload = _decode_provider_error_payload(body_text)
        if payload is not None:
            raise payload
        return body_text, content_type

    try:
        result = run_with_retries(
            request_body,
            policy=policy,
            cancellation_token=cancellation_token,
            sleep_fn=time.sleep,
        )
    except RetryExhausted as failure:
        raise _request_provider_error(
            provider,
            model,
            base_url,
            failure,
        ) from failure.cause
    body_text, content_type = result.value
    return (
        body_text,
        content_type,
        _provider_metadata(
            provider,
            model,
            base_url,
            result.attempts,
            result.retry_count,
            result.history,
        ),
    )


def _provider_metadata(
    provider, model, base_url, attempts, retry_count, retry_history=()
):
    metadata = {
        "provider_protocol": provider,
        "provider_model": model,
        "provider_base_url": sanitize_url(base_url),
        "provider_attempts": int(attempts),
        "provider_retry_count": int(retry_count),
    }
    if retry_history:
        metadata["provider_retry_history"] = [dict(item) for item in retry_history]
    return metadata


def _http_error_code(status):
    status = int(status)
    if status == 401 or status == 403:
        return "auth_error"
    if status == 408:
        return "timeout"
    if status == 429:
        return "rate_limited"
    if status >= 500:
        return "server_error"
    return "http_error"


def _transport_error_code(exc):
    reason = getattr(exc, "reason", None)
    text = f"{exc} {reason}".lower()
    if isinstance(exc, (TimeoutError, socket.timeout)) or isinstance(reason, (TimeoutError, socket.timeout)) or "timed out" in text:
        return "timeout"
    return "network_error"


def _retry_delay(attempt, headers):
    return calculate_retry_delay(attempt, headers)


def _retry_after_seconds(headers):
    return retry_after_seconds(headers)


def _request_provider_error(provider, model, base_url, failure):
    cause = failure.cause
    if isinstance(cause, ProviderError):
        return ProviderError(
            str(cause),
            provider=cause.provider or provider,
            model=cause.model or model,
            base_url=cause.base_url or base_url,
            code=cause.code,
            http_status=cause.http_status,
            retryable=cause.retryable,
            attempts=failure.attempts,
            retry_count=failure.retry_count,
            retry_history=failure.history,
            body_excerpt=cause.body_excerpt,
            cause_type=cause.cause_type,
        )
    if isinstance(cause, urllib.error.HTTPError):
        try:
            body = cause.read().decode("utf-8", errors="replace")
        except Exception:
            body = ""
        status = cause.code
        return ProviderError(
            f"{provider} provider request failed with HTTP {status}",
            provider=provider,
            model=model,
            base_url=base_url,
            code=_http_error_code(status),
            http_status=status,
            retryable=status in RETRYABLE_HTTP_STATUS,
            attempts=failure.attempts,
            retry_count=failure.retry_count,
            retry_history=failure.history,
            body_excerpt=body,
            cause_type=type(cause).__name__,
        )
    return ProviderError(
        f"{provider} provider request failed before a valid response",
        provider=provider,
        model=model,
        base_url=base_url,
        code=_transport_error_code(cause),
        retryable=True,
        attempts=failure.attempts,
        retry_count=failure.retry_count,
        retry_history=failure.history,
        cause_type=type(cause).__name__,
    )


def _decode_provider_error_payload(body_text):
    try:
        payload = json.loads(body_text)
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or not payload.get("error"):
        return None
    value = payload["error"]
    details = value if isinstance(value, dict) else {"message": str(value)}
    raw_code = details.get("code") or details.get("type") or "provider_error"
    code = _normalize_provider_error_code(raw_code)
    status = details.get("status", details.get("http_status"))
    try:
        status = int(status) if status is not None else None
    except (TypeError, ValueError):
        status = None
    message = details.get("message") or value
    return ProviderError(
        f"Provider returned an error: {message}",
        code=code,
        http_status=status,
        retryable=classify_retry({"code": code, "status": status}) is not None,
        body_excerpt=body_text,
    )


def _normalize_provider_error_code(value):
    code = str(value or "provider_error").strip().lower().replace("-", "_")
    if "rate" in code or "throttl" in code:
        return "rate_limited"
    if "overload" in code or "unavailable" in code:
        return "overloaded"
    if "timeout" in code:
        return "timeout"
    if "auth" in code or "permission" in code or "forbidden" in code:
        return "auth_error"
    if "context" in code or "length" in code or "token" in code:
        return "context_overflow"
    if "invalid" in code or "validation" in code or "schema" in code:
        return "invalid_request"
    if "server" in code or "internal" in code:
        return "server_error"
    return code


def _open_stream_with_retries(
    provider,
    model,
    base_url,
    request,
    timeout,
    *,
    cancellation_token=None,
    response_controller=None,
):
    def open_response(_attempt):
        response = urllib.request.urlopen(request, timeout=timeout)
        registration = (
            response_controller.register(response, cancellation_token)
            if response_controller is not None
            else None
        )
        return response, registration

    try:
        result = run_with_retries(
            open_response,
            policy=DEFAULT_RETRY_POLICY,
            cancellation_token=cancellation_token,
            sleep_fn=time.sleep,
        )
    except RetryExhausted as failure:
        raise _request_provider_error(
            provider,
            model,
            base_url,
            failure,
        ) from failure.cause
    response, registration = result.value
    return response, registration, _provider_metadata(
        provider,
        model,
        base_url,
        result.attempts,
        result.retry_count,
        result.history,
    )


def _provider_failure(provider, model, base_url, code, message, request_metadata=None, cause=None):
    request_metadata = request_metadata or {}
    error = ProviderError(
        message,
        provider=provider,
        model=model,
        base_url=base_url,
        code=code,
        retryable=False,
        attempts=request_metadata.get("provider_attempts", 1),
        retry_count=request_metadata.get("provider_retry_count", 0),
        retry_history=request_metadata.get("provider_retry_history", ()),
        cause_type=type(cause).__name__ if cause else "",
    )
    return error


class OpenAICompatibleModelClient:
    def __init__(
        self,
        model,
        base_url,
        api_key,
        temperature,
        timeout,
        strict_tools=False,
        reasoning_effort="",
        supports_explicit_prompt_cache=False,
    ):
        self.model = model
        self.base_url = _normalize_versioned_base_url(base_url)
        self.api_key = api_key
        self.temperature = temperature
        self.timeout = timeout
        self.strict_tools = bool(strict_tools)
        self.reasoning_effort = str(reasoning_effort or "").strip().lower()
        # Gateways can proxy cache-capable models under any host. Cache
        # capability follows the configured protocol, not a URL allowlist.
        self.supports_prompt_cache = True
        self.supports_explicit_prompt_cache = bool(
            supports_explicit_prompt_cache
        ) and "gpt-5.6" in self.model.lower()
        self.last_completion_metadata = {}
        self._http_responses = HttpResponseController()

    def abort(self):
        self._http_responses.abort()

    def complete(self, request, max_new_tokens, **kwargs):
        return self.complete_result(request, max_new_tokens, **kwargs).text

    def complete_result(
        self,
        request,
        max_new_tokens,
        prompt_cache_key=None,
        prompt_cache_retention=None,
        prompt_cache_prefix_chars=None,
        cancellation_token=None,
    ):
        self.last_completion_metadata = {}
        conversation = ensure_conversation(request)
        input_items, image_input_count = _openai_conversation_input(conversation)
        if self.supports_explicit_prompt_cache:
            _add_openai_prompt_cache_breakpoint(input_items, prompt_cache_prefix_chars)
        payload = {
            "model": self.model,
            "input": input_items,
            "max_output_tokens": max_new_tokens,
            "stream": False,
        }
        if conversation.tools:
            payload["tools"] = _openai_tool_definitions(conversation)
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.reasoning_effort:
            payload["reasoning"] = {"effort": self.reasoning_effort}
        # runtime 传入的是“稳定前缀”的签名，而不是整段 prompt 的签名。
        # 这样缓存复用针对的是稳定段，不会因为动态 history 每轮变化而失效。
        if self.supports_prompt_cache and prompt_cache_key:
            payload["prompt_cache_key"] = prompt_cache_key
        if self.supports_explicit_prompt_cache and prompt_cache_prefix_chars:
            payload["prompt_cache_options"] = {"mode": "explicit"}
        elif prompt_cache_retention:
            payload["prompt_cache_retention"] = prompt_cache_retention

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": OPENAI_COMPATIBLE_USER_AGENT,
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        request = urllib.request.Request(
            self.base_url + "/responses",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            body_text, content_type, request_metadata = _request_with_retries(
                "openai",
                self.model,
                self.base_url,
                request,
                self.timeout,
                cancellation_token=cancellation_token,
                response_controller=self._http_responses,
            )
        except ProviderError as exc:
            self.last_completion_metadata = exc.to_metadata()
            raise

        # 有些兼容后端返回普通 JSON，有些返回 SSE。
        # 这里两种都接住，并尽量统一抽取文本和 usage/cache 元数据。
        if content_type.startswith("text/event-stream") or body_text.lstrip().startswith("data:"):
            data = _extract_openai_response_from_sse(body_text)
        else:
            try:
                data = json.loads(body_text)
            except json.JSONDecodeError as exc:
                error = _provider_failure(
                    "openai",
                    self.model,
                    self.base_url,
                    "invalid_json",
                    "OpenAI-compatible error: backend returned non-JSON content that could not be parsed",
                    request_metadata,
                    cause=exc,
                )
                self.last_completion_metadata = error.to_metadata()
                raise error from exc
        if data.get("error"):
            error = _provider_failure(
                "openai",
                self.model,
                self.base_url,
                "provider_error",
                f"OpenAI-compatible error: {data['error']}",
                request_metadata,
            )
            self.last_completion_metadata = error.to_metadata()
            raise error
        self.last_completion_metadata = {
            "prompt_cache_supported": self.supports_prompt_cache,
            "prompt_cache_key": prompt_cache_key,
            "prompt_cache_retention": prompt_cache_retention,
            "image_input_count": image_input_count,
            **request_metadata,
            **_extract_usage_cache_details(data),
        }
        result = _openai_result(
            data,
            self.last_completion_metadata,
            "openai",
            self.model,
            self.base_url,
        )
        if result.text or result.tool_calls:
            return result
        error = _provider_failure(
            "openai",
            self.model,
            self.base_url,
            "empty_response",
            "OpenAI-compatible error: could not extract text from response",
            request_metadata,
        )
        self.last_completion_metadata = error.to_metadata()
        raise error

    def stream_result(
        self,
        request,
        max_new_tokens,
        *,
        cancellation_token=None,
        prompt_cache_key=None,
        prompt_cache_retention=None,
        prompt_cache_prefix_chars=None,
        **kwargs,
    ):
        """Stream OpenAI Responses SSE as provider-neutral model events."""

        del kwargs
        if cancellation_token is not None:
            cancellation_token.raise_if_cancelled()
        self.last_completion_metadata = {}
        conversation = ensure_conversation(request)
        input_items, image_input_count = _openai_conversation_input(conversation)
        if self.supports_prompt_cache and self.supports_explicit_prompt_cache:
            _add_openai_prompt_cache_breakpoint(input_items, prompt_cache_prefix_chars)
        payload = {
            "model": self.model,
            "input": input_items,
            "max_output_tokens": max_new_tokens,
            "stream": True,
        }
        if conversation.tools:
            payload["tools"] = _openai_tool_definitions(conversation)
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.reasoning_effort:
            payload["reasoning"] = {"effort": self.reasoning_effort}
        if self.supports_prompt_cache and prompt_cache_key:
            payload["prompt_cache_key"] = prompt_cache_key
        if (
            self.supports_prompt_cache
            and self.supports_explicit_prompt_cache
            and prompt_cache_prefix_chars
        ):
            payload["prompt_cache_options"] = {"mode": "explicit"}
        elif self.supports_prompt_cache and prompt_cache_retention:
            payload["prompt_cache_retention"] = prompt_cache_retention

        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "Cache-Control": "no-cache",
            "User-Agent": OPENAI_COMPATIBLE_USER_AGENT,
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        http_request = urllib.request.Request(
            self.base_url + "/responses",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        request_metadata = _provider_metadata(
            "openai", self.model, self.base_url, attempts=1, retry_count=0
        )
        metadata = {
            "prompt_cache_supported": self.supports_prompt_cache,
            "prompt_cache_key": prompt_cache_key,
            "prompt_cache_retention": prompt_cache_retention,
            "image_input_count": image_input_count,
            **request_metadata,
        }
        self.last_completion_metadata = metadata

        try:
            response, response_registration, request_metadata = _open_stream_with_retries(
                "openai",
                self.model,
                self.base_url,
                http_request,
                self.timeout,
                cancellation_token=cancellation_token,
                response_controller=self._http_responses,
            )
        except ProviderError as error:
            self.last_completion_metadata = error.to_metadata()
            raise
        metadata = {
            "prompt_cache_supported": self.supports_prompt_cache,
            "prompt_cache_key": prompt_cache_key,
            "prompt_cache_retention": prompt_cache_retention,
            "image_input_count": image_input_count,
            **request_metadata,
        }
        self.last_completion_metadata = metadata

        def build_result(data, result_metadata):
            return _openai_result(
                data,
                result_metadata,
                "openai",
                self.model,
                self.base_url,
            )

        def usage_metadata(usage):
            return _extract_usage_cache_details({"usage": usage or {}})

        def stream_error(message, *, cause=None):
            error = _provider_failure(
                "openai",
                self.model,
                self.base_url,
                "provider_error",
                message,
                metadata,
                cause=cause,
            )
            self.last_completion_metadata = error.to_metadata()
            return error

        managed_response = callable(getattr(response, "__enter__", None))
        response_context = response if managed_response else nullcontext(response)
        try:
            with response_context as body:
                content_type = (getattr(body, "headers", {}) or {}).get(
                    "Content-Type", ""
                )
                if content_type and not content_type.startswith(
                    "text/event-stream"
                ):
                    yield from self._json_stream_fallback(
                        body,
                        metadata,
                        build_result,
                        stream_error,
                    )
                else:
                    yield from decode_openai_stream(
                        body,
                        metadata=metadata,
                        build_result=build_result,
                        stop_reason_from_response=_openai_stop_reason,
                        usage_metadata=usage_metadata,
                        provider_error=stream_error,
                        cancellation_token=cancellation_token,
                    )
                response_registration.raise_if_cancelled()
                if cancellation_token is not None:
                    cancellation_token.raise_if_cancelled()
        except CancellationRequested:
            raise
        except ProviderError as exc:
            self.last_completion_metadata = exc.to_metadata()
            raise
        except Exception as exc:
            response_registration.raise_if_cancelled()
            if cancellation_token is not None:
                cancellation_token.raise_if_cancelled()
            raise stream_error(
                "OpenAI-compatible stream failed", cause=exc
            ) from exc
        finally:
            response_registration.release()

    def _json_stream_fallback(
        self,
        response,
        metadata,
        build_result,
        stream_error,
    ):
        try:
            data = json.loads(response.read().decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise stream_error(
                "OpenAI-compatible stream returned invalid JSON", cause=exc
            ) from exc
        if data.get("error"):
            raise stream_error(f"OpenAI-compatible error: {data['error']}")
        usage = _extract_usage_cache_details(data)
        metadata.update(usage)
        result = build_result(data, metadata)
        if not result.text and not result.tool_calls:
            raise stream_error(
                "OpenAI-compatible error: could not extract text from response"
            )
        yield ModelStreamEvent(kind="message_start", metadata=dict(metadata))
        if data.get("usage"):
            yield ModelStreamEvent(kind="usage", metadata=usage)
        yield ModelStreamEvent(
            kind="done",
            stop_reason=result.stop_reason,
            continuation=tuple(result.continuation or ()),
            metadata=dict(metadata),
            result=result,
        )


class AnthropicCompatibleModelClient:
    def __init__(
        self,
        model,
        base_url,
        api_key,
        temperature,
        timeout,
        strict_tools=False,
        reasoning_effort="",
    ):
        self.model = model
        self.base_url = _normalize_versioned_base_url(base_url)
        self.api_key = api_key
        self.temperature = temperature
        self.timeout = timeout
        self.strict_tools = bool(strict_tools)
        self.reasoning_effort = str(reasoning_effort or "").strip().lower()
        self.supports_prompt_cache = True
        self.last_completion_metadata = {}
        self._http_responses = HttpResponseController()

    def abort(self):
        self._http_responses.abort()

    def complete(self, request, max_new_tokens, **kwargs):
        return self.complete_result(request, max_new_tokens, **kwargs).text

    def complete_result(
        self,
        request,
        max_new_tokens,
        prompt_cache_key=None,
        prompt_cache_retention=None,
        prompt_cache_prefix_chars=None,
        cancellation_token=None,
    ):
        del prompt_cache_key, prompt_cache_retention
        self.last_completion_metadata = {}
        conversation = ensure_conversation(request)
        messages, image_input_count = _anthropic_conversation_messages(conversation)
        _add_anthropic_prompt_cache_breakpoint(messages, prompt_cache_prefix_chars)
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_new_tokens,
            "stream": False,
        }
        if conversation.tools:
            payload["tools"] = _anthropic_tool_definitions(conversation)
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.reasoning_effort:
            payload["output_config"] = {"effort": self.reasoning_effort}

        headers = {
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }
        if self.api_key:
            headers["x-api-key"] = self.api_key

        request = urllib.request.Request(
            self.base_url + "/messages",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            body_text, _content_type, request_metadata = _request_with_retries(
                "anthropic",
                self.model,
                self.base_url,
                request,
                self.timeout,
                cancellation_token=cancellation_token,
                response_controller=self._http_responses,
            )
        except ProviderError as exc:
            self.last_completion_metadata = exc.to_metadata()
            raise

        try:
            data = json.loads(body_text)
        except json.JSONDecodeError as exc:
            error = _provider_failure(
                "anthropic",
                self.model,
                self.base_url,
                "invalid_json",
                "Anthropic-compatible error: backend returned non-JSON content that could not be parsed",
                request_metadata,
                cause=exc,
            )
            self.last_completion_metadata = error.to_metadata()
            raise error from exc
        if data.get("error"):
            error = _provider_failure(
                "anthropic",
                self.model,
                self.base_url,
                "provider_error",
                f"Anthropic-compatible error: {data['error']}",
                request_metadata,
            )
            self.last_completion_metadata = error.to_metadata()
            raise error
        self.last_completion_metadata = {
            "image_input_count": image_input_count,
            **request_metadata,
            **_extract_usage_cache_details(data),
        }
        result = _anthropic_result(
            data,
            self.last_completion_metadata,
            self.model,
            self.base_url,
            request_metadata,
        )
        if result.text or result.tool_calls:
            return result
        error = _provider_failure(
            "anthropic",
            self.model,
            self.base_url,
            "empty_response",
            "Anthropic-compatible error: could not extract text from response",
            request_metadata,
        )
        self.last_completion_metadata = error.to_metadata()
        raise error

    def stream_result(
        self,
        request,
        max_new_tokens,
        *,
        cancellation_token=None,
        prompt_cache_key=None,
        prompt_cache_retention=None,
        prompt_cache_prefix_chars=None,
        **kwargs,
    ):
        """Stream Anthropic Messages SSE as provider-neutral model events."""

        del prompt_cache_key, prompt_cache_retention, kwargs
        if cancellation_token is not None:
            cancellation_token.raise_if_cancelled()
        self.last_completion_metadata = {}
        conversation = ensure_conversation(request)
        messages, image_input_count = _anthropic_conversation_messages(conversation)
        _add_anthropic_prompt_cache_breakpoint(messages, prompt_cache_prefix_chars)
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_new_tokens,
            "stream": True,
        }
        if conversation.tools:
            payload["tools"] = _anthropic_tool_definitions(conversation)
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.reasoning_effort:
            payload["output_config"] = {"effort": self.reasoning_effort}
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "anthropic-version": "2023-06-01",
        }
        if self.api_key:
            headers["x-api-key"] = self.api_key
        http_request = urllib.request.Request(
            self.base_url + "/messages",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        request_metadata = _provider_metadata(
            "anthropic", self.model, self.base_url, attempts=1, retry_count=0
        )
        metadata = {"image_input_count": image_input_count, **request_metadata}
        self.last_completion_metadata = metadata

        try:
            response, response_registration, request_metadata = _open_stream_with_retries(
                "anthropic",
                self.model,
                self.base_url,
                http_request,
                self.timeout,
                cancellation_token=cancellation_token,
                response_controller=self._http_responses,
            )
        except ProviderError as error:
            self.last_completion_metadata = error.to_metadata()
            raise
        metadata = {"image_input_count": image_input_count, **request_metadata}
        self.last_completion_metadata = metadata

        def stream_error(message, *, cause=None):
            error = _provider_failure(
                "anthropic",
                self.model,
                self.base_url,
                "provider_error",
                message,
                metadata,
                cause=cause,
            )
            self.last_completion_metadata = error.to_metadata()
            return error

        managed_response = callable(getattr(response, "__enter__", None))
        response_context = response if managed_response else nullcontext(response)
        try:
            with response_context as body:
                content_type = (getattr(body, "headers", {}) or {}).get(
                    "Content-Type", ""
                )
                if content_type and not content_type.startswith(
                    "text/event-stream"
                ):
                    yield from self._json_stream_fallback(
                        body, metadata, request_metadata, stream_error
                    )
                else:
                    yield from decode_anthropic_stream(
                        body,
                        metadata=metadata,
                        provider_error=stream_error,
                        cancellation_token=cancellation_token,
                    )
                response_registration.raise_if_cancelled()
                if cancellation_token is not None:
                    cancellation_token.raise_if_cancelled()
        except CancellationRequested:
            raise
        except ProviderError as exc:
            self.last_completion_metadata = exc.to_metadata()
            raise
        except Exception as exc:
            response_registration.raise_if_cancelled()
            if cancellation_token is not None:
                cancellation_token.raise_if_cancelled()
            raise stream_error("Anthropic-compatible stream failed", cause=exc) from exc
        finally:
            response_registration.release()

    def _json_stream_fallback(self, response, metadata, request_metadata, stream_error):
        try:
            data = json.loads(response.read().decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise stream_error(
                "Anthropic-compatible stream returned invalid JSON", cause=exc
            ) from exc
        if data.get("error"):
            raise stream_error(f"Anthropic-compatible error: {data['error']}")
        usage = _extract_usage_cache_details(data)
        metadata.update(usage)
        result = _anthropic_result(
            data, metadata, self.model, self.base_url, request_metadata
        )
        if not result.text and not result.tool_calls:
            raise stream_error(
                "Anthropic-compatible error: could not extract text from response"
            )
        yield ModelStreamEvent(kind="message_start", metadata=dict(metadata))
        if data.get("usage"):
            yield ModelStreamEvent(kind="usage", metadata=usage)
        yield ModelStreamEvent(
            kind="done",
            stop_reason=result.stop_reason,
            continuation=tuple(result.continuation or ()),
            metadata=dict(metadata),
            result=result,
        )
