"""Provider-neutral model stream events and deterministic aggregation."""

import json
from dataclasses import dataclass, field, replace

from ..cancellation import CancellationRequested
from .base import ModelResult, ToolCall, complete_model, normalize_stop_reason

STREAM_EVENT_KINDS = frozenset(
    {
        "message_start",
        "text_delta",
        "tool_call_delta",
        "usage",
        "done",
        "error",
    }
)


class ModelStreamProtocolError(RuntimeError):
    """Raised when a provider emits an incomplete or invalid event sequence."""


@dataclass(frozen=True)
class ModelStreamEvent:
    kind: str
    text_delta: str = ""
    tool_call_index: int = -1
    call_id_delta: str = ""
    name_delta: str = ""
    arguments_delta: str = ""
    stop_reason: str = ""
    continuation: tuple[dict, ...] = ()
    metadata: dict = field(default_factory=dict)
    result: ModelResult | None = None
    error: BaseException | None = None


class ModelStreamAccumulator:
    def __init__(self):
        self.started = False
        self.done = False
        self.text_parts = []
        self.tool_parts = {}
        self.metadata = {}
        self.stop_reason = ""
        self.continuation = ()
        self.completed_result = None

    def add(self, event):
        if not isinstance(event, ModelStreamEvent):
            raise ModelStreamProtocolError("model stream yielded a non-event value")
        if event.kind not in STREAM_EVENT_KINDS:
            raise ModelStreamProtocolError(
                f"unknown model stream event: {event.kind}"
            )
        if self.done:
            raise ModelStreamProtocolError("model stream emitted an event after done")
        if event.kind == "error":
            if isinstance(event.error, BaseException):
                raise event.error
            raise ModelStreamProtocolError("model stream error event has no exception")
        if event.kind == "message_start":
            if self.started:
                raise ModelStreamProtocolError("duplicate model stream message_start")
            self.started = True
            self.metadata.update(event.metadata or {})
            return
        if not self.started:
            raise ModelStreamProtocolError(
                f"model stream {event.kind} arrived before message_start"
            )
        if event.kind == "text_delta":
            self.text_parts.append(str(event.text_delta or ""))
        elif event.kind == "tool_call_delta":
            self._add_tool_delta(event)
        elif event.kind == "usage":
            self.metadata.update(event.metadata or {})
        elif event.kind == "done":
            self.done = True
            self.stop_reason = str(event.stop_reason or "")
            self.continuation = tuple(event.continuation or ())
            self.metadata.update(event.metadata or {})
            self.completed_result = event.result

    def finish(self, cancellation_token=None):
        if cancellation_token is not None:
            cancellation_token.raise_if_cancelled()
        if not self.done:
            raise ModelStreamProtocolError("model stream ended before done")
        if self.completed_result is not None:
            if self.text_parts or self.tool_parts:
                raise ModelStreamProtocolError(
                    "done result cannot be combined with preceding content deltas"
                )
            result = self.completed_result
            return replace(
                result,
                stop_reason=normalize_stop_reason(
                    self.stop_reason or result.stop_reason
                ),
                metadata={**dict(result.metadata or {}), **self.metadata},
            )
        return ModelResult(
            text="".join(self.text_parts),
            tool_calls=self._tool_calls(),
            stop_reason=normalize_stop_reason(self.stop_reason),
            continuation=self.continuation,
            metadata=dict(self.metadata),
        )

    def _add_tool_delta(self, event):
        index = int(event.tool_call_index)
        if index < 0:
            raise ModelStreamProtocolError("tool_call_delta requires a non-negative index")
        parts = self.tool_parts.setdefault(
            index, {"call_id": [], "name": [], "arguments": []}
        )
        parts["call_id"].append(str(event.call_id_delta or ""))
        parts["name"].append(str(event.name_delta or ""))
        parts["arguments"].append(str(event.arguments_delta or ""))

    def _tool_calls(self):
        calls = []
        for index in sorted(self.tool_parts):
            parts = self.tool_parts[index]
            call_id = "".join(parts["call_id"])
            name = "".join(parts["name"])
            raw_arguments = "".join(parts["arguments"]) or "{}"
            if not call_id or not name:
                raise ModelStreamProtocolError(
                    f"tool call {index} is missing call id or name"
                )
            try:
                arguments = json.loads(raw_arguments)
            except json.JSONDecodeError as exc:
                raise ModelStreamProtocolError(
                    f"tool call {index} has invalid JSON arguments"
                ) from exc
            if not isinstance(arguments, dict):
                raise ModelStreamProtocolError(
                    f"tool call {index} arguments must decode to an object"
                )
            calls.append(ToolCall(call_id=call_id, name=name, arguments=arguments))
        return tuple(calls)


def stream_model_events(
    model_client, request, max_new_tokens, *, cancellation_token=None, **kwargs
):
    """Yield native stream events or adapt an existing complete_result client."""
    if cancellation_token is not None:
        cancellation_token.raise_if_cancelled()
    stream_result = getattr(model_client, "stream_result", None)
    if callable(stream_result):
        try:
            for event in stream_result(
                request,
                max_new_tokens,
                cancellation_token=cancellation_token,
                **kwargs,
            ):
                if cancellation_token is not None:
                    cancellation_token.raise_if_cancelled()
                if not isinstance(event, ModelStreamEvent):
                    raise ModelStreamProtocolError(
                        "model stream yielded a non-event value"
                    )
                yield event
                if event.kind == "error":
                    return
        except CancellationRequested:
            raise
        except Exception as exc:
            yield ModelStreamEvent(kind="error", error=exc)
        return

    yield ModelStreamEvent(kind="message_start")
    try:
        result = complete_model(model_client, request, max_new_tokens, **kwargs)
        if cancellation_token is not None:
            cancellation_token.raise_if_cancelled()
    except CancellationRequested:
        raise
    except Exception as exc:
        yield ModelStreamEvent(kind="error", error=exc)
        return
    yield ModelStreamEvent(
        kind="done",
        stop_reason=result.stop_reason,
        metadata=dict(result.metadata or {}),
        result=result,
    )


def collect_model_stream(
    model_client, request, max_new_tokens, *, cancellation_token=None, **kwargs
):
    accumulator = ModelStreamAccumulator()
    for event in stream_model_events(
        model_client,
        request,
        max_new_tokens,
        cancellation_token=cancellation_token,
        **kwargs,
    ):
        accumulator.add(event)
    return accumulator.finish(cancellation_token)
