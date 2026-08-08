"""Engine integration for provider-neutral model stream events."""

from ..providers.streaming import ModelStreamAccumulator, stream_model_events

_ENGINE_EVENT_TYPES = {
    "message_start": "model_stream_start",
    "text_delta": "model_text_delta",
    "tool_call_delta": "model_tool_call_delta",
    "usage": "model_usage",
    "done": "model_stream_done",
}


def consume_model_stream(
    engine, task_state, request, max_new_tokens, **kwargs
):
    agent = engine.runtime
    token = agent.current_cancellation_token
    accumulator = ModelStreamAccumulator()
    expose_stream_events = callable(
        getattr(agent.model_client, "stream_result", None)
    )
    for event in stream_model_events(
        agent.model_client,
        request,
        max_new_tokens,
        cancellation_token=token,
        **kwargs,
    ):
        payload = _engine_event_payload(event, task_state.run_id)
        if expose_stream_events:
            agent.session_event_bus.emit(
                "model_stream_event",
                {**payload, "event": event.kind},
            )
        accumulator.add(event)
        if expose_stream_events and payload["type"]:
            yield payload
    return accumulator.finish(token)


def _engine_event_payload(event, run_id):
    payload = {
        "type": _ENGINE_EVENT_TYPES.get(event.kind, ""),
        "run_id": run_id,
    }
    if event.kind == "text_delta":
        payload["content"] = event.text_delta
    elif event.kind == "tool_call_delta":
        payload.update(
            {
                "tool_call_index": event.tool_call_index,
                "call_id_delta": event.call_id_delta,
                "name_delta": event.name_delta,
                "arguments_delta": event.arguments_delta,
            }
        )
    elif event.kind == "usage":
        payload["metadata"] = dict(event.metadata or {})
    elif event.kind == "done":
        payload["stop_reason"] = event.stop_reason
    return payload
