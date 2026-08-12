import json
from unittest.mock import patch

import pytest

from lite.cancellation import CancellationRequested, CancellationToken
from lite.providers import ModelConversation, ModelStreamEvent
from lite.providers.clients import OpenAICompatibleModelClient
from lite.providers.errors import ProviderError


class StreamingResponse:
    headers = {"Content-Type": "text/event-stream"}

    def __init__(self, events):
        self._body = "".join(
            f"event: {kind}\ndata: {json.dumps(payload)}\n\n"
            for kind, payload in events
        ).encode("utf-8") + b"data: [DONE]\n\n"
        self._offset = 0
        self.readline_calls = 0
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def readline(self):
        self.readline_calls += 1
        if self._offset >= len(self._body):
            return b""
        end = self._body.find(b"\n", self._offset) + 1
        if end == 0:
            end = len(self._body)
        line = self._body[self._offset:end]
        self._offset = end
        return line

    def __iter__(self):
        for offset in range(0, len(self._body), 7):
            yield self._body[offset : offset + 7]

    def close(self):
        self.closed = True


def make_client():
    return OpenAICompatibleModelClient(
        model="gpt-test",
        base_url="https://api.openai.com/v1",
        api_key="sk-test",
        temperature=None,
        timeout=30,
    )


def test_openai_streams_text_usage_incrementally_and_sets_payload():
    response = StreamingResponse(
        [
            ("response.created", {"type": "response.created", "response": {"id": "resp_1"}}),
            ("response.output_text.delta", {"type": "response.output_text.delta", "delta": "Hel"}),
            ("response.output_text.delta", {"type": "response.output_text.delta", "delta": "lo"}),
            (
                "response.completed",
                {
                    "type": "response.completed",
                    "response": {
                        "status": "completed",
                        "usage": {"input_tokens": 4, "output_tokens": 2, "total_tokens": 6},
                    },
                },
            ),
        ]
    )
    captured = {}

    def fake_urlopen(request, timeout):
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        captured["headers"] = dict(request.headers)
        captured["timeout"] = timeout
        return response

    client = make_client()
    with patch("urllib.request.urlopen", fake_urlopen):
        stream = client.stream_result(ModelConversation("Say hello"), 100)
        first = next(stream)
        assert first.kind == ModelStreamEvent(kind="message_start").kind
        assert response.readline_calls == 0
        events = [first, *stream]

    assert [event.kind for event in events] == [
        "message_start",
        "text_delta",
        "text_delta",
        "usage",
        "done",
    ]
    assert [event.text_delta for event in events if event.kind == "text_delta"] == [
        "Hel",
        "lo",
    ]
    assert events[-1].stop_reason == "completed"
    assert client.last_completion_metadata["input_tokens"] == 4
    assert client.last_completion_metadata["total_tokens"] == 6
    assert captured["payload"]["stream"] is True
    assert captured["headers"]["Accept"] == "text/event-stream"
    assert captured["timeout"] == 30
    assert response.closed
    assert response.readline_calls < len(response._body.splitlines()) + 2


def test_openai_gateway_stream_uses_cache_key_and_gpt_5_6_stable_prefix():
    response = StreamingResponse(
        [
            (
                "response.completed",
                {
                    "type": "response.completed",
                    "response": {"status": "completed", "output": []},
                },
            )
        ]
    )
    captured = {}

    def fake_urlopen(request, timeout):
        del timeout
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return response

    client = OpenAICompatibleModelClient(
        model="gpt-5.6-terra",
        base_url="https://gateway.example/v1",
        api_key="sk-test",
        temperature=None,
        timeout=30,
    )
    with patch("urllib.request.urlopen", fake_urlopen):
        list(
            client.stream_result(
                ModelConversation("stable\n\ndynamic"),
                100,
                prompt_cache_key="prefix-hash-123",
                prompt_cache_prefix_chars=len("stable"),
            )
        )

    payload = captured["payload"]
    assert payload["prompt_cache_key"] == "prefix-hash-123"
    assert payload["prompt_cache_options"] == {"mode": "explicit"}
    assert payload["input"][0]["content"] == [
        {
            "type": "input_text",
            "text": "stable",
            "prompt_cache_breakpoint": {"mode": "explicit"},
        },
        {"type": "input_text", "text": "\n\ndynamic"},
    ]


def test_openai_streams_function_call_deltas_and_replay_continuation():
    response = StreamingResponse(
        [
            (
                "response.output_item.added",
                {
                    "type": "response.output_item.added",
                    "output_index": 0,
                    "item": {
                        "type": "function_call",
                        "id": "fc_1",
                        "call_id": "call_1",
                        "name": "read_file",
                        "arguments": "",
                    },
                },
            ),
            (
                "response.function_call_arguments.delta",
                {
                    "type": "response.function_call_arguments.delta",
                    "output_index": 0,
                    "delta": '{"path":"README',
                },
            ),
            (
                "response.function_call_arguments.delta",
                {
                    "type": "response.function_call_arguments.delta",
                    "output_index": 0,
                    "delta": '.md"}',
                },
            ),
            (
                "response.output_item.done",
                {
                    "type": "response.output_item.done",
                    "output_index": 0,
                    "item": {
                        "type": "function_call",
                        "id": "fc_1",
                        "call_id": "call_1",
                        "name": "read_file",
                        "arguments": '{"path":"README.md"}',
                    },
                },
            ),
            (
                "response.completed",
                {
                    "type": "response.completed",
                    "response": {
                        "status": "completed",
                        "output": [
                            {
                                "type": "function_call",
                                "id": "fc_1",
                                "call_id": "call_1",
                                "name": "read_file",
                                "arguments": '{"path":"README.md"}',
                            }
                        ],
                    },
                },
            ),
        ]
    )

    with patch("urllib.request.urlopen", return_value=response):
        events = list(make_client().stream_result(ModelConversation("Inspect"), 100))

    call_events = [event for event in events if event.kind == "tool_call_delta"]
    assert call_events[0].call_id_delta == "call_1"
    assert call_events[0].name_delta == "read_file"
    assert "README" in "".join(event.arguments_delta for event in call_events)
    assert events[-1].continuation == (
        {
            "type": "function_call",
            "id": "fc_1",
            "call_id": "call_1",
            "name": "read_file",
            "arguments": '{"path":"README.md"}',
        },
    )


def test_openai_stream_normalizes_incomplete_response_and_does_not_hide_error():
    response = StreamingResponse(
        [
            ("response.created", {"type": "response.created", "response": {}}),
            (
                "response.incomplete",
                {
                    "type": "response.incomplete",
                    "response": {
                        "status": "incomplete",
                        "incomplete_details": {"reason": "max_output_tokens"},
                    },
                },
            ),
        ]
    )

    with patch("urllib.request.urlopen", return_value=response):
        events = list(make_client().stream_result(ModelConversation("Long"), 1))

    assert events[-1].kind == "done"
    assert events[-1].stop_reason == "length"

    failed = StreamingResponse(
        [("error", {"type": "error", "error": {"message": "bad gateway"}})]
    )
    client = make_client()
    with patch("urllib.request.urlopen", return_value=failed):
        with pytest.raises(ProviderError) as exc:
            list(client.stream_result(ModelConversation("Fail"), 1))

    assert exc.value.code == "provider_error"


def test_openai_stream_checks_cancellation_before_request_and_between_frames():
    token = CancellationToken()
    token.cancel()
    with patch("urllib.request.urlopen") as urlopen:
        with pytest.raises(CancellationRequested):
            list(make_client().stream_result(ModelConversation("Stop"), 1, cancellation_token=token))
    urlopen.assert_not_called()


def test_openai_accepts_chat_completion_style_sse_chunks():
    response = StreamingResponse(
        [
            (
                "",
                {
                    "choices": [
                        {"index": 0, "delta": {"role": "assistant", "content": "Hi"}}
                    ]
                },
            ),
            (
                "",
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_1",
                                        "function": {
                                            "name": "read_file",
                                            "arguments": '{"path":"README.md"}',
                                        },
                                    }
                                ]
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 3,
                        "completion_tokens": 4,
                        "total_tokens": 7,
                    },
                },
            ),
        ]
    )

    with patch("urllib.request.urlopen", return_value=response):
        events = list(make_client().stream_result(ModelConversation("Inspect"), 100))

    assert [event.text_delta for event in events if event.kind == "text_delta"] == ["Hi"]
    assert events[-1].stop_reason == "tool_calls"
    assert events[-1].continuation[0]["call_id"] == "call_1"
    assert events[-1].continuation[0]["arguments"] == '{"path":"README.md"}'


def test_openai_stream_adapts_json_response_from_non_streaming_gateway():
    class JsonResponse:
        headers = {"Content-Type": "application/json"}

        def read(self):
            return json.dumps({"status": "completed", "output_text": "fallback"}).encode(
                "utf-8"
            )

    client = make_client()
    with patch("urllib.request.urlopen", return_value=JsonResponse()):
        events = list(client.stream_result(ModelConversation("Fallback"), 10))

    assert [event.kind for event in events] == ["message_start", "done"]
    assert events[-1].result.text == "fallback"


def test_openai_stream_accepts_chunked_sse_iterators():
    response = StreamingResponse(
        [
            ("response.output_text.delta", {"type": "response.output_text.delta", "delta": "chunked"}),
            ("response.completed", {"type": "response.completed", "response": {"status": "completed"}}),
        ]
    )
    response.readline = None

    with patch("urllib.request.urlopen", return_value=response):
        events = list(make_client().stream_result(ModelConversation("Chunk"), 10))

    assert "".join(event.text_delta for event in events if event.kind == "text_delta") == "chunked"
