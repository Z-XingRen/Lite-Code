import json
from unittest.mock import patch

import pytest

from lite.cancellation import CancellationRequested, CancellationToken
from lite.providers import ModelConversation, collect_model_stream
from lite.providers.clients import AnthropicCompatibleModelClient
from lite.providers.errors import ProviderError


class StreamingResponse:
    headers = {"Content-Type": "text/event-stream"}

    def __init__(self, events):
        self._body = "".join(
            f"event: {kind}\ndata: {json.dumps(payload)}\n\n"
            for kind, payload in events
        ).encode("utf-8")
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
        line = self._body[self._offset:end]
        self._offset = end
        return line

    def close(self):
        self.closed = True


def make_client():
    return AnthropicCompatibleModelClient(
        model="claude-test",
        base_url="https://api.anthropic.com/v1",
        api_key="sk-ant-test",
        temperature=None,
        timeout=30,
    )


def test_anthropic_streams_text_and_usage_incrementally():
    response = StreamingResponse(
        [
            (
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": "msg_1",
                        "model": "claude-test",
                        "usage": {
                            "input_tokens": 4,
                            "output_tokens": 0,
                            "cache_read_input_tokens": 1,
                            "cache_creation_input_tokens": 2,
                        },
                    },
                },
            ),
            (
                "content_block_start",
                {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            ),
            (
                "content_block_delta",
                {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hel"}},
            ),
            (
                "content_block_delta",
                {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "lo"}},
            ),
            (
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn"},
                    "usage": {"output_tokens": 2},
                },
            ),
            ("message_stop", {"type": "message_stop"}),
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
        assert first.kind == "message_start"
        assert response.readline_calls == 0
        events = [first, *stream]

    assert [event.kind for event in events] == [
        "message_start",
        "usage",
        "text_delta",
        "text_delta",
        "usage",
        "done",
    ]
    assert "".join(event.text_delta for event in events if event.kind == "text_delta") == "Hello"
    assert events[-1].stop_reason == "end_turn"
    assert events[-1].continuation == (({"type": "text", "text": "Hello"}),)
    assert client.last_completion_metadata["input_tokens"] == 4
    assert client.last_completion_metadata["output_tokens"] == 2
    assert client.last_completion_metadata["cached_tokens"] == 1
    assert client.last_completion_metadata["cache_write_tokens"] == 2
    assert client.last_completion_metadata["cache_hit"] is True
    assert captured["payload"]["stream"] is True
    assert captured["headers"]["Accept"] == "text/event-stream"
    assert captured["headers"]["X-api-key"] == "sk-ant-test"
    assert captured["timeout"] == 30
    assert response.closed


def test_anthropic_stream_marks_stable_prefix_for_prompt_cache():
    response = StreamingResponse(
        [
            ("message_start", {"type": "message_start", "message": {"usage": {"input_tokens": 4}}}),
            ("message_stop", {"type": "message_stop"}),
        ]
    )
    captured = {}

    def fake_urlopen(request, timeout):
        del timeout
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return response

    with patch("urllib.request.urlopen", fake_urlopen):
        list(
            make_client().stream_result(
                ModelConversation("stable\n\ndynamic"),
                100,
                prompt_cache_prefix_chars=len("stable"),
            )
        )

    assert captured["payload"]["messages"][0]["content"] == [
        {
            "type": "text",
            "text": "stable",
            "cache_control": {"type": "ephemeral"},
        },
        {"type": "text", "text": "\n\ndynamic"},
    ]


def test_anthropic_stream_records_deepseek_cache_hits():
    response = StreamingResponse(
        [
            (
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "usage": {
                            "input_tokens": 1000,
                            "prompt_cache_hit_tokens": 800,
                        }
                    },
                },
            ),
            ("message_stop", {"type": "message_stop"}),
        ]
    )
    client = make_client()
    with patch("urllib.request.urlopen", return_value=response):
        list(client.stream_result(ModelConversation("hello"), 100))

    assert client.last_completion_metadata["cached_tokens"] == 800
    assert client.last_completion_metadata["cache_hit"] is True


def test_anthropic_streams_tool_use_and_replay_continuation():
    stream_events = [
        ("message_start", {"type": "message_start", "message": {"id": "msg_2", "usage": {"input_tokens": 1}}}),
        (
            "content_block_start",
            {"type": "content_block_start", "index": 0, "content_block": {"type": "tool_use", "id": "tool_1", "name": "read_file", "input": {}}},
        ),
        (
            "content_block_delta",
            {"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": '{"path":"README'}},
        ),
        (
            "content_block_delta",
            {"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": '.md"}'}},
        ),
        ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 3}}),
        ("message_stop", {"type": "message_stop"}),
    ]
    response = StreamingResponse(stream_events)
    with patch("urllib.request.urlopen", return_value=response):
        events = list(make_client().stream_result(ModelConversation("Inspect"), 100))

    calls = [event for event in events if event.kind == "tool_call_delta"]
    assert calls[0].call_id_delta == "tool_1"
    assert calls[0].name_delta == "read_file"
    assert "README" in "".join(event.arguments_delta for event in calls)
    assert events[-1].continuation == (
        {"type": "tool_use", "id": "tool_1", "name": "read_file", "input": {"path": "README.md"}},
    )
    with patch("urllib.request.urlopen", return_value=StreamingResponse(stream_events)):
        result = collect_model_stream(
            make_client(), ModelConversation("Inspect"), 100
        )
    assert result.tool_calls[0].call_id == "tool_1"
    assert result.tool_calls[0].arguments == {"path": "README.md"}


def test_anthropic_stream_normalizes_max_tokens_and_surfaces_error():
    response = StreamingResponse(
        [
            ("message_start", {"type": "message_start", "message": {"id": "msg_3"}}),
            ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "max_tokens"}}),
            ("message_stop", {"type": "message_stop"}),
        ]
    )
    with patch("urllib.request.urlopen", return_value=response):
        events = list(make_client().stream_result(ModelConversation("Long"), 1))
    assert events[-1].stop_reason == "length"

    failed = StreamingResponse([("error", {"type": "error", "error": {"message": "bad gateway"}})])
    client = make_client()
    with patch("urllib.request.urlopen", return_value=failed):
        with pytest.raises(ProviderError) as exc:
            list(client.stream_result(ModelConversation("Fail"), 1))
    assert exc.value.code == "provider_error"


def test_anthropic_stream_checks_cancellation_before_request_and_between_frames():
    token = CancellationToken()
    token.cancel()
    with patch("urllib.request.urlopen") as urlopen:
        with pytest.raises(CancellationRequested):
            list(make_client().stream_result(ModelConversation("Stop"), 1, cancellation_token=token))
    urlopen.assert_not_called()

    token = CancellationToken()
    response = StreamingResponse(
        [
            ("message_start", {"type": "message_start", "message": {"id": "msg_4"}}),
            ("ping", {"type": "ping"}),
            ("message_stop", {"type": "message_stop"}),
        ]
    )
    original_readline = response.readline

    def cancelling_readline():
        line = original_readline()
        if response.readline_calls == 4:
            token.cancel()
        return line

    response.readline = cancelling_readline
    with patch("urllib.request.urlopen", return_value=response):
        stream = make_client().stream_result(
            ModelConversation("Stop later"), 1, cancellation_token=token
        )
        assert next(stream).kind == "message_start"
        with pytest.raises(CancellationRequested):
            list(stream)
    assert response.closed


def test_anthropic_stream_adapts_json_response_from_non_streaming_gateway():
    class JsonResponse:
        headers = {"Content-Type": "application/json"}

        def read(self):
            return json.dumps({"content": [{"type": "text", "text": "fallback"}], "stop_reason": "end_turn"}).encode("utf-8")

    client = make_client()
    with patch("urllib.request.urlopen", return_value=JsonResponse()):
        events = list(client.stream_result(ModelConversation("Fallback"), 10))
    assert [event.kind for event in events] == ["message_start", "done"]
    assert events[-1].result.text == "fallback"
