import io
import urllib.error
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from lite.cancellation import CancellationRequested, CancellationToken
from lite.providers import ModelConversation
from lite.providers.clients import (
    AnthropicCompatibleModelClient,
    OpenAICompatibleModelClient,
)
from lite.providers.errors import ProviderError
from lite.providers.retry import (
    RetryExhausted,
    RetryPolicy,
    calculate_retry_delay,
    classify_retry,
    retry_after_seconds,
    run_with_retries,
)


def http_error(status, headers=None):
    return urllib.error.HTTPError(
        "https://example.test/v1/request",
        status,
        "provider error",
        headers or {},
        io.BytesIO(b'{"error":"busy"}'),
    )


@pytest.mark.parametrize(
    ("error", "reason"),
    [
        (http_error(408), "timeout"),
        (http_error(429), "rate_limited"),
        (http_error(500), "server_error"),
        (http_error(503), "server_error"),
        (http_error(400), None),
        (http_error(401), None),
        (http_error(501), None),
        (ProviderError("busy", code="rate_limited"), "rate_limited"),
        (ProviderError("overloaded", code="overloaded"), "overloaded"),
        (ProviderError("bad token", code="auth_error", retryable=True), None),
        (ProviderError("too long", code="context_overflow"), None),
        (urllib.error.URLError("connection reset"), "network_error"),
        (TimeoutError("timed out"), "timeout"),
    ],
)
def test_retry_classifier_table(error, reason):
    assert classify_retry(error) == reason


def test_retry_after_parses_ms_case_insensitively_and_http_date():
    assert retry_after_seconds({"Retry-After-Ms": "1250"}) == pytest.approx(1.25)
    assert retry_after_seconds({"retry-after": "2"}) == 2
    assert retry_after_seconds({"Retry-After": "-1"}) is None
    assert retry_after_seconds({"Retry-After": "n/a"}) is None

    target = datetime(2026, 8, 8, 0, 0, 5, tzinfo=timezone.utc).timestamp()
    now = datetime(2026, 8, 8, 0, 0, 0, tzinfo=timezone.utc).timestamp()
    assert retry_after_seconds(
        {"Retry-After": "Sat, 08 Aug 2026 00:00:05 GMT"}, now=now
    ) == pytest.approx(target - now)


def test_retry_delay_is_exponential_jittered_and_bounded():
    policy = RetryPolicy(
        max_retries=4,
        base_delay_seconds=1,
        max_delay_seconds=2.5,
        jitter_ratio=0.1,
    )
    assert calculate_retry_delay(0, policy=policy, random_fn=lambda: 0.5) == 1
    assert calculate_retry_delay(1, policy=policy, random_fn=lambda: 0.5) == 2
    assert calculate_retry_delay(4, policy=policy, random_fn=lambda: 0.5) == 2.5
    assert calculate_retry_delay(0, policy=policy, random_fn=lambda: 0) == pytest.approx(
        0.9
    )


def test_run_with_retries_records_history_and_attempt_budget_without_real_sleep():
    failures = [http_error(503), http_error(429)]
    attempts = []
    sleeps = []
    retry_events = []

    def operation(attempt):
        attempts.append(attempt)
        if failures:
            raise failures.pop(0)
        return "ok"

    result = run_with_retries(
        operation,
        policy=RetryPolicy(jitter_ratio=0),
        sleep_fn=sleeps.append,
        on_retry=retry_events.append,
    )

    assert result.value == "ok"
    assert attempts == [1, 2, 3]
    assert result.attempts == 3
    assert result.retry_count == 2
    assert [event["reason"] for event in retry_events] == [
        "server_error",
        "rate_limited",
    ]
    assert sleeps == [0.5, 1.0]
    assert result.history == tuple(retry_events)

    with pytest.raises(RetryExhausted) as exc:
        def always_unavailable(_attempt):
            raise http_error(503)

        run_with_retries(
            always_unavailable,
            policy=RetryPolicy(max_retries=1, jitter_ratio=0),
            sleep_fn=lambda _delay: None,
        )
    assert exc.value.attempts == 2
    assert exc.value.retry_count == 1
    assert isinstance(exc.value.cause, urllib.error.HTTPError)


def test_backoff_cancellation_is_immediate_and_does_not_start_next_attempt():
    token = CancellationToken()
    attempts = []

    def operation(attempt):
        attempts.append(attempt)
        raise http_error(503)

    def cancel_after_schedule(_event):
        token.cancel()

    with pytest.raises(CancellationRequested):
        run_with_retries(
            operation,
            cancellation_token=token,
            policy=RetryPolicy(jitter_ratio=0),
            on_retry=cancel_after_schedule,
        )
    assert attempts == [1]


def test_successful_http_response_with_transient_provider_error_is_retried():
    calls = {"count": 0}

    class Response:
        headers = {"Content-Type": "application/json"}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            calls["count"] += 1
            if calls["count"] == 1:
                return b'{"error":{"type":"rate_limit_error","message":"busy"}}'
            return b'{"output_text":"<final>ok</final>"}'

    client = OpenAICompatibleModelClient(
        model="test-model",
        base_url="https://example.test/v1",
        api_key="sk-test",
        temperature=None,
        timeout=10,
    )
    with patch("urllib.request.urlopen", return_value=Response()), patch(
        "lite.providers.clients.time.sleep"
    ):
        assert client.complete("hello", 20) == "<final>ok</final>"
    assert calls["count"] == 2
    assert client.last_completion_metadata["provider_retry_history"][0]["reason"] == (
        "rate_limited"
    )


def test_successful_http_response_with_context_error_is_not_retried():
    calls = {"count": 0}

    class Response:
        headers = {"Content-Type": "application/json"}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            calls["count"] += 1
            return b'{"error":{"type":"context_length_error","message":"too long"}}'

    client = OpenAICompatibleModelClient(
        model="test-model",
        base_url="https://example.test/v1",
        api_key="sk-test",
        temperature=None,
        timeout=10,
    )
    with patch("urllib.request.urlopen", return_value=Response()), pytest.raises(
        ProviderError
    ) as exc:
        client.complete("hello", 20)
    assert calls["count"] == 1
    assert exc.value.code == "context_overflow"
    assert exc.value.retryable is False


def test_non_streaming_request_passes_cancellation_through_backoff():
    token = CancellationToken()
    calls = []

    def fake_urlopen(_request, timeout):
        del timeout
        calls.append(True)
        raise http_error(503)

    def cancel_wait(_delay):
        token.cancel()
        return True

    token.wait = cancel_wait
    client = OpenAICompatibleModelClient(
        model="test-model",
        base_url="https://example.test/v1",
        api_key="sk-test",
        temperature=None,
        timeout=10,
    )
    with patch("urllib.request.urlopen", fake_urlopen), pytest.raises(
        CancellationRequested
    ):
        client.complete_result("hello", 20, cancellation_token=token)
    assert len(calls) == 1


class _SseResponse:
    headers = {"Content-Type": "text/event-stream"}

    def __init__(self):
        self._body = (
            b'data: {"type":"response.output_text.delta","delta":"ok"}\n\n'
            b'data: {"type":"response.completed","response":{"output_text":"ok"}}\n\n'
            b"data: [DONE]\n\n"
        )
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def readline(self):
        if not self._body:
            return b""
        line, _, self._body = self._body.partition(b"\n")
        return line + b"\n"

    def close(self):
        self.closed = True


@pytest.mark.parametrize(
    ("client_type", "path"),
    [
        (OpenAICompatibleModelClient, "/responses"),
        (AnthropicCompatibleModelClient, "/messages"),
    ],
)
def test_stream_initial_http_request_uses_shared_retry_policy(client_type, path):
    calls = []
    response = _SseResponse()

    def fake_urlopen(request, timeout):
        calls.append((request.full_url, timeout))
        if len(calls) == 1:
            raise http_error(503, {"Retry-After-Ms": "0"})
        return response

    client = client_type(
        model="test-model",
        base_url="https://example.test/v1",
        api_key="sk-test",
        temperature=None,
        timeout=10,
    )
    with patch("urllib.request.urlopen", fake_urlopen):
        events = list(client.stream_result(ModelConversation("hello"), 20))

    assert calls == [(f"https://example.test/v1{path}", 10)] * 2
    assert events[0].kind == "message_start"
    assert client.last_completion_metadata["provider_attempts"] == 2
    assert client.last_completion_metadata["provider_retry_count"] == 1
    assert client.last_completion_metadata["provider_retry_history"][0]["reason"] == (
        "server_error"
    )
    assert response.closed
