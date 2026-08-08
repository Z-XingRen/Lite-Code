import threading
from unittest.mock import patch

import pytest

from lite import Lite, SessionStore, WorkspaceContext
from lite.cancellation import CancellationRequested, CancellationToken
from lite.providers import ModelConversation
from lite.providers.clients import (
    AnthropicCompatibleModelClient,
    OpenAICompatibleModelClient,
)


CLIENT_TYPES = (OpenAICompatibleModelClient, AnthropicCompatibleModelClient)


class BlockingResponse:
    def __init__(self, *, content_type="text/event-stream"):
        self.headers = {"Content-Type": content_type}
        self.read_started = threading.Event()
        self.closed_event = threading.Event()
        self.closed = False
        self.readline_calls = 0
        self.read_calls = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def readline(self):
        self.readline_calls += 1
        self.read_started.set()
        self.closed_event.wait(5)
        raise OSError("response closed")

    def read(self):
        self.read_calls += 1
        self.read_started.set()
        self.closed_event.wait(5)
        raise OSError("response closed")

    def close(self):
        self.closed = True
        self.closed_event.set()


class EmptyResponse:
    headers = {"Content-Type": "text/event-stream"}

    def __init__(self):
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def readline(self):
        return b""

    def close(self):
        self.closed = True


def make_client(client_type):
    return client_type(
        model="test-model",
        base_url="https://example.test/v1",
        api_key="sk-test",
        temperature=None,
        timeout=30,
    )


def run_in_thread(operation):
    outcome = {}

    def target():
        try:
            outcome["value"] = operation()
        except BaseException as exc:
            outcome["error"] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    return thread, outcome


@pytest.mark.parametrize("client_type", CLIENT_TYPES)
def test_stream_token_cancel_closes_blocking_response_without_retry(client_type):
    response = BlockingResponse()
    token = CancellationToken()
    calls = []

    def fake_urlopen(_request, timeout):
        calls.append(timeout)
        return response

    client = make_client(client_type)
    with patch("urllib.request.urlopen", fake_urlopen):
        thread, outcome = run_in_thread(
            lambda: list(
                client.stream_result(
                    ModelConversation("wait"),
                    20,
                    cancellation_token=token,
                )
            )
        )
        assert response.read_started.wait(1)
        token.cancel()
        thread.join(1)
        stopped_after_cancel = not thread.is_alive()
        if thread.is_alive():
            response.close()
            thread.join(2)

    assert stopped_after_cancel
    assert not thread.is_alive()
    assert isinstance(outcome.get("error"), CancellationRequested)
    assert calls == [30]
    assert response.closed
    assert response.readline_calls == 1


@pytest.mark.parametrize("client_type", CLIENT_TYPES)
def test_non_streaming_token_cancel_closes_blocking_response_without_retry(
    client_type,
):
    response = BlockingResponse(content_type="application/json")
    token = CancellationToken()
    calls = []

    def fake_urlopen(_request, timeout):
        calls.append(timeout)
        return response

    client = make_client(client_type)
    with patch("urllib.request.urlopen", fake_urlopen):
        thread, outcome = run_in_thread(
            lambda: client.complete_result(
                ModelConversation("wait"),
                20,
                cancellation_token=token,
            )
        )
        assert response.read_started.wait(1)
        token.cancel()
        thread.join(1)
        stopped_after_cancel = not thread.is_alive()
        if thread.is_alive():
            response.close()
            thread.join(2)

    assert stopped_after_cancel
    assert not thread.is_alive()
    assert isinstance(outcome.get("error"), CancellationRequested)
    assert calls == [30]
    assert response.closed
    assert response.read_calls == 1


@pytest.mark.parametrize("client_type", CLIENT_TYPES)
@pytest.mark.parametrize("streaming", (True, False))
def test_client_abort_closes_active_response_without_retry(client_type, streaming):
    content_type = "text/event-stream" if streaming else "application/json"
    response = BlockingResponse(content_type=content_type)
    calls = []

    def fake_urlopen(_request, timeout):
        calls.append(timeout)
        return response

    client = make_client(client_type)
    if streaming:
        def operation():
            return list(client.stream_result(ModelConversation("wait"), 20))
    else:
        def operation():
            return client.complete_result(ModelConversation("wait"), 20)
    with patch("urllib.request.urlopen", fake_urlopen):
        thread, outcome = run_in_thread(operation)
        assert response.read_started.wait(1)
        client.abort()
        thread.join(1)
        stopped_after_abort = not thread.is_alive()
        if thread.is_alive():
            response.close()
            thread.join(2)

    assert stopped_after_abort
    assert not thread.is_alive()
    assert isinstance(outcome.get("error"), CancellationRequested)
    assert calls == [30]
    assert response.closed


@pytest.mark.parametrize("client_type", CLIENT_TYPES)
def test_runtime_abort_interrupts_active_provider_http_read(tmp_path, client_type):
    response = BlockingResponse()
    client = make_client(client_type)
    workspace = tmp_path / client_type.__name__
    workspace.mkdir()
    agent = Lite(
        model_client=client,
        workspace=WorkspaceContext.build(workspace),
        session_store=SessionStore(workspace / ".lite" / "sessions"),
        approval_policy="auto",
        auto_dream=False,
    )

    with patch("urllib.request.urlopen", return_value=response) as urlopen:
        thread, outcome = run_in_thread(
            lambda: list(agent.engine.run_turn("wait for the provider"))
        )
        assert response.read_started.wait(1)
        agent.abort_current_turn()
        thread.join(1)
        stopped_after_abort = not thread.is_alive()
        if thread.is_alive():
            response.close()
            thread.join(2)

    assert stopped_after_abort
    assert not thread.is_alive()
    assert "error" not in outcome
    assert next(event for event in outcome["value"] if event["type"] == "stop")[
        "content"
    ] == "Stopped after abort request."
    assert urlopen.call_count == 1
    assert response.closed
    assert [
        item["content"]
        for item in agent.session["history"]
        if item["role"] == "assistant"
    ] == ["Stopped after abort request."]


@pytest.mark.parametrize("client_type", CLIENT_TYPES)
def test_cancel_race_after_response_open_closes_before_reading(client_type):
    response = EmptyResponse()
    token = CancellationToken()

    def fake_urlopen(_request, timeout):
        assert timeout == 30
        token.cancel()
        return response

    with patch("urllib.request.urlopen", fake_urlopen), pytest.raises(
        CancellationRequested
    ):
        list(
            make_client(client_type).stream_result(
                ModelConversation("cancel at open"),
                20,
                cancellation_token=token,
            )
        )

    assert response.closed
