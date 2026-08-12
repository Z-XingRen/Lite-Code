import json

from lite import Lite, SessionStore, WorkspaceContext
from lite.providers import ModelStreamEvent
from lite.testing import ScriptedModelClient, ScriptedStreamingModelClient


def build_agent(tmp_path, client):
    (tmp_path / "README.md").write_text("stream fixture\n", encoding="utf-8")
    return Lite(
        model_client=client,
        workspace=WorkspaceContext.build(tmp_path),
        session_store=SessionStore(tmp_path / ".lite" / "sessions"),
        approval_policy="auto",
        auto_dream=False,
    )


def final_stream(*parts, metadata=None):
    return [
        ModelStreamEvent(kind="message_start"),
        *(ModelStreamEvent(kind="text_delta", text_delta=part) for part in parts),
        ModelStreamEvent(kind="usage", metadata=dict(metadata or {})),
        ModelStreamEvent(kind="done", stop_reason="stop"),
    ]


def test_engine_streams_text_usage_and_commits_only_completed_message(tmp_path):
    client = ScriptedStreamingModelClient(
        [
            final_stream(
                "Hel",
                "lo",
                metadata={
                    "input_tokens": 7,
                    "output_tokens": 2,
                    "total_tokens": 9,
                },
            )
        ]
    )
    agent = build_agent(tmp_path, client)

    events = list(agent.engine.run_turn("Say hello"))

    assert [
        event["content"] for event in events if event["type"] == "model_text_delta"
    ] == ["Hel", "lo"]
    assert next(event for event in events if event["type"] == "final")[
        "content"
    ] == "Hello"
    assert agent.last_completion_metadata == {
        "input_tokens": 7,
        "output_tokens": 2,
        "total_tokens": 9,
    }
    assert [
        item["content"]
        for item in agent.session["history"]
        if item["role"] == "assistant"
    ] == ["Hello"]


def test_engine_assembles_tool_argument_fragments_before_execution(tmp_path):
    client = ScriptedStreamingModelClient(
        [
            [
                ModelStreamEvent(kind="message_start"),
                ModelStreamEvent(
                    kind="tool_call_delta",
                    tool_call_index=0,
                    call_id_delta="call_",
                    name_delta="write_",
                    arguments_delta='{"path":"owned.txt",',
                ),
                ModelStreamEvent(
                    kind="tool_call_delta",
                    tool_call_index=0,
                    call_id_delta="stream",
                    name_delta="file",
                    arguments_delta='"content":"complete"}',
                ),
                ModelStreamEvent(kind="done", stop_reason="tool_use"),
            ],
            final_stream("Done."),
        ]
    )
    agent = build_agent(tmp_path, client)

    events = list(agent.engine.run_turn("Write the file"))

    assert (tmp_path / "owned.txt").read_text(encoding="utf-8") == "complete"
    tool_call = next(event for event in events if event["type"] == "tool_call")
    assert tool_call["call_id"] == "call_stream"
    assert tool_call["name"] == "write_file"
    assert tool_call["args"] == {"path": "owned.txt", "content": "complete"}
    assert client.requests[1].turns[0].tool_outputs[0].call_id == "call_stream"


def test_stream_error_discards_partial_assistant_message(tmp_path):
    client = ScriptedStreamingModelClient(
        [
            [
                ModelStreamEvent(kind="message_start"),
                ModelStreamEvent(kind="text_delta", text_delta="partial secret"),
                ModelStreamEvent(
                    kind="error", error=RuntimeError("fake stream failed")
                ),
            ]
        ]
    )
    agent = build_agent(tmp_path, client)

    events = list(agent.engine.run_turn("Fail while streaming"))

    stop = next(event for event in events if event["type"] == "stop")
    assert "model_client_error" in stop["content"]
    assert not any(
        item.get("content") == "partial secret"
        for item in agent.session["history"]
    )
    assert not any(item["role"] == "tool" for item in agent.session["history"])


def test_stream_without_done_is_rejected_without_committing_partial_text(tmp_path):
    client = ScriptedStreamingModelClient(
        [
            [
                ModelStreamEvent(kind="message_start"),
                ModelStreamEvent(kind="text_delta", text_delta="unfinished"),
            ]
        ]
    )
    agent = build_agent(tmp_path, client)

    events = list(agent.engine.run_turn("Return an incomplete stream"))

    stop = next(event for event in events if event["type"] == "stop")
    assert "model_client_error" in stop["content"]
    assert not any(
        item.get("content") == "unfinished" for item in agent.session["history"]
    )


def test_incomplete_stream_is_retried_once_and_can_recover(tmp_path):
    client = ScriptedStreamingModelClient(
        [
            [
                ModelStreamEvent(kind="message_start"),
                ModelStreamEvent(kind="text_delta", text_delta="unfinished"),
            ],
            final_stream("Recovered."),
        ]
    )
    agent = build_agent(tmp_path, client)

    events = list(agent.engine.run_turn("Recover an incomplete stream"))

    assert next(event for event in events if event["type"] == "final")[
        "content"
    ] == "Recovered."
    assert [
        item["content"]
        for item in agent.session["history"]
        if item["role"] == "assistant"
    ] == ["Recovered."]
    retry_events = [
        event
        for event in (
            json.loads(line)
            for line in agent.session_event_bus.path.read_text(
                encoding="utf-8"
            ).splitlines()
        )
        if event["event"] == "model_retry_scheduled"
    ]
    assert len(retry_events) == 1
    assert retry_events[0]["code"] == "ModelStreamProtocolError"


def test_abort_during_tool_arguments_discards_partial_call(tmp_path):
    client = ScriptedStreamingModelClient([])
    agent = build_agent(tmp_path, client)
    client.streams.append(
        [
            ModelStreamEvent(kind="message_start"),
            ModelStreamEvent(
                kind="tool_call_delta",
                tool_call_index=0,
                call_id_delta="call_partial",
                name_delta="write_file",
                arguments_delta='{"path":"must-not-exist.txt",',
            ),
            lambda _token: agent.abort_current_turn(),
            ModelStreamEvent(
                kind="tool_call_delta",
                tool_call_index=0,
                arguments_delta='"content":"unsafe"}',
            ),
            ModelStreamEvent(kind="done", stop_reason="tool_use"),
        ]
    )

    events = list(agent.engine.run_turn("Abort the partial call"))

    stop = next(event for event in events if event["type"] == "stop")
    assert stop["content"] == "Stopped after abort request."
    assert client.abort_count == 1
    assert not (tmp_path / "must-not-exist.txt").exists()
    assert not any(item["role"] == "tool" for item in agent.session["history"])


def test_engine_adapts_non_streaming_model_clients(tmp_path):
    client = ScriptedModelClient(["Compatibility result."])
    agent = build_agent(tmp_path, client)

    events = list(agent.engine.run_turn("Use the compatibility adapter"))

    assert next(event for event in events if event["type"] == "final")[
        "content"
    ] == "Compatibility result."
    assert len(client.requests) == 1
