"""Regression coverage for tool calls from truncated model responses."""

import json
from dataclasses import replace
from unittest.mock import patch

import pytest

from lite import Lite, SessionStore, WorkspaceContext
from lite.providers import ModelResult, ToolCall
from lite.providers.clients import (
    AnthropicCompatibleModelClient,
    OpenAICompatibleModelClient,
)
from lite.testing import ScriptedModelClient


class JsonResponse:
    headers = {"Content-Type": "application/json"}

    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def openai_client():
    return OpenAICompatibleModelClient(
        model="gpt-test",
        base_url="https://api.openai.com/v1",
        api_key="sk-test",
        temperature=None,
        timeout=30,
    )


def anthropic_client():
    return AnthropicCompatibleModelClient(
        model="claude-test",
        base_url="https://api.anthropic.com/v1",
        api_key="sk-test",
        temperature=None,
        timeout=30,
    )


@pytest.mark.parametrize("raw_stop_reason", ["length", "incomplete"])
def test_openai_normalizes_truncated_stop_reasons(raw_stop_reason):
    response = JsonResponse(
        {
            "status": raw_stop_reason,
            "output": [
                {
                    "type": "function_call",
                    "call_id": "call_openai",
                    "name": "write_file",
                    "arguments": '{"path":"result.txt","content":"partial"}',
                }
            ],
        }
    )

    with patch("urllib.request.urlopen", return_value=response):
        result = openai_client().complete_result("write a file", 100)

    assert result.stop_reason == "length"
    assert result.tool_calls[0].arguments == {
        "path": "result.txt",
        "content": "partial",
    }


def test_anthropic_normalizes_max_tokens_stop_reason():
    response = JsonResponse(
        {
            "stop_reason": "max_tokens",
            "content": [
                {
                    "type": "tool_use",
                    "id": "call_anthropic",
                    "name": "write_file",
                    "input": {"path": "result.txt", "content": "partial"},
                }
            ],
        }
    )

    with patch("urllib.request.urlopen", return_value=response):
        result = anthropic_client().complete_result("write a file", 100)

    assert result.stop_reason == "length"
    assert result.tool_calls[0].arguments == {
        "path": "result.txt",
        "content": "partial",
    }


def truncated_result(stop_reason):
    calls = (
        ToolCall(
            call_id="call_write",
            name="write_file",
            arguments={"path": "owned.txt", "content": "must not be written"},
        ),
        ToolCall(
            call_id="call_shell",
            name="run_shell",
            arguments={"command": "echo must-not-run", "timeout": 20},
        ),
        ToolCall(
            call_id="call_worker",
            name="agent",
            arguments={
                "description": "Must not launch",
                "prompt": "Do not execute this worker",
                "subagent_type": "Explore",
            },
        ),
    )
    continuation = tuple(
        {
            "type": "function_call",
            "call_id": call.call_id,
            "name": call.name,
            "arguments": json.dumps(call.arguments),
        }
        for call in calls
    )
    return ModelResult(
        tool_calls=calls,
        stop_reason=stop_reason,
        continuation=continuation,
    )


def build_agent(tmp_path, outputs):
    return Lite(
        model_client=ScriptedModelClient(outputs),
        workspace=WorkspaceContext.build(tmp_path),
        session_store=SessionStore(tmp_path / ".lite" / "sessions"),
        approval_policy="auto",
        auto_dream=False,
    )


@pytest.mark.parametrize("stop_reason", ["length", "max_tokens", "incomplete"])
def test_engine_rejects_truncated_tool_batch_without_effects(
    tmp_path, monkeypatch, stop_reason
):
    result = truncated_result(stop_reason)
    agent = build_agent(tmp_path, [result, "not consumed"])
    effects = {
        "permission": 0,
        "file": 0,
        "shell": 0,
        "worker": 0,
        "checkpoint": 0,
    }

    for call in result.tool_calls:
        agent.validate_tool(call.name, call.arguments)

    original_permission_check = agent.permission_checker.check

    def permission_check(*args, **kwargs):
        effects["permission"] += 1
        return original_permission_check(*args, **kwargs)

    def counted_effect(name, result_value="unexpected"):
        def invoke(*args, **kwargs):
            del args, kwargs
            effects[name] += 1
            return result_value

        return invoke

    monkeypatch.setattr(agent.permission_checker, "check", permission_check)
    monkeypatch.setattr(
        agent,
        "capture_workspace_snapshot",
        counted_effect("file", {}),
    )
    monkeypatch.setattr(
        agent,
        "create_checkpoint",
        counted_effect("checkpoint", {"checkpoint_id": "unexpected"}),
    )
    monkeypatch.setattr(
        agent.worker_manager,
        "spawn",
        counted_effect("worker", {"id": "unexpected"}),
    )
    agent.tools["write_file"] = replace(
        agent.tools["write_file"], runner=counted_effect("file")
    )
    agent.tools["run_shell"] = replace(
        agent.tools["run_shell"], runner=counted_effect("shell")
    )

    events = []
    stream = agent.engine.run_turn("reject truncated calls")
    try:
        for event in stream:
            events.append(event)
            if event["type"] == "model_requested" and event["attempts"] == 2:
                break
    finally:
        stream.close()

    assert effects == {
        "permission": 0,
        "file": 0,
        "shell": 0,
        "worker": 0,
        "checkpoint": 0,
    }
    assert not (tmp_path / "owned.txt").exists()
    assert agent.current_task_state.tool_steps == 0
    assert not any(event["type"] == "tool_call" for event in events)

    synthetic_events = [event for event in events if event["type"] == "tool_result"]
    assert [event["call_id"] for event in synthetic_events] == [
        "call_write",
        "call_shell",
        "call_worker",
    ]
    assert all(event["metadata"]["synthetic"] for event in synthetic_events)
    assert all("complete tool call" in event["content"] for event in synthetic_events)

    tool_history = [
        item for item in agent.session["history"] if item["role"] == "tool"
    ]
    assert [item["call_id"] for item in tool_history] == [
        "call_write",
        "call_shell",
        "call_worker",
    ]
    assert all(item["tool_error_code"] == "truncated_tool_call" for item in tool_history)

    conversation = agent.model_client.requests[0]
    assert [output.call_id for output in conversation.turns[0].tool_outputs] == [
        "call_write",
        "call_shell",
        "call_worker",
    ]
    assert all(output.is_error for output in conversation.turns[0].tool_outputs)
    assert conversation.turns[0].continuation == result.continuation


def test_pure_text_length_response_keeps_final_semantics(tmp_path):
    agent = build_agent(
        tmp_path,
        [ModelResult(text="Partial text remains visible.", stop_reason="length")],
    )

    assert agent.ask("return partial text") == "Partial text remains visible."
    assert not any(item["role"] == "tool" for item in agent.session["history"])
