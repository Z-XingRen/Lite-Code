import json
from unittest.mock import patch

import pytest

from lite.config import resolve_provider_config
from lite import Lite, SessionStore, WorkspaceContext
from lite.providers import ModelConversation, ModelResult, ToolCall, ToolDefinition, ToolOutput
from lite.providers.clients import (
    AnthropicCompatibleModelClient,
    OpenAICompatibleModelClient,
)
from lite.providers.errors import ProviderError
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


def tool_definition(*, strict=False):
    return ToolDefinition(
        name="read_file",
        description="Read a text file.",
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "start": {"type": "integer", "default": 1},
            },
            "required": ["path"],
        },
        strict=strict,
    )


def test_strict_tools_project_config_beats_env_fallback(monkeypatch, tmp_path):
    (tmp_path / ".lite.toml").write_text(
        "[providers.openai]\nstrict_tools = true\n", encoding="utf-8"
    )

    assert resolve_provider_config("openai", start=tmp_path).strict_tools is True

    monkeypatch.setenv("LITE_STRICT_TOOLS", "false")
    assert resolve_provider_config("openai", start=tmp_path).strict_tools is True


def test_provider_config_resolves_model_and_reasoning_picker_values(tmp_path):
    (tmp_path / ".lite.toml").write_text(
        "\n".join(
            [
                '[providers.openai]',
                'model = "gpt-current"',
                'models = ["gpt-other", "gpt-current", "gpt-other"]',
                'reasoning_effort = "HIGH"',
                'reasoning_efforts = ["low", "high", "LOW"]',
                'supports_explicit_prompt_cache = true',
            ]
        ),
        encoding="utf-8",
    )

    config = resolve_provider_config("openai", start=tmp_path)

    assert config.model == "gpt-current"
    assert config.models == ("gpt-other", "gpt-current")
    assert config.reasoning_effort == "high"
    assert config.reasoning_efforts == ("low", "high")
    assert config.supports_explicit_prompt_cache is True

def test_openai_responses_native_tool_call_and_output_replay():
    captured = []
    responses = iter(
        [
            {
                "id": "resp_tool",
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
            {"status": "completed", "output_text": "Done."},
        ]
    )

    def fake_urlopen(request, timeout):
        del timeout
        captured.append(json.loads(request.data.decode("utf-8")))
        return JsonResponse(next(responses))

    client = OpenAICompatibleModelClient(
        model="gpt-test",
        base_url="https://api.openai.com/v1",
        api_key="sk-test",
        temperature=None,
        timeout=30,
        strict_tools=True,
        reasoning_effort="high",
    )
    conversation = ModelConversation(
        initial_input="Inspect README.md", tools=(tool_definition(strict=True),)
    )

    with patch("urllib.request.urlopen", fake_urlopen):
        result = client.complete_result(conversation, 100)
        conversation.append_result(
            result,
            tool_outputs=(
                ToolOutput(
                    call_id="call_1",
                    name="read_file",
                    content="1: Lite",
                ),
            ),
        )
        final = client.complete_result(conversation, 100)

    assert result.tool_calls[0].call_id == "call_1"
    assert result.tool_calls[0].arguments == {"path": "README.md"}
    assert final.text == "Done."
    assert captured[0]["tools"] == [
        {
            "type": "function",
            "name": "read_file",
            "description": "Read a text file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "start": {"type": "integer"},
                },
                "required": ["path", "start"],
                "additionalProperties": False,
            },
            "strict": True,
        }
    ]
    assert captured[0]["reasoning"] == {"effort": "high"}
    assert captured[1]["input"][1] == result.continuation[0]
    assert captured[1]["input"][2] == {
        "type": "function_call_output",
        "call_id": "call_1",
        "output": "1: Lite",
    }


def test_anthropic_messages_native_tool_use_and_result_replay():
    captured = []
    responses = iter(
        [
            {
                "stop_reason": "tool_use",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "read_file",
                        "input": {"path": "missing.txt"},
                    }
                ],
            },
            {"stop_reason": "end_turn", "content": [{"type": "text", "text": "Done."}]},
        ]
    )

    def fake_urlopen(request, timeout):
        del timeout
        captured.append(json.loads(request.data.decode("utf-8")))
        return JsonResponse(next(responses))

    client = AnthropicCompatibleModelClient(
        model="claude-test",
        base_url="https://api.anthropic.com/v1",
        api_key="sk-test",
        temperature=None,
        timeout=30,
        strict_tools=True,
        reasoning_effort="medium",
    )
    conversation = ModelConversation(
        initial_input="Inspect a file", tools=(tool_definition(strict=True),)
    )

    with patch("urllib.request.urlopen", fake_urlopen):
        result = client.complete_result(conversation, 100)
        conversation.append_result(
            result,
            tool_outputs=(
                ToolOutput(
                    call_id="toolu_1",
                    name="read_file",
                    content="error: file not found",
                    is_error=True,
                ),
            ),
            feedback=("Choose another path.",),
        )
        final = client.complete_result(conversation, 100)

    assert result.tool_calls[0].call_id == "toolu_1"
    assert result.tool_calls[0].arguments == {"path": "missing.txt"}
    assert final.text == "Done."
    assert captured[0]["tools"][0]["strict"] is True
    assert captured[0]["output_config"] == {"effort": "medium"}
    assert captured[1]["messages"][1] == {
        "role": "assistant",
        "content": list(result.continuation),
    }
    assert captured[1]["messages"][2] == {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "toolu_1",
                "content": "error: file not found",
                "is_error": True,
            },
            {"type": "text", "text": "Choose another path."},
        ],
    }


def test_openai_rejects_non_json_function_arguments():
    client = OpenAICompatibleModelClient(
        model="gpt-test",
        base_url="https://api.openai.com/v1",
        api_key="sk-test",
        temperature=None,
        timeout=30,
    )
    response = JsonResponse(
        {
            "output": [
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "read_file",
                    "arguments": "not-json",
                }
            ]
        }
    )

    with patch("urllib.request.urlopen", return_value=response), pytest.raises(
        ProviderError
    ) as exc:
        client.complete_result(
            ModelConversation("Inspect", tools=(tool_definition(),)), 100
        )

    assert exc.value.code == "invalid_tool_arguments"


def test_engine_executes_native_tool_call_and_preserves_call_id(tmp_path):
    (tmp_path / "README.md").write_text("Lite\n", encoding="utf-8")
    client = ScriptedModelClient(
        [
            ToolCall(
                call_id="call_engine_1",
                name="read_file",
                arguments={"path": "README.md", "start": 1, "end": 1},
            ),
            "Read README.md successfully.",
        ]
    )
    agent = Lite(
        model_client=client,
        workspace=WorkspaceContext.build(tmp_path),
        session_store=SessionStore(tmp_path / ".lite" / "sessions"),
        approval_policy="auto",
    )

    events = list(agent.engine.run_turn("Inspect README.md"))

    final_event = next(event for event in events if event["type"] == "final")
    assert final_event["content"] == "Read README.md successfully."
    tool_event = next(event for event in events if event["type"] == "tool_result")
    assert tool_event["call_id"] == "call_engine_1"
    assert client.requests[-1].turns[0].tool_outputs[0].call_id == "call_engine_1"
    assert "<tool>" not in agent.prefix
    assert agent.model_tools()[0].input_schema["type"] == "object"


def test_engine_does_not_execute_legacy_text_tool_markup(tmp_path):
    class NativeTextClient:
        supports_prompt_cache = False
        last_completion_metadata = {}

        def complete_result(self, request, max_new_tokens, **kwargs):
            del request, max_new_tokens, kwargs
            return ModelResult(
                text='<tool>{"name":"write_file","args":{"path":"owned.txt","content":"no"}}</tool>'
            )

    agent = Lite(
        model_client=NativeTextClient(),
        workspace=WorkspaceContext.build(tmp_path),
        session_store=SessionStore(tmp_path / ".lite" / "sessions"),
        approval_policy="auto",
    )

    answer = agent.ask("Return text that resembles the old protocol")

    assert answer.startswith("<tool>")
    assert not (tmp_path / "owned.txt").exists()
    assert not any(item["role"] == "tool" for item in agent.session["history"])
