import copy

import pytest

from lite import Lite, SessionStore, WorkspaceContext
from lite.core.request_context import (
    HistoryHardeningError,
    harden_context_messages,
)
from lite.providers import ModelConversation, ModelResult, ToolCall
from lite.providers.clients import (
    _anthropic_conversation_messages,
    _openai_conversation_input,
)
from lite.testing import ScriptedModelClient


def build_agent(tmp_path, outputs, **kwargs):
    (tmp_path / "README.md").write_text("current context\n", encoding="utf-8")
    return Lite(
        model_client=ScriptedModelClient(outputs),
        workspace=WorkspaceContext.build(tmp_path),
        session_store=SessionStore(tmp_path / ".lite" / "sessions"),
        approval_policy="auto",
        auto_dream=False,
        **kwargs,
    )


def tool_call(call_id="call_1", name="read_file"):
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"call_id": call_id, "name": name, "arguments": {"path": "README.md"}}
        ],
    }


def tool_result(call_id="call_1", name="read_file"):
    return {
        "role": "tool",
        "call_id": call_id,
        "name": name,
        "content": "current context",
        "is_error": False,
    }


def test_engine_rebuilds_each_request_from_single_source_delta_and_current_tools(
    tmp_path,
):
    seen = []

    def transform(messages, cancellation_token):
        seen.append((copy.deepcopy(messages), cancellation_token))
        return messages

    agent = build_agent(
        tmp_path,
        [
            ModelResult(
                tool_calls=(
                    ToolCall("call_read", "read_file", {"path": "README.md", "start": 1, "end": 1}),
                ),
            ),
            ModelResult(text="Done."),
        ],
        context_transform=transform,
    )
    original_run_tool = agent.run_tool

    def run_tool_and_change_profile(name, args, **kwargs):
        result = original_run_tool(name, args, **kwargs)
        agent.tools.pop("write_file")
        return result

    agent.run_tool = run_tool_and_change_profile

    assert agent.ask("Read the current context") == "Done."

    first, second = agent.model_client.requests
    assert first is not second
    assert "[tool:read_file]" not in first.initial_input
    assert "[tool:read_file]" not in second.initial_input
    assert sum(
        message.get("role") == "tool"
        and "current context" in str(message.get("content", ""))
        for message in second.request_messages
    ) == 1
    assert "write_file" in {tool.name for tool in first.tools}
    assert "write_file" not in {tool.name for tool in second.tools}
    assert len(seen) == 2
    assert [message["role"] for message in seen[0][0]] == ["user"]
    assert [message["role"] for message in seen[1][0]] == [
        "user",
        "assistant",
        "tool",
    ]
    assert seen[0][1] is seen[1][1] is agent.current_cancellation_token
    request_context = agent.last_prompt_metadata["request_context"]
    assert request_context["transform_applied"] is True
    assert request_context["validated_tool_pairs"] == 1
    assert "current context" not in repr(request_context)


def test_context_transform_cancellation_stops_before_provider(tmp_path):
    client = ScriptedModelClient(["must not be called"])

    def cancel_transform(messages, cancellation_token):
        del messages
        cancellation_token.cancel()
        cancellation_token.raise_if_cancelled()

    agent = Lite(
        model_client=client,
        workspace=WorkspaceContext.build(tmp_path),
        session_store=SessionStore(tmp_path / ".lite" / "sessions"),
        approval_policy="auto",
        auto_dream=False,
        context_transform=cancel_transform,
    )

    events = list(agent.engine.run_turn("cancel while transforming"))

    assert client.requests == []
    assert next(event for event in events if event["type"] == "stop")["content"] == (
        "Stopped after abort request."
    )


def test_compaction_boundary_applies_at_the_next_turn(tmp_path):
    agent = build_agent(
        tmp_path,
        [
            ToolCall("call_read", "read_file", {"path": "README.md"}),
            "Done.",
        ],
    )
    for index in range(3):
        agent.record({"role": "user", "content": f"old request {index}"})
        agent.record({"role": "assistant", "content": f"old answer {index}"})
    original_run_tool = agent.run_tool

    def run_tool_and_compact(name, args, **kwargs):
        result = original_run_tool(name, args, **kwargs)
        agent.compact_history(trigger="manual", keep_recent_turns=1)
        return result

    agent.run_tool = run_tool_and_compact

    assert agent.ask("Read after compacting") == "Done."

    first, second = agent.model_client.requests
    assert "Compacted session summary:" not in first.initial_input
    assert "Compacted session summary:" not in second.initial_input

    agent.model_client.outputs.append("Next turn.")
    assert agent.ask("Continue after compacting") == "Next turn."
    assert "Compacted session summary:" in agent.model_client.requests[-1].initial_input


def test_hardening_merges_same_roles_stably_without_mutating_source():
    source = [
        {"role": "user", "content": "one"},
        {"role": "user", "content": "two"},
        {"role": "assistant", "content": "alpha"},
        {"role": "assistant", "content": "beta"},
        {"role": "user", "content": "next"},
    ]
    original = copy.deepcopy(source)

    hardened, report = harden_context_messages(source)
    hardened_again, second_report = harden_context_messages(hardened)

    assert source == original
    assert hardened == hardened_again
    assert [message["role"] for message in hardened] == [
        "user",
        "assistant",
        "user",
    ]
    assert hardened[0]["content"] == "one\n\ntwo"
    assert hardened[1]["content"] == "alpha\n\nbeta"
    assert report["merged_same_role"] == 2
    assert second_report["merged_same_role"] == 0


def test_hardening_inserts_reasoned_result_for_missing_tool_output():
    source = [
        {"role": "user", "content": "inspect"},
        tool_call(),
        {"role": "user", "content": "continue"},
    ]

    hardened, report = harden_context_messages(source)

    synthetic = hardened[2]
    assert synthetic["role"] == "tool"
    assert synthetic["call_id"] == "call_1"
    assert synthetic["name"] == "read_file"
    assert synthetic["is_error"] is True
    assert synthetic["synthetic"] is True
    assert synthetic["reason"] == "missing_tool_result"
    assert "missing" in synthetic["content"]
    assert report["synthetic_tool_results"] == 1


@pytest.mark.parametrize(
    ("messages", "fragment"),
    [
        ([tool_result()], "orphan"),
        ([{"role": "user", "content": "x"}, tool_result()], "orphan"),
        (
            [
                {"role": "user", "content": "x"},
                tool_call(),
                tool_result(name="list_files"),
            ],
            "name mismatch",
        ),
        (
            [
                {"role": "user", "content": "x"},
                tool_call(),
                tool_result(call_id="wrong"),
            ],
            "call id mismatch",
        ),
    ],
)
def test_hardening_rejects_unrecoverable_tool_histories(messages, fragment):
    with pytest.raises(HistoryHardeningError, match=fragment):
        harden_context_messages(messages)


def test_history_may_end_with_a_paired_tool_result():
    messages = [
        {"role": "user", "content": "inspect"},
        tool_call(),
        tool_result(),
    ]

    hardened, report = harden_context_messages(messages)

    assert [message["role"] for message in hardened] == [
        "user",
        "assistant",
        "tool",
    ]
    assert hardened[-1] == messages[-1]
    assert report["effective_first_role"] == "user"
    assert report["effective_last_role"] == "user"


def test_hardened_tool_history_is_valid_for_both_provider_protocols():
    hardened, _ = harden_context_messages(
        [
            {"role": "user", "content": "inspect"},
            tool_call(),
            tool_result(),
            {"role": "user", "content": "summarize"},
        ]
    )
    conversation = ModelConversation(
        initial_input="unused",
        request_messages=tuple(hardened),
    )

    openai_items, _ = _openai_conversation_input(conversation)
    anthropic_messages, _ = _anthropic_conversation_messages(conversation)

    openai_call = next(item for item in openai_items if item.get("type") == "function_call")
    openai_result = next(
        item for item in openai_items if item.get("type") == "function_call_output"
    )
    assert openai_call["call_id"] == openai_result["call_id"] == "call_1"
    assert [message["role"] for message in anthropic_messages] == [
        "user",
        "assistant",
        "user",
    ]
    tool_use = next(
        block
        for block in anthropic_messages[1]["content"]
        if block["type"] == "tool_use"
    )
    tool_output = next(
        block
        for block in anthropic_messages[2]["content"]
        if block["type"] == "tool_result"
    )
    assert tool_use["id"] == tool_output["tool_use_id"] == "call_1"


def test_unrecoverable_transform_does_not_call_provider_or_mutate_history(tmp_path):
    def invalid_transform(messages, cancellation_token):
        del messages, cancellation_token
        return [tool_result()]

    agent = build_agent(
        tmp_path,
        ["must not be called"],
        context_transform=invalid_transform,
    )
    canonical_before = copy.deepcopy(agent.session["history"])

    events = list(agent.engine.run_turn("reject invalid history"))

    assert agent.model_client.requests == []
    assert "history_hardening_error" in next(
        event["content"] for event in events if event["type"] == "stop"
    )
    assert canonical_before == []
    assert [item["role"] for item in agent.session["history"]] == [
        "user",
        "assistant",
    ]
    assert not any(item["role"] == "tool" for item in agent.session["history"])
