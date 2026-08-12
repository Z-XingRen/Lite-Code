import copy

from lite import Lite, SessionStore, WorkspaceContext
from lite.core.prompt_cache_projection import PROJECTION_VERSION
from lite.providers import ModelResult, ToolCall
from lite.testing import ScriptedModelClient


class CacheModelClient(ScriptedModelClient):
    def __init__(self, outputs, *, model="gpt-5.6-terra", base_url="https://one.example/v1"):
        super().__init__(outputs)
        self.supports_prompt_cache = True
        self.supports_append_prompt_cache = True
        self.model = model
        self.base_url = base_url
        self.provider = "openai"
        self.context_window = 200_000


def build_agent(tmp_path, outputs):
    (tmp_path / "README.md").write_text("cache projection\n", encoding="utf-8")
    client = CacheModelClient(outputs)
    agent = Lite(
        model_client=client,
        workspace=WorkspaceContext.build(tmp_path),
        session_store=SessionStore(tmp_path / ".lite" / "sessions"),
        approval_policy="auto",
        auto_dream=False,
    )
    return agent, client


def test_completed_turn_is_appended_to_next_provider_request(tmp_path):
    agent, client = build_agent(tmp_path, ["first answer", "second answer"])

    assert agent.ask("first request") == "first answer"
    first_messages = copy.deepcopy(client.requests[0].request_messages)
    assert agent.ask("second request") == "second answer"
    second_messages = client.requests[1].request_messages

    assert list(second_messages[: len(first_messages)]) == list(first_messages)
    assert second_messages[len(first_messages)] == {
        "role": "assistant",
        "content": "first answer",
        "continuation": (),
        "tool_calls": [],
    }
    assert second_messages[-1]["role"] == "user"
    assert "second request" in second_messages[-1]["content"]
    assert agent.last_prompt_metadata["cache_projection_reused"] is True
    assert agent.last_prompt_metadata["cache_projection_message_count"] == 2
    assert agent.session["prompt_cache_projection"]["version"] == PROJECTION_VERSION


def test_tool_chain_is_persisted_once_and_reused_next_turn(tmp_path):
    agent, client = build_agent(
        tmp_path,
        [
            ModelResult(
                tool_calls=(
                    ToolCall("call_read", "read_file", {"path": "README.md"}),
                )
            ),
            "tool answer",
            "next answer",
        ],
    )

    assert agent.ask("read the file") == "tool answer"
    assert agent.ask("continue") == "next answer"

    messages = client.requests[-1].request_messages
    assert sum(message.get("role") == "tool" for message in messages) == 1
    assert sum(
        call.get("call_id") == "call_read"
        for message in messages
        for call in message.get("tool_calls", [])
    ) == 1
    assert agent.last_prompt_metadata["cache_projection_reused"] is True


def test_projection_invalidates_after_manual_history_change(tmp_path):
    agent, client = build_agent(tmp_path, ["first answer", "second answer"])

    assert agent.ask("first request") == "first answer"
    agent.record({"role": "user", "content": "out-of-band history change"})
    assert agent.ask("second request") == "second answer"

    assert agent.last_prompt_metadata["cache_projection_reused"] is False
    assert agent.last_prompt_metadata["cache_projection_reason"] == "history"
    assert len(client.requests[-1].request_messages) == 1
    assert "out-of-band history change" in client.requests[-1].request_messages[0]["content"]


def test_projection_survives_session_resume(tmp_path):
    agent, _ = build_agent(tmp_path, ["first answer"])
    assert agent.ask("first request") == "first answer"
    session_id = agent.session["id"]

    resumed_client = CacheModelClient(["second answer"])
    resumed = Lite.from_session(
        model_client=resumed_client,
        workspace=WorkspaceContext.build(tmp_path),
        session_store=agent.session_store,
        session_id=session_id,
        approval_policy="auto",
        auto_dream=False,
    )

    assert resumed.ask("second request") == "second answer"
    assert resumed.last_prompt_metadata["cache_projection_reused"] is True
    assert resumed.last_prompt_metadata["cache_projection_message_count"] == 2


def test_projection_invalidates_when_gateway_url_changes(tmp_path):
    agent, client = build_agent(tmp_path, ["first answer", "second answer"])
    assert agent.ask("first request") == "first answer"

    client.base_url = "https://two.example/v1"
    assert agent.ask("second request") == "second answer"

    assert agent.last_prompt_metadata["cache_projection_reused"] is False
    assert agent.last_prompt_metadata["cache_projection_reason"] == "identity"


def test_projection_invalidates_when_model_changes(tmp_path):
    agent, client = build_agent(tmp_path, ["first answer", "second answer"])
    assert agent.ask("first request") == "first answer"
    first_cache_key = agent.last_prompt_metadata["prompt_cache_key"]

    client.model = "gpt-5.5"
    assert agent.ask("second request") == "second answer"

    assert agent.last_prompt_metadata["cache_projection_reused"] is False
    assert agent.last_prompt_metadata["cache_projection_reason"] == "identity"
    assert agent.last_prompt_metadata["cache_projection_generation"] == 2
    assert agent.last_prompt_metadata["prompt_cache_key"] != first_cache_key
    assert len(client.requests[-1].request_messages) == 1


def test_workspace_change_appends_context_refresh_without_reset(tmp_path):
    agent, client = build_agent(
        tmp_path, ["first answer", "second answer", "third answer"]
    )
    assert agent.ask("first request") == "first answer"
    routing_cache_key = agent.last_prompt_metadata["prompt_cache_key"]

    (tmp_path / "README.md").write_text(
        "cache projection\nlatest workspace fact\n", encoding="utf-8"
    )
    assert agent.ask("second request") == "second answer"

    request = client.requests[-1]
    assert agent.last_prompt_metadata["cache_projection_reused"] is True
    assert agent.last_prompt_metadata["cache_projection_reason"] == "context_refresh"
    assert agent.last_prompt_metadata["cache_projection_context_refreshed"] is True
    assert agent.last_prompt_metadata["cache_projection_generation"] == 1
    assert agent.last_prompt_metadata["prompt_cache_key"] == routing_cache_key
    assert (
        agent.last_prompt_metadata["prompt_cache_context_key"] != routing_cache_key
    )
    assert [message["role"] for message in request.request_messages[:2]] == [
        "user",
        "assistant",
    ]
    assert request.request_messages[0]["content"] == agent.session[
        "prompt_cache_projection"
    ]["messages"][0]["content"]
    assert request.request_messages[1]["content"] == "first answer"
    assert "latest workspace fact" in request.request_messages[-1]["content"]
    assert "Context refresh:" in request.request_messages[-1]["content"]
    assert "Transcript:" not in request.request_messages[-1]["content"]

    refreshed_messages = copy.deepcopy(request.request_messages)
    assert agent.ask("third request") == "third answer"

    next_request = client.requests[-1]
    assert list(next_request.request_messages[: len(refreshed_messages)]) == list(
        refreshed_messages
    )
    assert agent.last_prompt_metadata["cache_projection_reason"] == "append"
    assert agent.last_prompt_metadata["cache_projection_context_refreshed"] is False
    assert agent.last_prompt_metadata["cache_projection_generation"] == 1
    assert agent.last_prompt_metadata["prompt_cache_key"] == routing_cache_key


def test_projection_invalidates_when_tool_signature_changes(tmp_path):
    agent, client = build_agent(tmp_path, ["first answer", "second answer"])
    assert agent.ask("first request") == "first answer"

    agent.set_tool_profile("readonly")
    assert agent.ask("second request") == "second answer"

    assert agent.last_prompt_metadata["cache_projection_reused"] is False
    assert agent.last_prompt_metadata["cache_projection_reason"] == "identity"


def test_projection_is_disabled_for_custom_context_transform(tmp_path):
    agent, client = build_agent(tmp_path, ["first answer", "second answer"])
    agent.context_transform = lambda messages, token: messages

    assert agent.ask("first request") == "first answer"
    assert agent.ask("second request") == "second answer"

    assert agent.last_prompt_metadata["cache_projection_reused"] is False
    assert agent.last_prompt_metadata["cache_projection_reason"] == "unsupported"
    assert len(client.requests[-1].request_messages) == 1


def test_projection_is_disabled_by_prompt_cache_feature_flag(tmp_path):
    agent, client = build_agent(tmp_path, ["first answer", "second answer"])
    agent.feature_flags["prompt_cache"] = False

    assert agent.ask("first request") == "first answer"
    assert agent.ask("second request") == "second answer"

    assert agent.last_prompt_metadata["cache_projection_reason"] == "unsupported"
    assert "prompt_cache_projection" not in agent.session
    assert len(client.requests[-1].request_messages) == 1


def test_projection_invalidates_after_context_compaction(tmp_path):
    agent, _ = build_agent(tmp_path, ["first answer"])
    assert agent.ask("first request") == "first answer"

    metadata = {
        "prompt_cache_key": agent.last_prompt_metadata["prompt_cache_key"],
        "prompt_cache_prefix_chars": agent.last_prompt_metadata[
            "prompt_cache_prefix_chars"
        ],
        "auto_compacted": True,
    }
    from lite.core.prompt_cache_projection import prepare_prompt_cache_turn

    turn = prepare_prompt_cache_turn(
        agent,
        full_prompt="full compacted prompt",
        append_delta="next request",
        metadata=metadata,
        history=agent.session["history"],
    )

    assert turn["base_messages"] == []
    assert metadata["cache_projection_reason"] == "compaction"


def test_projection_starts_new_generation_at_context_budget(tmp_path):
    agent, client = build_agent(tmp_path, ["first answer", "second answer"])
    assert agent.ask("first request") == "first answer"

    agent.context_manager.total_budget = 1
    assert agent.ask("second request") == "second answer"

    assert agent.last_prompt_metadata["cache_projection_reused"] is False
    assert agent.last_prompt_metadata["cache_projection_reason"] == "budget"
    assert agent.last_prompt_metadata["cache_projection_generation"] == 2
    assert len(client.requests[-1].request_messages) == 1
