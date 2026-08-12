import json

from lite import Lite, SessionStore, WorkspaceContext
from lite.providers import ProviderError
from lite.testing import ScriptedModelClient


def build_agent(tmp_path, outputs):
    for name, marker in (("one.txt", "ONE_RESULT"), ("two.txt", "TWO_RESULT"), ("three.txt", "THREE_RESULT")):
        (tmp_path / name).write_text(marker + "\n", encoding="utf-8")
    return Lite(
        model_client=ScriptedModelClient(outputs),
        workspace=WorkspaceContext.build(tmp_path),
        session_store=SessionStore(tmp_path / ".lite" / "sessions"),
        approval_policy="auto",
        max_steps=4,
        feature_flags={"frozen_base_context": True},
    )


def test_three_tool_results_are_once_in_each_provider_request(tmp_path):
    agent = build_agent(
        tmp_path,
        [
            {"call_id": "c1", "name": "read_file", "args": {"path": "one.txt"}},
            {"call_id": "c2", "name": "read_file", "args": {"path": "two.txt"}},
            {"call_id": "c3", "name": "read_file", "args": {"path": "three.txt"}},
            "<final>verified</final>",
        ],
    )
    agent.record({"role": "user", "content": "A prior completed request."})

    assert agent.ask("Read the three files and report their markers.") == "verified"
    requests = agent.model_client.requests
    assert len(requests) == 4
    for request in requests:
        serialized = json.dumps(request.request_messages, ensure_ascii=False, sort_keys=True)
        assert serialized.count("ONE_RESULT") <= 1
        assert serialized.count("TWO_RESULT") <= 1
        assert serialized.count("THREE_RESULT") <= 1
        assert request.context_metadata["duplicate_tool_result_count"] == 0
        assert request.context_metadata["assembled_input_chars"] > 0
        assert request.context_metadata["context_source"] == "session_projection_plus_turn_delta"

    assert requests[-1].request_messages[-1]["role"] == "tool"
    assert agent.last_prompt_metadata["base_context_hash"]
    assert agent.last_prompt_metadata["session_projection_event_count"] >= 1
    assert agent.last_prompt_metadata["current_turn_delta_count"] == 3
    assert agent.last_prompt_metadata["provider_turn_count"] == 3


def assert_single_source_requests(agent):
    """Check native call/output pairing for every request in a scripted run."""

    for request in agent.model_client.requests:
        output_counts = {}
        for turn in request.turns:
            for output in turn.tool_outputs:
                output_counts[output.call_id] = output_counts.get(output.call_id, 0) + 1
        assert all(count == 1 for count in output_counts.values())
        assert request.context_metadata["duplicate_tool_result_count"] == 0
        assert request.context_metadata["context_source"] == (
            "session_projection_plus_turn_delta"
        )


def test_single_source_survives_compaction(tmp_path):
    agent = build_agent(
        tmp_path,
        [
            {"call_id": "c1", "name": "read_file", "args": {"path": "one.txt"}},
            {"call_id": "c2", "name": "read_file", "args": {"path": "two.txt"}},
            {"call_id": "c3", "name": "read_file", "args": {"path": "three.txt"}},
            "<final>compacted</final>",
        ],
    )
    for index in range(5):
        agent.record({"role": "user", "content": f"old request {index}"})
        agent.record({"role": "assistant", "content": f"old answer {index}"})
    agent.compact_history(trigger="manual", keep_recent_turns=2)

    assert agent.ask("Read the three files after compaction") == "compacted"
    assert_single_source_requests(agent)


def test_single_source_survives_resume(tmp_path):
    first = build_agent(tmp_path, ["<final>seeded</final>"])
    assert first.ask("Seed the resumable session") == "seeded"

    resumed = Lite.from_session(
        model_client=ScriptedModelClient(
            [
                {"call_id": "c1", "name": "read_file", "args": {"path": "one.txt"}},
                {"call_id": "c2", "name": "read_file", "args": {"path": "two.txt"}},
                {"call_id": "c3", "name": "read_file", "args": {"path": "three.txt"}},
                "<final>resumed</final>",
            ]
        ),
        workspace=WorkspaceContext.build(tmp_path),
        session_store=first.session_store,
        session_id=first.session["id"],
        approval_policy="auto",
        feature_flags={"frozen_base_context": True},
    )

    assert resumed.ask("Continue after resume") == "resumed"
    assert_single_source_requests(resumed)


def test_single_source_survives_provider_retry(tmp_path):
    agent = build_agent(
        tmp_path,
        [
            ProviderError("temporary empty response", code="empty_response"),
            {"call_id": "c1", "name": "read_file", "args": {"path": "one.txt"}},
            {"call_id": "c2", "name": "read_file", "args": {"path": "two.txt"}},
            {"call_id": "c3", "name": "read_file", "args": {"path": "three.txt"}},
            "<final>retried</final>",
        ],
    )

    assert agent.ask("Read the files despite a provider retry") == "retried"
    assert len(agent.model_client.requests) == 5
    assert_single_source_requests(agent)


def test_single_source_survives_final_only_recovery(tmp_path):
    agent = build_agent(
        tmp_path,
        [
            {"call_id": "c1", "name": "read_file", "args": {"path": "one.txt"}},
            "<final>final-only</final>",
        ],
    )
    agent.max_steps = 1

    assert agent.ask("Read one file and finish within the step budget") == "final-only"
    assert agent.model_client.requests[-1].tools == ()
    assert_single_source_requests(agent)
