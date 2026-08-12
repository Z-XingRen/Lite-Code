import json

from lite import Lite, SessionStore, WorkspaceContext
from lite.testing import ScriptedModelClient


def build_agent(tmp_path, outputs, *, optimized=None):
    (tmp_path / "input.txt").write_text("input\n", encoding="utf-8")
    feature_flags = (
        {}
        if optimized is None
        else {"journal_checkpoint_policy": bool(optimized)}
    )
    return Lite(
        model_client=ScriptedModelClient(outputs),
        workspace=WorkspaceContext.build(tmp_path),
        session_store=SessionStore(tmp_path / ".lite" / "sessions"),
        approval_policy="auto",
        feature_flags=feature_flags,
        max_steps=2,
    )


def trace(agent):
    path = next((agent.current_run_dir.parent).glob("*/trace.jsonl"))
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_read_only_tool_skips_checkpoint_and_reports_write_count(tmp_path):
    agent = build_agent(
        tmp_path,
        [
            {"call_id": "read-1", "name": "read_file", "args": {"path": "input.txt"}},
            "<final>read</final>",
        ],
        optimized=True,
    )

    assert agent.ask("Read input.") == "read"
    events = trace(agent)
    tool_event = next(event for event in events if event.get("event") == "tool_executed")
    assert tool_event["persistence_write_count"] >= 1
    assert not any(
        event.get("event") == "checkpoint_created"
        and event.get("trigger") == "tool_executed"
        for event in events
    )


def test_successful_write_tool_still_creates_checkpoint(tmp_path):
    agent = build_agent(
        tmp_path,
        [
            {
                "call_id": "write-1",
                "name": "write_file",
                "args": {"path": "output.txt", "content": "written\n"},
            },
            "<final>written</final>",
        ],
        optimized=True,
    )

    assert agent.ask("Write output.") == "written"
    events = trace(agent)
    assert any(
        event.get("event") == "checkpoint_created"
        and event.get("trigger") == "tool_executed"
        for event in events
    )


def test_read_only_write_count_is_at_least_sixty_percent_lower_than_legacy(tmp_path):
    def run(optimized, name):
        root = tmp_path / name
        root.mkdir()
        agent = build_agent(
            root,
            [
                {"call_id": "read-1", "name": "read_file", "args": {"path": "input.txt"}},
                "<final>read</final>",
            ],
            optimized=optimized,
        )
        agent.ask("Read input.")
        events = trace(agent)
        return next(event["persistence_write_count"] for event in events if event.get("event") == "tool_executed")

    legacy = run(False, "legacy")
    optimized = run(True, "optimized")
    assert legacy > 0
    assert optimized <= legacy * 0.4


def test_checkpoint_write_policy_is_enabled_by_default(tmp_path):
    agent = build_agent(tmp_path, [], optimized=None)

    assert agent.feature_enabled("journal_checkpoint_policy") is True


def test_failed_read_only_tool_keeps_explicit_governance_evidence(tmp_path):
    agent = build_agent(
        tmp_path,
        [
            {"call_id": "bad-read", "name": "read_file", "args": {}},
            "<final>handled</final>",
        ],
        optimized=True,
    )

    assert agent.ask("Attempt the read.") == "handled"
    events = trace(agent)
    decisions = [
        event
        for event in events
        if event.get("event") == "governance_decision"
    ]
    assert any(event.get("reason_code") == "invalid_arguments" for event in decisions)
    session_events = [
        json.loads(line)
        for line in agent.session_event_bus.path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert any(
        event.get("event") == "tool_finished"
        and event.get("tool_error_code") == "invalid_arguments"
        for event in session_events
    )
