import json
from pathlib import Path
from unittest.mock import patch

from benchmarks.agent_runtime_baseline import benchmark_session
from lite import Lite, SessionStore, WorkspaceContext
from lite.core.session_migration import (
    AUTHORITY_JOURNAL,
    authority_marker_path,
    read_authority_marker,
)
from lite.core.session_journal import SessionJournalWriter
from lite.testing import ScriptedModelClient


LEGACY_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "session_legacy_v0.json"


def legacy_session(session_id="legacy-session"):
    return {
        "id": session_id,
        "created_at": "2026-08-08T00:00:00+00:00",
        "workspace_root": "C:/workspace",
        "history": [{"role": "user", "content": "before"}],
        "memory": {"working": [], "durable": []},
    }


def build_legacy_agent(tmp_path):
    (tmp_path / "README.md").write_text("demo\n", encoding="utf-8")
    store = SessionStore(tmp_path / ".lite" / "sessions")
    value = legacy_session()
    store.save(value)
    legacy_before = store.path(value["id"]).read_bytes()
    agent = Lite.from_session(
        ScriptedModelClient([]),
        WorkspaceContext.build(tmp_path),
        store,
        value["id"],
        approval_policy="auto",
        auto_dream=False,
    )
    return agent, store, value, legacy_before


def test_legacy_session_load_is_read_only_until_first_runtime_write(tmp_path):
    agent, store, value, before = build_legacy_agent(tmp_path)
    legacy_path = store.path(value["id"])

    assert agent.session["history"] == value["history"]
    assert agent.session_journal_writer is None
    assert read_authority_marker(store, value["id"]) is None
    assert legacy_path.read_bytes() == before

    agent.record({"role": "assistant", "content": "continued"})

    assert agent.session_journal_writer is not None
    assert read_authority_marker(store, value["id"])["authority"] == AUTHORITY_JOURNAL
    assert legacy_path.read_bytes() == before
    assert store.load(value["id"])["history"][-1]["content"] == "continued"
    agent.session_journal_writer.close()


def test_cutover_session_resume_opens_writer_and_never_uses_legacy_save(tmp_path):
    agent, store, value, _before = build_legacy_agent(tmp_path)
    agent.record({"role": "assistant", "content": "migrated"})
    agent.session_journal_writer.close()

    resumed = Lite.from_session(
        ScriptedModelClient([]),
        WorkspaceContext.build(tmp_path),
        store,
        value["id"],
        approval_policy="auto",
        auto_dream=False,
    )
    assert resumed.session_journal_writer is not None
    with patch.object(store, "save", side_effect=AssertionError("legacy rewrite")):
        resumed.enter_plan_mode("compatibility")
        resumed.todo_ledger.add("keep running")

    assert resumed.session_journal_writer.state.session["runtime_mode"]["mode"] == "plan"
    assert resumed.session_journal_writer.state.session["todos"]["items"]
    assert store.load(value["id"])["runtime_mode"]["mode"] == "plan"
    resumed.session_journal_writer.close()


def test_runtime_session_switch_keeps_legacy_load_read_only(tmp_path):
    agent, store, value, _before = build_legacy_agent(tmp_path)
    other = legacy_session("other-session")
    other["history"] = []
    store.save(other)
    before = store.path(other["id"]).read_bytes()

    agent.resume_session(other["id"])

    assert agent.session_journal_writer is None
    assert agent.session["id"] == other["id"]
    assert store.path(other["id"]).read_bytes() == before
    agent.record({"role": "user", "content": "resume"})
    assert authority_marker_path(store, other["id"]).exists()
    agent.session_journal_writer.close()


def test_direct_tool_migrates_before_recording_the_effect(tmp_path):
    agent, store, value, before = build_legacy_agent(tmp_path)

    result = agent.run_tool("read_file", {"path": "README.md", "start": 1, "end": 5})

    records = [
        json.loads(line)
        for line in store.journal_path(value["id"]).read_text(encoding="utf-8").splitlines()
    ]
    assert "demo" in result
    assert store.path(value["id"]).read_bytes() == before
    assert [record["kind"] for record in records if record["kind"].startswith("effect_")][
        -2:
    ] == ["effect_intent", "effect_result"]
    agent.session_journal_writer.close()


def test_legacy_fixture_remains_json_compatible_after_journal_migration(tmp_path):
    (tmp_path / "README.md").write_text("demo\n", encoding="utf-8")
    store = SessionStore(tmp_path / ".lite" / "sessions")
    before = LEGACY_FIXTURE_PATH.read_bytes()
    value = json.loads(before)
    store.path(value["id"]).write_bytes(before)
    agent = Lite.from_session(
        ScriptedModelClient([]),
        WorkspaceContext.build(tmp_path),
        store,
        value["id"],
        approval_policy="auto",
        auto_dream=False,
    )
    agent.record({"role": "assistant", "content": "json-compatible"})
    loaded = json.loads(store.path(value["id"]).read_text(encoding="utf-8"))

    assert loaded == value
    assert store.path(value["id"]).read_bytes() == before
    assert store.load(value["id"])["history"][-1]["content"] == "json-compatible"
    agent.session_journal_writer.close()


def test_session_benchmark_reports_legacy_and_journal_io():
    result = benchmark_session(25, recovery_runs=3)

    assert result["history_records"] == 25
    assert result["legacy"]["append"]["samples"] == 25
    assert result["journal"]["append"]["samples"] == 25
    assert result["legacy"]["recovery"]["samples"] == 3
    assert result["journal"]["recovery"]["samples"] == 3
    assert result["journal"]["cumulative_write_bytes"] < result["legacy"][
        "cumulative_write_bytes"
    ]
    assert 0 < result["journal_to_legacy_cumulative_write_ratio"] < 1


def test_session_listing_uses_journal_authority_without_legacy_json(tmp_path):
    store = SessionStore(tmp_path / "sessions")
    first = legacy_session("first")
    second = legacy_session("second")
    store.save(first)
    store.save(second)
    store.migrate_session(first["id"])
    writer = SessionJournalWriter.open(store.journal_path(first["id"]))
    try:
        writer.append_history({"role": "assistant", "content": "latest"})
    finally:
        writer.close()
    store.path(first["id"]).unlink()

    assert store.latest() == first["id"]
    assert [row["id"] for row in store.list_sessions()] == ["first", "second"]
    assert store.load(first["id"])["history"][-1]["content"] == "latest"
