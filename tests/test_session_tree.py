import hashlib
import json

import pytest

from lite import Lite, SessionStore, WorkspaceContext
from lite.core.session_journal import (
    JournalCorruption,
    JournalSchemaError,
    SessionJournalWriter,
    restore_session_journal,
)
from lite.core.session_journal_schema import canonical_json
from lite.testing import ScriptedModelClient


def session(history=None):
    return {
        "id": "tree-session",
        "created_at": "2026-08-10T00:00:00+00:00",
        "workspace_root": "C:/workspace",
        "history": list(history or []),
    }


def test_rewind_then_append_creates_a_branch_without_deleting_old_nodes(tmp_path):
    path = tmp_path / "tree.journal.jsonl"
    writer = SessionJournalWriter.create(path, session())
    try:
        first = writer.append_message({"role": "user", "content": "first"})
        abandoned = writer.append_message({"role": "assistant", "content": "old"})

        writer.move_head(first.entry_id, reason="test-rewind")
        replacement = writer.append_message(
            {"role": "assistant", "content": "new"}
        )

        assert writer.state.session["history"] == [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "new"},
        ]
        assert writer.state.tree.entries[abandoned.entry_id].parent_id == first.entry_id
        assert writer.state.tree.entries[replacement.entry_id].parent_id == first.entry_id
        assert writer.state.tree.children[first.entry_id] == [
            abandoned.entry_id,
            replacement.entry_id,
        ]
    finally:
        writer.close()

    restored = restore_session_journal(path).state
    assert restored.tree.active_head == replacement.entry_id
    assert abandoned.entry_id in restored.tree.entries
    assert restored.session["history"][-1]["content"] == "new"


def test_compaction_is_a_node_and_rewind_restores_pre_compaction_context(tmp_path):
    path = tmp_path / "compact-tree.journal.jsonl"
    writer = SessionJournalWriter.create(path, session())
    try:
        first = writer.append_message({"role": "user", "content": "large input"})
        second = writer.append_message({"role": "assistant", "content": "large output"})
        compacted = [{"role": "system", "kind": "compact_summary", "content": "summary"}]

        boundary = writer.append_compaction(compacted, metadata={"source_count": 2})

        assert boundary.parent_id == second.entry_id
        assert writer.state.session["history"] == compacted
        writer.move_head(second.entry_id, reason="inspect-pre-compaction")
        assert [item["content"] for item in writer.state.session["history"]] == [
            "large input",
            "large output",
        ]
        assert first.entry_id in writer.state.tree.entries
    finally:
        writer.close()


def test_effect_result_can_atomically_commit_a_tool_exchange_tree_delta(tmp_path):
    path = tmp_path / "effect-tree.journal.jsonl"
    writer = SessionJournalWriter.create(path, session())
    try:
        entry_id = "entry_tool_exchange"
        with writer.effect(
            "tool",
            call_id="call-1",
            request={"name": "read_file", "args": {"path": "README.md"}},
        ) as effect:
            effect.complete(
                "ok",
                {"content": "hello"},
                tree_delta={
                    "expected_head": None,
                    "entries": [
                        {
                            "entry_id": entry_id,
                            "parent_id": None,
                            "entry_type": "tool_exchange",
                            "turn_id": "turn-1",
                            "run_id": "run-1",
                            "created_at": "2026-08-10T00:00:01+00:00",
                            "data": {
                                "assistant": {
                                    "role": "assistant",
                                    "content": "",
                                    "tool_calls": [{"call_id": "call-1", "name": "read_file"}],
                                },
                                "results": [
                                    {
                                        "role": "tool",
                                        "call_id": "call-1",
                                        "name": "read_file",
                                        "content": "hello",
                                    }
                                ],
                            },
                        }
                    ],
                },
            )

        assert writer.state.tree.active_head == entry_id
        assert [item["role"] for item in writer.state.session["history"]] == ["tool"]
        operation = next(iter(writer.state.completed_operations.values()))
        assert operation.tree_entry_ids == (entry_id,)
        records = [json.loads(line) for line in path.read_text().splitlines()]
        assert records[-1]["kind"] == "effect_result"
        assert records[-1]["payload"]["tree_delta"]["entries"][0]["entry_id"] == entry_id
    finally:
        writer.close()


def test_runtime_tree_commands_switch_context_and_future_records_branch(tmp_path):
    store = SessionStore(tmp_path / ".lite" / "sessions")
    agent = Lite(
        model_client=ScriptedModelClient([]),
        workspace=WorkspaceContext.build(tmp_path),
        session_store=store,
        approval_policy="auto",
    )
    agent.record({"role": "user", "content": "one"})
    first = agent.session_tree_rows()[-1]["entry_id"]
    agent.record({"role": "assistant", "content": "old"})
    old = agent.session_tree_rows()[-1]["entry_id"]

    agent.move_session_head(first)
    agent.record({"role": "assistant", "content": "new"})

    rows = {row["entry_id"]: row for row in agent.session_tree_rows()}
    assert rows[old]["active"] is False
    assert agent.session["history"][-1]["content"] == "new"
    assert len(rows[first]["children"]) == 2
    from lite.cli import handle_repl_command

    handled, should_exit, label_output = handle_repl_command(agent, "/label preferred")
    assert (handled, should_exit) == (True, False)
    assert "labeled" in label_output
    assert "preferred" in handle_repl_command(agent, "/tree")[2]
    assert "active head:" in handle_repl_command(agent, "/rewind 1")[2]
    assert "active branch:" in handle_repl_command(agent, "/branch preferred")[2]
    agent.close()


def test_tree_snapshot_restores_branch_index_labels_and_active_head(tmp_path):
    path = tmp_path / "snapshot-tree.journal.jsonl"
    writer = SessionJournalWriter.create(path, session())
    first = writer.append_message({"role": "user", "content": "one"})
    old = writer.append_message({"role": "assistant", "content": "old"})
    writer.move_head(first.entry_id, reason="fork")
    new = writer.append_message({"role": "assistant", "content": "new"})
    writer.label_head("preferred")
    writer.write_snapshot()
    writer.close()

    restored = restore_session_journal(path)

    assert restored.used_snapshot is True
    assert restored.state.tree.active_head == new.entry_id
    assert restored.state.tree.labels == {"preferred": new.entry_id}
    assert restored.state.tree.children[first.entry_id] == [old.entry_id, new.entry_id]


def test_snapshot_without_tree_projection_falls_back_to_full_journal_replay(tmp_path):
    path = tmp_path / "old-snapshot.journal.jsonl"
    writer = SessionJournalWriter.create(path, session())
    first = writer.append_message({"role": "user", "content": "one"})
    old = writer.append_message({"role": "assistant", "content": "old"})
    writer.move_head(first.entry_id, reason="fork")
    new = writer.append_message({"role": "assistant", "content": "new"})
    snapshot_path = writer.write_snapshot()
    writer.close()

    document = json.loads(snapshot_path.read_text(encoding="utf-8"))
    document["state"].pop("tree")
    body = {key: value for key, value in document.items() if key != "checksum"}
    document["checksum"] = hashlib.sha256(
        canonical_json(body).encode("utf-8")
    ).hexdigest()
    snapshot_path.write_text(canonical_json(document) + "\n", encoding="utf-8")

    restored = restore_session_journal(path)

    assert restored.used_snapshot is False
    assert old.entry_id in restored.state.tree.entries
    assert new.entry_id in restored.state.tree.entries


def test_effect_tree_delta_rejects_stale_head_before_writing_result(tmp_path):
    path = tmp_path / "stale-effect-tree.journal.jsonl"
    writer = SessionJournalWriter.create(path, session())
    writer.append_message({"role": "user", "content": "anchor"})
    intent = writer.begin_effect(
        "tool",
        call_id="call-1",
        request={"name": "read_file"},
        replay_policy="interrupt",
        operation_id="operation-1",
    )
    try:
        with pytest.raises(JournalCorruption, match="tree delta head mismatch"):
            writer.finish_effect(
                intent,
                outcome="ok",
                result={},
                tree_delta={"expected_head": None, "entries": []},
            )
        records = [json.loads(line) for line in path.read_text().splitlines()]
        assert records[-1]["kind"] == "effect_intent"
        assert writer.state.open_operation is not None
    finally:
        writer.close()


def test_tool_exchange_requires_one_ordered_result_per_assistant_call(tmp_path):
    path = tmp_path / "invalid-exchange.journal.jsonl"
    writer = SessionJournalWriter.create(path, session())
    try:
        with pytest.raises(JournalSchemaError, match="exactly match"):
            writer.append_tree_entry(
                "tool_exchange",
                {
                    "assistant": {
                        "role": "assistant",
                        "tool_calls": [
                            {"call_id": "call-1"},
                            {"call_id": "call-2"},
                        ],
                    },
                    "results": [{"role": "tool", "call_id": "call-2"}],
                },
            )
        assert len(writer.state.tree.entries) == 0
    finally:
        writer.close()


def test_runtime_tool_effect_uses_atomic_tool_exchange_record(tmp_path):
    (tmp_path / "README.md").write_text("hello\n", encoding="utf-8")
    store = SessionStore(tmp_path / ".lite" / "sessions")
    agent = Lite(
        model_client=ScriptedModelClient([]),
        workspace=WorkspaceContext.build(tmp_path),
        session_store=store,
        approval_policy="auto",
    )

    result = agent.run_tool(
        "read_file", {"path": "README.md", "start": 1, "end": 5}, call_id="call-1"
    )

    assert "hello" in result
    records = [
        json.loads(line)
        for line in agent.session_journal_writer.path.read_text(encoding="utf-8").splitlines()
    ]
    tool_result = next(
        record
        for record in records
        if record["kind"] == "effect_result"
        and record["payload"]["effect_type"] == "tool"
    )
    assert tool_result["payload"]["tree_delta"]["entries"][0]["entry_type"] == "tool_exchange"
    assert [item["role"] for item in agent.session["history"]] == ["tool"]
    assert agent.session_tree_rows()[-1]["entry_type"] == "tool_exchange"
    agent.close()


def test_parallel_runtime_batch_commits_one_ordered_tool_exchange(tmp_path):
    (tmp_path / "README.md").write_text("hello\n", encoding="utf-8")
    store = SessionStore(tmp_path / ".lite" / "sessions")
    agent = Lite(
        model_client=ScriptedModelClient(
            [
                [
                    {
                        "call_id": "call-read",
                        "name": "read_file",
                        "args": {"path": "README.md", "start": 1, "end": 5},
                    },
                    {
                        "call_id": "call-list",
                        "name": "list_files",
                        "args": {"path": "."},
                    },
                ],
                "done",
            ]
        ),
        workspace=WorkspaceContext.build(tmp_path),
        session_store=store,
        approval_policy="auto",
    )

    assert agent.ask("inspect") == "done"

    records = [
        json.loads(line)
        for line in agent.session_journal_writer.path.read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    tool_results = [
        record
        for record in records
        if record["kind"] == "effect_result"
        and record["payload"]["effect_type"] == "tool"
    ]
    assert len(tool_results) == 1
    entry = tool_results[0]["payload"]["tree_delta"]["entries"][0]
    assert [
        result["call_id"] for result in entry["data"]["results"]
    ] == ["call-read", "call-list"]
    assert [
        call["call_id"] for call in entry["data"]["assistant"]["tool_calls"]
    ] == ["call-read", "call-list"]
    assert [item["call_id"] for item in agent.session["history"] if item["role"] == "tool"] == [
        "call-read",
        "call-list",
    ]
    agent.close()
