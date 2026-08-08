from pathlib import Path

import pytest

from lite import Lite, SessionStore, WorkspaceContext
from lite.core.workspace_change_tracker import WorkspaceChangeTracker
from lite.testing import ScriptedModelClient


def build_agent(tmp_path):
    (tmp_path / "README.md").write_text("demo\n", encoding="utf-8")
    return Lite(
        model_client=ScriptedModelClient([]),
        workspace=WorkspaceContext.build(tmp_path),
        session_store=SessionStore(tmp_path / ".lite" / "sessions"),
        approval_policy="auto",
    )


def test_tracker_reports_create_modify_delete_binary_and_unicode_paths(tmp_path):
    tracker = WorkspaceChangeTracker(tmp_path)
    unicode_path = "目录/новый-файл.bin"

    token = tracker.begin("write_file", {"path": unicode_path})
    target = tmp_path / Path(unicode_path)
    target.parent.mkdir(parents=True)
    target.write_bytes(b"\x00\xff\x01")
    affected, summary = tracker.finish(token)

    assert affected == [unicode_path]
    assert summary == [f"created:{unicode_path}"]

    token = tracker.begin("write_file", {"path": unicode_path})
    target.write_bytes(b"\x00\xff\x02")
    affected, summary = tracker.finish(token)
    assert affected == [unicode_path]
    assert summary == [f"modified:{unicode_path}"]

    token = tracker.begin("write_file", {"path": unicode_path})
    target.unlink()
    affected, summary = tracker.finish(token)
    assert affected == [unicode_path]
    assert summary == [f"deleted:{unicode_path}"]


def test_tracker_reports_both_sides_of_known_rename(tmp_path):
    source = "旧目录/old.bin"
    destination = "新目录/new.bin"
    source_path = tmp_path / Path(source)
    source_path.parent.mkdir(parents=True)
    source_path.write_bytes(b"binary")
    tracker = WorkspaceChangeTracker(tmp_path)

    token = tracker.begin(
        "rename",
        {"path": source, "new_path": destination},
    )
    destination_path = tmp_path / Path(destination)
    destination_path.parent.mkdir(parents=True)
    source_path.rename(destination_path)
    affected, summary = tracker.finish(token)

    assert affected == sorted([source, destination])
    assert summary == [f"created:{destination}", f"deleted:{source}"]


def test_transparent_write_tools_do_not_capture_full_workspace(tmp_path, monkeypatch):
    agent = build_agent(tmp_path)

    def fail_legacy_snapshot():
        raise AssertionError("transparent tools must not hash the whole workspace")

    monkeypatch.setattr(agent, "capture_workspace_snapshot", fail_legacy_snapshot)

    assert agent.run_tool(
        "write_file", {"path": "目录/notes.txt", "content": "hello\n"}
    ).startswith("wrote ")
    assert agent._last_tool_result_metadata["affected_paths"] == ["目录/notes.txt"]

    assert agent.run_tool(
        "patch_file",
        {"path": "目录/notes.txt", "old_text": "hello", "new_text": "updated"},
    ).startswith("patched ")
    assert agent._last_tool_result_metadata["affected_paths"] == ["目录/notes.txt"]


def test_tracker_rejects_paths_outside_workspace(tmp_path):
    tracker = WorkspaceChangeTracker(tmp_path)

    with pytest.raises(ValueError, match="path escapes workspace"):
        tracker.begin("write_file", {"path": "../outside.txt"})
