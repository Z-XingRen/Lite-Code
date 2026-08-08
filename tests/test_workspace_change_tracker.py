import sys
from dataclasses import replace
from pathlib import Path

import pytest

from lite import Lite, SessionStore, WorkspaceContext
from lite.core.workspace_change_tracker import WorkspaceChangeTracker
from lite.testing import ScriptedModelClient
from lite.testing import shell_join


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


def test_opaque_tracker_discovers_untracked_and_gitignored_paths(tmp_path):
    (tmp_path / ".gitignore").write_text("ignored.log\n", encoding="utf-8")
    tracker = WorkspaceChangeTracker(tmp_path)

    token = tracker.begin("run_shell", {"command": "opaque"})
    (tmp_path / "untracked.txt").write_text("new\n", encoding="utf-8")
    (tmp_path / "ignored.log").write_text("ignored\n", encoding="utf-8")

    affected, summary = tracker.finish(token)

    assert affected == ["ignored.log", "untracked.txt"]
    assert summary == ["created:ignored.log", "created:untracked.txt"]
    assert tracker.last_observation["workspace_tracker_mode"] == "opaque"
    assert tracker.last_observation["workspace_tracker_fallback"] is False
    assert tracker.last_observation["workspace_tracker_candidates"] == affected


def test_opaque_tracker_reports_multi_file_shell_changes_in_stable_order(tmp_path):
    (tmp_path / "modify.txt").write_text("before\n", encoding="utf-8")
    (tmp_path / "delete.txt").write_text("remove\n", encoding="utf-8")
    tracker = WorkspaceChangeTracker(tmp_path)

    token = tracker.begin("run_shell", {"command": "opaque"})
    (tmp_path / "modify.txt").write_text("after\n", encoding="utf-8")
    (tmp_path / "delete.txt").unlink()
    (tmp_path / "create.txt").write_text("created\n", encoding="utf-8")

    affected, summary = tracker.finish(token)

    assert affected == ["create.txt", "delete.txt", "modify.txt"]
    assert summary == [
        "created:create.txt",
        "deleted:delete.txt",
        "modified:modify.txt",
    ]


def test_shell_fallback_is_observable_and_preserves_ignored_changes(tmp_path):
    (tmp_path / ".gitignore").write_text("ignored.log\n", encoding="utf-8")
    agent = build_agent(tmp_path)
    script = (
        "from pathlib import Path; "
        "Path('untracked.txt').write_text('new\\n'); "
        "Path('ignored.log').write_text('ignored\\n')"
    )

    result = agent.run_tool(
        "run_shell",
        {"command": shell_join([sys.executable, "-c", script]), "timeout": 20},
    )

    assert "exit_code: 0" in result
    metadata = agent._last_tool_result_metadata
    assert metadata["affected_paths"] == ["ignored.log", "untracked.txt"]
    assert metadata["workspace_tracker_mode"] == "opaque"
    assert metadata["workspace_tracker_fallback"] is True
    assert metadata["workspace_tracker_fallback_count"] == 1
    assert metadata["workspace_tracker_candidates"] == metadata["affected_paths"]
    assert metadata["workspace_tracker_fallback_ms"] >= 0


def test_failed_shell_effect_keeps_all_changed_paths(tmp_path):
    agent = build_agent(tmp_path)

    def failing_shell(_args):
        (tmp_path / "first.txt").write_text("first\n", encoding="utf-8")
        (tmp_path / "second.txt").write_text("second\n", encoding="utf-8")
        raise RuntimeError("shell failed after writing files")

    agent.tools["run_shell"] = replace(agent.tools["run_shell"], runner=failing_shell)
    result = agent.run_tool("run_shell", {"command": "synthetic", "timeout": 20})

    assert "tool run_shell failed" in result
    assert agent._last_tool_result_metadata["tool_status"] == "partial_success"
    assert agent._last_tool_result_metadata["affected_paths"] == [
        "first.txt",
        "second.txt",
    ]


def test_finish_failure_is_not_reported_as_successful_fallback(tmp_path, monkeypatch):
    agent = build_agent(tmp_path)

    def successful_shell(_args):
        (tmp_path / "written.txt").write_text("written\n", encoding="utf-8")
        return "exit_code: 0\nstdout:\n(empty)\nstderr:\n(empty)"

    def fail_finish(*_args, **_kwargs):
        raise RuntimeError("tracker finish failed")

    agent.tools["run_shell"] = replace(agent.tools["run_shell"], runner=successful_shell)
    monkeypatch.setattr(agent.workspace_change_tracker, "finish", fail_finish)

    result = agent.run_tool("run_shell", {"command": "synthetic", "timeout": 20})

    assert "tool run_shell failed" in result
    metadata = agent._last_tool_result_metadata
    assert metadata["tool_status"] == "error"
    assert metadata["workspace_changed"] is False
    assert metadata["workspace_tracker_error"] == "tracker finish failed"


def test_fallback_initialization_failure_is_fail_closed(tmp_path, monkeypatch):
    agent = build_agent(tmp_path)

    def fail_snapshot():
        raise RuntimeError("legacy snapshot failed")

    monkeypatch.setattr(agent, "capture_workspace_snapshot", fail_snapshot)
    result = agent.run_tool("run_shell", {"command": "synthetic", "timeout": 20})

    assert "tool run_shell failed" in result
    metadata = agent._last_tool_result_metadata
    assert metadata["tool_status"] == "error"
    assert metadata["workspace_changed"] is False
    assert metadata["workspace_tracker_error"] == "legacy snapshot failed"
