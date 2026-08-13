"""Tests for artifact-backed retention of long tool results."""

import hashlib
import json
import sys

import pytest

from lite import Lite, SessionStore, WorkspaceContext
from lite.core.context_manager import ContextManager
from lite.core.run_store import RunStore
from lite.core.task_state import TaskState
from lite.core.tool_result_artifacts import (
    DEFAULT_TOOL_OUTPUT_LIMITS,
    prepare_tool_result_observation,
)
from lite.testing import ScriptedModelClient, read_jsonl, shell_join
from lite.tools.base import RegisteredTool


def build_agent(tmp_path, outputs=None, **kwargs):
    (tmp_path / "README.md").write_text("hello world\n", encoding="utf-8")
    workspace = WorkspaceContext.build(tmp_path)
    store = SessionStore(tmp_path / ".lite" / "sessions")
    return Lite(
        model_client=ScriptedModelClient(outputs or []),
        workspace=workspace,
        session_store=store,
        approval_policy="auto",
        **kwargs,
    )


def prepare_observation(tmp_path, name, full_result):
    agent = build_agent(tmp_path)
    task_state = TaskState.create(
        run_id=f"run-{name}",
        task_id=f"task-{name}",
        user_request="test adaptive tool output",
    )
    agent.current_task_state = task_state
    agent.current_run_dir = agent.run_store.start_run(task_state)
    observation, metadata = prepare_tool_result_observation(agent, name, full_result)
    artifact_path = tmp_path / metadata["full_output_artifact"]
    return observation, metadata, artifact_path


def assert_within_inline_limits(observation, metadata):
    assert len(observation.encode("utf-8")) <= metadata["max_bytes"]
    assert len(observation.splitlines()) <= metadata["max_lines"]


def test_long_single_line_uses_utf8_safe_head_tail_preview(tmp_path):
    full_result = "HEAD-" + ("x" * 12000) + "-TAIL"

    observation, metadata, artifact_path = prepare_observation(
        tmp_path, "agent", full_result
    )

    assert metadata["truncated"] is True
    assert metadata["truncation_strategy"] == "head_tail"
    assert "HEAD-" in observation
    assert "-TAIL" in observation
    assert f"{metadata['omitted_bytes']} bytes" in observation
    assert metadata["omitted_bytes"] > 0
    assert "read_file" in observation
    assert metadata["full_output_artifact"] in observation
    assert artifact_path.read_bytes() == full_result.encode("utf-8")
    assert_within_inline_limits(observation, metadata)


def test_multiline_read_preview_keeps_complete_head_lines(tmp_path):
    source_lines = [f"line-{index:03d}-" + ("x" * 80) for index in range(500)]
    full_result = "\n".join(source_lines)

    observation, metadata, artifact_path = prepare_observation(
        tmp_path, "read_file", full_result
    )

    assert metadata["truncation_strategy"] == "head"
    assert source_lines[0] in observation
    assert source_lines[-1] not in observation
    preview_lines = [line for line in observation.splitlines() if line.startswith("line-")]
    assert preview_lines
    assert all(line in source_lines for line in preview_lines)
    assert metadata["omitted_lines"] == len(source_lines) - len(preview_lines)
    assert f"{metadata['omitted_lines']} lines" in observation
    assert artifact_path.read_bytes() == full_result.encode("utf-8")
    assert_within_inline_limits(observation, metadata)


def test_line_budget_alone_creates_an_artifact_and_omits_lines(tmp_path):
    source_lines = [f"line-{index:03d}" for index in range(300)]
    full_result = "\n".join(source_lines)
    assert len(full_result.encode("utf-8")) < DEFAULT_TOOL_OUTPUT_LIMITS.max_bytes

    observation, metadata, artifact_path = prepare_observation(
        tmp_path, "search", full_result
    )

    assert metadata["truncated"] is True
    assert metadata["truncation_strategy"] == "head"
    assert metadata["omitted_lines"] > 0
    assert source_lines[0] in observation
    assert source_lines[-1] not in observation
    assert artifact_path.read_bytes() == full_result.encode("utf-8")
    assert_within_inline_limits(observation, metadata)


def test_traceback_preview_prioritizes_error_tail(tmp_path):
    full_result = "\n".join(
        ["HEAD", "Traceback (most recent call last):"]
        + [f'  File "module_{index}.py", line {index}, in function' for index in range(500)]
        + ["RuntimeError: TAIL"]
    )

    observation, metadata, _ = prepare_observation(
        tmp_path, "run_shell", full_result
    )

    assert metadata["truncation_strategy"] == "tail"
    assert "RuntimeError: TAIL" in observation
    assert 'File "module_499.py"' in observation
    assert "\nHEAD\n" not in f"\n{observation}\n"
    assert_within_inline_limits(observation, metadata)


@pytest.mark.parametrize("character", ["汉", "\U0001f642"])
def test_utf8_preview_never_splits_multibyte_characters(tmp_path, character):
    full_result = "HEAD-" + (character * 8000) + "-TAIL"

    observation, metadata, artifact_path = prepare_observation(
        tmp_path, "agent", full_result
    )

    observation.encode("utf-8").decode("utf-8")
    assert "\ufffd" not in observation
    assert artifact_path.read_bytes() == full_result.encode("utf-8")
    assert metadata["original_bytes"] == len(full_result.encode("utf-8"))
    assert_within_inline_limits(observation, metadata)


def test_empty_output_is_not_artifact_backed_or_truncated(tmp_path):
    agent = build_agent(tmp_path)

    observation, metadata = prepare_tool_result_observation(agent, "run_shell", "")

    assert observation == ""
    assert metadata["truncated"] is False
    assert metadata["original_bytes"] == 0
    assert metadata["original_lines"] == 0
    assert metadata["omitted_bytes"] == 0
    assert metadata["omitted_lines"] == 0
    assert metadata["full_output_artifact"] == ""
    assert metadata["max_bytes"] == DEFAULT_TOOL_OUTPUT_LIMITS.max_bytes
    assert metadata["max_lines"] == DEFAULT_TOOL_OUTPUT_LIMITS.max_lines


def test_long_shell_output_is_clipped_and_full_output_is_saved_as_run_artifact(tmp_path):
    script = "print('x'*6000)"
    command = shell_join([sys.executable, "-c", script])
    agent = build_agent(
        tmp_path,
        [
            f'<tool>{{"name":"run_shell","args":{{"command":{json.dumps(command)},"timeout":20}}}}</tool>',
            "<final>captured</final>",
        ],
    )

    assert agent.ask("produce long shell output") == "captured"

    tool_item = next(item for item in agent.session["history"] if item["role"] == "tool" and item["name"] == "run_shell")
    assert len(tool_item["content"].encode("utf-8")) <= DEFAULT_TOOL_OUTPUT_LIMITS.max_bytes
    assert len(tool_item["content"].splitlines()) <= DEFAULT_TOOL_OUTPUT_LIMITS.max_lines
    assert "full output saved:" in tool_item["content"]
    native_output = agent.model_client.requests[1].turns[0].tool_outputs[0]
    assert "full output saved:" in native_output.content

    report = json.loads((agent.current_run_dir / "report.json").read_text(encoding="utf-8"))
    artifact_path = report["runtime_reminders"][0]["artifact_path"] if report["runtime_reminders"] else agent._last_tool_result_metadata["full_output_artifact"]
    full_output = (tmp_path / artifact_path).read_text(encoding="utf-8")
    assert "x" * 6000 in full_output
    assert tool_item["content_sha256"] == hashlib.sha256(full_output.encode("utf-8")).hexdigest()

    trace_events = read_jsonl(agent.current_run_dir / "trace.jsonl")
    tool_event = next(event for event in trace_events if event["event"] == "tool_executed")
    assert tool_event["full_output_artifact"] == artifact_path
    assert tool_event["content_sha256"] == tool_item["content_sha256"]


def test_run_shell_status_is_parsed_from_full_result_before_artifact_rendering(tmp_path):
    agent = build_agent(
        tmp_path,
        [
            '<tool>{"name":"run_shell","args":{"command":"synthetic long failure","timeout":20}}</tool>',
            "<final>captured</final>",
        ],
    )
    long_stdout = "x" * 6000
    agent.tools["run_shell"] = RegisteredTool(
        name="run_shell",
        schema={"command": "str", "timeout": "int=20"},
        description="Synthetic shell command.",
        risky=True,
        runner=lambda args: f"stdout:\n{long_stdout}\nexit_code: 1\nstderr:\nboom",
    )

    assert agent.ask("run synthetic shell") == "captured"

    trace_events = read_jsonl(agent.current_run_dir / "trace.jsonl")
    tool_event = next(event for event in trace_events if event["event"] == "tool_executed")
    assert tool_event["status"] == "error"
    assert tool_event["tool_error_code"] == "tool_failed"
    assert tool_event["full_output_artifact"]
    assert tool_event["truncation_strategy"] == "tail"


def test_long_tool_output_artifact_ref_survives_external_run_store(tmp_path):
    external_runs = tmp_path.parent / f"{tmp_path.name}-external-runs"
    agent = build_agent(
        tmp_path,
        [
            '<tool>{"name":"run_shell","args":{"command":"synthetic long output","timeout":20}}</tool>',
            "<final>captured</final>",
        ],
        run_store=RunStore(external_runs),
    )
    agent.tools["run_shell"] = RegisteredTool(
        name="run_shell",
        schema={"command": "str", "timeout": "int=20"},
        description="Synthetic shell command.",
        risky=True,
        runner=lambda args: "exit_code: 0\nstdout:\n" + ("x" * 6000),
    )

    assert agent.ask("run synthetic shell") == "captured"

    tool_item = next(
        item
        for item in agent.session["history"]
        if item["role"] == "tool" and item["name"] == "run_shell"
    )
    artifact_ref = tool_item["artifact_ref"]
    assert artifact_ref
    assert (external_runs.parent / artifact_ref).exists()


def test_long_read_file_result_is_artifact_backed_when_history_is_microcompacted(tmp_path):
    large_text = "\n".join(f"line-{index} " + ("x" * 40) for index in range(120))
    (tmp_path / "large.txt").write_text(large_text, encoding="utf-8")
    agent = build_agent(
        tmp_path,
        [
            '<tool>{"name":"read_file","args":{"path":"large.txt","start":1,"end":120}}</tool>',
            "<final>read</final>",
            "<final>committed</final>",
        ],
    )

    assert agent.ask("read the large file") == "read"
    tool_item = next(item for item in agent.session["history"] if item["role"] == "tool")
    original_history_content = tool_item["content"]
    artifact_ref = tool_item["artifact_ref"]

    assert artifact_ref.endswith(".txt")
    assert tool_item["original_chars"] > len(original_history_content)
    assert "line-119" in (tmp_path / artifact_ref).read_text(encoding="utf-8")

    for index in range(4):
        agent.record({"role": "user", "content": f"later user {index}"})
        agent.record({"role": "assistant", "content": f"later answer {index}"})

    before_history = json.dumps(agent.session["history"], sort_keys=True)
    prompt, metadata = ContextManager(agent).build("continue")

    persisted_tool_item = next(
        item for item in agent.session["history"] if item.get("artifact_ref") == artifact_ref
    )
    assert json.dumps(agent.session["history"], sort_keys=True) == before_history
    assert "context_replacements" not in agent.session
    assert persisted_tool_item["content"] == original_history_content
    assert artifact_ref in prompt
    assert "line-119" not in prompt
    assert metadata["history"]["microcompact_artifact_refs"] == [artifact_ref]
    assert metadata["history"]["microcompact_saved_chars"] > 0
    assert metadata["history"]["proposed_replacements"]

    assert agent.ask("continue") == "committed"

    event_id = persisted_tool_item["event_id"]
    assert agent.session["context_replacements"][event_id]["content_sha256"] == persisted_tool_item["content_sha256"]
    assert agent.session["context_replacements"][event_id]["artifact_ref"] == artifact_ref


def test_recent_long_tool_result_is_not_microcompact_stubbed(tmp_path):
    large_text = "\n".join(f"line-{index} " + ("x" * 40) for index in range(120))
    (tmp_path / "large.txt").write_text(large_text, encoding="utf-8")
    agent = build_agent(
        tmp_path,
        [
            '<tool>{"name":"read_file","args":{"path":"large.txt","start":1,"end":120}}</tool>',
            "<final>read</final>",
        ],
    )

    assert agent.ask("read the large file") == "read"
    prompt, metadata = ContextManager(agent).build("continue")

    assert "read_file output saved:" not in prompt
    assert "full output saved:" in prompt
    assert metadata["history"]["microcompact_artifact_refs"] == []


def test_microcompact_keeps_old_tool_result_tied_to_current_changed_path(tmp_path):
    large_text = "\n".join(f"line-{index} " + ("x" * 40) for index in range(120))
    (tmp_path / "large.txt").write_text(large_text, encoding="utf-8")
    agent = build_agent(
        tmp_path,
        [
            '<tool>{"name":"read_file","args":{"path":"large.txt","start":1,"end":120}}</tool>',
            "<final>read</final>",
        ],
    )

    assert agent.ask("read the large file") == "read"
    for index in range(4):
        agent.record({"role": "user", "content": f"later user {index}"})
        agent.record({"role": "assistant", "content": f"later answer {index}"})
    agent.current_task_state.changed_paths = ["large.txt"]

    prompt, metadata = ContextManager(agent).build("continue")

    assert "read_file output saved:" not in prompt
    assert "full output saved:" in prompt
    assert metadata["history"]["microcompact_artifact_refs"] == []


def test_microcompact_keeps_latest_failed_tool_result_visible(tmp_path):
    script = "for i in range(140): print(f'FAIL-{i}')\nraise SystemExit(1)"
    command = shell_join([sys.executable, "-c", f"exec({script!r})"])
    agent = build_agent(
        tmp_path,
        [
            f'<tool>{{"name":"run_shell","args":{{"command":{json.dumps(command)},"timeout":20}}}}</tool>',
            "<final>captured failure</final>",
        ],
    )

    assert agent.ask("capture a long failure") == "captured failure"
    for index in range(4):
        agent.record({"role": "user", "content": f"later-{index}"})
        agent.record({"role": "assistant", "content": f"done-{index}"})

    prompt, metadata = ContextManager(agent).build("continue")

    assert "FAIL-0" in prompt
    assert "run_shell output saved:" not in prompt
    assert metadata["history"]["microcompact_artifact_refs"] == []


def test_microcompact_keeps_latest_workspace_changing_tool_result_visible(tmp_path):
    script = "\n".join(
        [
            "from pathlib import Path",
            "Path('notes').mkdir(exist_ok=True)",
            "Path('notes/out.txt').write_text('ok\\n')",
            "for i in range(140): print(f'CHANGED-{i}')",
        ]
    )
    command = shell_join([sys.executable, "-c", f"exec({script!r})"])
    agent = build_agent(
        tmp_path,
        [
            f'<tool>{{"name":"run_shell","args":{{"command":{json.dumps(command)},"timeout":20}}}}</tool>',
            "<final>captured change</final>",
        ],
    )

    assert agent.ask("capture a long workspace change") == "captured change"
    for index in range(4):
        agent.record({"role": "user", "content": f"later-{index}"})
        agent.record({"role": "assistant", "content": f"done-{index}"})

    prompt, metadata = ContextManager(agent).build("continue")

    assert "CHANGED-0" in prompt
    assert "run_shell output saved:" not in prompt
    assert metadata["history"]["microcompact_artifact_refs"] == []
