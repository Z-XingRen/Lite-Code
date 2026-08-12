"""Acceptance tests for final-readiness behavior inside real engine turns."""

import json
import sys

from lite import Lite, SessionStore, WorkspaceContext
from lite.testing import ScriptedModelClient, shell_join


def build_agent(tmp_path, outputs, **kwargs):
    (tmp_path / "README.md").write_text("demo\n", encoding="utf-8")
    workspace = WorkspaceContext.build(tmp_path)
    store = SessionStore(tmp_path / ".lite" / "sessions")
    return Lite(
        model_client=ScriptedModelClient(outputs),
        workspace=workspace,
        session_store=store,
        approval_policy="auto",
        **kwargs,
    )


def read_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_warn_final_readiness_allows_low_pressure_missing_provider_usage(tmp_path):
    agent = build_agent(
        tmp_path,
        ["<final>done</final>"],
        final_readiness_mode="warn",
    )

    events = list(agent.engine.run_turn("answer directly"))

    assert [event["type"] for event in events] == [
        "turn_started",
        "model_requested",
        "model_parsed",
        "final",
        "turn_finished",
    ]


def test_warn_final_readiness_records_one_decision_without_an_extra_model_call(tmp_path):
    agent = build_agent(
        tmp_path,
        [
            '<tool name="write_file" path="notes/result.txt"><content>ok\n</content></tool>',
            "<final>Done without verification.</final>",
        ],
        final_readiness_mode="warn",
        max_steps=3,
    )

    events = list(agent.engine.run_turn("write the result"))

    assert not [event for event in events if event["type"] == "runtime_notice"]
    assert events[-2]["type"] == "final"
    assert events[-2]["content"] == "Done without verification."

    trace = read_jsonl(agent.current_run_dir / "trace.jsonl")
    readiness = [event for event in trace if event["event"] == "final_readiness_decision"]
    assert [(event["decision"], event["action"]) for event in readiness] == [
        ("warn", "none")
    ]
    assert "reminder_already_sent" not in readiness[0]
    report = json.loads(
        (agent.current_run_dir / "report.json").read_text(encoding="utf-8")
    )
    assert report["evidence_summaries"]["final_readiness_summary"]["warn_count"] == 1
    assert "remind_count" not in report["evidence_summaries"]["final_readiness_summary"]
    assert (
        report["evidence_summaries"]["final_readiness_summary"]["schema_version"]
        == "lite.final_readiness_summary.v1"
    )


def test_enforce_final_readiness_repairs_once_then_blocks_unverified_changes(tmp_path):
    agent = build_agent(
        tmp_path,
        [
            '<tool name="write_file" path="notes/result.txt"><content>ok\n</content></tool>',
            "<final>Done without verification.</final>",
            "<final>Still done without verification.</final>",
        ],
        final_readiness_mode="enforce",
        max_steps=2,
    )

    events = list(agent.engine.run_turn("write the result"))

    notices = [event for event in events if event["type"] == "runtime_notice"]
    assert len(notices) == 1
    assert "Verification did not succeed" in notices[0]["content"]
    stop_event = next(event for event in events if event["type"] == "stop")
    assert "Verification did not succeed" in stop_event["content"]
    assert events[-1]["stop_reason"] == "final_gate_blocked"

    trace = read_jsonl(agent.current_run_dir / "trace.jsonl")
    readiness = [event for event in trace if event["event"] == "final_readiness_decision"]
    assert [(event["decision"], event["action"]) for event in readiness] == [
        ("repair", "runtime_notice"),
        ("block", "block"),
    ]
    assert readiness[-1]["completion_contract"]["repair_attempt_count"] == 1
    assert readiness[-1]["completion_contract"]["unresolved_codes"] == [
        "verification_required"
    ]

    report = json.loads(
        (agent.current_run_dir / "report.json").read_text(encoding="utf-8")
    )
    assert report["status"] == "stopped"
    assert report["stop_reason"] == "final_gate_blocked"
    assert report["evidence_summaries"]["final_readiness_summary"]["block_count"] == 1


def test_enforce_completion_contract_accepts_verification_after_repair_notice(tmp_path):
    compile_command = shell_join([sys.executable, "-m", "py_compile", "app.py"])
    agent = build_agent(
        tmp_path,
        [
            '<tool name="write_file" path="app.py"><content>VALUE = 1\n</content></tool>',
            "<final>Done before verification.</final>",
            f'<tool>{{"name":"run_shell","args":{{"command":{json.dumps(compile_command)},"timeout":20}}}}</tool>',
            "<final>Verified.</final>",
        ],
        final_readiness_mode="enforce",
        max_steps=3,
    )

    events = list(agent.engine.run_turn("change app.py and verify it"))

    notices = [event for event in events if event["type"] == "runtime_notice"]
    assert len(notices) == 1
    assert events[-2]["type"] == "final"
    assert events[-2]["content"] == "Verified."
    report = json.loads(
        (agent.current_run_dir / "report.json").read_text(encoding="utf-8")
    )
    contract = report["evidence_summaries"]["completion_contract"]
    assert contract["ready"] is True
    assert contract["unresolved_codes"] == []
    assert contract["repair_attempt_count"] == 1


def test_enforce_final_readiness_blocks_partial_success_workspace_changes(tmp_path):
    command = shell_join(
        [
            sys.executable,
            "-c",
            "from pathlib import Path; Path('notes/result.txt').parent.mkdir(exist_ok=True); "
            "Path('notes/result.txt').write_text('partial\\n'); raise SystemExit(1)",
        ]
    )
    agent = build_agent(
        tmp_path,
        [
            f'<tool>{{"name":"run_shell","args":{{"command":{json.dumps(command)},"timeout":20}}}}</tool>',
            "<final>Partial write is fine.</final>",
        ],
        final_readiness_mode="enforce",
        max_steps=2,
    )

    events = list(agent.engine.run_turn("write the result with shell"))

    stop_event = next(event for event in events if event["type"] == "stop")
    assert "partially succeeded" in stop_event["content"]
    assert events[-1]["stop_reason"] == "final_gate_blocked"

    trace = read_jsonl(agent.current_run_dir / "trace.jsonl")
    tool_event = next(event for event in trace if event["event"] == "tool_executed")
    assert tool_event["status"] == "partial_success"
    assert tool_event["workspace_changed"] is True

    readiness = [event for event in trace if event["event"] == "final_readiness_decision"]
    assert [(event["decision"], event["action"]) for event in readiness] == [
        ("block", "block")
    ]


def test_enforce_readiness_accepts_fresh_structured_compile_receipt(
    tmp_path,
):
    compile_command = shell_join([sys.executable, "-m", "py_compile", "app.py"])
    agent = build_agent(
        tmp_path,
        [
            '<tool name="write_file" path="app.py"><content>VALUE = 1\n</content></tool>',
            '<tool>{"name":"run_shell","args":{"command":"pytest missing_tests -q","timeout":20}}</tool>',
            f'<tool>{{"name":"run_shell","args":{{"command":{json.dumps(compile_command)},"timeout":20}}}}</tool>',
            "<final>Compile succeeded.</final>",
        ],
        final_readiness_mode="enforce",
        max_steps=4,
    )

    events = list(agent.engine.run_turn("change code and verify it"))

    assert not [event for event in events if event["type"] == "runtime_notice"]
    assert events[-2]["type"] == "final"
    assert events[-2]["content"] == "Compile succeeded."
    report = json.loads(
        (agent.current_run_dir / "report.json").read_text(encoding="utf-8")
    )
    signal = report["evidence_summaries"]["verification_signal"]
    assert signal["state"] == "passed"
    assert signal["command_class"] == "compile"
    assert signal["test_state"] == "failed"
    assert signal["last_successful_verification_sequence"] > signal["last_mutation_sequence"]


def test_warn_final_readiness_records_net_negative_llm_compaction(tmp_path):
    agent = build_agent(
        tmp_path,
        [
            """## Goal
Continue the large task.

## Files Read
- README.md

## Next Steps
- Finish the task.
            """,
            "<final>Done after compact.</final>",
        ],
        final_readiness_mode="warn",
        max_steps=2,
    )
    agent.model_client.context_window = 1000
    agent.model_client.last_completion_metadata = {
        "input_tokens": 500,
        "output_tokens": 500,
        "total_tokens": 1000,
    }
    for index in range(5):
        agent.record({"role": "user", "content": f"request {index} " + ("x" * 900)})
        agent.record({"role": "assistant", "content": f"answer {index} " + ("y" * 900)})

    events = list(agent.engine.run_turn("finish"))

    assert not any(event["type"] == "runtime_notice" for event in events)
    assert events[-2]["type"] == "final"
    trace = read_jsonl(agent.current_run_dir / "trace.jsonl")
    readiness = [event for event in trace if event["event"] == "final_readiness_decision"]
    assert any(event["decision"] == "warn" for event in readiness)
    assert any("compact_net_negative" in event["reasons"] for event in readiness)
