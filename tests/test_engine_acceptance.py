"""End-to-end engine acceptance tests for user-visible turn behavior."""

import json
import sys

from lite.testing import ScriptedModelClient, read_jsonl, shell_join
from lite import Lite, SessionStore, WorkspaceContext
from lite.providers import ProviderError


def build_agent(tmp_path, outputs, **kwargs):
    (tmp_path / "README.md").write_text("demo\n", encoding="utf-8")
    workspace = WorkspaceContext.build(tmp_path)
    store = SessionStore(tmp_path / ".lite" / "sessions")
    kwargs.setdefault("feature_flags", {"multi_agent": True})
    return Lite(
        model_client=ScriptedModelClient(outputs),
        workspace=workspace,
        session_store=store,
        approval_policy="auto",
        **kwargs,
    )


def test_engine_streams_a_real_session_with_tool_artifacts(tmp_path):
    agent = build_agent(
        tmp_path,
        [
            '<tool name="write_file" path="notes/result.txt"><content>ok\n</content></tool>',
            "<final>Wrote it.</final>",
        ],
    )

    events = list(agent.engine.run_turn("create the result file"))

    assert [event["type"] for event in events] == [
        "turn_started",
        "model_requested",
        "model_parsed",
        "tool_call",
        "tool_result",
        "model_requested",
        "model_parsed",
        "final",
        "turn_finished",
    ]
    assert events[-2]["content"] == "Wrote it."
    assert (tmp_path / "notes" / "result.txt").read_text(encoding="utf-8") == "ok\n"

    persisted_events = read_jsonl(agent.session_event_bus.path)
    event_names = [event["event"] for event in persisted_events]
    assert event_names.count("context_orchestrator_decision") == 1
    assert event_names.count("context_usage_recorded") == 1
    assert "tool_started" in event_names
    assert "tool_finished" in event_names
    tool_finished = next(
        event for event in persisted_events if event["event"] == "tool_finished"
    )
    assert tool_finished["tool_name"] == "write_file"
    assert tool_finished["status"] == "ok"
    assert tool_finished["workspace_changed"] is True
    assert event_names[-4:] == [
        "model_requested",
        "model_parsed",
        "assistant_message",
        "turn_finished",
    ]

    report_path = agent.current_run_dir / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "completed"
    assert report["final_answer"] == "Wrote it."


def test_workspace_mutation_adds_completion_contract_feedback(tmp_path):
    agent = build_agent(
        tmp_path,
        [
            '<tool name="write_file" path="app.py"><content>VALUE = 1\n</content></tool>',
            "<final>Done.</final>",
        ],
    )

    agent.ask("change app.py")

    feedback = agent.model_client.requests[-1].turns[0].feedback
    assert feedback == (
        "Completion contract: the workspace changed. Run focused verification "
        "before returning the final answer.",
    )


def test_engine_reports_context_budget_summary_from_prompt_metadata(tmp_path):
    agent = build_agent(tmp_path, ["<final>Done.</final>"])

    list(agent.engine.run_turn("summarize context usage"))

    report = json.loads(
        (agent.current_run_dir / "report.json").read_text(encoding="utf-8")
    )
    summary = report["evidence_summaries"]["context_budget_summary"]
    usage = report["prompt_metadata"]["context_usage"]
    assert summary["schema_version"] == "lite.context_budget_summary.v1"
    assert summary["budget_unit"] == "tokens_estimated"
    assert summary["token_estimator"] == "context_usage_analyzer"
    assert summary["estimated_tokens"] == usage["total_estimated_tokens"]
    assert summary["effective_window"] == (
        usage["context_window"] - usage["reserved_output_tokens"]
    )
    assert summary["prompt_changed_by_phase_3"] is False
    assert summary["reductions"] == []
    assert "pressure_tier" in summary
    assert "usage_source" in summary
    assert summary["snip_count"] == 0
    assert summary["prune_count"] == 0
    assert summary["summary_called"] is False
    assert summary["summary_delta_event_count"] == 0
    assert summary["replacement_cache_hits"] == 0
    assert summary["replacement_records_created"] == 0
    assert summary["replacement_ledger_enabled"] is True
    assert summary["provider_usage_available"] is False
    assert summary["saved_chars"] == 0
    assert summary["cached_tokens"] == 0


def test_engine_backfills_current_provider_usage_into_report(tmp_path):
    from lite.providers import ModelResult

    agent = build_agent(
        tmp_path,
        [
            ModelResult(
                text="<final>Done.</final>",
                metadata={
                    "input_tokens": 8192,
                    "output_tokens": 16,
                    "cached_tokens": 4608,
                    "cache_write_tokens": 1024,
                },
            )
        ],
    )

    list(agent.engine.run_turn("summarize context usage"))

    report = json.loads(
        (agent.current_run_dir / "report.json").read_text(encoding="utf-8")
    )
    usage = report["prompt_metadata"]["context_usage"]
    summary = report["evidence_summaries"]["context_budget_summary"]
    assert usage["usage_source"] == "actual"
    assert usage["actual_input_tokens"] == 8192
    assert usage["cached_tokens"] == 4608
    assert usage["cache_write_tokens"] == 1024
    assert summary["provider_usage_available"] is True
    assert summary["actual_input_tokens"] == 8192
    assert summary["cached_tokens"] == 4608
    assert summary["cache_write_tokens"] == 1024


def test_engine_records_provider_error_as_failed_run(tmp_path):
    agent = build_agent(
        tmp_path,
        [
            ProviderError(
                "rate limited",
                provider="openai",
                model="gpt-test",
                base_url="https://example.test/v1",
                code="rate_limited",
                http_status=429,
                retryable=True,
                attempts=3,
                retry_count=2,
            )
        ],
    )

    events = list(agent.engine.run_turn("call a rate limited provider"))

    assert events[-2]["type"] == "stop"
    assert "rate_limited" in events[-2]["content"]
    assert events[-2]["content"].startswith("模型错误")
    report = json.loads(
        (agent.current_run_dir / "report.json").read_text(encoding="utf-8")
    )
    assert report["status"] == "failed"
    assert report["stop_reason"] == "model_error"
    assert report["prompt_metadata"]["provider_error"]["code"] == "rate_limited"
    assert report["prompt_metadata"]["provider_error"]["retry_count"] == 2

    trace_events = read_jsonl(agent.current_run_dir / "trace.jsonl")
    model_error = next(
        event for event in trace_events if event["event"] == "model_error"
    )
    assert model_error["error"]["http_status"] == 429

    persisted_events = read_jsonl(agent.session_event_bus.path)
    assert any(
        event["event"] == "model_error" and event["code"] == "rate_limited"
        for event in persisted_events
    )


def test_worker_notification_drained_during_turn_is_streamed(tmp_path):
    agent = build_agent(
        tmp_path,
        [
            '<tool>{"name":"agent","args":{"description":"Inspect","prompt":"Read README","subagent_type":"Explore"}}</tool>',
            "<final>Child done.</final>",
            "<final>Parent done.</final>",
        ],
        max_steps=3,
    )

    events = list(agent.engine.run_turn("delegate and continue"))

    notifications = [
        event for event in events if event["type"] == "worker_notification"
    ]
    assert len(notifications) == 1
    assert "<task-id>agent_1</task-id>" in notifications[0]["content"]


def test_verification_signal_passes_after_workspace_verification(tmp_path):
    command = shell_join([sys.executable, "-m", "compileall", "notes"])
    agent = build_agent(
        tmp_path,
        [
            '<tool name="write_file" path="notes/result.py"><content>VALUE = 1\n</content></tool>',
            f'<tool>{{"name":"run_shell","args":{{"command":{json.dumps(command)},"timeout":20}}}}</tool>',
            "<final>Verified.</final>",
        ],
        max_steps=3,
    )

    events = list(agent.engine.run_turn("write and verify python code"))

    assert events[-2]["content"] == "Verified."
    report = json.loads(
        (agent.current_run_dir / "report.json").read_text(encoding="utf-8")
    )
    signal = report["evidence_summaries"]["verification_signal"]
    assert signal["schema_version"] == "lite.verification_signal.v1"
    assert signal["state"] == "passed"
    assert signal["command"] == command
    assert signal["command_class"] == "compile"
    assert signal["after_last_workspace_change"] is True
    assert signal["changed_paths_present"] is True
    assert signal["covers_changed_paths"] is False
    assert signal["coverage_confidence"] == "unknown"
    assert "notes/result.py" in signal["changed_paths"]


def test_verify_tool_runs_default_python_check_and_records_receipt(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        "[tool.pytest.ini_options]\ntestpaths = ['tests']\n",
        encoding="utf-8",
    )
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_smoke.py").write_text(
        "def test_smoke():\n    assert True\n",
        encoding="utf-8",
    )
    agent = build_agent(
        tmp_path,
        [
            {"name": "verify", "args": {}},
            "Verification passed.",
        ],
        max_steps=2,
    )

    events = list(agent.engine.run_turn("verify this Python project"))

    assert events[-2]["content"] == "Verification passed."
    tool_result = next(event for event in events if event["type"] == "tool_result")
    receipt = tool_result["metadata"]["verification_receipt"]
    assert receipt["schema_version"] == "lite.verification_receipt.v1"
    assert receipt["command_class"] == "test"
    assert receipt["exit_code"] == 0
    assert receipt["command"].startswith(shell_join([sys.executable]))
    assert "-m pytest -q" in receipt["command"]
    assert receipt["workspace_revision"].startswith("sha256:")

    report = json.loads(
        (agent.current_run_dir / "report.json").read_text(encoding="utf-8")
    )
    signal = report["evidence_summaries"]["verification_signal"]
    assert signal["state"] == "passed"
    assert signal["test_state"] == "passed"
    assert signal["verification_receipt"] == receipt


def test_verify_tool_rejects_non_verification_commands_before_execution(tmp_path):
    agent = build_agent(tmp_path, [])

    result = agent.run_tool(
        "verify",
        {"command": "echo bypass>owned.txt", "timeout": 20},
    )

    assert "not recognized" in result
    assert not (tmp_path / "owned.txt").exists()
    assert agent._last_tool_result_metadata["tool_status"] == "rejected"
    assert agent._last_tool_result_metadata["tool_error_code"] == "invalid_arguments"
