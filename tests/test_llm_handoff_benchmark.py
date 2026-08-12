import json
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from lite.evaluation.context_cost import generate_report, run_paired_experiment
from lite.evaluation import context_cost


ROOT = Path(__file__).resolve().parents[1]
TASKS_PATH = ROOT / "benchmarks" / "long_session_tasks.json"


def _load_long_session_tasks():
    return json.loads(TASKS_PATH.read_text(encoding="utf-8"))["tasks"]


def test_long_session_tasks_define_five_fixture_backed_tasks():
    tasks = _load_long_session_tasks()
    expected_scripted_counts = {
        "multi-file-refactor": 10,
        "debug-and-fix": 8,
        "add-endpoint-with-test": 12,
        "config-migration": 10,
        "dependency-upgrade": 12,
    }

    assert len(tasks) == 5
    assert {task["category"] for task in tasks} == {"long_session"}
    assert {task["id"] for task in tasks} == {
        "multi-file-refactor",
        "debug-and-fix",
        "add-endpoint-with-test",
        "config-migration",
        "dependency-upgrade",
    }
    for task in tasks:
        assert (ROOT / task["fixture_repo"]).is_dir()
        assert 8 <= int(task["step_budget"]) <= 24
        assert len(task["scripted_outputs"]) == expected_scripted_counts[task["id"]]


def test_run_paired_experiment_scripted_populates_llm_handoff_metrics(tmp_path):
    tasks = _load_long_session_tasks()[:1]

    payload = run_paired_experiment(
        tasks=tasks,
        variants=["full_orchestrator", "full_orchestrator_with_llm_handoff"],
        mode="scripted",
        provider=None,
        repetitions=1,
        output_dir=tmp_path / "work",
    )

    rows = payload["rows"]
    assert {row["variant"] for row in rows} == {
        "full_orchestrator",
        "full_orchestrator_with_llm_handoff",
    }
    assert all(row["status"] == "completed" for row in rows)
    assert all(row["verification_status"] == "passed" for row in rows)

    handoff_rows = [
        row for row in rows if row["variant"] == "full_orchestrator_with_llm_handoff"
    ]
    assert handoff_rows
    assert handoff_rows[0]["compact_summary_mode"] == "llm"
    assert isinstance(handoff_rows[0]["compact_call_input_tokens"], int)
    assert isinstance(handoff_rows[0]["compact_call_output_tokens"], int)
    assert isinstance(handoff_rows[0]["compact_net_benefit_tokens"], int)
    assert handoff_rows[0]["usage"]["input_tokens"] >= (
        handoff_rows[0]["prompt_estimated_tokens"]
        + handoff_rows[0]["compact_call_input_tokens"]
    )
    assert handoff_rows[0]["usage"]["output_tokens"] >= handoff_rows[0]["compact_call_output_tokens"]
    assert payload["summary"]["estimated_proxy_only"]["claimable_cost_win"] is False


def test_live_mode_routes_through_provider_client_factory(tmp_path):
    tasks = _load_long_session_tasks()[:1]
    calls = []

    def provider_client_factory(*, provider, task, variant, repeat):
        calls.append((provider, task["id"], variant, repeat))
        from lite.evaluation.context_cost import _LongSessionScriptedClient

        return _LongSessionScriptedClient(task["scripted_outputs"])

    payload = run_paired_experiment(
        tasks=tasks,
        variants=["full_orchestrator"],
        mode="live",
        provider="deepseek",
        repetitions=1,
        output_dir=tmp_path / "work",
        provider_client_factory=provider_client_factory,
    )

    assert calls == [("deepseek", "multi-file-refactor", "full_orchestrator", 0)]
    assert payload["pricing_profile"] == "llm-handoff-live-configured"
    assert payload["rows"][0]["layer"] == "live"
    assert payload["rows"][0]["verification_status"] == "passed"


def test_live_mode_without_provider_config_reports_clear_blocked_error(monkeypatch, tmp_path):
    import lite.evaluation.context_cost as context_cost

    monkeypatch.setattr(
        context_cost,
        "resolve_provider_config",
        lambda provider, start=".": type(
            "Config",
            (),
            {
                "name": provider,
                "protocol": "openai",
                "api_key": "",
                "base_url": "https://example.test/v1",
                "model": "test-model",
            },
        )(),
    )

    with pytest.raises(RuntimeError, match="live provider config blocked: API key missing"):
        run_paired_experiment(
            tasks=_load_long_session_tasks()[:1],
            variants=["full_orchestrator"],
            mode="live",
            provider="deepseek",
            repetitions=1,
            output_dir=tmp_path / "work",
        )


def test_long_session_benchmark_verifier_can_downgrade_runtime_pass(tmp_path):
    task = dict(_load_long_session_tasks()[1])
    task["verifier"] = "python3 -c \"raise SystemExit(1)\""

    payload = run_paired_experiment(
        tasks=[task],
        variants=["full_orchestrator"],
        mode="scripted",
        provider=None,
        repetitions=1,
        output_dir=tmp_path / "work",
    )

    row = payload["rows"][0]
    assert row["status"] == "completed"
    assert row["verification_status"] == "failed"


def test_generate_report_includes_llm_handoff_comparison():
    payload = {
        "summary": {},
        "pricing": {},
        "rows": [
            {
                "task_id": "task-a",
                "variant": "full_orchestrator",
                "cost_usd": 0.01,
                "compact_net_benefit_tokens": None,
                "compact_summary_mode": "deterministic",
            },
            {
                "task_id": "task-a",
                "variant": "full_orchestrator_with_llm_handoff",
                "cost_usd": 0.012,
                "compact_net_benefit_tokens": -15,
                "compact_summary_mode": "llm",
            },
        ],
    }

    report = generate_report(payload, include_llm_handoff_comparison=True)

    assert "## LLM Handoff vs Deterministic Comparison" in report
    assert "| task-a |" in report
    assert "Median net benefit: -15 tokens" in report
    assert "Net-negative tasks: task-a" in report


def test_generate_report_includes_all_repeats_in_llm_handoff_comparison():
    payload = {
        "summary": {},
        "pricing": {},
        "rows": [
            {
                "task_id": "task-a",
                "repeat": 0,
                "variant": "full_orchestrator",
                "cost_usd": 0.01,
                "compact_net_benefit_tokens": None,
                "compact_summary_mode": "deterministic",
            },
            {
                "task_id": "task-a",
                "repeat": 0,
                "variant": "full_orchestrator_with_llm_handoff",
                "cost_usd": 0.012,
                "compact_net_benefit_tokens": -10,
                "compact_summary_mode": "llm",
            },
            {
                "task_id": "task-a",
                "repeat": 1,
                "variant": "full_orchestrator",
                "cost_usd": 0.02,
                "compact_net_benefit_tokens": None,
                "compact_summary_mode": "deterministic",
            },
            {
                "task_id": "task-a",
                "repeat": 1,
                "variant": "full_orchestrator_with_llm_handoff",
                "cost_usd": 0.018,
                "compact_net_benefit_tokens": 30,
                "compact_summary_mode": "llm",
            },
        ],
    }

    report = generate_report(payload, include_llm_handoff_comparison=True)

    assert "| Task | Repeat | Deterministic Cost | LLM Handoff Cost | Net Benefit | Mode Used |" in report
    assert "| task-a | 0 |" in report
    assert "| task-a | 1 |" in report
    assert "Median net benefit: 10 tokens" in report
    assert "Positive net benefit: 50%" in report
    assert "Negative net benefit: 50%" in report
    assert "Net-negative tasks: task-a#0" in report


def test_fixture_verifiers_pass_after_scripted_correct_state(tmp_path):
    tasks = _load_long_session_tasks()
    payload = run_paired_experiment(
        tasks=tasks,
        variants=["full_orchestrator"],
        mode="scripted",
        provider=None,
        repetitions=1,
        output_dir=tmp_path / "work",
    )

    assert len(payload["rows"]) == 5
    assert all(row["verification_status"] == "passed" for row in payload["rows"])


def test_long_session_row_timeout_cancels_turn_and_closes_agent(monkeypatch, tmp_path):
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    (fixture / "README.md").write_text("demo\n", encoding="utf-8")
    ask_started = threading.Event()
    ask_released = threading.Event()
    run_finished = threading.Event()
    lifecycle = {"aborted": False, "closed": False}

    class BlockingAgent:
        def __init__(self, **_kwargs):
            self.current_run_dir = None
            self.context_orchestrator = SimpleNamespace(
                _compact_request=lambda _metadata, _snapshot: (None, None, None)
            )

        def record(self, _message):
            return None

        def ask(self, _prompt):
            ask_started.set()
            ask_released.wait()

        def abort_current_turn(self):
            lifecycle["aborted"] = True
            ask_released.set()

        def close(self):
            lifecycle["closed"] = True

    monkeypatch.setattr(context_cost, "Lite", BlockingAgent)
    monkeypatch.setattr(
        context_cost,
        "run_verifier",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0),
    )
    monkeypatch.setattr(
        context_cost,
        "extract_usage_from_artifacts",
        lambda *_args, **_kwargs: "row",
    )
    task = {
        "id": "timeout",
        "fixture_repo": str(fixture),
        "scripted_outputs": [],
        "allowed_tools": [],
        "step_budget": 1,
        "row_timeout": 0,
        "prompt": "block",
        "verifier": "true",
    }
    result = []

    def run_task():
        try:
            result.append(
                context_cost._run_long_session_task(
                    task,
                    variant="full_orchestrator",
                    repeat=0,
                    mode="scripted",
                    provider=None,
                    provider_client_factory=None,
                    output_dir=tmp_path / "work",
                    pricing=context_cost.DEFAULT_PROXY_PRICING,
                )
            )
        finally:
            run_finished.set()

    runner = threading.Thread(target=run_task)
    runner.start()
    assert ask_started.wait(2)
    finished_without_external_release = run_finished.wait(1)
    ask_released.set()
    runner.join(2)

    assert finished_without_external_release is True
    assert result == ["row"]
    assert lifecycle == {"aborted": True, "closed": True}


def test_llm_handoff_benchmark_cli_scripted_smoke(tmp_path):
    output_dir = tmp_path / "artifacts"
    tasks_path = tmp_path / "tasks.json"
    tasks_path.write_text(
        json.dumps({"tasks": _load_long_session_tasks()[:1]}),
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_llm_handoff_benchmark.py",
            "--mode",
            "scripted",
            "--output-dir",
            str(output_dir),
            "--tasks",
            str(tasks_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    assert f"Results: {output_dir / 'results.json'}" in result.stdout
    assert (output_dir / "results.json").is_file()
    assert "## LLM Handoff vs Deterministic Comparison" in (
        output_dir / "report.md"
    ).read_text(encoding="utf-8")
