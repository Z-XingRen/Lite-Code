import json
import os
from unittest.mock import patch

from lite.evaluation.metrics import (
    _provider_profile,
    main as metrics_main,
    run_context_ablation_v2,
    run_memory_fidelity_v1,
    run_memory_ablation_v2,
    run_recovery_ablation_v2,
    write_benchmark_core_report,
)


def test_run_context_ablation_v2_writes_expected_artifact(tmp_path):
    artifact_path = tmp_path / "artifacts" / "context-ablation-v2.json"

    artifact = run_context_ablation_v2(
        artifact_path=artifact_path,
        repetitions=1,
    )

    assert artifact_path.exists()
    assert artifact["artifact_type"] == "context-ablation-v2"
    assert artifact["config_count"] == 12
    assert len(artifact["configs"]) == 12
    assert "current_request_preserved_rate" in artifact["summary"]


def test_metrics_cli_context_ab_writes_artifacts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    assert metrics_main(["--run", "context_ab"]) == 0

    assert (tmp_path / "artifacts" / "context-ab-v1" / "results.json").is_file()
    assert (tmp_path / "artifacts" / "context-ab-v1" / "report.md").is_file()


def test_provider_profile_uses_toml_routing_and_legacy_env_secret(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".lite.toml").write_text(
        "\n".join(
            [
                "[providers.deepseek]",
                'protocol = "anthropic"',
                'api_key = "sk-project-deepseek"',
                'model = "deepseek-v4-pro"',
                'base_url = "https://api.deepseek.com/anthropic"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with patch.dict(
        os.environ,
        {
            "LITE_DEEPSEEK_API_KEY": "sk-legacy-deepseek",
            "LITE_DEEPSEEK_MODEL": "legacy-deepseek-model",
            "LITE_DEEPSEEK_API_BASE": "https://legacy.deepseek.example/anthropic",
        },
        clear=True,
    ):
        profile = _provider_profile("deepseek")

    assert profile["status"] == "ready"
    assert profile["api_key"] == "sk-legacy-deepseek"
    assert profile["model"] == "deepseek-v4-pro"
    assert profile["base_url"] == "https://api.deepseek.com/anthropic"


def test_run_memory_ablation_v2_writes_expected_artifact(tmp_path):
    artifact_path = tmp_path / "artifacts" / "memory-ablation-v2.json"

    artifact = run_memory_ablation_v2(
        artifact_path=artifact_path,
        repetitions=1,
    )

    assert artifact_path.exists()
    assert artifact["artifact_type"] == "memory-ablation-v2"
    assert artifact["task_count"] == 12
    assert set(artifact["variants"]) == {"memory_on", "memory_off", "memory_irrelevant"}
    assert "memory_hit_rate" in artifact["variants"]["memory_on"]


def test_memory_fidelity_irrelevant_memory_present_category(tmp_path):
    artifact = run_memory_fidelity_v1(tmp_path / "artifacts" / "memory-fidelity-v1.json")
    row = next(row for row in artifact["rows"] if row["category"] == "irrelevant_memory_present")

    assert row["passed"]
    assert not row["distractor_selected"]


def test_memory_fidelity_superseded_fact_category(tmp_path):
    artifact = run_memory_fidelity_v1(tmp_path / "artifacts" / "memory-fidelity-v1.json")
    row = next(row for row in artifact["rows"] if row["category"] == "superseded_fact")

    assert row["passed"]
    assert row["new_fact_selected"]
    assert row["old_fact_superseded"]


def test_memory_fidelity_secret_shaped_category(tmp_path):
    artifact = run_memory_fidelity_v1(tmp_path / "artifacts" / "memory-fidelity-v1.json")
    row = next(row for row in artifact["rows"] if row["category"] == "secret_shaped")

    assert row["passed"]
    assert not row["secret_selected"]


def test_run_memory_fidelity_v1_writes_expected_artifact(tmp_path):
    artifact_path = tmp_path / "artifacts" / "memory-fidelity-v1.json"

    artifact = run_memory_fidelity_v1(artifact_path)

    assert artifact_path.exists()
    assert artifact["artifact_type"] == "memory-fidelity-v1"
    assert artifact["summary"]["irrelevant_injection_rate"] == 0
    assert artifact["summary"]["supersede_success_rate"] == 1
    assert artifact["summary"]["secret_exposure_rate"] == 0
    assert artifact["summary"]["stale_detection_rate"] == 1
    assert artifact["summary"]["stale_use_rate"] == 0
    assert artifact["summary"]["poison_quarantine_rate"] == 1
    assert artifact["summary"]["benign_recall_retention_rate"] == 1
    assert artifact["schema_version"] == 1
    assert {row["category"] for row in artifact["rows"]} == {
        "irrelevant_memory_present",
        "superseded_fact",
        "secret_shaped",
        "stale_evidence",
        "prompt_injection",
    }


def test_memory_fidelity_stale_and_prompt_injection_categories(tmp_path):
    artifact = run_memory_fidelity_v1(tmp_path / "artifacts" / "memory-fidelity-v1.json")
    stale = next(row for row in artifact["rows"] if row["category"] == "stale_evidence")
    poison = next(row for row in artifact["rows"] if row["category"] == "prompt_injection")

    assert stale["passed"]
    assert stale["stale_detected"]
    assert not stale["stale_selected"]
    assert poison["passed"]
    assert poison["attack_quarantined"]
    assert poison["benign_selected"]


def test_run_recovery_ablation_v2_writes_expected_artifact(tmp_path):
    artifact_path = tmp_path / "artifacts" / "recovery-ablation-v2.json"

    artifact = run_recovery_ablation_v2(
        artifact_path=artifact_path,
        repetitions=1,
    )

    assert artifact_path.exists()
    assert artifact["artifact_type"] == "recovery-ablation-v2"
    assert artifact["schema_version"] == 2
    assert artifact["task_count"] == 11
    assert set(artifact["variants"]) == {"resume_enabled", "resume_disabled"}
    assert set(artifact["variants"]["resume_enabled"]["summary"]) >= {
        "resume_success_rate",
        "stale_reanchor_rate",
        "workspace_drift_detection_rate",
        "resume_false_accept_rate",
        "resumption_success_rate",
        "first_action_correctness",
        "todo_continuity_rate",
    }
    assert artifact["variants"]["resume_enabled"]["summary"]["resumption_success_rate"] >= 0.8
    assert artifact["variants"]["resume_enabled"]["summary"]["first_action_correctness"] >= 0.8
    assert artifact["variants"]["resume_enabled"]["summary"]["todo_continuity_rate"] >= 0.8


def write_core_report_artifacts(artifact_dir, *, include_context_ab=False):
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "harness-regression-v2.json": {
            "summary": {
                "total_tasks": 12,
                "pass_rate": 1.0,
                "within_budget_rate": 1.0,
                "verifier_pass_rate": 1.0,
            },
            "failure_category_counts": {},
        },
        "context-ablation-v2.json": {
            "config_count": 12,
            "summary": {
                "avg_full_prompt_chars": 1000.0,
                "avg_raw_prompt_chars": 2000.0,
                "avg_prompt_compression_ratio": 0.5,
                "max_prompt_compression_ratio": 0.6,
                "current_request_preserved_rate": 1.0,
            },
        },
        "memory-ablation-v2.json": {
            "variants": {
                "memory_on": {
                    "repeated_reads": 0,
                    "avg_tool_steps": 1.0,
                    "correct_rate": 1.0,
                    "memory_hit_rate": 1.0,
                },
                "memory_off": {"repeated_reads": 12},
            }
        },
        "memory-fidelity-v1.json": {
            "summary": {
                "pass_rate": 1.0,
                "irrelevant_injection_rate": 0.0,
                "supersede_success_rate": 1.0,
                "secret_exposure_rate": 0.0,
                "stale_detection_rate": 1.0,
                "stale_use_rate": 0.0,
                "poison_quarantine_rate": 1.0,
                "benign_recall_retention_rate": 1.0,
            }
        },
        "recovery-ablation-v2.json": {
            "variants": {
                "resume_enabled": {
                    "summary": {
                        "resume_success_rate": 1.0,
                        "stale_reanchor_rate": 1.0,
                        "workspace_drift_detection_rate": 1.0,
                        "resume_false_accept_rate": 0.0,
                        "resumption_success_rate": 1.0,
                        "first_action_correctness": 1.0,
                        "todo_continuity_rate": 1.0,
                    }
                }
            }
        },
    }
    if include_context_ab:
        artifacts["context-ab-v1.json"] = {
            "summary": {
                "estimated_proxy_only": {
                    "paired_task_count": 4,
                    "median_cost_delta_pct": -0.2,
                    "claimable_cost_win": True,
                    "quality_regression_count": 0,
                }
            }
        }
    paths = {}
    for name, payload in artifacts.items():
        path = artifact_dir / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        paths[name] = path
    return paths


def test_write_benchmark_core_report_marks_resume_safe_metrics(tmp_path):
    artifacts = write_core_report_artifacts(tmp_path / "artifacts")

    report_path = tmp_path / "docs" / "metrics" / "lite-benchmark-core-report.md"
    report_text = write_benchmark_core_report(
        report_path=report_path,
        harness_artifact_path=artifacts["harness-regression-v2.json"],
        context_artifact_path=artifacts["context-ablation-v2.json"],
        memory_artifact_path=artifacts["memory-ablation-v2.json"],
        recovery_artifact_path=artifacts["recovery-ablation-v2.json"],
        fidelity_artifact_path=artifacts["memory-fidelity-v1.json"],
    )

    assert report_path.exists()
    assert "可以安全写进简历的指标" in report_text
    assert "只适合放文档/面试展开的指标" in report_text
    assert "resume_success_rate" in report_text
    assert "resumption_success_rate" in report_text
    assert "first_action_correctness" in report_text
    assert "todo_continuity_rate" in report_text
    assert "memory_hit_rate" in report_text
    assert "Context Efficiency Under Follow-up" in report_text
    assert "Memory Fidelity" in report_text


def test_write_benchmark_core_report_includes_optional_context_ab(tmp_path):
    artifacts = write_core_report_artifacts(
        tmp_path / "artifacts", include_context_ab=True
    )

    report_text = write_benchmark_core_report(
        report_path=tmp_path / "docs" / "metrics" / "lite-benchmark-core-report.md",
        harness_artifact_path=artifacts["harness-regression-v2.json"],
        context_artifact_path=artifacts["context-ablation-v2.json"],
        memory_artifact_path=artifacts["memory-ablation-v2.json"],
        recovery_artifact_path=artifacts["recovery-ablation-v2.json"],
        fidelity_artifact_path=artifacts["memory-fidelity-v1.json"],
        context_ab_artifact_path=artifacts["context-ab-v1.json"],
    )

    assert "Context A/B (Scripted)" in report_text
    assert "claimable_cost_win：True" in report_text


def test_write_benchmark_core_report_falls_back_to_local_artifacts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    local_artifacts = tmp_path / "_local" / "benchmark" / "artifacts"
    write_core_report_artifacts(local_artifacts)

    report_text = write_benchmark_core_report()

    assert "Harness Regression" in report_text
    assert "Context Efficiency Under Follow-up" in report_text
    assert "Memory Fidelity" in report_text
