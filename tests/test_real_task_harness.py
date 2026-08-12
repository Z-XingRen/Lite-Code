import json
import subprocess
import sys
from pathlib import Path

from lite.evaluation.real_task_harness import (
    METRIC_FIELDS,
    load_manifest,
    write_results,
)
from scripts.run_real_task_harness import feature_flags_for_task


ROOT = Path(__file__).resolve().parents[1]


def test_real_task_harness_script_starts_from_repository_root():
    completed = subprocess.run(
        [sys.executable, "scripts/run_real_task_harness.py", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_real_task_manifest_is_fixed_and_covers_required_workflows():
    manifest = load_manifest(ROOT)

    assert manifest["seed"] == 20260811
    assert manifest["repetitions"] == 3
    assert len(manifest["tasks"]) == 16
    assert {task["scenario"] for task in manifest["tasks"]} >= {
        "single_file_bug",
        "cross_file_refactor",
        "failure_diagnosis",
        "new_interface_and_tests",
        "lint_typecheck_build",
        "tool_failure_recovery",
        "long_context",
        "max_steps_final",
    }
    assert all(task["fixture_repo"] and task["grader"] for task in manifest["tasks"])


def test_real_task_features_enable_multi_agent_only_for_explicit_worker_tasks():
    baseline_task = {"allowed_tools": ["read_file", "patch_file"]}
    worker_task = {"allowed_tools": ["read_file", "agent", "task_stop"]}

    assert feature_flags_for_task(baseline_task, "optimized")["multi_agent"] is False
    assert feature_flags_for_task(worker_task, "optimized")["multi_agent"] is True


def test_real_task_results_write_jsonl_and_task_level_markdown(tmp_path):
    row = {
        "task_id": "F01_pricing",
        "variant": "baseline",
        "repeat": 0,
        **{
            field: value
            for field, value in {
                "task_success": True,
                "verification_success": True,
                "scope_violation": False,
                "changed_paths": ["src/pricing.py"],
                "model_call_count": 2,
                "input_tokens": 100,
                "output_tokens": 20,
                "tool_call_count": 1,
                "duplicate_tool_result_count": 0,
                "checkpoint_count": 0,
                "persistence_write_count": 2,
                "wall_time": 0.25,
                "final_stop_reason": "final_answer_returned",
            }.items()
        },
    }

    paths = write_results([row], tmp_path)
    assert paths["jsonl"].is_file()
    assert paths["markdown"].is_file()
    written = [
        json.loads(line)
        for line in paths["jsonl"].read_text(encoding="utf-8").splitlines()
    ]
    assert written == [row]
    markdown = paths["markdown"].read_text(encoding="utf-8")
    assert "F01_pricing" in markdown
    assert all(field in markdown for field in METRIC_FIELDS)
    assert "task_success" in markdown
