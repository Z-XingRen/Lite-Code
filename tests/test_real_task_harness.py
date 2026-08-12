import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from lite.evaluation.real_task_harness import (
    METRIC_FIELDS,
    load_manifest,
    result_matrix_keys,
    validate_result_matrix,
    write_results,
)
from scripts import run_real_task_harness
from scripts.run_real_task_harness import VARIANT_FLAGS, feature_flags_for_task


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


def test_real_task_variants_isolate_context_and_persistence_effects():
    assert VARIANT_FLAGS == {
        "baseline": {
            "context_reduction": True,
            "frozen_base_context": False,
            "journal_checkpoint_policy": False,
        },
        "frozen_context": {
            "context_reduction": True,
            "frozen_base_context": True,
            "journal_checkpoint_policy": False,
        },
        "journal_policy": {
            "context_reduction": True,
            "frozen_base_context": False,
            "journal_checkpoint_policy": True,
        },
        "optimized": {
            "context_reduction": True,
            "frozen_base_context": True,
            "journal_checkpoint_policy": True,
        },
    }


def test_real_task_identity_locks_runtime_commit_source_and_config(monkeypatch):
    monkeypatch.setattr(run_real_task_harness, "_git_commit", lambda: "commit-a")
    monkeypatch.setattr(
        run_real_task_harness,
        "_runtime_source_sha256",
        lambda: "sha256:runtime-a",
    )
    monkeypatch.setattr(
        run_real_task_harness,
        "_file_sha256",
        lambda _path: "sha256:config-a",
    )
    manifest = {
        "benchmark_id": "real-task-test",
        "seed": 7,
        "tasks": [{"id": "F01_pricing"}],
    }
    config = type(
        "Config",
        (),
        {
            "name": "openai",
            "protocol": "openai",
            "model": "gpt-test",
            "reasoning_effort": "high",
            "base_url": "https://example.test/v1",
            "api_key": "secret",
        },
    )()

    identity = run_real_task_harness.build_identity(
        manifest, config, ("baseline",), 3
    )

    assert identity["schema_version"] == "lite.real_task_identity.v2"
    assert identity["git_commit"] == "commit-a"
    assert identity["runtime_source_sha256"] == "sha256:runtime-a"
    assert identity["config_sha256"] == "sha256:config-a"

    monkeypatch.setattr(
        run_real_task_harness,
        "_runtime_source_sha256",
        lambda: "sha256:runtime-b",
    )
    with pytest.raises(RuntimeError, match="changed during evaluation"):
        run_real_task_harness.assert_identity_unchanged(
            identity, manifest, config, ("baseline",), 3
        )


def test_runtime_identity_ignores_generated_cache_files(tmp_path, monkeypatch):
    monkeypatch.setattr(run_real_task_harness, "ROOT", tmp_path)
    monkeypatch.setattr(
        run_real_task_harness,
        "RUNTIME_IDENTITY_PATHS",
        (Path("lite"),),
    )
    source = tmp_path / "lite" / "runtime.py"
    source.parent.mkdir(parents=True)
    source.write_text("VALUE = 1\n", encoding="utf-8")
    before = run_real_task_harness._runtime_source_sha256()

    cache = source.parent / "__pycache__" / "runtime.pyc"
    cache.parent.mkdir()
    cache.write_bytes(b"generated")
    assert run_real_task_harness._runtime_source_sha256() == before

    source.write_text("VALUE = 2\n", encoding="utf-8")
    assert run_real_task_harness._runtime_source_sha256() != before


def test_real_task_results_write_jsonl_and_task_level_markdown(tmp_path):
    row = _result_row("baseline")

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


def test_real_task_result_matrix_rejects_duplicate_and_unexpected_rows():
    expected = {
        ("F01_pricing", 0, "baseline"),
        ("F01_pricing", 0, "optimized"),
    }
    baseline = _result_row("baseline")

    with pytest.raises(ValueError, match="duplicate result rows"):
        validate_result_matrix([baseline, baseline], expected)

    with pytest.raises(ValueError, match="unexpected result rows"):
        validate_result_matrix([_result_row("other")], expected)


def test_real_task_summary_cannot_present_an_incomplete_matrix_as_baseline(tmp_path):
    expected = result_matrix_keys(
        [{"id": "F01_pricing"}], ("baseline", "optimized"), 1
    )
    paths = write_results(
        [_result_row("baseline")], tmp_path, expected_keys=expected
    )

    assert paths["complete"] is False
    incomplete = paths["markdown"].read_text(encoding="utf-8")
    assert "Incomplete" in incomplete
    assert "1/2 result rows" in incomplete
    assert "| task_id |" not in incomplete

    with pytest.raises(ValueError, match="missing result rows"):
        write_results(
            [_result_row("baseline")],
            tmp_path,
            expected_keys=expected,
            require_complete=True,
        )

    paths = write_results(
        [_result_row("baseline"), _result_row("optimized")],
        tmp_path,
        expected_keys=expected,
        require_complete=True,
    )
    assert paths["complete"] is True
    jsonl_bytes = paths["jsonl"].read_bytes()
    digest = hashlib.sha256(jsonl_bytes).hexdigest()
    complete = paths["markdown"].read_text(encoding="utf-8")
    assert "2/2 result rows" in complete
    assert f"`sha256:{digest}`" in complete
    assert "| task_id |" in complete


def _result_row(variant):
    return {
        "task_id": "F01_pricing",
        "variant": variant,
        "repeat": 0,
        "task_success": True,
        "verification_success": True,
        "scope_violation": False,
        "changed_paths": ["src/pricing.py"],
        "model_call_count": 2,
        "input_tokens": 100,
        "cached_tokens": 40,
        "billable_input_tokens": 60,
        "output_tokens": 20,
        "tool_call_count": 1,
        "duplicate_tool_result_count": 0,
        "checkpoint_count": 0,
        "persistence_write_count": 2,
        "wall_time": 0.25,
        "final_stop_reason": "final_answer_returned",
        "failure_category": "none",
    }
