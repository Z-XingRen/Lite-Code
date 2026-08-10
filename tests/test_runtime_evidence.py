import json

import pytest

from lite.evaluation.runtime_evidence import (
    CRASH_PHASES,
    EFFECT_TYPES,
    render_runtime_evidence_markdown,
    run_effect_recovery_matrix,
    run_journal_scaling_benchmark,
    run_workspace_tracker_benchmark,
    write_runtime_evidence,
)
from scripts.run_runtime_evidence import main


def test_workspace_tracker_benchmark_reports_timing_and_exact_path_parity():
    result = run_workspace_tracker_benchmark(
        file_counts=(20, 40),
        changed_counts=(1, 5),
        measured_runs=2,
        warmup_runs=1,
        file_bytes=16,
    )

    assert result["schema_version"] == "lite.workspace_tracker_benchmark.v1"
    assert result["summary"] == {
        "scenario_count": 4,
        "path_observation_count": 16,
        "path_exact_match_count": 16,
        "path_exact_rate": 1.0,
    }
    assert result["headline"]["file_count"] == 40
    assert result["headline"]["changed_file_count"] == 1
    assert result["headline"]["path_parity_rate"] == 1.0
    assert result["gates"]["passed"] is True
    for row in result["scenarios"]:
        assert row["legacy"]["timing"]["samples"] == 2
        assert row["incremental"]["timing"]["samples"] == 2
        assert row["comparison"]["path_parity_rate"] == 1.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"file_counts": (), "changed_counts": (1,)},
        {"file_counts": (10,), "changed_counts": (11,)},
        {"file_counts": (10,), "changed_counts": (1,), "measured_runs": 0},
        {"file_counts": (10,), "changed_counts": (1, 1)},
    ],
)
def test_workspace_tracker_benchmark_rejects_invalid_matrix(kwargs):
    with pytest.raises(ValueError):
        run_workspace_tracker_benchmark(**kwargs)


def test_effect_recovery_matrix_covers_effect_phase_repetition_product():
    result = run_effect_recovery_matrix(repetitions=2, sync=False)

    expected = len(EFFECT_TYPES) * len(CRASH_PHASES) * 2
    assert result["schema_version"] == "lite.effect_recovery_matrix.v1"
    assert result["config"]["expected_trial_count"] == expected
    assert result["summary"]["trial_count"] == expected
    assert result["summary"]["safe_resolution_rate"] == 1.0
    assert result["summary"]["state_replay_match_rate"] == 1.0
    assert result["summary"]["duplicate_side_effect_rate"] == 0.0
    assert result["gates"]["passed"] is True
    assert set(result["by_effect"]) == set(EFFECT_TYPES)
    assert set(result["by_phase"]) == set(CRASH_PHASES)
    assert all(row["trial_count"] == 6 for row in result["by_effect"].values())
    assert all(row["trial_count"] == 12 for row in result["by_phase"].values())


def test_runtime_evidence_renderer_and_writer_preserve_scope(tmp_path):
    workspace = run_workspace_tracker_benchmark(
        file_counts=(10,),
        changed_counts=(1,),
        measured_runs=1,
        warmup_runs=0,
        file_bytes=8,
    )
    recovery = run_effect_recovery_matrix(repetitions=1, sync=False)

    markdown = render_runtime_evidence_markdown(workspace, recovery)
    assert "target paths are known" in markdown
    assert "Journal recovery only" in markdown
    assert "18 trials" in markdown

    paths = write_runtime_evidence(tmp_path, workspace, recovery)
    assert json.loads(paths["workspace_json"].read_text(encoding="utf-8"))[
        "gates"
    ]["passed"] is True
    assert json.loads(paths["recovery_json"].read_text(encoding="utf-8"))[
        "gates"
    ]["passed"] is True
    assert paths["markdown"].read_text(encoding="utf-8") == markdown


def test_journal_scaling_benchmark_reports_bounded_growth_and_correct_replay():
    result = run_journal_scaling_benchmark(
        record_counts=(20, 100, 200), sample_window=10
    )

    assert result["schema_version"] == "lite.journal_scaling_benchmark.v1"
    assert [row["record_count"] for row in result["scenarios"]] == [20, 100, 200]
    assert all(row["state_correct"] for row in result["scenarios"])
    assert result["gates"]["passed"] is True


def test_runtime_evidence_cli_writes_all_artifacts(tmp_path):
    output = tmp_path / "evidence"

    exit_code = main(
        [
            "--output-dir",
            str(output),
            "--workspace-file-counts",
            "10",
            "--workspace-changed-counts",
            "1",
            "--workspace-runs",
            "1",
            "--workspace-warmups",
            "0",
            "--workspace-file-bytes",
            "8",
            "--recovery-repetitions",
            "1",
            "--journal-record-counts",
            "10,20,30",
            "--journal-sample-window",
            "5",
            "--no-journal-sync",
        ]
    )

    assert exit_code == 0
    assert (output / "workspace-tracker.json").exists()
    assert (output / "effect-recovery.json").exists()
    assert (output / "journal-scaling.json").exists()
    assert (output / "summary.md").exists()
