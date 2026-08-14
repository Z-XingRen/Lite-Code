import json

import pytest

from lite.evaluation.tool_scheduler_benchmark import (
    BenchmarkConfig,
    parse_batch_sizes,
    percentile,
    run_benchmark,
    write_reports,
)


def test_percentile_interpolates_small_samples():
    assert percentile([10], 0.95) == 10
    assert percentile([0, 10], 0.50) == 5
    assert percentile([0, 10], 0.95) == pytest.approx(9.5)


def test_parse_batch_sizes_rejects_non_batches():
    assert parse_batch_sizes("2,4,8") == (2, 4, 8)
    with pytest.raises(Exception):
        parse_batch_sizes("1")


def test_small_engine_benchmark_writes_raw_samples_and_reports(tmp_path):
    output_dir = tmp_path / "artifacts"
    config = BenchmarkConfig(
        batch_sizes=(2, 4),
        delay_ms=5,
        warmups=1,
        repeats=2,
        safety_trials=2,
        seed=20260813,
        output_dir=output_dir,
    )

    report = run_benchmark(config)
    json_path, markdown_path = write_reports(report, output_dir)

    assert report["validation"]["passed"] is True
    assert len(report["performance"]["samples"]) == 4
    assert len(report["performance"]["warmup_samples"]) == 2
    assert report["performance"]["invariants"]["violation_count"] == 0
    for sample in report["performance"]["samples"]:
        assert sample["execution_order"] in (
            ["sequential", "parallel"],
            ["parallel", "sequential"],
        )
        assert sample["sequential"]["result_order"] == sample["request_order"]
        assert sample["parallel"]["result_order"] == sample["request_order"]
        assert isinstance(sample["paired_speedup"], float)
        assert isinstance(sample["run_turn_paired_speedup"], float)

    assert report["safety"]["total_trials"] == 8
    assert all(value == 0 for value in report["safety"]["counters"].values())
    assert json_path.is_file()
    assert markdown_path.is_file()
    saved = json.loads(json_path.read_text(encoding="utf-8"))
    assert saved["config"]["seed"] == 20260813
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "性能对比" in markdown
    assert "安全性测试" in markdown
    assert "简历候选表述" in markdown
