"""Deterministic long-session gates kept out of the fast unit-test lane."""

import pytest

from lite.evaluation.runtime_evidence import (
    run_effect_recovery_matrix,
    run_journal_scaling_benchmark,
)


@pytest.mark.stress
def test_ten_thousand_record_journal_has_bounded_growth():
    result = run_journal_scaling_benchmark()

    assert result["gates"]["passed"] is True
    assert result["headline"]["append_growth_ratio"] <= 2.5
    assert result["scenarios"][-1]["state_correct"] is True


@pytest.mark.stress
def test_recovery_matrix_remains_idempotent_across_repetitions():
    result = run_effect_recovery_matrix(repetitions=3, sync=True)

    assert result["gates"]["passed"] is True
    assert result["summary"]["trial_count"] == 54
