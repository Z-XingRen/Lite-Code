import json
from types import SimpleNamespace

import pytest

from lite.evaluation.prompt_cache_turn_harness import (
    RESULT_FIELDS,
    VARIANTS,
    load_manifest,
    result_matrix_keys,
    row_from_turns,
    summarize_results,
    validate_result_matrix,
    write_results,
)
from lite.providers import ModelResult
from lite.testing import ScriptedModelClient
from scripts import run_prompt_cache_turn_harness


ROOT = run_prompt_cache_turn_harness.ROOT


class CacheModelClient(ScriptedModelClient):
    def __init__(self, outputs):
        super().__init__(outputs)
        self.supports_prompt_cache = True
        self.supports_append_prompt_cache = True
        self.model = "gpt-test"
        self.base_url = "https://example.test/v1"
        self.provider = "openai"
        self.context_window = 200_000


def test_prompt_cache_turn_manifest_is_fixed_and_multi_scenario():
    manifest = load_manifest(ROOT)

    assert manifest["repetitions"] == 3
    assert [scenario["id"] for scenario in manifest["scenarios"]] == [
        "append",
        "workspace_refresh",
        "session_resume",
    ]
    assert all(len(scenario["turns"]) == 2 for scenario in manifest["scenarios"])
    assert VARIANTS == ("full_prompt", "append_projection")


def test_prompt_cache_row_requires_expected_projection_and_clean_requests():
    scenario = {
        "id": "workspace_refresh",
        "between_turns": "workspace_refresh",
        "expected_projection_reason": "context_refresh",
    }
    turns = [
        _turn(answer_match=True, prompt_cache_key="stable", cached_tokens=0),
        _turn(
            answer_match=True,
            prompt_cache_key="stable",
            cached_tokens=80,
            cache_projection_reused=True,
            cache_projection_reason="context_refresh",
            cache_projection_context_refreshed=True,
        ),
    ]

    row = row_from_turns(
        scenario, turns, variant="append_projection", repeat=0
    )

    assert row["behavior_pass"] is True
    assert row["initial_projection_match"] is True
    assert row["provider_prompt_cache_controls_enabled"] is True
    assert row["usage_complete"] is True
    assert row["provider_cache_hit"] is True
    assert row["prompt_cache_key_stable"] is True
    assert row["billable_input_tokens"] == 120

    turns[-1]["duplicate_tool_result_count"] = 1
    failed = row_from_turns(
        scenario, turns, variant="append_projection", repeat=0
    )
    assert failed["behavior_pass"] is False

    turns[0]["cache_projection_reused"] = True
    turns[0]["cache_projection_reason"] = "append"
    failed_initial = row_from_turns(
        scenario, turns, variant="append_projection", repeat=0
    )
    assert failed_initial["initial_projection_match"] is False
    assert failed_initial["behavior_pass"] is False


def test_prompt_cache_result_matrix_and_summary_are_completeness_bound(tmp_path):
    scenarios = [{"id": "append"}]
    expected = result_matrix_keys(scenarios, VARIANTS, 1)
    baseline = _row("full_prompt")

    paths = write_results([baseline], tmp_path, expected_keys=expected)

    assert paths["complete"] is False
    assert paths["summary"].is_file()
    partial_summary = json.loads(
        paths["summary"].read_text(encoding="utf-8")
    )
    assert partial_summary["row_count"] == 1
    assert partial_summary["matrix"]["complete"] is False
    incomplete = paths["markdown"].read_text(encoding="utf-8")
    assert "Incomplete" in incomplete
    assert "1/2 result rows" in incomplete

    with pytest.raises(ValueError, match="missing prompt-cache result rows"):
        validate_result_matrix([baseline], expected, require_complete=True)

    paths = write_results(
        [baseline, _row("append_projection")],
        tmp_path,
        expected_keys=expected,
        require_complete=True,
    )
    assert paths["complete"] is True
    complete_summary = json.loads(
        paths["summary"].read_text(encoding="utf-8")
    )
    assert complete_summary["matrix"]["complete"] is True
    assert complete_summary["overall"]["row_count"] == 2
    assert "append_projection" in complete_summary["by_variant"]
    markdown = paths["markdown"].read_text(encoding="utf-8")
    assert all(field in markdown for field in RESULT_FIELDS)


def test_prompt_cache_summary_aggregates_rates_and_token_means():
    rows = [
        {
            "variant": "append_projection",
            "scenario": "append",
            "behavior_pass": True,
            "usage_complete": True,
            "provider_cache_hit": True,
            "prompt_cache_key_stable": True,
            "cache_projection_reused": True,
            "billable_input_tokens": 100,
            "second_turn_cached_tokens": 80,
        },
        {
            "variant": "append_projection",
            "scenario": "workspace_refresh",
            "behavior_pass": False,
            "usage_complete": True,
            "provider_cache_hit": False,
            "prompt_cache_key_stable": True,
            "cache_projection_reused": True,
            "billable_input_tokens": 200,
            "second_turn_cached_tokens": 0,
        },
    ]

    summary = summarize_results(rows)

    assert summary["overall"] == {
        "row_count": 2,
        "behavior_pass_rate": 0.5,
        "usage_complete_rate": 1.0,
        "provider_cache_hit_rate": 0.5,
        "prompt_cache_key_stable_rate": 1.0,
        "projection_reuse_rate": 1.0,
        "mean_billable_input_tokens": 150.0,
        "mean_second_turn_cached_tokens": 40.0,
    }
    assert summary["by_scenario"]["append"]["behavior_pass_rate"] == 1.0


@pytest.mark.parametrize(
    ("scenario_id", "expected_reason", "context_refreshed"),
    [
        ("append", "append", False),
        ("workspace_refresh", "context_refresh", True),
        ("session_resume", "append", False),
    ],
)
def test_prompt_cache_runner_executes_cross_turn_projection_scenarios(
    tmp_path, monkeypatch, scenario_id, expected_reason, context_refreshed
):
    scenarios = {
        scenario["id"]: scenario for scenario in load_manifest(ROOT)["scenarios"]
    }
    scenario = scenarios[scenario_id]
    first, second = scenario["turns"]
    outputs = [
        ModelResult(text=first["expected_answer"], metadata=_usage(100, 0)),
        ModelResult(text=second["expected_answer"], metadata=_usage(120, 80)),
    ]
    clients = (
        [CacheModelClient([outputs[0]]), CacheModelClient([outputs[1]])]
        if scenario_id == "session_resume"
        else [CacheModelClient(outputs)]
    )
    monkeypatch.setattr(
        run_prompt_cache_turn_harness,
        "make_client",
        lambda _config: clients.pop(0),
    )
    config = SimpleNamespace(name="openai")

    row = run_prompt_cache_turn_harness.run_scenario(
        scenario,
        0,
        "append_projection",
        tmp_path,
        config,
        timeout=30,
    )

    assert row["behavior_pass"] is True
    assert row["initial_projection_match"] is True
    assert row["cache_projection_reused"] is True
    assert row["cache_projection_reason"] == expected_reason
    assert row["cache_projection_context_refreshed"] is context_refreshed
    assert row["second_turn_cached_tokens"] == 80
    assert row["billable_input_tokens"] == 140
    assert row["model_call_count"] == 2
    assert row["tool_call_count"] == 0
    assert len(row["turns"]) == 2


def test_prompt_cache_runner_full_prompt_control_disables_projection(
    tmp_path, monkeypatch
):
    scenario = load_manifest(ROOT)["scenarios"][0]
    outputs = [
        ModelResult(
            text=turn["expected_answer"],
            metadata=_usage(100, 0),
        )
        for turn in scenario["turns"]
    ]
    monkeypatch.setattr(
        run_prompt_cache_turn_harness,
        "make_client",
        lambda _config: CacheModelClient(outputs),
    )

    row = run_prompt_cache_turn_harness.run_scenario(
        scenario,
        0,
        "full_prompt",
        tmp_path,
        SimpleNamespace(name="openai"),
        timeout=30,
    )

    assert row["behavior_pass"] is True
    assert row["initial_projection_match"] is True
    assert row["provider_prompt_cache_controls_enabled"] is False
    assert row["cache_projection_reused"] is False
    assert row["cache_projection_reason"] == "unsupported"


def test_prompt_cache_runner_script_starts_from_repository_root():
    with pytest.raises(SystemExit) as exc:
        run_prompt_cache_turn_harness.main(["--help"])

    assert exc.value.code == 0


def test_prompt_cache_identity_rejects_unbound_or_changed_partial_results(tmp_path):
    identity = {"schema_version": "test", "git_commit": "commit-a"}
    results = tmp_path / "results.jsonl"
    results.write_text("{}\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="have no evaluation identity"):
        run_prompt_cache_turn_harness.ensure_evaluation_identity(tmp_path, identity)

    results.unlink()
    path = run_prompt_cache_turn_harness.ensure_evaluation_identity(tmp_path, identity)
    assert path.is_file()
    assert run_prompt_cache_turn_harness.ensure_evaluation_identity(
        tmp_path, identity
    ) == path

    with pytest.raises(RuntimeError, match="identity changed"):
        run_prompt_cache_turn_harness.ensure_evaluation_identity(
            tmp_path,
            {**identity, "git_commit": "commit-b"},
        )


def test_prompt_cache_identity_distinguishes_smoke_and_formal_modes():
    manifest = {
        "benchmark_id": "prompt-cache-test",
        "scenarios": [{"id": "append"}],
    }
    config = SimpleNamespace(
        name="openai",
        protocol="openai",
        model="gpt-test",
        reasoning_effort="medium",
        base_url="https://example.test/v1",
        api_key="secret",
    )

    smoke = run_prompt_cache_turn_harness.build_identity(
        manifest, config, VARIANTS, 1, mode="smoke"
    )
    formal = run_prompt_cache_turn_harness.build_identity(
        manifest, config, VARIANTS, 3, mode="formal"
    )

    assert smoke["mode"] == "smoke"
    assert formal["mode"] == "formal"
    assert smoke != formal


def test_prompt_cache_smoke_rejects_non_fixed_overrides(monkeypatch):
    monkeypatch.setattr(
        run_prompt_cache_turn_harness,
        "provider_metadata",
        lambda: pytest.fail("provider config should not be loaded"),
    )

    with pytest.raises(ValueError, match="fixes repetitions at 1"):
        run_prompt_cache_turn_harness.main(["--smoke", "--repetitions", "2"])
    with pytest.raises(ValueError, match="fixes the scenario at append"):
        run_prompt_cache_turn_harness.main(
            ["--smoke", "--scenario-ids", "session_resume"]
        )
    with pytest.raises(ValueError, match="requires both fixed variants"):
        run_prompt_cache_turn_harness.main(
            ["--smoke", "--variants", "append_projection"]
        )


def _usage(input_tokens, cached_tokens):
    return {
        "provider_protocol": "openai",
        "provider_model": "gpt-test",
        "input_tokens": input_tokens,
        "cached_tokens": cached_tokens,
        "output_tokens": 5,
    }


def _turn(**overrides):
    value = {
        "answer_match": True,
        "status": "completed",
        "stop_reason": "final_answer_returned",
        "usage_complete": True,
        "model_call_count": 1,
        "input_tokens": 100,
        "cached_tokens": 0,
        "prompt_cache_key": "key",
        "provider_prompt_cache_controls_enabled": True,
        "provider_prompt_cache_key": "key",
        "cache_projection_reused": False,
        "cache_projection_reason": "missing",
        "cache_projection_generation": 1,
        "cache_projection_message_count": 0,
        "cache_projection_chars": 0,
        "cache_projection_context_refreshed": False,
        "provider_prompt_chars": 500,
        "tool_call_count": 0,
        "duplicate_tool_result_count": 0,
    }
    value.update(overrides)
    return value


def _row(variant):
    return {
        "task_id": "append",
        "scenario": "append",
        "variant": variant,
        "repeat": 0,
        **{field: 0 for field in RESULT_FIELDS},
    }
