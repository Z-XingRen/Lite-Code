"""Evidence adapter for fixed cross-turn prompt-cache evaluations."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from .run_evidence import RunEvidence


MANIFEST_PATH = Path("benchmarks/prompt_cache_turns_v1.json")
VARIANTS = ("full_prompt", "append_projection")
MINIMUM_CLAIMABLE_PAIR_COUNT = 9
MINIMUM_CLAIMABLE_PAIRS_PER_SCENARIO = 3
REQUIRED_CLAIMABLE_SCENARIOS = frozenset(
    {"append", "workspace_refresh", "session_resume"}
)
RESULT_FIELDS = (
    "execution_position",
    "pair_execution_order",
    "behavior_pass",
    "usage_complete",
    "provider_cache_hit",
    "prompt_cache_key_stable",
    "provider_prompt_cache_controls_enabled",
    "initial_projection_match",
    "model_call_count",
    "input_tokens",
    "cached_tokens",
    "billable_input_tokens",
    "second_turn_cached_tokens",
    "cache_projection_reused",
    "cache_projection_reason",
    "cache_projection_generation",
    "cache_projection_message_count",
    "cache_projection_chars",
    "cache_projection_context_refreshed",
    "provider_prompt_chars",
    "tool_call_count",
    "duplicate_tool_result_count",
)


def load_manifest(root):
    """Load and validate the fixed two-turn scenario manifest."""

    root = Path(root).resolve()
    manifest = json.loads((root / MANIFEST_PATH).read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "lite.prompt_cache_turns.v1":
        raise ValueError("unknown prompt-cache turn manifest schema")
    scenarios = list(manifest.get("scenarios", []) or [])
    if len(scenarios) < 3:
        raise ValueError("prompt-cache turn manifest requires at least 3 scenarios")
    if int(manifest.get("repetitions", 0) or 0) < 3:
        raise ValueError("prompt-cache turn repetitions must be at least 3")
    allowed_actions = {"none", "workspace_refresh", "session_resume"}
    for scenario in scenarios:
        if len(list(scenario.get("turns", []) or [])) != 2:
            raise ValueError("each prompt-cache scenario must contain exactly 2 turns")
        if scenario.get("between_turns") not in allowed_actions:
            raise ValueError("unknown prompt-cache between-turn action")
    return manifest


def turn_evidence(workspace, run_id):
    """Project one exact run into cache and usage evidence."""

    evidence = RunEvidence.for_run(workspace, run_id)
    report = dict(evidence.report or {})
    summary = dict(
        (report.get("evidence_summaries", {}) or {}).get(
            "context_budget_summary", {}
        )
        or {}
    )
    metadata = dict(report.get("prompt_metadata", {}) or {})
    completion_events = [
        event
        for event in evidence.trace_events
        if event.get("event") == "model_parsed"
    ]
    completion_metadata = [
        dict(event.get("completion_metadata", {}) or {})
        for event in completion_events
    ]
    usage_complete = bool(completion_metadata) and all(
        _is_provider_usage(item) for item in completion_metadata
    )
    input_tokens = sum(int(item.get("input_tokens", 0) or 0) for item in completion_metadata)
    cached_tokens = sum(
        int(item.get("cached_tokens", 0) or 0) for item in completion_metadata
    )
    return {
        "run_id": str(run_id),
        "status": str(report.get("status", "")),
        "stop_reason": str(report.get("stop_reason", "")),
        "model_call_count": len(completion_events),
        "usage_complete": usage_complete,
        "input_tokens": input_tokens,
        "cached_tokens": cached_tokens,
        "cache_write_tokens": sum(
            int(item.get("cache_write_tokens", 0) or 0)
            for item in completion_metadata
        ),
        "output_tokens": sum(
            int(item.get("output_tokens", 0) or 0) for item in completion_metadata
        ),
        "prompt_cache_key": str(metadata.get("prompt_cache_key", "")),
        "provider_prompt_cache_controls_enabled": bool(
            metadata.get("provider_prompt_cache_controls_enabled", False)
        ),
        "provider_prompt_cache_key": str(
            metadata.get("provider_prompt_cache_key", "") or ""
        ),
        "cache_projection_reused": bool(
            summary.get(
                "cache_projection_reused",
                metadata.get("cache_projection_reused", False),
            )
        ),
        "cache_projection_reason": str(
            summary.get(
                "cache_projection_reason",
                metadata.get("cache_projection_reason", "none"),
            )
            or "none"
        ),
        "cache_projection_generation": int(
            summary.get(
                "cache_projection_generation",
                metadata.get("cache_projection_generation", 0),
            )
            or 0
        ),
        "cache_projection_message_count": int(
            summary.get(
                "cache_projection_message_count",
                metadata.get("cache_projection_message_count", 0),
            )
            or 0
        ),
        "cache_projection_chars": int(
            summary.get(
                "cache_projection_chars",
                metadata.get("cache_projection_chars", 0),
            )
            or 0
        ),
        "cache_projection_context_refreshed": bool(
            summary.get(
                "cache_projection_context_refreshed",
                metadata.get("cache_projection_context_refreshed", False),
            )
        ),
        "provider_prompt_chars": int(
            summary.get(
                "provider_prompt_chars",
                metadata.get("provider_prompt_chars", 0),
            )
            or 0
        ),
        "tool_call_count": sum(
            event.get("event") == "tool_executed" for event in evidence.trace_events
        ),
        "duplicate_tool_result_count": int(
            metadata.get("duplicate_tool_result_count", 0) or 0
        ),
    }


def row_from_turns(scenario, turns, *, variant, repeat, errors=None):
    """Aggregate two completed turn records into one stable result row."""

    turns = [dict(turn) for turn in turns]
    second = turns[-1] if turns else {}
    input_tokens = sum(int(turn.get("input_tokens", 0) or 0) for turn in turns)
    cached_tokens = sum(int(turn.get("cached_tokens", 0) or 0) for turn in turns)
    expected_reused = variant == "append_projection"
    expected_reason = (
        str(scenario.get("expected_projection_reason", "append"))
        if expected_reused
        else "unsupported"
    )
    expected_initial_reason = "missing" if expected_reused else "unsupported"
    answer_match = bool(turns) and all(turn.get("answer_match") for turn in turns)
    expected_context_refresh = bool(
        expected_reused and scenario.get("between_turns") == "workspace_refresh"
    )
    completed_turns = bool(turns) and all(
        turn.get("status") == "completed"
        and turn.get("stop_reason") == "final_answer_returned"
        for turn in turns
    )
    first = turns[0] if turns else {}
    initial_projection_match = (
        not bool(first.get("cache_projection_reused"))
        and str(first.get("cache_projection_reason", ""))
        == expected_initial_reason
    )
    projection_match = (
        bool(second.get("cache_projection_reused")) == expected_reused
        and str(second.get("cache_projection_reason", "")) == expected_reason
        and bool(second.get("cache_projection_context_refreshed"))
        == expected_context_refresh
    )
    duplicate_count = sum(
        int(turn.get("duplicate_tool_result_count", 0) or 0) for turn in turns
    )
    prompt_cache_keys = [
        str(turn.get("provider_prompt_cache_key", "")) for turn in turns
    ]
    prompt_cache_key_stable = bool(prompt_cache_keys) and all(
        prompt_cache_keys
    ) and len(set(prompt_cache_keys)) == 1
    provider_prompt_cache_controls_enabled = bool(
        second.get("provider_prompt_cache_controls_enabled", False)
    )
    model_call_count = sum(
        int(turn.get("model_call_count", 0) or 0) for turn in turns
    )
    tool_call_count = sum(
        int(turn.get("tool_call_count", 0) or 0) for turn in turns
    )
    return {
        "task_id": str(scenario["id"]),
        "scenario": str(scenario["id"]),
        "variant": str(variant),
        "repeat": int(repeat),
        "behavior_pass": bool(
            answer_match
            and completed_turns
            and initial_projection_match
            and projection_match
            and duplicate_count == 0
            and model_call_count == 2
            and tool_call_count == 0
            and (not expected_reused or prompt_cache_key_stable)
            and provider_prompt_cache_controls_enabled == expected_reused
            and len(turns) == 2
        ),
        "answer_match": answer_match,
        "projection_match": projection_match,
        "initial_projection_match": initial_projection_match,
        "usage_complete": bool(turns) and all(
            turn.get("usage_complete") for turn in turns
        ),
        "provider_cache_hit": int(second.get("cached_tokens", 0) or 0) > 0,
        "prompt_cache_key_stable": prompt_cache_key_stable,
        "provider_prompt_cache_controls_enabled": (
            provider_prompt_cache_controls_enabled
        ),
        "model_call_count": model_call_count,
        "input_tokens": input_tokens,
        "cached_tokens": cached_tokens,
        "billable_input_tokens": max(0, input_tokens - cached_tokens),
        "second_turn_cached_tokens": int(second.get("cached_tokens", 0) or 0),
        "cache_projection_reused": bool(second.get("cache_projection_reused", False)),
        "cache_projection_reason": str(
            second.get("cache_projection_reason", "none") or "none"
        ),
        "cache_projection_generation": int(
            second.get("cache_projection_generation", 0) or 0
        ),
        "cache_projection_message_count": int(
            second.get("cache_projection_message_count", 0) or 0
        ),
        "cache_projection_chars": int(second.get("cache_projection_chars", 0) or 0),
        "cache_projection_context_refreshed": bool(
            second.get("cache_projection_context_refreshed", False)
        ),
        "provider_prompt_chars": int(second.get("provider_prompt_chars", 0) or 0),
        "tool_call_count": tool_call_count,
        "duplicate_tool_result_count": duplicate_count,
        "turns": turns,
        "errors": list(errors or []),
    }


def result_matrix_keys(scenarios, variants, repetitions):
    return frozenset(
        (str(scenario["id"]), repeat, str(variant))
        for repeat in range(int(repetitions))
        for scenario in scenarios
        for variant in variants
    )


def validate_result_matrix(rows, expected_keys, *, require_complete=False):
    expected = frozenset(expected_keys)
    counts = Counter(
        (
            str(row.get("scenario", "")),
            int(row.get("repeat", 0)),
            str(row.get("variant", "")),
        )
        for row in rows
    )
    duplicates = {key for key, count in counts.items() if count > 1}
    if duplicates:
        raise ValueError("duplicate prompt-cache result rows")
    unexpected = set(counts) - expected
    if unexpected:
        raise ValueError("unexpected prompt-cache result rows")
    missing = expected - set(counts)
    if require_complete and missing:
        raise ValueError("missing prompt-cache result rows")
    return {
        "complete": not missing,
        "completed_count": len(counts),
        "expected_count": len(expected),
    }


def write_results(
    rows,
    output_dir,
    *,
    expected_keys=None,
    require_complete=False,
    evaluation_identity=None,
):
    """Write hash-bound raw rows and a matrix-aware Markdown summary."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ordered = sorted(
        (dict(row) for row in rows),
        key=lambda row: (
            str(row.get("scenario", "")),
            str(row.get("variant", "")),
            int(row.get("repeat", 0)),
        ),
    )
    matrix = None
    if expected_keys is not None:
        matrix = validate_result_matrix(
            ordered, expected_keys, require_complete=require_complete
        )
    jsonl_text = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in ordered
    )
    jsonl_path = output_dir / "results.jsonl"
    _atomic_write_text(jsonl_path, jsonl_text)
    digest = hashlib.sha256(jsonl_text.encode("utf-8")).hexdigest()
    summary = summarize_results(
        ordered,
        matrix=matrix,
        results_digest=f"sha256:{digest}",
        evaluation_identity=evaluation_identity,
    )
    summary_path = output_dir / "summary.json"
    _atomic_write_text(
        summary_path,
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    markdown_path = output_dir / "summary.md"
    claimability = summary["claimability"]
    claimability_lines = [
        "## Claimability",
        "",
        f"- Claimable: {claimability['claimable']}",
        f"- Evaluation mode: {claimability['evaluation_mode'] or 'unknown'}",
        f"- Matrix complete: {claimability['matrix_complete']}",
        (
            "- Usage completeness: "
            f"{claimability['usage_completeness']}"
        ),
        (
            "- Minimum pair count: "
            f"{claimability['minimum_pair_count']}"
        ),
        (
            "- Scenario coverage satisfied: "
            f"{claimability['scenario_coverage_satisfied']}"
        ),
        (
            "- Scenario pair counts: "
            f"{claimability['scenario_pair_counts']}"
        ),
        (
            "- Order balance satisfied: "
            f"{claimability['order_balance_satisfied']}"
        ),
        (
            "- Per-scenario order balance satisfied: "
            f"{claimability['scenario_order_balance_satisfied']}"
        ),
        (
            "- Behavior regressions: "
            f"{claimability['behavior_regression_count']}"
        ),
        (
            "- Behavior-valid pairs: "
            f"{claimability['behavior_valid_pair_count']}"
        ),
        (
            "- Reasons: "
            + (
                ", ".join(claimability["claimability_reasons"])
                or "none"
            )
        ),
        "",
    ]
    if matrix is not None and not matrix["complete"]:
        lines = [
            "# Prompt Cache Turn Evaluation (Incomplete)",
            "",
            f"Status: {matrix['completed_count']}/{matrix['expected_count']} result rows.",
            "",
            f"Results SHA-256: `sha256:{digest}`",
            "",
            *claimability_lines,
        ]
        text = "\n".join(lines)
        _atomic_write_text(markdown_path, text)
        return {
            "jsonl": jsonl_path,
            "summary": summary_path,
            "markdown": markdown_path,
            "complete": False,
        }

    columns = ("scenario", "variant", "repeat", *RESULT_FIELDS)
    lines = ["# Prompt Cache Turn Evaluation", ""]
    if matrix is not None:
        lines.extend(
            [
                f"Status: Complete, {matrix['completed_count']}/{matrix['expected_count']} result rows.",
                "",
                f"Results SHA-256: `sha256:{digest}`",
                "",
            ]
        )
    lines.extend(claimability_lines)
    paired = summary["paired"]
    order_strata = paired["by_execution_order"]
    scenario_strata = paired["by_scenario"]
    lines.extend(
        [
            "## Paired comparison",
            "",
            f"- Complete pairs: {paired['pair_count']}",
            f"- Usage-complete pairs: {paired['usage_complete_pair_count']}",
            f"- Control-first pairs: {paired['control_first_pair_count']}",
            f"- Projection-first pairs: {paired['projection_first_pair_count']}",
            f"- Inconsistent-order pairs: {paired['inconsistent_order_pair_count']}",
            f"- Behavior regressions: {paired['behavior_regression_count']}",
            f"- Break-even pairs: {paired['break_even_pair_count']}",
            (
                "- Mean billable input delta: "
                f"{paired['mean_billable_input_delta_tokens']} tokens "
                f"({paired['mean_billable_input_delta_pct']})"
            ),
            (
                "- Mean second-turn cached-token delta: "
                f"{paired['mean_second_turn_cached_tokens_delta']}"
            ),
            "",
            "## Execution-order strata",
            "",
            (
                "Projection-first minus control-first mean billable-token delta: "
                f"{paired['projection_first_minus_control_first_mean_billable_input_delta_tokens']}"
            ),
            "",
            (
                "| Execution order | Pairs | Usage-complete | Behavior regressions | "
                "Break-even rate | Mean billable delta | Mean billable delta rate |"
            ),
            "| --- | --- | --- | --- | --- | --- | --- |",
            *(
                "| "
                + " | ".join(
                    (
                        order,
                        str(metrics["pair_count"]),
                        str(metrics["usage_complete_pair_count"]),
                        str(metrics["behavior_regression_count"]),
                        str(metrics["break_even_pair_rate"]),
                        str(metrics["mean_billable_input_delta_tokens"]),
                        str(metrics["mean_billable_input_delta_pct"]),
                    )
                )
                + " |"
                for order, metrics in order_strata.items()
            ),
            "",
            "## Scenario strata",
            "",
            (
                "| Scenario | Pairs | Usage-complete | Behavior regressions | "
                "Break-even rate | Mean billable delta | Mean billable delta rate |"
            ),
            "| --- | --- | --- | --- | --- | --- | --- |",
            *(
                "| "
                + " | ".join(
                    (
                        scenario,
                        str(metrics["pair_count"]),
                        str(metrics["usage_complete_pair_count"]),
                        str(metrics["behavior_regression_count"]),
                        str(metrics["break_even_pair_rate"]),
                        str(metrics["mean_billable_input_delta_tokens"]),
                        str(metrics["mean_billable_input_delta_pct"]),
                    )
                )
                + " |"
                for scenario, metrics in scenario_strata.items()
            ),
            "",
            "## Rows",
            "",
        ]
    )
    lines.extend(
        [
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join("---" for _ in columns) + " |",
        ]
    )
    for row in ordered:
        lines.append(
            "| "
            + " | ".join(_markdown_cell(row.get(column, "")) for column in columns)
            + " |"
        )
    _atomic_write_text(markdown_path, "\n".join(lines) + "\n")
    return {
        "jsonl": jsonl_path,
        "summary": summary_path,
        "markdown": markdown_path,
        "complete": True,
    }


def summarize_results(
    rows,
    *,
    matrix=None,
    results_digest="",
    evaluation_identity=None,
):
    """Build machine-readable aggregate metrics without hiding raw rows."""

    ordered = [dict(row) for row in rows]
    paired = paired_metrics(ordered)
    return {
        "schema_version": "lite.prompt_cache_turn_summary.v8",
        "results_sha256": str(results_digest),
        "matrix": dict(matrix or {}),
        "row_count": len(ordered),
        "overall": _aggregate_rows(ordered),
        "by_variant": _aggregate_groups(ordered, "variant"),
        "by_scenario": _aggregate_groups(ordered, "scenario"),
        "paired": paired,
        "claimability": claimability_metrics(
            matrix=matrix,
            paired=paired,
            evaluation_identity=evaluation_identity,
        ),
    }


def claimability_metrics(*, matrix, paired, evaluation_identity=None):
    """Decide whether aggregate results support a formal comparative claim."""

    identity = dict(evaluation_identity or {})
    mode = str(identity.get("mode", ""))
    order_policy = str(identity.get("execution_order_policy", ""))
    pair_count = int(paired.get("pair_count", 0) or 0)
    usage_complete_pair_count = int(
        paired.get("usage_complete_pair_count", 0) or 0
    )
    control_first = int(paired.get("control_first_pair_count", 0) or 0)
    projection_first = int(
        paired.get("projection_first_pair_count", 0) or 0
    )
    inconsistent_order = int(
        paired.get("inconsistent_order_pair_count", 0) or 0
    )
    behavior_regressions = int(
        paired.get("behavior_regression_count", 0) or 0
    )
    behavior_valid_pair_count = int(
        paired.get("behavior_valid_pair_count", 0) or 0
    )
    scenario_metrics = dict(paired.get("by_scenario", {}) or {})
    scenario_pair_counts = {
        scenario: int(metrics.get("pair_count", 0) or 0)
        for scenario, metrics in sorted(scenario_metrics.items())
    }
    missing_required_scenarios = sorted(
        REQUIRED_CLAIMABLE_SCENARIOS - set(scenario_pair_counts)
    )
    underrepresented_scenarios = sorted(
        scenario
        for scenario in REQUIRED_CLAIMABLE_SCENARIOS
        if scenario_pair_counts.get(scenario, 0)
        < MINIMUM_CLAIMABLE_PAIRS_PER_SCENARIO
    )
    scenario_coverage_satisfied = bool(
        not missing_required_scenarios and not underrepresented_scenarios
    )
    scenario_order_balance = {
        scenario: bool(
            metrics.get("order_balance_satisfied", False)
        )
        for scenario, metrics in sorted(scenario_metrics.items())
    }
    scenario_order_balance_satisfied = bool(
        scenario_coverage_satisfied
        and all(
            scenario_order_balance.get(scenario, False)
            for scenario in REQUIRED_CLAIMABLE_SCENARIOS
        )
    )
    matrix_data = dict(matrix or {})
    matrix_complete = bool(matrix_data.get("complete", False))
    expected_count = int(matrix_data.get("expected_count", 0) or 0)
    paired_matrix_complete = bool(
        matrix_complete and expected_count and pair_count * len(VARIANTS) == expected_count
    )
    usage_completeness = (
        round(usage_complete_pair_count / pair_count, 6)
        if pair_count
        else 0.0
    )
    order_balance_satisfied = bool(
        inconsistent_order == 0
        and control_first > 0
        and projection_first > 0
        and abs(control_first - projection_first) <= 1
    )

    reasons = []
    if not identity:
        reasons.append("evaluation_identity_missing")
    if mode == "smoke":
        reasons.append("smoke_preflight_only")
    elif mode != "formal":
        reasons.append("formal_mode_required")
    if order_policy != "counterbalanced_v1":
        reasons.append("counterbalanced_order_policy_required")
    if not matrix_complete:
        reasons.append("complete_matrix_required")
    if not paired_matrix_complete:
        reasons.append("complete_paired_matrix_required")
    if pair_count < MINIMUM_CLAIMABLE_PAIR_COUNT:
        reasons.append("minimum_pair_count_not_met")
    if missing_required_scenarios:
        reasons.append("required_scenarios_missing")
    if underrepresented_scenarios:
        reasons.append("minimum_scenario_pair_count_not_met")
    if usage_complete_pair_count != pair_count or not pair_count:
        reasons.append("usage_incomplete")
    if behavior_regressions:
        reasons.append("behavior_regressions_present")
    if behavior_valid_pair_count != pair_count or not pair_count:
        reasons.append("behavior_validation_failed")
    if inconsistent_order:
        reasons.append("inconsistent_execution_order")
    if not order_balance_satisfied:
        reasons.append("execution_order_balance_required")
    if not scenario_order_balance_satisfied:
        reasons.append("scenario_execution_order_balance_required")

    return {
        "claimable": not reasons,
        "claimability_reasons": reasons,
        "evaluation_mode": mode,
        "execution_order_policy": order_policy,
        "minimum_pair_count": MINIMUM_CLAIMABLE_PAIR_COUNT,
        "minimum_pairs_per_scenario": MINIMUM_CLAIMABLE_PAIRS_PER_SCENARIO,
        "pair_count": pair_count,
        "required_scenarios": sorted(REQUIRED_CLAIMABLE_SCENARIOS),
        "scenario_pair_counts": scenario_pair_counts,
        "missing_required_scenarios": missing_required_scenarios,
        "underrepresented_scenarios": underrepresented_scenarios,
        "scenario_coverage_satisfied": scenario_coverage_satisfied,
        "scenario_order_balance": scenario_order_balance,
        "scenario_order_balance_satisfied": scenario_order_balance_satisfied,
        "matrix_complete": matrix_complete,
        "paired_matrix_complete": paired_matrix_complete,
        "usage_completeness": usage_completeness,
        "behavior_valid_pair_count": behavior_valid_pair_count,
        "behavior_completeness": (
            round(behavior_valid_pair_count / pair_count, 6)
            if pair_count
            else 0.0
        ),
        "behavior_regression_count": behavior_regressions,
        "inconsistent_order_pair_count": inconsistent_order,
        "order_balance_satisfied": order_balance_satisfied,
    }


def paired_metrics(rows):
    """Compare append projection with its matched full-prompt control."""

    index = {
        (
            str(row.get("scenario", "")),
            int(row.get("repeat", 0)),
            str(row.get("variant", "")),
        ): row
        for row in rows
    }
    pair_keys = sorted(
        {
            (scenario, repeat)
            for scenario, repeat, _variant in index
        }
    )
    pairs = []
    for scenario, repeat in pair_keys:
        control = index.get((scenario, repeat, "full_prompt"))
        treatment = index.get((scenario, repeat, "append_projection"))
        if not control or not treatment:
            continue
        usage_complete = bool(
            control.get("usage_complete") and treatment.get("usage_complete")
        )
        control_billable = int(control.get("billable_input_tokens", 0) or 0)
        treatment_billable = int(treatment.get("billable_input_tokens", 0) or 0)
        delta_tokens = treatment_billable - control_billable
        delta_pct = (
            round(delta_tokens / control_billable, 6)
            if usage_complete and control_billable
            else None
        )
        behavior_regression = bool(
            control.get("behavior_pass") and not treatment.get("behavior_pass")
        )
        behavior_valid = bool(
            control.get("behavior_pass") and treatment.get("behavior_pass")
        )
        break_even = bool(
            usage_complete
            and treatment_billable < control_billable
            and treatment.get("behavior_pass")
            and not behavior_regression
        )
        control_order = str(control.get("pair_execution_order", ""))
        treatment_order = str(treatment.get("pair_execution_order", ""))
        execution_order_consistent = bool(
            control_order and control_order == treatment_order
        )
        pair_execution_order = control_order if execution_order_consistent else ""
        pairs.append(
            {
                "scenario": scenario,
                "repeat": repeat,
                "pair_execution_order": pair_execution_order,
                "execution_order_consistent": execution_order_consistent,
                "usage_complete": usage_complete,
                "control_behavior_pass": bool(control.get("behavior_pass")),
                "treatment_behavior_pass": bool(treatment.get("behavior_pass")),
                "behavior_regression": behavior_regression,
                "behavior_valid": behavior_valid,
                "control_billable_input_tokens": control_billable,
                "treatment_billable_input_tokens": treatment_billable,
                "billable_input_delta_tokens": delta_tokens,
                "billable_input_delta_pct": delta_pct,
                "second_turn_cached_tokens_delta": int(
                    treatment.get("second_turn_cached_tokens", 0) or 0
                )
                - int(control.get("second_turn_cached_tokens", 0) or 0),
                "break_even": break_even,
            }
        )
    complete_usage = [pair for pair in pairs if pair["usage_complete"]]
    pct_pairs = [
        pair
        for pair in complete_usage
        if pair["billable_input_delta_pct"] is not None
    ]
    control_first_pairs = [
        pair
        for pair in pairs
        if pair["pair_execution_order"]
        == "full_prompt_then_append_projection"
    ]
    projection_first_pairs = [
        pair
        for pair in pairs
        if pair["pair_execution_order"]
        == "append_projection_then_full_prompt"
    ]
    order_strata = {
        "control_first": _summarize_pairs(control_first_pairs),
        "projection_first": _summarize_pairs(projection_first_pairs),
    }
    scenarios = sorted({pair["scenario"] for pair in pairs})
    scenario_strata = {
        scenario: _summarize_pairs(
            [pair for pair in pairs if pair["scenario"] == scenario]
        )
        for scenario in scenarios
    }
    return {
        "pair_count": len(pairs),
        "usage_complete_pair_count": len(complete_usage),
        "control_first_pair_count": len(control_first_pairs),
        "projection_first_pair_count": len(projection_first_pairs),
        "inconsistent_order_pair_count": sum(
            not pair["execution_order_consistent"] for pair in pairs
        ),
        "behavior_regression_count": sum(
            pair["behavior_regression"] for pair in pairs
        ),
        "behavior_valid_pair_count": sum(
            pair["behavior_valid"] for pair in pairs
        ),
        "break_even_pair_count": sum(pair["break_even"] for pair in pairs),
        "break_even_pair_rate": _pair_rate(pairs, "break_even"),
        "mean_billable_input_delta_tokens": _pair_mean(
            complete_usage, "billable_input_delta_tokens"
        ),
        "mean_billable_input_delta_pct": _pair_mean(
            pct_pairs, "billable_input_delta_pct"
        ),
        "mean_second_turn_cached_tokens_delta": _pair_mean(
            complete_usage, "second_turn_cached_tokens_delta"
        ),
        "by_execution_order": order_strata,
        "by_scenario": scenario_strata,
        "projection_first_minus_control_first_mean_billable_input_delta_tokens": (
            _metric_difference(
                order_strata["projection_first"],
                order_strata["control_first"],
                "mean_billable_input_delta_tokens",
            )
        ),
        "projection_first_minus_control_first_mean_billable_input_delta_pct": (
            _metric_difference(
                order_strata["projection_first"],
                order_strata["control_first"],
                "mean_billable_input_delta_pct",
            )
        ),
        "pairs": pairs,
    }


def _summarize_pairs(pairs):
    complete_usage = [pair for pair in pairs if pair["usage_complete"]]
    pct_pairs = [
        pair
        for pair in complete_usage
        if pair["billable_input_delta_pct"] is not None
    ]
    control_first_pair_count = sum(
        pair["pair_execution_order"]
        == "full_prompt_then_append_projection"
        for pair in pairs
    )
    projection_first_pair_count = sum(
        pair["pair_execution_order"]
        == "append_projection_then_full_prompt"
        for pair in pairs
    )
    inconsistent_order_pair_count = sum(
        not pair["execution_order_consistent"] for pair in pairs
    )
    return {
        "pair_count": len(pairs),
        "usage_complete_pair_count": len(complete_usage),
        "control_first_pair_count": control_first_pair_count,
        "projection_first_pair_count": projection_first_pair_count,
        "inconsistent_order_pair_count": inconsistent_order_pair_count,
        "order_balance_satisfied": bool(
            inconsistent_order_pair_count == 0
            and control_first_pair_count > 0
            and projection_first_pair_count > 0
            and abs(control_first_pair_count - projection_first_pair_count) <= 1
        ),
        "behavior_regression_count": sum(
            pair["behavior_regression"] for pair in pairs
        ),
        "behavior_valid_pair_count": sum(
            pair["behavior_valid"] for pair in pairs
        ),
        "break_even_pair_count": sum(pair["break_even"] for pair in pairs),
        "break_even_pair_rate": _pair_rate(pairs, "break_even"),
        "mean_billable_input_delta_tokens": _pair_mean(
            complete_usage, "billable_input_delta_tokens"
        ),
        "mean_billable_input_delta_pct": _pair_mean(
            pct_pairs, "billable_input_delta_pct"
        ),
        "mean_second_turn_cached_tokens_delta": _pair_mean(
            complete_usage, "second_turn_cached_tokens_delta"
        ),
    }


def _metric_difference(left, right, field):
    left_value = left.get(field)
    right_value = right.get(field)
    if left_value is None or right_value is None:
        return None
    return round(float(left_value) - float(right_value), 6)


def _aggregate_groups(rows, field):
    groups = {}
    for row in rows:
        groups.setdefault(str(row.get(field, "")), []).append(row)
    return {
        key: _aggregate_rows(groups[key]) for key in sorted(groups)
    }


def _aggregate_rows(rows):
    count = len(rows)
    if not count:
        return {
            "row_count": 0,
            "behavior_pass_rate": 0.0,
            "usage_complete_rate": 0.0,
            "provider_cache_hit_rate": 0.0,
            "prompt_cache_key_stable_rate": 0.0,
            "projection_reuse_rate": 0.0,
            "mean_billable_input_tokens": 0.0,
            "mean_second_turn_cached_tokens": 0.0,
        }
    return {
        "row_count": count,
        "behavior_pass_rate": _rate(rows, "behavior_pass"),
        "usage_complete_rate": _rate(rows, "usage_complete"),
        "provider_cache_hit_rate": _rate(rows, "provider_cache_hit"),
        "prompt_cache_key_stable_rate": _rate(rows, "prompt_cache_key_stable"),
        "projection_reuse_rate": _rate(rows, "cache_projection_reused"),
        "mean_billable_input_tokens": _mean(rows, "billable_input_tokens"),
        "mean_second_turn_cached_tokens": _mean(
            rows, "second_turn_cached_tokens"
        ),
    }


def _rate(rows, field):
    return round(
        sum(bool(row.get(field, False)) for row in rows) / len(rows),
        6,
    )


def _mean(rows, field):
    return round(
        sum(float(row.get(field, 0) or 0) for row in rows) / len(rows),
        6,
    )


def _pair_rate(pairs, field):
    if not pairs:
        return 0.0
    return round(sum(bool(pair[field]) for pair in pairs) / len(pairs), 6)


def _pair_mean(pairs, field):
    if not pairs:
        return None
    return round(sum(float(pair[field]) for pair in pairs) / len(pairs), 6)


def _atomic_write_text(path, content):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8", newline="")
    temporary.replace(path)


def _is_provider_usage(metadata):
    return (
        metadata.get("provider_protocol") is not None
        and metadata.get("provider_model") is not None
        and metadata.get("input_tokens") is not None
        and metadata.get("output_tokens") is not None
        and metadata.get("synthetic") is not True
    )


def _markdown_cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")
