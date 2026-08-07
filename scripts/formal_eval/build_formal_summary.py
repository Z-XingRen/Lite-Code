"""Build the unified formal-evaluation JSON and Markdown reports."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts" / "formal-evaluation-20260806"
OUTPUT_JSON = ARTIFACTS / "formal-summary.json"
OUTPUT_MD = ARTIFACTS / "formal-summary.md"


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def rate(count: int, total: int) -> dict[str, float | int]:
    return {
        "count": int(count),
        "total": int(total),
        "rate": count / total if total else 0.0,
    }


def summarize_excluded_attempts(root: Path) -> dict[str, Any]:
    traces = []
    totals = {
        "model_calls": 0,
        "input_tokens": 0,
        "cached_tokens": 0,
        "output_tokens": 0,
        "provider_errors": 0,
    }
    for path in sorted(root.glob("**/trace.jsonl")):
        trace = {
            "trace_path": str(path.relative_to(ROOT)),
            "model_calls": 0,
            "input_tokens": 0,
            "cached_tokens": 0,
            "output_tokens": 0,
            "provider_errors": [],
        }
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            metadata = event.get("completion_metadata")
            if event.get("event") == "model_parsed" and isinstance(metadata, dict):
                trace["model_calls"] += 1
                for field in ("input_tokens", "cached_tokens", "output_tokens"):
                    trace[field] += int(metadata.get(field) or 0)
            if event.get("event") == "model_error":
                error = event.get("error")
                if isinstance(error, dict):
                    trace["provider_errors"].append(
                        {
                            "code": error.get("code"),
                            "http_status": error.get("http_status"),
                        }
                    )
        if trace["model_calls"] or trace["provider_errors"]:
            traces.append(trace)
            for field in (
                "model_calls",
                "input_tokens",
                "cached_tokens",
                "output_tokens",
            ):
                totals[field] += trace[field]
            totals["provider_errors"] += len(trace["provider_errors"])
    totals["billable_input_tokens"] = max(
        0, totals["input_tokens"] - totals["cached_tokens"]
    )
    return {
        "excluded_from_primary_metrics": True,
        "attempt_count": len(traces),
        **totals,
        "attempts": traces,
    }


def usage_rollup(
    usages: list[dict[str, Any]], *, default_model_calls: int = 0
) -> dict[str, int]:
    result = {
        "usage_records": len(usages),
        "model_calls": sum(
            int(usage.get("model_call_count", default_model_calls) or 0)
            for usage in usages
        ),
        "input_tokens": sum(int(usage.get("input_tokens") or 0) for usage in usages),
        "cached_tokens": sum(int(usage.get("cached_tokens") or 0) for usage in usages),
        "output_tokens": sum(int(usage.get("output_tokens") or 0) for usage in usages),
    }
    result["billable_input_tokens"] = max(
        0, result["input_tokens"] - result["cached_tokens"]
    )
    return result


def actual_row_usages(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        usage = row.get("usage")
        if not isinstance(usage, dict):
            continue
        is_actual = (
            usage.get("usage_source") == "actual"
            if "usage_source" in usage
            else usage.get("actual") is True
        )
        if is_actual:
            result.append(usage)
    return result


def combine_usage(sources: dict[str, dict[str, int]]) -> dict[str, int]:
    fields = (
        "usage_records",
        "model_calls",
        "input_tokens",
        "cached_tokens",
        "billable_input_tokens",
        "output_tokens",
    )
    return {
        field: sum(source[field] for source in sources.values()) for field in fields
    }


def combination(n: int, k: int) -> int:
    return math.comb(n, k) if 0 <= k <= n else 0


def pass_at_k(rows: list[dict], field: str, repetitions: int) -> dict[str, Any]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["task_id"])].append(row)
    if any(len(bucket) != repetitions for bucket in grouped.values()):
        raise RuntimeError(f"{field}: incomplete repetitions in quality results")
    counts = [
        sum(bool(row.get(field)) for row in bucket) for bucket in grouped.values()
    ]
    metrics = {}
    for k in range(1, repetitions + 1):
        denominator = combination(repetitions, k)
        values = [
            1 - combination(repetitions - correct, k) / denominator
            for correct in counts
        ]
        metrics[f"pass@{k}"] = sum(values) / len(values)
    metrics["tasks_with_any_pass"] = sum(correct > 0 for correct in counts)
    metrics["tasks_all_repetitions_pass"] = sum(
        correct == repetitions for correct in counts
    )
    return metrics


def summarize_quality() -> dict[str, Any]:
    payload = load_json(ARTIFACTS / "quality" / "results.json")
    rows = payload["rows"]
    repetitions = int(payload["repetitions"])
    keys = [(str(row["task_id"]), int(row["repeat"])) for row in rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError("quality results contain duplicate task/repeat keys")
    total = len(rows)
    by_category = {}
    for category in sorted({str(row["category"]) for row in rows}):
        bucket = [row for row in rows if row["category"] == category]
        by_category[category] = {
            "trials": len(bucket),
            "target_pass": rate(
                sum(bool(row["target_pass"]) for row in bucket), len(bucket)
            ),
            "regression_pass": rate(
                sum(bool(row["regression_pass"]) for row in bucket), len(bucket)
            ),
            "scc": rate(sum(bool(row["scc"]) for row in bucket), len(bucket)),
        }
    return {
        "status": "complete",
        "model": payload["model"],
        "trials": total,
        "unique_trials": len(set(keys)),
        "target_pass": rate(sum(bool(row["target_pass"]) for row in rows), total),
        "regression_pass": rate(
            sum(bool(row["regression_pass"]) for row in rows), total
        ),
        "scc": rate(sum(bool(row["scc"]) for row in rows), total),
        "scope_pass": rate(sum(bool(row["scope_pass"]) for row in rows), total),
        "finalization_pass": rate(
            sum(bool(row["finalization_pass"]) for row in rows), total
        ),
        "budget_pass": rate(sum(bool(row["budget_pass"]) for row in rows), total),
        "safety_pass": rate(sum(bool(row["safety_pass"]) for row in rows), total),
        "actual_usage": rate(
            sum(row["usage"].get("usage_source") == "actual" for row in rows),
            total,
        ),
        "error_trials": sum(bool(row.get("errors")) for row in rows),
        "timeout_trials": sum(
            "timeout" in " ".join(row.get("errors", [])).lower() for row in rows
        ),
        "pass_at_k": {
            field: pass_at_k(rows, field, repetitions)
            for field in ("target_pass", "regression_pass", "scc")
        },
        "by_category": by_category,
    }


def summarize_context() -> dict[str, Any]:
    payload = load_json(ARTIFACTS / "context-ab" / "results.json")
    rows = payload["rows"]
    keys = [
        (str(row["task_id"]), int(row["repeat"]), str(row["variant"])) for row in rows
    ]
    if len(keys) != 60 or len(set(keys)) != 60:
        raise RuntimeError("context A/B must contain 60 unique rows")
    metrics = dict(payload["formal_metrics"])
    metrics.pop("pairs", None)
    return {
        "status": "complete",
        "rows": len(rows),
        "completed": sum(row["status"] == "completed" for row in rows),
        "stopped": sum(row["status"] == "stopped" for row in rows),
        "verification_pass": rate(
            sum(row["verification_status"] == "passed" for row in rows), len(rows)
        ),
        "actual_usage": rate(
            sum(row["usage"]["usage_source"] == "actual" for row in rows), len(rows)
        ),
        "excluded_attempt_audit": summarize_excluded_attempts(
            ARTIFACTS / "context-ab" / "failed-attempts"
        ),
        "formal_metrics": metrics,
    }


def summarize_multi_agent() -> dict[str, Any]:
    payload = load_json(ARTIFACTS / "multi-agent-ab" / "results.json")
    rows = payload["rows"]
    keys = [
        (str(row["task_id"]), int(row["repeat"]), str(row["variant"])) for row in rows
    ]
    if len(keys) != 48 or len(set(keys)) != 48:
        raise RuntimeError("single/multi A/B must contain 48 unique rows")
    actual = sum(row["usage"]["usage_source"] == "actual" for row in rows)
    by_variant = {}
    for variant in ("single", "multi"):
        bucket = [row for row in rows if row["variant"] == variant]
        total = len(bucket)
        by_variant[variant] = {
            "rows": total,
            **{
                field: rate(sum(bool(row[field]) for row in bucket), total)
                for field in (
                    "target_pass",
                    "regression_pass",
                    "scc",
                    "scope_pass",
                    "finalization_pass",
                    "budget_pass",
                    "safety_pass",
                )
            },
            "actual_usage": rate(
                sum(row["usage"]["usage_source"] == "actual" for row in bucket),
                total,
            ),
            "worker_evidence": (
                rate(sum("worker_started" in row["events"] for row in bucket), total)
                if variant == "multi"
                else None
            ),
            "mean_input_tokens": sum(row["usage"]["input_tokens"] for row in bucket)
            / total,
            "mean_billable_input_tokens": sum(
                max(
                    0,
                    row["usage"]["input_tokens"] - row["usage"]["cached_tokens"],
                )
                for row in bucket
            )
            / total,
            "mean_wall_time_ms": sum(row["wall_time_ms"] for row in bucket) / total,
            "mean_duplicate_reads": sum(row["duplicate_read_count"] for row in bucket)
            / total,
            "write_conflict": rate(
                sum(row["write_conflict_count"] > 0 for row in bucket), total
            ),
        }
    pairs = payload["pairs"]
    return {
        "status": "complete",
        "rows": len(rows),
        **{
            field: rate(sum(bool(row[field]) for row in rows), len(rows))
            for field in (
                "target_pass",
                "regression_pass",
                "scc",
                "scope_pass",
                "finalization_pass",
                "budget_pass",
                "safety_pass",
            )
        },
        "actual_usage": rate(actual, len(rows)),
        "excluded_attempt_audit": summarize_excluded_attempts(
            ARTIFACTS / "multi-agent-ab" / "failed-attempts"
        ),
        "summary": payload["summary"],
        "by_variant": by_variant,
        "paired_rows": len(pairs),
        "paired_comparison": {
            "multi_scc_better": sum(
                pair["multi_scc"] and not pair["single_scc"] for pair in pairs
            ),
            "single_scc_better": sum(
                pair["single_scc"] and not pair["multi_scc"] for pair in pairs
            ),
            "same_scc": sum(pair["single_scc"] == pair["multi_scc"] for pair in pairs),
            "mean_billable_input_delta": sum(
                pair["billable_input_delta"] for pair in pairs
            )
            / len(pairs),
            "mean_wall_time_delta_ms": sum(pair["wall_time_delta_ms"] for pair in pairs)
            / len(pairs),
            "mean_duplicate_read_delta": sum(
                pair["duplicate_read_delta"] for pair in pairs
            )
            / len(pairs),
        },
    }


def _trial_reward(trial: dict[str, Any]) -> dict[str, float | int] | None:
    verifier = trial.get("verifier_result")
    rewards = verifier.get("rewards") if isinstance(verifier, dict) else None
    return rewards if isinstance(rewards, dict) else None


def _recovered_lite_usage(trial_result_path: Path) -> dict[str, Any] | None:
    usage_path = (
        trial_result_path.parent
        / "agent"
        / "timeout-recovery"
        / "usage.json"
    )
    if not usage_path.is_file():
        return None
    usage = load_json(usage_path)
    return usage if usage.get("usage_source") == "actual" else None


def _attach_recovery_job(
    base: dict[str, Any], recovery: dict[str, Any]
) -> dict[str, Any]:
    if recovery.get("status") == "missing":
        return base
    effective: dict[str, float | int | None] = {}
    for row in base.get("task_results", []):
        rewards = row.get("rewards")
        effective[row["task_id"]] = next(iter(rewards.values())) if rewards else None
    for row in recovery.get("task_results", []):
        rewards = row.get("rewards")
        if rewards:
            effective[row["task_id"]] = next(iter(rewards.values()))
    verified = [value for value in effective.values() if value is not None]
    return {
        **base,
        "recovery_job": recovery,
        "effective_verified_trials": len(verified),
        "effective_passed_trials": sum(value == 1 for value in verified),
        "effective_mean_primary_reward": (
            sum(float(value) for value in verified) / len(verified)
            if verified
            else None
        ),
    }


def _harbor_job(
    subset_name: str,
    agent: str,
    expected: int,
    *,
    job_name: str | None = None,
) -> dict[str, Any]:
    job_name = job_name or (
        f"{subset_name}-{agent}-gpt-5-5" if agent == "lite" else f"{subset_name}-oracle"
    )
    result_path = ARTIFACTS / subset_name / "jobs" / job_name / "result.json"
    if not result_path.is_file():
        return {"status": "missing", "result_path": str(result_path)}
    job_payload = load_json(result_path)
    trial_paths = sorted(result_path.parent.glob("*/result.json"))
    trials = [load_json(path) for path in trial_paths]
    stats = job_payload.get("stats")
    if not isinstance(stats, dict):
        raise RuntimeError(f"invalid Harbor job stats: {result_path}")
    task_names = [str(trial["task_name"]) for trial in trials]
    rewards = [_trial_reward(trial) for trial in trials]
    primary_rewards = [next(iter(item.values())) if item else None for item in rewards]
    actual_usage = 0
    actual_usages = []
    recovered_usage_tasks = []
    for trial_path, trial in zip(trial_paths, trials, strict=True):
        agent_result = trial.get("agent_result")
        metadata = (
            agent_result.get("metadata") if isinstance(agent_result, dict) else None
        )
        lite_usage = metadata.get("lite_usage") if isinstance(metadata, dict) else None
        recovered = False
        if not (
            isinstance(lite_usage, dict)
            and lite_usage.get("usage_source") == "actual"
        ):
            lite_usage = _recovered_lite_usage(trial_path)
            recovered = lite_usage is not None
        if isinstance(lite_usage, dict) and lite_usage.get("usage_source") == "actual":
            actual_usage += 1
            actual_usages.append(lite_usage)
            if recovered:
                recovered_usage_tasks.append(str(trial["task_name"]))
    stopped_path = ARTIFACTS / subset_name / f"{agent}.stopped.json"
    stopped = load_json(stopped_path) if stopped_path.is_file() else None
    return {
        "status": (
            "complete"
            if job_payload.get("finished_at")
            and stats.get("n_completed_trials") == expected
            and len(trials) == expected
            else "incomplete"
        ),
        "result_path": str(result_path),
        "expected_trials": expected,
        "trials": len(trials),
        "unique_tasks": len(set(task_names)),
        "job_completed_trials": stats.get("n_completed_trials"),
        "job_errored_trials": stats.get("n_errored_trials"),
        "job_cancelled_trials": stats.get("n_cancelled_trials"),
        "job_retries": stats.get("n_retries"),
        "verified_trials": sum(reward is not None for reward in rewards),
        "passed_trials": sum(reward == 1 for reward in primary_rewards),
        "mean_primary_reward": (
            sum(float(reward) for reward in primary_rewards if reward is not None)
            / sum(reward is not None for reward in primary_rewards)
            if any(reward is not None for reward in primary_rewards)
            else None
        ),
        "errored_trials": sum(
            trial.get("exception_info") is not None for trial in trials
        ),
        "actual_usage_trials": actual_usage if agent == "lite" else None,
        "recovered_usage_trials": (
            len(recovered_usage_tasks) if agent == "lite" else None
        ),
        "recovered_usage_tasks": recovered_usage_tasks if agent == "lite" else None,
        "usage": usage_rollup(actual_usages) if agent == "lite" else None,
        "stopped": stopped,
        "task_results": [
            {
                "task_id": trial["task_name"],
                "trial_name": trial["trial_name"],
                "rewards": reward,
                "exception_type": (
                    trial["exception_info"].get("exception_type")
                    if isinstance(trial.get("exception_info"), dict)
                    else None
                ),
            }
            for trial, reward in zip(trials, rewards, strict=True)
        ],
    }


def summarize_harbor() -> dict[str, Any]:
    subsets = {
        "terminal-bench-20": 20,
        "swebench-verified-10": 10,
    }
    result = {
        subset: {
            "oracle": _harbor_job(subset, "oracle", expected),
            "lite": _harbor_job(subset, "lite", expected),
        }
        for subset, expected in subsets.items()
    }
    recovery = _harbor_job(
        "terminal-bench-20",
        "oracle",
        1,
        job_name="terminal-bench-20-oracle-recovery-winning",
    )
    result["terminal-bench-20"]["oracle"] = _attach_recovery_job(
        result["terminal-bench-20"]["oracle"], recovery
    )
    return result


def build_summary() -> dict[str, Any]:
    qualification = load_json(ARTIFACTS / "quality" / "qualification.json")
    quality_payload = load_json(ARTIFACTS / "quality" / "results.json")
    security = load_json(ARTIFACTS / "agentdojo-style-20" / "results.json")
    long_memory = load_json(ARTIFACTS / "longmemeval-50" / "results.json")
    context_payload = load_json(ARTIFACTS / "context-ab" / "results.json")
    multi_payload = load_json(ARTIFACTS / "multi-agent-ab" / "results.json")
    context = summarize_context()
    multi = summarize_multi_agent()
    harbor = summarize_harbor()
    primary_usage = {
        "coding_quality": usage_rollup(actual_row_usages(quality_payload["rows"])),
        "agentdojo_style_security": usage_rollup(actual_row_usages(security["rows"])),
        "longmemeval_generation": usage_rollup(actual_row_usages(long_memory["rows"])),
        "longmemeval_judge": usage_rollup(
            [row["judge_usage"] for row in long_memory["rows"]],
            default_model_calls=1,
        ),
        "context_ab": usage_rollup(actual_row_usages(context_payload["rows"])),
        "single_multi_agent_ab": usage_rollup(actual_row_usages(multi_payload["rows"])),
    }
    for subset, jobs in harbor.items():
        lite_usage = jobs["lite"].get("usage")
        if isinstance(lite_usage, dict):
            primary_usage[f"harbor_{subset}_lite"] = lite_usage
    excluded_usage = {
        "context_ab_failed_retries": {
            "usage_records": context["excluded_attempt_audit"]["attempt_count"],
            **{
                field: context["excluded_attempt_audit"][field]
                for field in (
                    "model_calls",
                    "input_tokens",
                    "cached_tokens",
                    "billable_input_tokens",
                    "output_tokens",
                )
            },
        },
        "single_multi_agent_failed_retries": {
            "usage_records": multi["excluded_attempt_audit"]["attempt_count"],
            **{
                field: multi["excluded_attempt_audit"][field]
                for field in (
                    "model_calls",
                    "input_tokens",
                    "cached_tokens",
                    "billable_input_tokens",
                    "output_tokens",
                )
            },
        },
    }
    primary_totals = combine_usage(primary_usage)
    excluded_totals = combine_usage(excluded_usage)
    return {
        "schema_version": "lite-formal-evaluation-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_policy": {
            "provider_profile": "openai",
            "model": "gpt-5.5",
            "reasoning_effort": "medium",
            "credential_source": "project .env (value omitted)",
        },
        "grader_qualification": qualification["qualification"],
        "coding_quality": summarize_quality(),
        "agentdojo_style_security": security["summary"],
        "longmemeval_oracle_50": {
            **long_memory["summary"],
            "selection": long_memory["selection"],
            "leaderboard_comparable": False,
        },
        "context_ab": context,
        "single_multi_agent_ab": multi,
        "harbor": harbor,
        "usage_audit": {
            "actual_primary_sources": primary_usage,
            "excluded_attempt_sources": excluded_usage,
            "actual_primary_total": primary_totals,
            "excluded_attempt_total": excluded_totals,
            "actual_total_including_excluded": {
                field: primary_totals[field] + excluded_totals[field]
                for field in primary_totals
            },
        },
        "limitations": [
            "LongMemEval uses a fixed oracle subset and a same-model gpt-5.5 judge; it is not an official leaderboard score.",
            "AgentDojo-style is a local paired safety set, not the upstream AgentDojo benchmark leaderboard.",
            "Terminal-Bench and SWE-bench results are fixed deterministic subsets, not full-dataset leaderboard scores.",
            "External benchmark expansion was intentionally stopped on 2026-08-07 after the evaluation scope was refocused on controllability, traceability, token A/B gates, and recovery evidence; Terminal-Bench Lite is partial and SWE-bench was not started.",
            "Functional pass, regression pass, and full execution-loop SCC are reported separately.",
        ],
    }


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.2f}%"


def render_markdown(payload: dict[str, Any]) -> str:
    quality = payload["coding_quality"]
    security = payload["agentdojo_style_security"]
    memory = payload["longmemeval_oracle_50"]
    context = payload["context_ab"]
    multi = payload["single_multi_agent_ab"]
    usage = payload["usage_audit"]
    lines = [
        "# Lite Formal Evaluation",
        "",
        f"Generated: `{payload['generated_at']}`",
        "",
        "## Coding quality",
        "",
        f"- Functional target pass: {quality['target_pass']['count']}/{quality['trials']} ({_pct(quality['target_pass']['rate'])})",
        f"- Regression pass: {quality['regression_pass']['count']}/{quality['trials']} ({_pct(quality['regression_pass']['rate'])})",
        f"- Full SCC: {quality['scc']['count']}/{quality['trials']} ({_pct(quality['scc']['rate'])})",
        f"- Scope / finalization / budget / safety: {_pct(quality['scope_pass']['rate'])} / {_pct(quality['finalization_pass']['rate'])} / {_pct(quality['budget_pass']['rate'])} / {_pct(quality['safety_pass']['rate'])}",
        f"- Actual usage: {quality['actual_usage']['count']}/{quality['trials']}",
        "",
        "## Safety and memory",
        "",
        f"- AgentDojo-style: benign utility {_pct(security['benign_utility_rate'])}, injection utility {_pct(security['injection_utility_rate'])}, attack success {_pct(security['attack_success_rate'])}, secret exposure {_pct(security['secret_exposure_rate'])}.",
        f"- LongMemEval oracle subset: {memory['sample_count']} questions, same-model judge accuracy {_pct(memory['judge_accuracy'])}, actual usage {memory['actual_usage_rows']}/{memory['sample_count']}.",
        "",
        "## A/B evaluations",
        "",
        f"- Context A/B: {context['formal_metrics']['pair_count']} pairs; mean billable input delta {_pct(context['formal_metrics']['mean_billable_delta_pct'])}; 95% CI {context['formal_metrics']['billable_delta_95ci']}; quality regressions {context['formal_metrics']['quality_regression_count']}; claimable reduction `{str(context['formal_metrics']['claimable_reduction']).lower()}`.",
        f"- Context excluded retry usage: {context['excluded_attempt_audit']['model_calls']} model calls, {context['excluded_attempt_audit']['input_tokens']} input / {context['excluded_attempt_audit']['cached_tokens']} cached / {context['excluded_attempt_audit']['output_tokens']} output tokens; excluded from paired metrics.",
        f"- Single vs multi-agent: {multi['paired_rows']} pairs; overall target / regression / SCC {_pct(multi['target_pass']['rate'])} / {_pct(multi['regression_pass']['rate'])} / {_pct(multi['scc']['rate'])}; single SCC {_pct(multi['by_variant']['single']['scc']['rate'])}, multi SCC {_pct(multi['by_variant']['multi']['scc']['rate'])}.",
        f"- Single vs multi-agent excluded attempts: {multi['excluded_attempt_audit']['provider_errors']} provider errors; excluded from primary metrics.",
        "",
        "## Harbor subsets",
        "",
    ]
    for subset, result in payload["harbor"].items():
        oracle = result["oracle"]
        lite = result["lite"]
        oracle_passed = oracle.get("effective_passed_trials", oracle.get("passed_trials", 0))
        recovery_note = (
            f", recovery {oracle['recovery_job'].get('passed_trials', 0)}/{oracle['recovery_job'].get('trials', 0)}"
            if isinstance(oracle.get("recovery_job"), dict)
            else ""
        )
        stopped_note = "; intentionally stopped after scope refocus" if lite.get("stopped") else ""
        lines.append(
            f"- {subset}: oracle `{oracle['status']}` ({oracle_passed}/{oracle.get('expected_trials', oracle.get('trials', 0))}{recovery_note}); Lite `{lite['status']}` ({lite.get('passed_trials', 0)}/{lite.get('trials', 0)}), actual usage {lite.get('actual_usage_trials', 0)}/{lite.get('trials', 0)}{stopped_note}."
        )
    lines.extend(
        [
            "",
            "## Usage audit",
            "",
            f"- Primary actual usage: {usage['actual_primary_total']['model_calls']} model calls; {usage['actual_primary_total']['input_tokens']} input / {usage['actual_primary_total']['cached_tokens']} cached / {usage['actual_primary_total']['billable_input_tokens']} billable input / {usage['actual_primary_total']['output_tokens']} output tokens.",
            f"- Excluded attempts: {usage['excluded_attempt_total']['model_calls']} model calls; {usage['excluded_attempt_total']['input_tokens']} input / {usage['excluded_attempt_total']['cached_tokens']} cached / {usage['excluded_attempt_total']['billable_input_tokens']} billable input / {usage['excluded_attempt_total']['output_tokens']} output tokens.",
            f"- Actual total including excluded attempts: {usage['actual_total_including_excluded']['model_calls']} model calls; {usage['actual_total_including_excluded']['input_tokens']} input / {usage['actual_total_including_excluded']['cached_tokens']} cached / {usage['actual_total_including_excluded']['billable_input_tokens']} billable input / {usage['actual_total_including_excluded']['output_tokens']} output tokens.",
            "",
            "## Limitations",
            "",
            *[f"- {item}" for item in payload["limitations"]],
            "",
        ]
    )
    return "\n".join(lines)


def _assert_complete(payload: dict[str, Any]) -> None:
    quality = payload["coding_quality"]
    if quality["trials"] != 72 or quality["actual_usage"]["count"] != 72:
        raise RuntimeError("incomplete coding-quality results or usage")
    context = payload["context_ab"]
    if context["rows"] != 60 or context["actual_usage"]["count"] != 60:
        raise RuntimeError("incomplete context A/B results or usage")
    multi = payload["single_multi_agent_ab"]
    if multi["rows"] != 48 or multi["actual_usage"]["count"] != 48:
        raise RuntimeError("incomplete single/multi-agent results or usage")
    security = payload["agentdojo_style_security"]
    if security["run_count"] != 40 or security["actual_usage_rows"] != 40:
        raise RuntimeError("incomplete AgentDojo-style results or usage")
    memory = payload["longmemeval_oracle_50"]
    if memory["sample_count"] != 50 or memory["actual_usage_rows"] != 50:
        raise RuntimeError("incomplete LongMemEval results or usage")
    for subset, result in payload["harbor"].items():
        for agent, job in result.items():
            if job["status"] != "complete":
                raise RuntimeError(f"incomplete Harbor result: {subset}/{agent}")
            if agent == "lite" and job["actual_usage_trials"] != job["trials"]:
                raise RuntimeError(f"incomplete Lite usage: {subset}/{agent}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args(argv)
    payload = build_summary()
    if args.require_complete:
        _assert_complete(payload)
    OUTPUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    OUTPUT_MD.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps({"json": str(OUTPUT_JSON), "markdown": str(OUTPUT_MD)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
