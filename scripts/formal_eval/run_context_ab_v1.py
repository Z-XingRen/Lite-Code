from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

from lite.config import load_project_env, resolve_provider_config
from lite.evaluation.context_cost import (
    DEFAULT_PROXY_PRICING,
    _run_long_session_task,
    build_result_payload,
    generate_report,
)
from lite.providers import OpenAICompatibleModelClient

ROOT = Path(__file__).resolve().parents[2]
TASKS = ROOT / "benchmarks/formal_v1/long_session_tasks_10.json"
OUT = ROOT / "artifacts/formal-evaluation-20260806/context-ab"
VARIANTS = ["no_context_reduction", "full_orchestrator"]


def client_factory(**_):
    load_project_env(ROOT, override=True)
    config = resolve_provider_config(
        "openai", start=ROOT, config_path=ROOT / ".lite.toml"
    )
    if config.model != "gpt-5.5" or not config.api_key:
        raise RuntimeError("context A/B requires configured gpt-5.5")
    return OpenAICompatibleModelClient(
        model=config.model,
        base_url=config.base_url,
        api_key=config.api_key,
        temperature=0.0,
        timeout=300,
        strict_tools=config.strict_tools,
        reasoning_effort=config.reasoning_effort,
    )


def bootstrap_ci(values, *, samples=5000, seed=20260806):
    if not values:
        return [None, None]
    rng = random.Random(seed)
    means = []
    for _ in range(samples):
        means.append(statistics.mean(rng.choice(values) for _ in values))
    means.sort()
    return [
        round(means[int(0.025 * (samples - 1))], 6),
        round(means[int(0.975 * (samples - 1))], 6),
    ]


def paired_metrics(rows):
    index = {(r["task_id"], int(r["repeat"]), r["variant"]): r for r in rows}
    pairs = []
    for task_id, repeat, _ in sorted(
        {(r["task_id"], int(r["repeat"]), "") for r in rows}
    ):
        control = index.get((task_id, repeat, "no_context_reduction"))
        treatment = index.get((task_id, repeat, "full_orchestrator"))
        if not control or not treatment:
            continue
        cu, tu = control["usage"], treatment["usage"]
        cb = max(0, int(cu["input_tokens"]) - int(cu["cached_tokens"]))
        tb = max(0, int(tu["input_tokens"]) - int(tu["cached_tokens"]))
        ct, tt = int(cu["input_tokens"]), int(tu["input_tokens"])
        quality_regression = (
            control["verification_status"] == "passed"
            and treatment["verification_status"] != "passed"
        ) or (control["status"] == "completed" and treatment["status"] != "completed")
        pairs.append(
            {
                "task_id": task_id,
                "repeat": repeat,
                "control_billable_input": cb,
                "treatment_billable_input": tb,
                "billable_delta_pct": ((tb - cb) / cb) if cb else None,
                "total_delta_pct": ((tt - ct) / ct) if ct else None,
                "control_verification": control["verification_status"],
                "treatment_verification": treatment["verification_status"],
                "quality_regression": quality_regression,
                "break_even": tb < cb and not quality_regression,
                "usage_complete": cu["usage_source"] == "actual"
                and tu["usage_source"] == "actual",
            }
        )
    valid = [
        p for p in pairs if p["usage_complete"] and p["billable_delta_pct"] is not None
    ]
    billable = [p["billable_delta_pct"] for p in valid]
    total = [p["total_delta_pct"] for p in valid if p["total_delta_pct"] is not None]
    return {
        "pair_count": len(pairs),
        "actual_usage_pair_count": len(valid),
        "usage_completeness": len(valid) / len(pairs) if pairs else 0,
        "mean_billable_delta_pct": statistics.mean(billable) if billable else None,
        "median_billable_delta_pct": statistics.median(billable) if billable else None,
        "billable_delta_95ci": bootstrap_ci(billable),
        "mean_total_delta_pct": statistics.mean(total) if total else None,
        "total_delta_95ci": bootstrap_ci(total),
        "quality_regression_count": sum(p["quality_regression"] for p in pairs),
        "break_even_pair_count": sum(p["break_even"] for p in pairs),
        "break_even_pair_rate": sum(p["break_even"] for p in pairs) / len(pairs)
        if pairs
        else 0,
        "claimable_reduction": bool(
            valid
            and len(valid) / len(pairs) >= 0.95
            and not any(p["quality_regression"] for p in pairs)
            and bootstrap_ci(billable)[1] < 0
        ),
        "pairs": pairs,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output-dir", default=str(OUT))
    args = parser.parse_args()
    tasks = json.loads(TASKS.read_text(encoding="utf-8"))["tasks"]
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    partial = out / "rows.partial.json"
    rows = json.loads(partial.read_text(encoding="utf-8")) if partial.is_file() else []
    completed = {(r["task_id"], int(r["repeat"]), r["variant"]) for r in rows}
    for repeat in range(args.repetitions):
        for task in tasks:
            for variant in VARIANTS:
                key = (task["id"], repeat, variant)
                if key in completed:
                    continue
                print(
                    json.dumps(
                        {
                            "event": "start",
                            "task": task["id"],
                            "repeat": repeat,
                            "variant": variant,
                        }
                    ),
                    flush=True,
                )
                row = _run_long_session_task(
                    dict(task),
                    variant=variant,
                    repeat=repeat,
                    mode="live",
                    provider="openai",
                    provider_client_factory=client_factory,
                    output_dir=out / "work",
                    pricing=DEFAULT_PROXY_PRICING,
                )
                rows.append(row.to_dict())
                partial.write_text(
                    json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                print(
                    json.dumps(
                        {
                            "event": "done",
                            "task": task["id"],
                            "repeat": repeat,
                            "variant": variant,
                            "status": row.status,
                            "verification": row.verification_status,
                        }
                    ),
                    flush=True,
                )
    payload = build_result_payload(
        [_row_from_dict(r) for r in rows],
        pricing_profile="gpt-5.5-live-configured",
        pricing=DEFAULT_PROXY_PRICING,
        treatment="full_orchestrator",
        control="no_context_reduction",
    )
    metrics = paired_metrics(payload["rows"])
    payload["formal_metrics"] = metrics
    (out / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "report.md").write_text(
        generate_report(payload)
        + "\n\n## Formal paired inference\n\n```json\n"
        + json.dumps(
            {k: v for k, v in metrics.items() if k != "pairs"},
            ensure_ascii=False,
            indent=2,
        )
        + "\n```\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "results": str(out / "results.json"),
                "formal_metrics": {k: v for k, v in metrics.items() if k != "pairs"},
            },
            ensure_ascii=False,
        )
    )


def _row_from_dict(payload):
    from lite.evaluation.context_cost import CostUsage, ExperimentRow

    data = dict(payload)
    data["usage"] = CostUsage(**data["usage"])
    return ExperimentRow(**data)


if __name__ == "__main__":
    main()
