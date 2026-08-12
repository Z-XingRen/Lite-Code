from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from lite.evaluation.context_cost import (
    DEFAULT_PROXY_PRICING,
    _run_long_session_task,
    build_result_payload,
    generate_report,
)
try:
    from .run_lite_quality_v1 import (
        configured_temperature,
        make_client,
        provider_metadata,
    )
except ImportError:  # direct script execution
    from run_lite_quality_v1 import (
        configured_temperature,
        make_client,
        provider_metadata,
    )

ROOT = Path(__file__).resolve().parents[2]
TASKS = ROOT / "benchmarks/formal_v1/long_session_tasks_10.json"
OUT = ROOT / "artifacts/formal-evaluation-20260806/context-ab"
VARIANTS = ["no_context_reduction", "full_orchestrator"]


def client_factory(**_):
    return make_client(provider_metadata())


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
    minimum_claim_pairs = 10
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
        "minimum_claim_pairs": minimum_claim_pairs,
        "claimable_reduction": bool(
            len(valid) >= minimum_claim_pairs
            and len(valid) / len(pairs) >= 0.95
            and not any(p["quality_regression"] for p in pairs)
            and bootstrap_ci(billable)[1] < 0
        ),
        "pairs": pairs,
    }


def ensure_evaluation_identity(out, config, tasks, repetitions):
    identity = {
        "schema_version": "lite.context_ab_identity.v1",
        "model": config.model,
        "reasoning_effort": config.reasoning_effort,
        "protocol": config.protocol,
        "task_ids": [task["id"] for task in tasks],
        "repetitions": int(repetitions),
        "variants": list(VARIANTS),
        "task_source_sha256": "sha256:"
        + hashlib.sha256(TASKS.read_bytes()).hexdigest(),
    }
    manifest_path = out / "evaluation-manifest.json"
    partial_path = out / "rows.partial.json"
    if manifest_path.is_file():
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        if prior.get("identity") != identity:
            raise RuntimeError(
                "context A/B output identity changed; use a fresh output directory"
            )
        return
    adopted = False
    if partial_path.is_file():
        results_path = out / "results.json"
        if not results_path.is_file():
            raise RuntimeError(
                "context A/B partial rows have no evaluation identity or results"
            )
        prior_results = json.loads(results_path.read_text(encoding="utf-8"))
        prior_model = prior_results.get("model", {})
        observed_keys = {
            (row["task_id"], int(row["repeat"]), row["variant"])
            for row in prior_results.get("rows", [])
        }
        expected_keys = {
            (task["id"], repeat, variant)
            for task in tasks
            for repeat in range(int(repetitions))
            for variant in VARIANTS
        }
        if (
            prior_model.get("model") != config.model
            or prior_model.get("reasoning_effort") != config.reasoning_effort
            or observed_keys != expected_keys
        ):
            raise RuntimeError(
                "context A/B partial rows belong to a different evaluation identity"
            )
        adopted = True
    manifest_path.write_text(
        json.dumps(
            {
                "identity": identity,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "adopted_pre_identity_results": adopted,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output-dir", default=str(OUT))
    parser.add_argument("--task-ids", default="")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    tasks = json.loads(TASKS.read_text(encoding="utf-8"))["tasks"]
    requested = [
        item.strip() for item in args.task_ids.split(",") if item.strip()
    ]
    if requested:
        by_id = {task["id"]: task for task in tasks}
        missing = [task_id for task_id in requested if task_id not in by_id]
        if missing:
            raise ValueError(f"unknown context A/B task ids: {', '.join(missing)}")
        tasks = [by_id[task_id] for task_id in requested]
    if args.limit > 0:
        tasks = tasks[: args.limit]
    if not tasks:
        raise ValueError("context A/B task selection is empty")
    config = provider_metadata()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    ensure_evaluation_identity(out, config, tasks, args.repetitions)
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
                    provider=config.name,
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
        pricing_profile=f"{config.model}-live-configured",
        pricing=DEFAULT_PROXY_PRICING,
        treatment="full_orchestrator",
        control="no_context_reduction",
    )
    metrics = paired_metrics(payload["rows"])
    payload["formal_metrics"] = metrics
    payload["model"] = {
        "source": str(ROOT / ".lite.toml"),
        "provider": config.name,
        "protocol": config.protocol,
        "model": config.model,
        "reasoning_effort": config.reasoning_effort,
        "strict_tools": config.strict_tools,
        "temperature": configured_temperature(config),
        "base_url_hostname": urlparse(config.base_url).hostname,
        "api_key_present": bool(config.api_key),
    }
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
