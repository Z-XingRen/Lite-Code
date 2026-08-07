from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.formal_eval.run_lite_quality_v1 import (  # noqa: E402
    ROOT,
    load_manifest,
    provider_metadata,
    run_trial,
)

OUT = ROOT / "artifacts/formal-evaluation-20260806/multi-agent-ab"


def trace_metrics(workspace):
    reads = []
    writers = defaultdict(set)
    for trace in Path(workspace).glob(".lite/runs/*/trace.jsonl"):
        for line in trace.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("event") != "tool_executed":
                continue
            if event.get("name") == "read_file":
                reads.append(str((event.get("args") or {}).get("path", "")))
            if event.get("name") in {"write_file", "patch_file"}:
                for path in event.get("affected_paths", []) or []:
                    writers[str(path)].add(str(event.get("turn_id", "")))
    counts = Counter(path for path in reads if path)
    return {
        "read_count": len(reads),
        "duplicate_read_count": sum(max(0, count - 1) for count in counts.values()),
        "write_conflict_count": sum(1 for turns in writers.values() if len(turns) > 1),
        "conflicted_paths": sorted(
            path for path, turns in writers.items() if len(turns) > 1
        ),
    }


def variant_task(task, variant):
    item = dict(task)
    item["required_trace_events"] = []
    if variant == "single":
        item["allowed_tools"] = [
            name
            for name in task["allowed_tools"]
            if name not in {"agent", "send_message", "task_stop"}
        ]
        item["prompt"] = (
            task["prompt"]
            + " Solve this in the parent agent without launching a subagent."
        )
    else:
        scopes = json.dumps(task["expected_changed_paths"])
        item["allowed_tools"] = list(
            dict.fromkeys(
                list(task["allowed_tools"]) + ["agent", "send_message", "task_stop"]
            )
        )
        item["step_budget"] = int(task["step_budget"]) + 5
        item["required_trace_events"] = ["worker_started"]
        item["prompt"] = (
            "Delegate repository inspection and implementation to one worker subagent using write_scope="
            + scopes
            + ". "
            "After the worker reports, the parent must inspect the result, run the tests, make only necessary corrections, and final. "
            + task["prompt"]
        )
    return item


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output-dir", default=str(OUT))
    parser.add_argument("--retry-non-actual", action="store_true")
    args = parser.parse_args()
    manifest = load_manifest()
    tasks = manifest["tasks"][:8]
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    partial = out / "rows.partial.json"
    rows = json.loads(partial.read_text(encoding="utf-8")) if partial.is_file() else []
    if args.retry_non_actual:
        rows = [row for row in rows if row["usage"]["usage_source"] == "actual"]
        partial.write_text(
            json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    done = {(r["task_id"], int(r["repeat"]), r["variant"]) for r in rows}
    config = provider_metadata()
    for repeat in range(args.repetitions):
        for task in tasks:
            for variant in ("single", "multi"):
                key = (task["id"], repeat, variant)
                if key in done:
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
                row = run_trial(
                    variant_task(task, variant), repeat, out / variant, config
                )
                row["variant"] = variant
                row.update(trace_metrics(row["workspace"]))
                rows.append(row)
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
                            "scc": row["scc"],
                        }
                    ),
                    flush=True,
                )
    summary = {
        "task_count": len(tasks),
        "repetitions": args.repetitions,
        "run_count": len(rows),
        "variants": {},
    }
    for variant in ("single", "multi"):
        bucket = [r for r in rows if r["variant"] == variant]
        summary["variants"][variant] = {
            "n": len(bucket),
            "scc_rate": sum(r["scc"] for r in bucket) / len(bucket),
            "mean_input_tokens": statistics.mean(
                r["usage"]["input_tokens"] for r in bucket
            ),
            "mean_billable_input_tokens": statistics.mean(
                max(0, r["usage"]["input_tokens"] - r["usage"]["cached_tokens"])
                for r in bucket
            ),
            "mean_wall_time_ms": statistics.mean(r["wall_time_ms"] for r in bucket),
            "mean_duplicate_reads": statistics.mean(
                r["duplicate_read_count"] for r in bucket
            ),
            "write_conflict_rate": sum(r["write_conflict_count"] > 0 for r in bucket)
            / len(bucket),
            "scope_violation_rate": sum(not r["scope_pass"] for r in bucket)
            / len(bucket),
            "actual_usage_rate": sum(
                r["usage"]["usage_source"] == "actual" for r in bucket
            )
            / len(bucket),
        }
    pairs = []
    index = {(r["task_id"], r["repeat"], r["variant"]): r for r in rows}
    for task in tasks:
        for repeat in range(args.repetitions):
            s = index[(task["id"], repeat, "single")]
            m = index[(task["id"], repeat, "multi")]
            pairs.append(
                {
                    "task_id": task["id"],
                    "repeat": repeat,
                    "single_scc": s["scc"],
                    "multi_scc": m["scc"],
                    "billable_input_delta": max(
                        0, m["usage"]["input_tokens"] - m["usage"]["cached_tokens"]
                    )
                    - max(0, s["usage"]["input_tokens"] - s["usage"]["cached_tokens"]),
                    "wall_time_delta_ms": m["wall_time_ms"] - s["wall_time_ms"],
                    "duplicate_read_delta": m["duplicate_read_count"]
                    - s["duplicate_read_count"],
                }
            )
    payload = {"summary": summary, "pairs": pairs, "rows": rows}
    (out / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {"summary": summary, "results": str(out / "results.json")},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
