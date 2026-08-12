"""Thin result adapter for fixed real-task evaluations."""

from __future__ import annotations

import json
from pathlib import Path

from .run_evidence import RunEvidence


MANIFEST_PATH = Path("benchmarks/real_tasks_v1.json")
METRIC_FIELDS = (
    "task_success",
    "verification_success",
    "scope_violation",
    "changed_paths",
    "model_call_count",
    "input_tokens",
    "output_tokens",
    "tool_call_count",
    "duplicate_tool_result_count",
    "checkpoint_count",
    "persistence_write_count",
    "wall_time",
    "final_stop_reason",
)


def load_manifest(root):
    """Load the fixed task selection and resolve its existing formal fixtures."""

    root = Path(root).resolve()
    manifest = json.loads((root / MANIFEST_PATH).read_text(encoding="utf-8"))
    source_path = root / manifest["source_manifest"]
    source = json.loads(source_path.read_text(encoding="utf-8"))
    by_id = {task["id"]: task for task in source["tasks"]}
    tasks = []
    for selection in manifest["tasks"]:
        source_id = selection["source_id"]
        if source_id not in by_id:
            raise ValueError(f"unknown real-task source id: {source_id}")
        tasks.append({**by_id[source_id], **selection})
    if not 12 <= len(tasks) <= 20:
        raise ValueError("real-task manifest must contain 12 to 20 tasks")
    if int(manifest.get("repetitions", 0)) < 3:
        raise ValueError("real-task repetitions must be at least 3")
    return {**manifest, "tasks": tasks}


def row_from_trial(task, trial, *, variant, repeat):
    """Project one completed formal trial into the stable harness schema."""

    evidence = RunEvidence.latest(Path(trial["workspace"]))
    events = evidence.trace_events
    usage = dict(trial.get("usage", {}) or {})
    model_calls = int(usage.get("model_call_count", 0) or 0)
    if not model_calls:
        model_calls = sum(event.get("event") == "model_requested" for event in events)
    persistence_writes = max(
        [int(event.get("persistence_write_count", 0) or 0) for event in events] or [0]
    )
    return {
        "task_id": task["id"],
        "scenario": task["scenario"],
        "variant": str(variant),
        "repeat": int(repeat),
        "task_success": bool(trial.get("scc")),
        "verification_success": bool(
            trial.get("target_pass") and trial.get("regression_pass")
        ),
        "scope_violation": not bool(trial.get("scope_pass")),
        "changed_paths": list(trial.get("changed_paths", []) or []),
        "model_call_count": model_calls,
        "input_tokens": int(usage.get("input_tokens", 0) or 0),
        "output_tokens": int(usage.get("output_tokens", 0) or 0),
        "tool_call_count": sum(
            event.get("event") == "tool_executed" for event in events
        ),
        "duplicate_tool_result_count": sum(
            int(event.get("duplicate_tool_result_count", 0) or 0)
            for event in events
            if event.get("event") == "prompt_built"
        ),
        "checkpoint_count": sum(
            event.get("event") == "checkpoint_created" for event in events
        ),
        "persistence_write_count": persistence_writes,
        "wall_time": round(float(trial.get("wall_time_ms", 0) or 0) / 1000, 6),
        "final_stop_reason": evidence.stop_reason(),
        "errors": list(trial.get("errors", []) or []),
        "usage_source": str(usage.get("usage_source", "none")),
    }


def write_results(rows, output_dir):
    """Write raw JSONL rows and a Markdown table without hiding task variance."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ordered = sorted(
        (dict(row) for row in rows),
        key=lambda row: (
            str(row.get("task_id", "")),
            str(row.get("variant", "")),
            int(row.get("repeat", 0)),
        ),
    )
    jsonl = output_dir / "results.jsonl"
    jsonl.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in ordered
        ),
        encoding="utf-8",
    )
    markdown = output_dir / "summary.md"
    columns = ("task_id", "variant", "repeat", *METRIC_FIELDS)
    lines = [
        "# Lite Real Task Evaluation",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in ordered:
        lines.append(
            "| "
            + " | ".join(_markdown_cell(row.get(column, "")) for column in columns)
            + " |"
        )
    markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"jsonl": jsonl, "markdown": markdown}


def _markdown_cell(value):
    if isinstance(value, (list, dict)):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value).replace("|", "\\|").replace("\n", " ")
