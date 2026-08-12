"""Thin result adapter for fixed real-task evaluations."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
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
    "cached_tokens",
    "billable_input_tokens",
    "cache_projection_reused",
    "cache_projection_reason",
    "cache_projection_generation",
    "cache_projection_message_count",
    "cache_projection_chars",
    "cache_projection_context_refreshed",
    "provider_prompt_chars",
    "output_tokens",
    "tool_call_count",
    "duplicate_tool_result_count",
    "checkpoint_count",
    "persistence_write_count",
    "wall_time",
    "final_stop_reason",
    "failure_category",
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
    projection = _cache_projection_from_evidence(evidence)
    usage = dict(trial.get("usage", {}) or {})
    model_calls = int(usage.get("model_call_count", 0) or 0)
    input_tokens = int(usage.get("input_tokens", 0) or 0)
    cached_tokens = int(usage.get("cached_tokens", 0) or 0)
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
        "input_tokens": input_tokens,
        "cached_tokens": cached_tokens,
        "billable_input_tokens": max(0, input_tokens - cached_tokens),
        **projection,
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
        "failure_category": str(
            trial.get("failure_category", "incomplete_evidence")
        ),
        "errors": list(trial.get("errors", []) or []),
        "usage_source": str(usage.get("usage_source", "none")),
    }


def _cache_projection_from_evidence(evidence):
    """Read turn-level cache projection evidence with a trace fallback."""

    summary = dict(
        (evidence.report.get("evidence_summaries", {}) or {}).get(
            "context_budget_summary", {}
        )
        or {}
    )
    report_metadata = dict(evidence.report.get("prompt_metadata", {}) or {})
    trace_metadata = {}
    for event in evidence.trace_events:
        if event.get("event") == "prompt_built":
            trace_metadata = dict(event.get("prompt_metadata", {}) or {})
    values = {}
    defaults = {
        "cache_projection_reused": False,
        "cache_projection_reason": "none",
        "cache_projection_generation": 0,
        "cache_projection_message_count": 0,
        "cache_projection_chars": 0,
        "cache_projection_context_refreshed": False,
        "provider_prompt_chars": 0,
    }
    for key, default in defaults.items():
        if key in summary:
            values[key] = summary[key]
        elif key in report_metadata:
            values[key] = report_metadata[key]
        else:
            values[key] = trace_metadata.get(key, default)
    values["cache_projection_reused"] = bool(values["cache_projection_reused"])
    values["cache_projection_reason"] = str(values["cache_projection_reason"] or "none")
    for key in (
        "cache_projection_generation",
        "cache_projection_message_count",
        "cache_projection_chars",
        "provider_prompt_chars",
    ):
        values[key] = int(values[key] or 0)
    values["cache_projection_context_refreshed"] = bool(
        values["cache_projection_context_refreshed"]
    )
    return values


def result_matrix_keys(tasks, variants, repetitions):
    """Return the fixed task, repeat, and variant keys for one evaluation."""

    return frozenset(
        (str(task["id"]), repeat, str(variant))
        for repeat in range(int(repetitions))
        for task in tasks
        for variant in variants
    )


def validate_result_matrix(rows, expected_keys, *, require_complete=False):
    """Reject ambiguous rows and report whether the expected matrix is complete."""

    expected = frozenset(expected_keys)
    counts = Counter(
        (
            str(row.get("task_id", "")),
            int(row.get("repeat", 0)),
            str(row.get("variant", "")),
        )
        for row in rows
    )
    seen = set(counts)
    duplicates = {key for key, count in counts.items() if count > 1}
    if duplicates:
        raise ValueError(f"duplicate result rows: {_format_keys(duplicates)}")
    unexpected = seen - expected
    if unexpected:
        raise ValueError(f"unexpected result rows: {_format_keys(unexpected)}")
    missing = expected - seen
    if require_complete and missing:
        raise ValueError(f"missing result rows: {_format_keys(missing)}")
    return {
        "complete": not missing,
        "completed_count": len(seen),
        "expected_count": len(expected),
        "missing_keys": missing,
    }


def write_results(rows, output_dir, *, expected_keys=None, require_complete=False):
    """Atomically write raw rows and a completeness-bound Markdown summary."""

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
    matrix = None
    if expected_keys is not None:
        matrix = validate_result_matrix(
            ordered,
            expected_keys,
            require_complete=require_complete,
        )
    jsonl = output_dir / "results.jsonl"
    jsonl_text = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        for row in ordered
    )
    _atomic_write_text(jsonl, jsonl_text)
    results_digest = hashlib.sha256(jsonl_text.encode("utf-8")).hexdigest()
    markdown = output_dir / "summary.md"
    if matrix is not None and not matrix["complete"]:
        lines = [
            "# Lite Real Task Evaluation (Incomplete)",
            "",
            (
                f"Status: {matrix['completed_count']}/{matrix['expected_count']} "
                "result rows are present. This artifact is not a valid baseline."
            ),
            "",
            f"Results SHA-256: `sha256:{results_digest}`",
        ]
        _atomic_write_text(markdown, "\n".join(lines) + "\n")
        return {"jsonl": jsonl, "markdown": markdown, "complete": False}

    columns = ("task_id", "variant", "repeat", *METRIC_FIELDS)
    lines = [
        "# Lite Real Task Evaluation",
        "",
    ]
    if matrix is not None:
        lines.extend(
            [
                (
                    f"Status: Complete, {matrix['completed_count']}/"
                    f"{matrix['expected_count']} result rows are present."
                ),
                "",
                f"Results SHA-256: `sha256:{results_digest}`",
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
    _atomic_write_text(markdown, "\n".join(lines) + "\n")
    return {"jsonl": jsonl, "markdown": markdown, "complete": True}


def _atomic_write_text(path, content):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8", newline="")
    temporary.replace(path)


def _format_keys(keys):
    return ", ".join(
        f"{task_id}/{repeat}/{variant}"
        for task_id, repeat, variant in sorted(keys)
    )


def _markdown_cell(value):
    if isinstance(value, (list, dict)):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value).replace("|", "\\|").replace("\n", " ")
