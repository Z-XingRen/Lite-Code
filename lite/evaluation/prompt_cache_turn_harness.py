"""Evidence adapter for fixed cross-turn prompt-cache evaluations."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from .run_evidence import RunEvidence


MANIFEST_PATH = Path("benchmarks/prompt_cache_turns_v1.json")
VARIANTS = ("full_prompt", "append_projection")
RESULT_FIELDS = (
    "behavior_pass",
    "usage_complete",
    "provider_cache_hit",
    "prompt_cache_key_stable",
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
        str(turn.get("prompt_cache_key", "")) for turn in turns
    ]
    prompt_cache_key_stable = bool(prompt_cache_keys) and len(set(prompt_cache_keys)) == 1
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


def write_results(rows, output_dir, *, expected_keys=None, require_complete=False):
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
    markdown_path = output_dir / "summary.md"
    if matrix is not None and not matrix["complete"]:
        text = (
            "# Prompt Cache Turn Evaluation (Incomplete)\n\n"
            f"Status: {matrix['completed_count']}/{matrix['expected_count']} result rows.\n\n"
            f"Results SHA-256: `sha256:{digest}`\n"
        )
        _atomic_write_text(markdown_path, text)
        return {"jsonl": jsonl_path, "markdown": markdown_path, "complete": False}

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
    return {"jsonl": jsonl_path, "markdown": markdown_path, "complete": True}


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
