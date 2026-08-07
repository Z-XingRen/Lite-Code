"""UTF-8 byte/line budgets and preview selection for tool output."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ToolOutputLimits:
    max_bytes: int
    max_lines: int


DEFAULT_TOOL_OUTPUT_LIMITS = ToolOutputLimits(max_bytes=4096, max_lines=200)
TOOL_OUTPUT_LIMITS = {
    "inspect_image": ToolOutputLimits(max_bytes=12000, max_lines=400),
}

_HEAD_TOOLS = frozenset({"list_files", "read_file", "search"})
_EXIT_CODE_PATTERN = re.compile(r"(?:^|\n)exit_code:\s*(-?\d+)", re.IGNORECASE)
_NOTICE_BYTE_RESERVE = 768
_NOTICE_LINES = 4


def describe_tool_output(text, limits):
    payload = text.encode("utf-8")
    original_lines = _line_count(text)
    truncated = len(payload) > limits.max_bytes or original_lines > limits.max_lines
    return {
        "original_chars": len(text),
        "original_bytes": len(payload),
        "original_lines": original_lines,
        "content_sha256": hashlib.sha256(payload).hexdigest(),
        "full_output_artifact": "",
        "artifact_retention": "run",
        "truncated": truncated,
        "truncation_strategy": "full",
        "omitted_bytes": 0,
        "omitted_lines": 0,
        "max_bytes": limits.max_bytes,
        "max_lines": limits.max_lines,
        "inline_bytes": len(payload),
        "inline_lines": original_lines,
    }


def truncate_tool_output(text, *, name, limits, artifact_ref):
    strategy = _truncation_strategy(name, text)
    observation, omitted_bytes, omitted_lines = _render_preview(
        text,
        strategy=strategy,
        limits=limits,
        artifact_ref=artifact_ref,
    )
    return observation, {
        "truncation_strategy": strategy,
        "omitted_bytes": omitted_bytes,
        "omitted_lines": omitted_lines,
        "inline_bytes": len(observation.encode("utf-8")),
        "inline_lines": _line_count(observation),
    }


def _truncation_strategy(name, text):
    if _looks_like_error(text):
        return "tail"
    if name in _HEAD_TOOLS:
        return "head"
    return "head_tail"


def _looks_like_error(text):
    lowered = text.lower().lstrip()
    if lowered.startswith("error:") or "traceback (most recent call last):" in lowered:
        return True
    return any(int(code) != 0 for code in _EXIT_CODE_PATTERN.findall(text))


def _render_preview(text, *, strategy, limits, artifact_ref):
    max_content_lines = max(0, limits.max_lines - _NOTICE_LINES)
    content_byte_budget = max(0, limits.max_bytes - _NOTICE_BYTE_RESERVE)
    while True:
        head, tail, retained_bytes, retained_lines = _select_content(
            text,
            strategy=strategy,
            max_bytes=content_byte_budget,
            max_lines=max_content_lines,
        )
        omitted_bytes = max(0, len(text.encode("utf-8")) - retained_bytes)
        omitted_lines = max(0, _line_count(text) - retained_lines)
        observation = _format_preview(
            head,
            tail,
            strategy=strategy,
            artifact_ref=artifact_ref,
            omitted_bytes=omitted_bytes,
            omitted_lines=omitted_lines,
        )
        byte_overflow = len(observation.encode("utf-8")) - limits.max_bytes
        line_overflow = _line_count(observation) - limits.max_lines
        if byte_overflow <= 0 and line_overflow <= 0:
            return observation, omitted_bytes, omitted_lines
        next_byte_budget = max(0, content_byte_budget - max(1, byte_overflow))
        next_line_budget = max(0, max_content_lines - max(0, line_overflow))
        budgets_unchanged = next_byte_budget == content_byte_budget
        budgets_unchanged &= next_line_budget == max_content_lines
        if budgets_unchanged:
            raise ValueError("tool output limits are too small for the truncation notice")
        content_byte_budget = next_byte_budget
        max_content_lines = next_line_budget


def _select_content(text, *, strategy, max_bytes, max_lines):
    lines = text.splitlines()
    if len(lines) <= 1:
        return _select_single_line(text, strategy, max_bytes)
    return _select_complete_lines(lines, strategy, max_bytes, max_lines)


def _select_single_line(text, strategy, max_bytes):
    if strategy == "head":
        head = [_utf8_prefix(text, max_bytes)] if max_bytes else []
        tail = []
    elif strategy == "tail":
        head = []
        tail = [_utf8_suffix(text, max_bytes)] if max_bytes else []
    else:
        head_budget = max_bytes // 2
        tail_budget = max_bytes - head_budget
        head = [_utf8_prefix(text, head_budget)] if head_budget else []
        tail = [_utf8_suffix(text, tail_budget)] if tail_budget else []
    head = [part for part in head if part]
    tail = [part for part in tail if part]
    retained = sum(len(part.encode("utf-8")) for part in [*head, *tail])
    return head, tail, retained, 1 if head or tail else 0


def _select_complete_lines(lines, strategy, max_bytes, max_lines):
    if strategy == "head":
        head_indices = _take_head_indices(lines, max_bytes, max_lines)
        tail_indices = []
    elif strategy == "tail":
        head_indices = []
        tail_indices = _take_tail_indices(lines, max_bytes, max_lines)
    else:
        head_indices = _take_head_indices(lines, max_bytes // 2, max_lines // 2)
        head_bytes = _joined_line_bytes(lines, head_indices)
        tail_indices = _take_tail_indices(
            lines,
            max_bytes - head_bytes,
            max_lines - len(head_indices),
            excluded=set(head_indices),
        )
    head = [lines[index] for index in head_indices]
    tail = [lines[index] for index in tail_indices]
    retained = _joined_line_bytes(lines, head_indices) + _joined_line_bytes(
        lines, tail_indices
    )
    return head, tail, retained, len(set(head_indices + tail_indices))


def _take_head_indices(lines, max_bytes, max_lines):
    selected = []
    for index in range(len(lines)):
        if len(selected) >= max_lines:
            break
        candidate = selected + [index]
        if _joined_line_bytes(lines, candidate) > max_bytes:
            break
        selected = candidate
    return selected


def _take_tail_indices(lines, max_bytes, max_lines, excluded=None):
    excluded = excluded or set()
    selected = []
    for index in range(len(lines) - 1, -1, -1):
        if index in excluded:
            continue
        if len(selected) >= max_lines:
            break
        candidate = [index] + selected
        if _joined_line_bytes(lines, candidate) > max_bytes:
            break
        selected = candidate
    return selected


def _joined_line_bytes(lines, indices):
    if not indices:
        return 0
    return len("\n".join(lines[index] for index in indices).encode("utf-8"))


def _format_preview(head, tail, *, strategy, artifact_ref, omitted_bytes, omitted_lines):
    if artifact_ref:
        header = f"full output saved: {artifact_ref}"
        footer = "use read_file with the artifact path above to inspect the full output."
    else:
        header = "full output unavailable; preview only"
        footer = "rerun the tool with a narrower range or query to inspect omitted output."
    marker = f"...[omitted {omitted_bytes} bytes, {omitted_lines} lines]..."
    label = strategy.replace("_", "+")
    return "\n".join([header, f"preview ({label}):", *head, marker, *tail, footer])


def _utf8_prefix(text, max_bytes):
    return text.encode("utf-8")[:max_bytes].decode("utf-8", errors="ignore")


def _utf8_suffix(text, max_bytes):
    payload = text.encode("utf-8")
    return payload[-max_bytes:].decode("utf-8", errors="ignore") if max_bytes else ""


def _line_count(text):
    return len(str(text).splitlines()) if text else 0
