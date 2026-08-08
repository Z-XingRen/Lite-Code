"""Tool failure classification shared by the execution boundary."""

import subprocess

from ..cancellation import CancellationRequested


def classify_tool_failure(exc, *, workspace_changed):
    cancelled = isinstance(exc, CancellationRequested)
    timed_out = isinstance(exc, subprocess.TimeoutExpired)
    if cancelled:
        error_code = "tool_cancelled"
    elif timed_out:
        error_code = "tool_timeout"
    elif workspace_changed:
        error_code = "tool_partial_success"
    else:
        error_code = "tool_failed"
    return {
        "status": "partial_success" if workspace_changed else "error",
        "error_code": error_code,
        "cancellation_requested": cancelled,
        "timed_out": timed_out,
    }
