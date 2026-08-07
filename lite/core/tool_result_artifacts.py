"""Artifact persistence boundary for adaptive tool-result previews."""

from __future__ import annotations

from .tool_output_truncation import (
    DEFAULT_TOOL_OUTPUT_LIMITS,
    TOOL_OUTPUT_LIMITS,
    ToolOutputLimits,
    describe_tool_output,
    truncate_tool_output,
)

__all__ = [
    "DEFAULT_TOOL_OUTPUT_LIMITS",
    "TOOL_OUTPUT_LIMITS",
    "ToolOutputLimits",
    "prepare_tool_result_observation",
]


def prepare_tool_result_observation(agent, name, full_result):
    full_result = str(full_result)
    limits = TOOL_OUTPUT_LIMITS.get(name, DEFAULT_TOOL_OUTPUT_LIMITS)
    metadata = describe_tool_output(full_result, limits)
    if not metadata["truncated"]:
        return full_result, metadata

    task_state = getattr(agent, "current_task_state", None)
    artifact_ref = ""
    if task_state is not None:
        path = agent.run_store.write_text_artifact(
            task_state, f"{name}-output", full_result
        )
        artifact_ref = agent.run_store.artifact_ref(task_state, path)
        metadata["full_output_artifact"] = artifact_ref

    observation, truncation_metadata = truncate_tool_output(
        full_result,
        name=name,
        limits=limits,
        artifact_ref=artifact_ref,
    )
    metadata.update(truncation_metadata)
    return observation, metadata
