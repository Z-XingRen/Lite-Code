"""Staged tool authorization and observation outside the control loop."""

import re
from dataclasses import dataclass, field

from .governance import record_governance_decision
from .tool_failures import classify_tool_failure
from .tool_policy import ToolPolicyChecker
from .tool_repetition import repeated_tool_call_metadata
from .tool_result_artifacts import prepare_tool_result_observation

_COMMAND_TOOLS = frozenset({"run_shell", "verify"})


@dataclass
class PreparedToolCall:
    name: str
    args: dict
    call_id: str
    tool: object | None
    ready: bool = False
    result: str = ""
    metadata: dict = field(default_factory=dict)
    tracking_token: object | None = None
    before_snapshot: object | None = None


def prepare_tool_call(agent, name, args, *, call_id=""):
    """Run ordered lookup, validation, authorization, and tracking setup."""

    args = args or {}
    tool = agent.tools.get(name)
    prepared = PreparedToolCall(name, args, str(call_id or ""), tool)
    if tool is None:
        prepared.result = f"error: unknown tool '{name}'"
        prepared.metadata = tool_result_metadata(
            None,
            status="rejected",
            error_code="unknown_tool",
            risk_level="high",
            read_only=False,
        )
        record_governance_decision(
            agent,
            name,
            args,
            decision="deny",
            reason_code="unknown_tool",
            decision_type="tool_lookup",
        )
        return prepared
    try:
        agent.validate_tool(name, args)
    except Exception as exc:
        example = agent.tool_example(name)
        prepared.result = f"error: invalid arguments for {name}: {exc}"
        if example:
            prepared.result += f"\nexample: {example}"
        security_event = "path_escape" if "path escapes workspace" in str(exc) else ""
        prepared.metadata = tool_result_metadata(
            tool,
            status="rejected",
            error_code="invalid_arguments",
            security_event_type=security_event,
        )
        record_governance_decision(
            agent,
            name,
            args,
            decision="deny",
            reason_code="invalid_arguments",
            decision_type="tool_validation",
            original_reason=str(exc),
            security_event_type=security_event,
        )
        return prepared
    if agent.repeated_tool_call(name, args):
        prepared.result = (
            f"error: repeated identical tool call for {name}; "
            "choose a different tool or return a final answer"
        )
        prepared.metadata = repeated_tool_call_metadata(tool)
        record_governance_decision(
            agent,
            name,
            args,
            decision="deny",
            reason_code="repeated_identical_call",
            decision_type="tool_repetition",
        )
        return prepared
    decision = agent.permission_checker.check(tool, args, call_id=call_id or None)
    emit_permission_decision(agent, tool, args, decision)
    permission_reason = (
        "read_only_violation"
        if not decision.allowed and getattr(agent, "read_only", False)
        else decision.reason
    )
    record_governance_decision(
        agent,
        name,
        args,
        decision=decision.decision,
        reason_code=permission_reason,
        decision_type="permission",
        original_reason=decision.reason,
        security_event_type=decision.security_event_type
        or ("read_only_block" if permission_reason == "read_only_violation" else ""),
    )
    if not decision.allowed:
        prepared.result = permission_error(agent, tool, decision)
        prepared.metadata = tool_result_metadata(
            tool,
            status="rejected",
            error_code=decision.reason,
            security_event_type=decision.security_event_type,
        )
        return prepared
    policy = ToolPolicyChecker(agent).check(tool, args)
    emit_tool_policy_decision(agent, tool, args, policy)
    record_governance_decision(
        agent,
        name,
        args,
        decision=policy.decision,
        reason_code=policy.reason,
        decision_type="tool_policy",
        original_reason=policy.reason,
        security_event_type="tool_policy" if not policy.allowed else "",
    )
    if not policy.allowed:
        prepared.result = policy.message
        prepared.metadata = tool_result_metadata(
            tool,
            status="rejected",
            error_code=policy.reason,
            security_event_type="tool_policy",
        )
        agent.record_process_note_for_tool(name, prepared.metadata)
        return prepared
    try:
        prepared.tracking_token, prepared.before_snapshot = (
            agent.prepare_workspace_change(tool, args)
        )
    except Exception as exc:
        prepared.result, prepared.metadata = finish_tool_call(
            agent, prepared, error=exc, consume_pending=False
        )
        return prepared
    prepared.ready = True
    return prepared


def finish_tool_call(
    agent, prepared, *, execution_result=None, error=None, consume_pending=True
):
    """Create the ordered observation for an already executed tool effect."""

    tool = prepared.tool
    name = prepared.name
    try:
        if error is not None:
            raise error
        full_result = execution_result.content
        pending_metadata = {}
        if consume_pending:
            pending_metadata = dict(
                getattr(agent, "_pending_tool_result_metadata", {}) or {}
            )
            agent._pending_tool_result_metadata = {}
        exit_code = run_shell_exit_code(full_result) if name in _COMMAND_TOOLS else 0
        result, artifact_metadata = prepare_tool_result_observation(
            agent, name, full_result
        )
        affected_paths, diff_summary = agent.complete_workspace_change(
            tool,
            prepared.tracking_token,
            prepared.before_snapshot,
            full_result,
        )
        workspace_changed = bool(affected_paths)
        status, error_code = "ok", ""
        if name in _COMMAND_TOOLS and exit_code != 0:
            status = "partial_success" if workspace_changed else "error"
            error_code = "tool_partial_success" if workspace_changed else "tool_failed"
        agent.update_memory_after_tool(name, prepared.args, result)
        metadata = tool_result_metadata(
            tool,
            status=status,
            error_code=error_code,
            affected_paths=affected_paths,
            workspace_changed=workspace_changed,
            workspace_fingerprint=agent.workspace.fingerprint(),
            diff_summary=diff_summary,
            **artifact_metadata,
            **pending_metadata,
            **getattr(agent, "_last_workspace_tracking_metadata", {}),
        )
        agent.record_process_note_for_tool(name, metadata)
        return result, metadata
    except Exception as exc:
        affected_paths, diff_summary = agent.complete_workspace_change(
            tool, prepared.tracking_token, prepared.before_snapshot
        )
        workspace_changed = bool(affected_paths)
        security_event = "path_escape" if "path escapes workspace" in str(exc) else ""
        if name in _COMMAND_TOOLS and "sandbox required but unavailable" in str(exc):
            record_governance_decision(
                agent,
                name,
                prepared.args,
                decision="deny",
                reason_code="sandbox_rejected_command",
                decision_type="sandbox",
                original_reason=str(exc),
                security_event_type="sandbox",
            )
        metadata = tool_result_metadata(
            tool,
            **classify_tool_failure(exc, workspace_changed=workspace_changed),
            security_event_type=security_event,
            affected_paths=affected_paths,
            workspace_changed=workspace_changed,
            workspace_fingerprint=agent.workspace.fingerprint(),
            diff_summary=diff_summary,
            **getattr(agent, "_last_workspace_tracking_metadata", {}),
        )
        agent.record_process_note_for_tool(name, metadata)
        return f"error: tool {name} failed: {exc}", metadata


def run_shell_exit_code(result):
    match = re.search(r"exit_code:\s*(-?\d+)", str(result))
    return int(match.group(1)) if match else 0


def tool_result_metadata(
    tool,
    *,
    status,
    error_code="",
    security_event_type="",
    risk_level=None,
    read_only=None,
    affected_paths=None,
    workspace_changed=False,
    workspace_fingerprint=None,
    diff_summary=None,
    **extra,
):
    metadata = {
        "tool_status": status,
        "tool_error_code": error_code,
        "security_event_type": security_event_type,
        "risk_level": risk_level
        if risk_level is not None
        else ("high" if tool.risky else "low"),
        "read_only": read_only if read_only is not None else tool.read_only,
        "affected_paths": list(affected_paths or []),
        "workspace_changed": bool(workspace_changed),
        "diff_summary": list(diff_summary or []),
        **extra,
    }
    if workspace_fingerprint is not None:
        metadata["workspace_fingerprint"] = workspace_fingerprint
    return metadata


def emit_permission_decision(agent, tool, args, decision):
    if _defer_read_only_projection(agent, tool):
        return
    agent.session_event_bus.emit(
        "permission_decision",
        {
            "tool_name": tool.name,
            "decision": decision.decision,
            "reason": decision.reason,
            "security_event_type": decision.security_event_type,
            "tool_profile": agent.active_tool_profile.name,
            "args": args or {},
        },
    )


def emit_tool_policy_decision(agent, tool, args, decision):
    if _defer_read_only_projection(agent, tool):
        return
    agent.session_event_bus.emit(
        "tool_policy_decision",
        {
            "tool_name": tool.name,
            "decision": decision.decision,
            "reason": decision.reason,
            "args": args or {},
        },
    )


def _defer_read_only_projection(agent, tool):
    return bool(
        getattr(agent, "feature_enabled", lambda _name: False)(
            "journal_checkpoint_policy"
        )
        and getattr(tool, "read_only", False)
    )


def permission_error(agent, tool, decision):
    if decision.reason == "plan_mode_path_mismatch":
        return (
            "error: plan mode can only write the active plan artifact "
            f"({agent.plan_mode.plan_path})"
        )
    if decision.reason == "plan_mode_tool_not_allowed":
        return (
            "error: plan mode only allows read-only tools or writing the active "
            f"plan artifact ({agent.plan_mode.plan_path})"
        )
    if decision.reason == "write_scope_mismatch":
        return f"error: worker write_scope does not allow {tool.name} on this path"
    if decision.reason in {"approval_denied", "tool_not_allowed"}:
        return f"error: approval denied for {tool.name}"
    return f"error: permission denied for {tool.name}: {decision.reason}"
