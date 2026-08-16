"""Per-run governance evidence for tool decisions.

Tool execution records allow, warn, and deny decisions here so reports can
explain what the runtime permitted or blocked. This module summarizes decisions
but does not enforce policy itself.
"""

GOVERNANCE_SUMMARY_SCHEMA = "lite.governance_summary.v2"

HARD_SAFETY_DENIAL_REASONS = {
    "hard_deny",
    "plan_mode_path_mismatch",
    "plan_mode_tool_not_allowed",
    "read_only_violation",
    "sandbox_rejected_command",
    "scope_violation",
    "secret_exposure",
    "unsafe_command_executed",
    "worker_scope_violation",
    "workspace_escape",
    "write_scope_mismatch",
    "write_scope_shell_blocked",
    "write_scope_verify_expansion_blocked",
}
HARD_SAFETY_EVENT_TYPES = HARD_SAFETY_DENIAL_REASONS | {
    "path_escape",
    "plan_mode_write_guard",
    "read_only_block",
    "write_scope_guard",
}
RECOVERABLE_POLICY_DENIAL_REASONS = {
    "invalid_arguments",
    "prior_read_required",
    "repeated_identical_call",
    "shell_search_should_use_tool",
}
_GENERIC_POLICY_SECURITY_EVENTS = {"", "approval_denied", "tool_policy"}


def record_governance_decision(
    agent,
    tool_name,
    args,
    *,
    decision,
    reason_code,
    decision_type,
    original_reason="",
    security_event_type="",
    effects=None,
    source="tool_executor",
):
    task_state = getattr(agent, "current_task_state", None)
    if task_state is None:
        return None
    tool = getattr(agent, "tools", {}).get(str(tool_name))
    defer_projection = bool(
        getattr(agent, "feature_enabled", lambda _name: False)(
            "journal_checkpoint_policy"
        )
        and getattr(tool, "read_only", False)
    )
    payload = {
            "decision": str(decision),
            "decision_type": str(decision_type),
            "reason_code": str(reason_code),
            "original_reason": str(original_reason or reason_code),
            "security_event_type": str(security_event_type),
            "effects": list(effects or []),
            "tool_name": str(tool_name),
            "tool_profile": getattr(agent.active_tool_profile, "name", ""),
            "read_only": bool(getattr(agent, "read_only", False)),
            "args": args or {},
            "source": source,
    }
    if defer_projection:
        deferred = task_state.evidence_summaries.setdefault(
            "deferred_governance_decisions", []
        )
        deferred.append(
            {
                key: payload[key]
                for key in ("decision", "decision_type", "reason_code", "security_event_type")
            }
        )
        return payload
    return agent.emit_trace(task_state, "governance_decision", payload)


def reduce_governance_summary(summary, event):
    summary = dict(summary or {})
    summary["schema_version"] = GOVERNANCE_SUMMARY_SCHEMA
    decision = str(event.get("decision", ""))
    reason = str(event.get("reason_code", ""))
    decision_type = str(event.get("decision_type", ""))
    key = f"{decision}_count"
    summary[key] = int(summary.get(key, 0) or 0) + 1
    for missing in ("allow_count", "deny_count", "warn_count"):
        summary.setdefault(missing, 0)
    summary.setdefault("hard_safety_denial_count", 0)
    summary.setdefault("recoverable_policy_denial_count", 0)
    summary.setdefault("unclassified_policy_denial_count", 0)
    type_counts = dict(summary.get("decision_type_counts", {}) or {})
    type_counts[decision_type] = type_counts.get(decision_type, 0) + 1
    summary["decision_type_counts"] = type_counts
    reasons = dict(summary.get("reasons", {}) or {})
    reasons[reason] = reasons.get(reason, 0) + 1
    summary["reasons"] = reasons
    if decision == "deny":
        summary["last_denied_reason"] = reason
        denial_class = governance_denial_class(event)
        count_key = f"{denial_class}_denial_count"
        summary[count_key] = int(summary.get(count_key, 0) or 0) + 1
        reason_key = f"{denial_class}_denial_reasons"
        classified_reasons = dict(summary.get(reason_key, {}) or {})
        classified_reasons[reason] = classified_reasons.get(reason, 0) + 1
        summary[reason_key] = classified_reasons
        summary[f"last_{denial_class}_denial_reason"] = reason
    return summary


def governance_denial_class(event):
    """Classify denials through explicit lists and fail closed on unknown safety."""

    reason = str(event.get("reason_code", ""))
    security_event = str(event.get("security_event_type", ""))
    if reason in HARD_SAFETY_DENIAL_REASONS:
        return "hard_safety"
    if security_event in HARD_SAFETY_EVENT_TYPES:
        return "hard_safety"
    if reason in RECOVERABLE_POLICY_DENIAL_REASONS:
        # A validation error can still carry path_escape or another hard event;
        # hard events are checked first so the recoverable reason cannot mask it.
        return "recoverable_policy"
    if security_event not in _GENERIC_POLICY_SECURITY_EVENTS:
        return "hard_safety"
    return "unclassified_policy"
