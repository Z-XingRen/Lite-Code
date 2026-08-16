"""Final-readiness reason catalog.

Reason codes live here so user-facing messages, severities, and tests stay in
one place. The catalog is intentionally data-only; gate behavior belongs in
final_readiness.py.
"""

FINAL_READINESS_SUMMARY_SCHEMA = "lite.final_readiness_summary.v1"

READINESS_REASONS = {
    "verification_required": (
        "hard",
        "Verification did not succeed after the last workspace mutation.",
    ),
    "hard_safety_denial": (
        "hard",
        "A hard safety policy permanently denied an unsafe action in this run.",
    ),
    "recoverable_policy_denial": (
        "soft",
        "A recoverable tool-policy error occurred; use the correct tool or arguments.",
    ),
    "unclassified_policy_denial": (
        "hard",
        "An unclassified tool-policy denial occurred and must be reviewed before completion.",
    ),
    "partial_success_workspace_changed": (
        "hard",
        "A tool partially succeeded and changed the workspace.",
    ),
    "missing_required_artifact": (
        "hard",
        "A required output artifact mentioned in the request is still missing.",
    ),
    "unresolved_high_priority_todo": (
        "soft",
        "A current-run high priority todo is still unresolved.",
    ),
    "context_pressure_without_reduction": (
        "soft",
        "Context pressure is high and no successful reduction was recorded.",
    ),
    "tier3_summary_without_delta": (
        "soft",
        "Tier 3 context summary ran but had no new delta to summarize.",
    ),
    "replacement_ledger_disabled_under_pressure": (
        "soft",
        "Context pressure is high but the replacement ledger is disabled.",
    ),
    "provider_real_token_usage_unavailable": (
        "soft",
        "Provider real token usage was unavailable; context pressure used estimates.",
    ),
    "compact_net_negative": ("soft", "LLM compaction cost more tokens than it saved."),
    "compact_summary_quality_low": ("soft", "Compaction summary lacks concrete next steps or file references."),
    "context_pressure_compaction_failed": ("hard", "Context pressure is extreme but compaction yielded no token savings."),
}


def reason_severity(reason):
    return READINESS_REASONS.get(str(reason), ("soft", str(reason)))[0]


def reason_message(reason):
    return READINESS_REASONS.get(str(reason), ("soft", str(reason)))[1]
