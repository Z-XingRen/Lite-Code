"""Reduce trace events into TaskState evidence summaries.

This module is the bridge from append-only trace facts to compact report-ready
state. It does not re-read trace files; runtime consumers call it as events are
emitted during a run.
"""

from .final_readiness import reduce_final_readiness_summary
from .governance import (
    HARD_SAFETY_DENIAL_REASONS,
    HARD_SAFETY_EVENT_TYPES,
    reduce_governance_summary,
)
from .context_budget_summary import (
    context_budget_summary,
    update_from_completion,
    update_from_orchestrator,
)
from .turn_transitions import reduce_transition_summary
from .verification import reduce_verification_signal


def update_evidence_summaries(summaries, event, changed_paths=None):
    summaries = dict(summaries or {})
    if event.get("event") == "loop_transition":
        summaries["transition_summary"] = reduce_transition_summary(
            summaries.get("transition_summary", {}), event
        )
    elif event.get("event") == "prompt_built":
        summaries["context_budget_summary"] = context_budget_summary(
            event.get("prompt_metadata", {})
        )
    elif event.get("event") == "context_orchestrator_decision":
        summaries["context_budget_summary"] = update_from_orchestrator(
            summaries.get("context_budget_summary", {}), event
        )
    elif event.get("event") == "model_parsed":
        summaries["context_budget_summary"] = update_from_completion(
            summaries.get("context_budget_summary", {}), event
        )
    elif event.get("event") == "governance_decision":
        summaries["governance_summary"] = reduce_governance_summary(
            summaries.get("governance_summary", {}), event
        )
    elif event.get("event") == "governance_batch":
        governance = summaries.get("governance_summary", {})
        for decision in event.get("decisions", []) or []:
            governance = reduce_governance_summary(governance, decision)
        summaries["governance_summary"] = governance
    elif event.get("event") == "tool_executed":
        summaries["verification_signal"] = reduce_verification_signal(
            summaries.get("verification_signal", {}), event, changed_paths or []
        )
        security_event = str(event.get("security_event_type", ""))
        if (
            security_event in HARD_SAFETY_EVENT_TYPES
            and event.get("tool_status") != "rejected"
        ):
            summaries["governance_summary"] = reduce_governance_summary(
                summaries.get("governance_summary", {}),
                {
                    "decision": "deny",
                    "decision_type": "executed_safety_event",
                    "reason_code": security_event,
                    "security_event_type": security_event,
                },
            )
    elif event.get("event") in HARD_SAFETY_DENIAL_REASONS:
        event_name = str(event.get("event", ""))
        summaries["governance_summary"] = reduce_governance_summary(
            summaries.get("governance_summary", {}),
            {
                "decision": "deny",
                "decision_type": "runtime_safety_event",
                "reason_code": event_name,
                "security_event_type": event_name,
            },
        )
    elif event.get("event") == "final_readiness_decision":
        summaries["final_readiness_summary"] = reduce_final_readiness_summary(
            summaries.get("final_readiness_summary", {}), event
        )
    return summaries
