"""Canonical completion contract and final-answer policy over TaskState evidence."""

from .final_readiness_artifacts import (
    extract_required_artifact_paths as extract_required_artifact_paths,
    summarize_required_artifacts as summarize_required_artifacts,
)
from .final_readiness_reasons import (
    FINAL_READINESS_SUMMARY_SCHEMA,
    reason_message,
    reason_severity,
)
from .final_readiness_tools import readiness_reasons

VALID_MODES = {"off", "warn", "enforce"}
COMPLETION_CONTRACT_SCHEMA = "lite.completion_contract.v1"
LEGACY_MODE_ALIASES = {
    "soft": "warn",
    "strict": "enforce",
    "verify": "enforce",
}


def evaluate_final_readiness(task_state, mode, workspace_root=None):
    summaries = dict(task_state.evidence_summaries or {})
    if summaries.pop("final_readiness_state", None) is not None:
        task_state.evidence_summaries = summaries
    requested_mode = str(mode or "warn")
    mode = LEGACY_MODE_ALIASES.get(requested_mode, requested_mode)
    if mode not in VALID_MODES:
        mode = "warn"
    reasons = (
        []
        if mode == "off"
        else readiness_reasons(task_state, workspace_root=workspace_root)
    )
    contract = build_completion_contract(task_state, reasons)
    decision = "allow"
    action = "none"
    if reasons and mode == "warn":
        decision = "warn"
    elif reasons and mode == "enforce":
        if any(reason_severity(reason) == "hard" for reason in reasons):
            decision, action = "block", "block"
        else:
            decision = "warn"
    result = {
        "mode": mode,
        "decision": decision,
        "reasons": reasons,
        "action": action,
        "completion_contract": contract,
        "required_artifact_summary": dict(
            (task_state.evidence_summaries or {}).get("required_artifact_summary", {})
            or {}
        ),
    }
    contract["policy_allows_final"] = action != "block"
    _store_completion_contract(task_state, contract)
    return result


def build_completion_contract(task_state, reasons):
    previous = dict(
        (task_state.evidence_summaries or {}).get("completion_contract", {}) or {}
    )
    unresolved = [
        {
            "code": reason,
            "severity": reason_severity(reason),
            "message": reason_message(reason),
        }
        for reason in reasons
    ]
    blocking_codes = [
        item["code"] for item in unresolved if item["severity"] == "hard"
    ]
    verification = dict(
        (task_state.evidence_summaries or {}).get("verification_signal", {}) or {}
    )
    return {
        "schema_version": COMPLETION_CONTRACT_SCHEMA,
        "ready": not blocking_codes,
        "unresolved": unresolved,
        "unresolved_codes": [item["code"] for item in unresolved],
        "blocking_codes": blocking_codes,
        "repair_attempt_count": int(previous.get("repair_attempt_count", 0) or 0),
        "changed_paths": list(task_state.changed_paths or []),
        "verification": {
            "required": bool(task_state.changed_paths),
            "state": str(verification.get("state", "unknown")),
            "fresh": "verification_required" not in reasons,
        },
        "required_artifacts": dict(
            (task_state.evidence_summaries or {}).get(
                "required_artifact_summary", {}
            )
            or {}
        ),
    }


def record_completion_repair_attempt(task_state, contract):
    contract = dict(contract or {})
    contract["repair_attempt_count"] = int(
        contract.get("repair_attempt_count", 0) or 0
    ) + 1
    contract["policy_allows_final"] = False
    _store_completion_contract(task_state, contract)
    return contract


def _store_completion_contract(task_state, contract):
    summaries = dict(task_state.evidence_summaries or {})
    summaries["completion_contract"] = dict(contract or {})
    task_state.evidence_summaries = summaries


def readiness_notice(decision):
    messages = [reason_message(reason) for reason in decision.get("reasons", [])]
    text = "\n".join(f"- {message}" for message in messages) or "- Readiness warning."
    if decision.get("action") == "block":
        return f"Final answer blocked by runtime readiness gate:\n{text}"
    return (
        "Before final answer, address this runtime readiness issue:\n"
        f"{text}\nReturn final again only after addressing it or explaining why it is unavailable."
    )


def reduce_final_readiness_summary(summary, event):
    summary = dict(summary or {})
    summary.pop("remind_count", None)
    summary.setdefault("schema_version", FINAL_READINESS_SUMMARY_SCHEMA)
    decision = str(event.get("decision", ""))
    summary[f"{decision}_count"] = int(summary.get(f"{decision}_count", 0) or 0) + 1
    for missing in ("allow_count", "warn_count", "block_count"):
        summary.setdefault(missing, 0)
    summary["last_decision"] = decision
    summary["last_reasons"] = list(event.get("reasons", []) or [])
    return summary
