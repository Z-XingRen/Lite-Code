"""Unit tests for final-readiness gate decisions and notices."""

from lite.core.final_readiness import (
    VALID_MODES,
    evaluate_final_readiness,
    extract_required_artifact_paths,
    readiness_notice,
)
from lite.core.final_readiness_tools import readiness_reasons
from lite.core.task_state import TaskState


def task_state():
    return TaskState.create(task_id="task_1", run_id="run_1", user_request="demo")


def test_enforce_mode_uses_one_canonical_verification_reason_and_sequence_freshness():
    assert VALID_MODES == {"off", "warn", "enforce"}
    state = task_state()
    state.changed_paths = ["src/app.py"]
    state.evidence_summaries = {
        "verification_signal": {
            "state": "passed",
            "last_mutation_sequence": 8,
            "last_successful_verification_sequence": 8,
            "verification_receipt": {
                "schema_version": "lite.verification_receipt.v1",
                "exit_code": 0,
            },
        }
    }

    stale = evaluate_final_readiness(state, "enforce")

    assert stale["decision"] == "block"
    assert stale["action"] == "block"
    assert stale["reasons"] == ["verification_required"]

    state.evidence_summaries["verification_signal"][
        "last_successful_verification_sequence"
    ] = 9
    fresh = evaluate_final_readiness(state, "enforce")
    assert fresh["decision"] == "allow"
    assert fresh["reasons"] == []


def test_legacy_modes_normalize_without_creating_reminder_state():
    expected = {
        "soft": ("warn", "warn", "none"),
        "strict": ("enforce", "block", "block"),
        "verify": ("enforce", "block", "block"),
    }
    for legacy_mode, outcome in expected.items():
        state = task_state()
        state.changed_paths = ["src/app.py"]
        state.evidence_summaries = {
            "final_readiness_state": {
                "reminded_reason_signatures": ["legacy-signature"]
            },
            "verification_signal": {
                "state": "failed",
                "last_mutation_sequence": 2,
                "last_successful_verification_sequence": 1,
            }
        }

        decision = evaluate_final_readiness(state, legacy_mode)

        assert (decision["mode"], decision["decision"], decision["action"]) == outcome
        assert decision["reasons"] == ["verification_required"]
        assert "reason_signature" not in decision
        assert "reminder_already_sent" not in decision
        assert "final_readiness_state" not in state.evidence_summaries


def test_readiness_reasons_emit_one_verification_fact():
    state = task_state()
    state.changed_paths = ["src/app.py"]
    state.evidence_summaries = {
        "verification_signal": {
            "state": "failed",
            "last_mutation_sequence": 4,
            "last_successful_verification_sequence": 3,
        }
    }

    assert readiness_reasons(state) == ["verification_required"]


def test_final_readiness_detects_unresolved_current_run_high_priority_todo():
    state = task_state()
    state.todo_changes = [
        {
            "action": "add",
            "todo": {
                "id": "todo_1",
                "priority": "high",
                "status": "pending",
            },
        }
    ]

    decision = evaluate_final_readiness(state, "enforce")

    assert decision["decision"] == "warn"
    assert decision["action"] == "none"
    assert decision["reasons"] == ["unresolved_high_priority_todo"]


def test_final_readiness_uses_latest_current_run_todo_state():
    state = task_state()
    state.todo_changes = [
        {
            "action": "add",
            "todo": {"id": "todo_1", "priority": "high", "status": "pending"},
        },
        {
            "action": "update",
            "todo": {"id": "todo_1", "priority": "high", "status": "done"},
        },
    ]

    decision = evaluate_final_readiness(state, "enforce")

    assert decision["decision"] == "allow"
    assert decision["reasons"] == []


def test_final_readiness_detects_unreduced_context_pressure():
    state = task_state()
    state.evidence_summaries = {
        "context_budget_summary": {
            "pressure_ratio": 0.98,
            "reductions": [],
        }
    }

    decision = evaluate_final_readiness(state, "enforce")

    assert decision["decision"] == "warn"
    assert decision["action"] == "none"
    assert decision["reasons"] == ["context_pressure_without_reduction"]


def test_final_readiness_allows_context_pressure_after_successful_reduction():
    state = task_state()
    state.evidence_summaries = {
        "context_budget_summary": {
            "pressure_ratio": 0.98,
            "reductions": [{"source": "microcompact", "saved_chars": 100}],
        }
    }

    decision = evaluate_final_readiness(state, "enforce")

    assert decision["decision"] == "allow"
    assert decision["reasons"] == []


def test_final_readiness_reports_context_observability_gaps():
    state = task_state()
    state.evidence_summaries = {
        "context_budget_summary": {
            "pressure_tier": "tier3_summary",
            "pressure_ratio": 0.96,
            "reductions": [{"source": "microcompact", "saved_chars": 100}],
            "summary_called": True,
            "summary_delta_event_count": 0,
            "replacement_ledger_enabled": False,
            "provider_usage_available": False,
        }
    }

    decision = evaluate_final_readiness(state, "enforce")

    assert decision["decision"] == "warn"
    assert decision["reasons"] == [
        "tier3_summary_without_delta",
        "replacement_ledger_disabled_under_pressure",
        "provider_real_token_usage_unavailable",
    ]


def test_final_readiness_allows_missing_provider_usage_at_low_pressure():
    state = task_state()
    state.evidence_summaries = {
        "context_budget_summary": {
            "pressure_tier": "tier0_observe",
            "pressure_ratio": 0.2,
            "provider_usage_available": False,
        }
    }

    decision = evaluate_final_readiness(state, "enforce")

    assert decision["decision"] == "allow"
    assert decision["reasons"] == []


def test_final_readiness_warns_on_negative_llm_compact_net_benefit():
    state = task_state()
    state.evidence_summaries = {
        "context_budget_summary": {
            "summary_mode": "llm",
            "compact_net_benefit_tokens": -25,
        }
    }

    decision = evaluate_final_readiness(state, "enforce")

    assert decision["decision"] == "warn"
    assert decision["reasons"] == ["compact_net_negative"]


def test_final_readiness_allows_non_negative_or_unknown_compact_net_benefit():
    for net in (0, 50, None):
        state = task_state()
        state.evidence_summaries = {
            "context_budget_summary": {
                "summary_mode": "llm",
                "compact_net_benefit_tokens": net,
            }
        }

        decision = evaluate_final_readiness(state, "enforce")

        assert decision["decision"] == "allow"
        assert decision["reasons"] == []


def test_final_readiness_warns_on_low_quality_llm_compact_summary():
    state = task_state()
    state.evidence_summaries = {
        "context_budget_summary": {
            "summary_mode": "llm",
            "compact_summary_has_next_steps": True,
            "compact_summary_has_file_references": False,
        }
    }

    decision = evaluate_final_readiness(state, "enforce")

    assert decision["decision"] == "warn"
    assert decision["reasons"] == ["compact_summary_quality_low"]


def test_final_readiness_ignores_deterministic_compact_summary_quality():
    state = task_state()
    state.evidence_summaries = {
        "context_budget_summary": {
            "summary_mode": "deterministic",
            "compact_summary_has_next_steps": False,
            "compact_summary_has_file_references": False,
        }
    }

    decision = evaluate_final_readiness(state, "enforce")

    assert decision["decision"] == "allow"
    assert decision["reasons"] == []


def test_final_readiness_blocks_tier3_compaction_without_token_savings():
    state = task_state()
    state.evidence_summaries = {
        "context_budget_summary": {
            "pressure_tier": "tier3_summary",
            "pressure_ratio": 0.90,
            "pre_compact_estimated_tokens": 1200,
            "post_compact_estimated_tokens": 1200,
            "reductions": [{"source": "microcompact", "saved_chars": 1}],
        }
    }

    decision = evaluate_final_readiness(state, "enforce")

    assert decision["decision"] == "block"
    assert decision["action"] == "block"
    assert decision["reasons"] == ["context_pressure_compaction_failed"]


def test_final_readiness_blocks_partial_success_workspace_change():
    state = task_state()
    state.runtime_reminders = [
        {
            "event": "tool_executed",
            "tool": "run_shell",
            "status": "partial_success",
            "workspace_changed": True,
            "affected_paths": ["notes/result.txt"],
        }
    ]

    decision = evaluate_final_readiness(state, "enforce")

    assert decision["decision"] == "block"
    assert decision["action"] == "block"
    assert decision["reasons"] == ["partial_success_workspace_changed"]


def test_enforce_uses_fresh_receipt_sequence_instead_of_command_class():
    state = task_state()
    state.changed_paths = ["src/app.py"]
    state.evidence_summaries = {
        "verification_signal": {
            "state": "passed",
            "command_class": "compile",
            "test_state": "failed",
            "last_mutation_sequence": 3,
            "last_successful_verification_sequence": 4,
        }
    }

    decision = evaluate_final_readiness(state, "enforce")

    assert decision["decision"] == "allow"
    assert decision["action"] == "none"
    assert decision["reasons"] == []


def test_enforce_readiness_allows_fresh_test_receipt_after_code_changes():
    state = task_state()
    state.changed_paths = ["src/app.py"]
    state.evidence_summaries = {
        "verification_signal": {
            "state": "passed",
            "command_class": "test",
            "test_state": "passed",
            "last_mutation_sequence": 3,
            "last_successful_verification_sequence": 4,
        }
    }

    decision = evaluate_final_readiness(state, "enforce")

    assert decision["decision"] == "allow"
    assert decision["reasons"] == []


def test_required_artifact_extraction_tracks_output_directory(tmp_path):
    prompt = f"""
输入文件：
- `provider_capabilities.json`

请完成以下产物，全部写入 `{tmp_path}/out/`：
1. `provider_scorecard.json`
2. `openclaw_config_patch.json`
3. `failover_playbook.md`

## 执行约束
- 不要修改 `test_config.py`
"""

    paths = extract_required_artifact_paths(prompt, tmp_path)

    assert paths == [
        "out/provider_scorecard.json",
        "out/openclaw_config_patch.json",
        "out/failover_playbook.md",
    ]


def test_required_artifact_extraction_ignores_negated_output_requests(tmp_path):
    prompt = "Do not create `forbidden.py`.\n请生成 `final_report.md`。"

    paths = extract_required_artifact_paths(prompt, tmp_path)

    assert paths == ["final_report.md"]


def test_required_artifact_extraction_keeps_mixed_input_output_line_scoped(tmp_path):
    prompt = "Use input file `source.json` and produce `result.json`."

    paths = extract_required_artifact_paths(prompt, tmp_path)

    assert paths == ["result.json"]


def test_required_artifact_extraction_clears_output_context_at_plain_constraints(tmp_path):
    prompt = f"""
Create `final_report.md` under `{tmp_path}/out/`.

Constraints:
- keep `config.yaml` unchanged
"""

    paths = extract_required_artifact_paths(prompt, tmp_path)

    assert paths == ["out/final_report.md"]


def test_required_artifact_extraction_ignores_do_not_modify_after_output(tmp_path):
    prompt = "Please create `final_report.md`.\nDo not modify `test_config.py`."

    paths = extract_required_artifact_paths(prompt, tmp_path)

    assert paths == ["final_report.md"]


def test_final_readiness_detects_missing_required_artifacts(tmp_path):
    state = task_state()
    state.user_request = "请生成 `final_report.md` 和 `progress.md`。"
    (tmp_path / "progress.md").write_text("done\n", encoding="utf-8")

    decision = evaluate_final_readiness(state, "warn", workspace_root=tmp_path)

    assert decision["mode"] == "warn"
    assert decision["decision"] == "warn"
    assert decision["action"] == "none"
    assert decision["reasons"] == ["missing_required_artifact"]
    summary = decision["required_artifact_summary"]
    assert summary["declared_paths"] == ["final_report.md", "progress.md"]
    assert summary["missing_paths"] == ["final_report.md"]


def test_readiness_notice_uses_catalog_messages_not_raw_codes():
    notice = readiness_notice(
        {
            "action": "block",
            "reasons": ["verification_required"],
        }
    )

    assert "verification_required" not in notice
    assert "Verification did not succeed" in notice
    assert "last workspace mutation" in notice


def test_final_readiness_summary_has_schema_version():
    from lite.core.final_readiness import reduce_final_readiness_summary

    summary = reduce_final_readiness_summary({}, {"decision": "warn", "reasons": []})

    assert summary["schema_version"] == "lite.final_readiness_summary.v1"
    assert "remind_count" not in summary
