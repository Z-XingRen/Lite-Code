"""Soft entropy trends and hard runtime architecture boundaries."""

import json
from pathlib import Path

from lite.architecture import line_trend, module_line_trends


CORE_MODULE_TREND_BUDGETS = {
    "lite/core/runtime.py": 950,
    "lite/core/before_final_hooks.py": 140,
    "lite/core/evidence_summaries.py": 90,
    "lite/core/final_readiness.py": 120,
    "lite/core/final_readiness_artifacts.py": 160,
    "lite/core/final_readiness_context.py": 60,
    "lite/core/final_readiness_reasons.py": 60,
    "lite/core/final_readiness_tools.py": 100,
    "lite/core/governance.py": 80,
    "lite/core/runtime_events.py": 90,
    "lite/core/runtime_consumers.py": 90,
    "lite/core/artifacts.py": 130,
    "lite/core/task_state.py": 140,
    "lite/core/todo_ledger.py": 120,
    "lite/core/worker_manager.py": 220,
    "lite/core/context_manager.py": 420,
    "lite/core/context_budget_summary.py": 130,
    "lite/core/context_handoff.py": 240,
    "lite/core/context_orchestrator.py": 210,
    "lite/core/context_pressure.py": 140,
    "lite/core/context_report.py": 140,
    "lite/core/context_retention.py": 90,
    "lite/core/context_replacements.py": 160,
    "lite/core/context_sections.py": 170,
    "lite/core/context_usage.py": 130,
    "lite/core/compact.py": 250,
    "lite/core/compact_summary.py": 130,
    "lite/core/completion_governance.py": 240,
    "lite/core/engine.py": 470,
    "lite/core/model_errors.py": 100,
    "lite/core/model_router.py": 40,
    "lite/core/permissions.py": 140,
    "lite/core/tool_policy.py": 90,
    "lite/core/plan_mode.py": 140,
    "lite/core/tool_executor.py": 181,
    "lite/core/tool_profiles.py": 80,
    "lite/core/tool_result_artifacts.py": 60,
    "lite/core/turn_transitions.py": 90,
    "lite/core/verification.py": 80,
    "lite/core/turn_history.py": 280,
    "lite/core/media_history.py": 20,
    "lite/features/skills.py": 220,
    "lite/features/skills_bundled.py": 120,
    "lite/features/skills_runtime.py": 140,
    "lite/tools/registry.py": 360,
    "lite/tools/todos.py": 80,
    "lite/tools/agents.py": 90,
}


def test_core_module_entropy_is_a_reported_trend(record_property):
    root = Path(__file__).resolve().parents[1]
    report = module_line_trends(root, CORE_MODULE_TREND_BUDGETS)

    assert set(report) == set(CORE_MODULE_TREND_BUDGETS)
    assert all(item["status"] in {"within_trend", "review"} for item in report.values())
    record_property("module_line_trends", json.dumps(report, sort_keys=True))


def test_one_extra_registry_line_requests_review_without_failing_the_suite():
    assert line_trend(361, 360) == {
        "lines": 361,
        "trend_budget": 360,
        "delta": 1,
        "status": "review",
    }
