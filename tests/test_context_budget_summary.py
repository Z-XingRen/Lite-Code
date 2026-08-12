from lite.core.context_budget_summary import (
    context_budget_summary,
    update_from_completion,
    update_from_orchestrator,
)


def test_context_budget_summary_includes_compact_call_usage_and_net_benefit():
    summary = context_budget_summary(
        {
            "context_usage": {
                "context_window": 1000,
                "reserved_output_tokens": 100,
                "total_estimated_tokens": 600,
            },
            "context_orchestrator": {
                "summary_called": True,
                "compact_call_usage": {
                    "input_tokens": 80,
                    "output_tokens": 20,
                    "total_tokens": 100,
                    "cached_tokens": 5,
                    "model": "test-model",
                    "provider": "openai",
                },
                "pre_compact_estimated_tokens": 900,
                "post_compact_estimated_tokens": 600,
            },
        }
    )

    assert summary["compact_call_usage"]["total_tokens"] == 100
    assert summary["compact_net_benefit_tokens"] == 200


def test_context_budget_summary_preserves_negative_net_benefit():
    summary = context_budget_summary(
        {
            "context_usage": {
                "context_window": 1000,
                "reserved_output_tokens": 100,
                "total_estimated_tokens": 650,
            },
            "context_orchestrator": {
                "compact_call_usage": {"total_tokens": 100},
                "pre_compact_estimated_tokens": 600,
                "post_compact_estimated_tokens": 650,
            },
        }
    )

    assert summary["compact_net_benefit_tokens"] == -150


def test_update_from_orchestrator_carries_compact_call_usage():
    summary = update_from_orchestrator(
        {},
        {
            "context_orchestrator": {
                "compact_call_usage": {"total_tokens": 12},
                "pre_compact_estimated_tokens": 40,
                "post_compact_estimated_tokens": 20,
            }
        },
    )

    assert summary["compact_call_usage"] == {"total_tokens": 12}
    assert summary["compact_net_benefit_tokens"] == 8


def test_context_budget_summary_preserves_turn_cache_projection_evidence():
    summary = context_budget_summary(
        {
            "context_usage": {},
            "cache_projection_reused": True,
            "cache_projection_reason": "context_refresh",
            "cache_projection_generation": 3,
            "cache_projection_message_count": 8,
            "cache_projection_chars": 4096,
            "cache_projection_context_refreshed": True,
            "provider_prompt_chars": 1200,
        }
    )

    updated = update_from_completion(
        summary,
        {
            "context_usage": {"actual_input_tokens": 500},
            "completion_metadata": {"cached_tokens": 400},
        },
    )

    assert updated["cache_projection_reused"] is True
    assert updated["cache_projection_reason"] == "context_refresh"
    assert updated["cache_projection_generation"] == 3
    assert updated["cache_projection_message_count"] == 8
    assert updated["cache_projection_chars"] == 4096
    assert updated["cache_projection_context_refreshed"] is True
    assert updated["provider_prompt_chars"] == 1200
