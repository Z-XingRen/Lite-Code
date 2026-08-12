"""Context budget evidence reducer."""

CONTEXT_BUDGET_SCHEMA = "lite.context_budget_summary.v1"


def context_budget_summary(metadata):
    usage = dict(metadata.get("context_usage", {}) or {})
    orchestrator = dict(metadata.get("context_orchestrator", {}) or {})
    history = dict(metadata.get("history", {}) or {})
    window = int(usage.get("context_window", 0) or 0)
    reserved = int(usage.get("reserved_output_tokens", 0) or 0)
    effective_window = max(0, window - reserved)
    estimated_tokens = int(usage.get("total_estimated_tokens", 0) or 0)
    reductions = [
        *[_section_reduction(item) for item in metadata.get("budget_reductions", []) or []],
        *_microcompact_reductions(metadata),
    ]
    compact_call_usage = _compact_call_usage(orchestrator)
    return {
        "schema_version": CONTEXT_BUDGET_SCHEMA,
        "budget_unit": "tokens_estimated",
        "token_estimator": "context_usage_analyzer",
        "estimated_tokens": estimated_tokens,
        "effective_window": effective_window,
        "reserved_output_tokens": reserved,
        "pressure_ratio": round(estimated_tokens / effective_window, 4)
        if effective_window
        else 0,
        "reductions": reductions,
        "pressure_tier": orchestrator.get("pressure_tier") or usage.get("pressure_tier", ""),
        "usage_source": orchestrator.get("usage_source") or usage.get("usage_source", ""),
        "provider_usage_available": usage.get("actual_input_tokens") is not None,
        "snip_count": sum(1 for item in reductions if item.get("source") == "section_reduction"),
        "prune_count": sum(1 for item in reductions if item.get("source") == "microcompact"),
        "summary_called": bool(orchestrator.get("summary_called", False)),
        "summary_mode": str(orchestrator.get("summary_mode", "")),
        "summary_delta_event_count": int(orchestrator.get("summary_delta_event_count", 0) or 0),
        "compact_call_usage": compact_call_usage,
        "compact_net_benefit_tokens": _compact_net_benefit(orchestrator, compact_call_usage),
        "compact_summary_has_next_steps": orchestrator.get("compact_summary_has_next_steps"),
        "compact_summary_has_file_references": orchestrator.get("compact_summary_has_file_references"),
        "pre_compact_estimated_tokens": int(orchestrator.get("pre_compact_estimated_tokens", 0) or 0),
        "post_compact_estimated_tokens": int(orchestrator.get("post_compact_estimated_tokens", 0) or 0),
        "replacement_cache_hits": int(orchestrator.get("replacement_cache_hits", 0) or 0),
        "replacement_records_created": int(orchestrator.get("replacement_records_created", 0) or 0),
        "replacement_ledger_enabled": bool(orchestrator.get("replacement_ledger_enabled", False)),
        "saved_chars": _saved_chars(metadata, history, orchestrator),
        "cached_tokens": int(usage.get("cached_tokens", 0) or 0),
        **_cache_projection_fields(metadata),
        "prompt_changed_by_phase_3": False,
    }


def update_from_orchestrator(summary, event):
    summary = dict(summary or {})
    orchestrator = dict(event.get("context_orchestrator", {}) or {})
    usage = dict(event.get("context_usage", {}) or {})
    compact_call_usage = _compact_call_usage(orchestrator)
    summary.update(
        {
            "pressure_tier": orchestrator.get("pressure_tier") or usage.get("pressure_tier", ""),
            "usage_source": orchestrator.get("usage_source") or usage.get("usage_source", ""),
            "provider_usage_available": usage.get("actual_input_tokens") is not None,
            "summary_called": bool(orchestrator.get("summary_called", False)),
            "summary_mode": str(orchestrator.get("summary_mode", "")),
            "summary_delta_event_count": int(orchestrator.get("summary_delta_event_count", 0) or 0),
            "compact_call_usage": compact_call_usage,
            "compact_net_benefit_tokens": _compact_net_benefit(orchestrator, compact_call_usage),
            "compact_summary_has_next_steps": orchestrator.get("compact_summary_has_next_steps"),
            "compact_summary_has_file_references": orchestrator.get("compact_summary_has_file_references"),
            "pre_compact_estimated_tokens": int(orchestrator.get("pre_compact_estimated_tokens", 0) or 0),
            "post_compact_estimated_tokens": int(orchestrator.get("post_compact_estimated_tokens", 0) or 0),
            "replacement_cache_hits": int(orchestrator.get("replacement_cache_hits", 0) or 0),
            "replacement_records_created": int(orchestrator.get("replacement_records_created", 0) or 0),
            "replacement_ledger_enabled": bool(orchestrator.get("replacement_ledger_enabled", False)),
            "cached_tokens": int(usage.get("cached_tokens", 0) or 0),
        }
    )
    return summary


def update_from_completion(summary, event):
    summary = dict(summary or {})
    usage = dict(event.get("context_usage", {}) or {})
    completion = dict(event.get("completion_metadata", {}) or {})
    summary.update(
        {
            "pressure_tier": usage.get("pressure_tier", ""),
            "pressure_ratio": float(usage.get("pressure_ratio", 0) or 0),
            "usage_source": usage.get("usage_source", ""),
            "provider_usage_available": usage.get("actual_input_tokens")
            is not None,
            "actual_input_tokens": usage.get("actual_input_tokens"),
            "cached_tokens": int(completion.get("cached_tokens", 0) or 0),
            "cache_write_tokens": int(
                completion.get("cache_write_tokens", 0) or 0
            ),
        }
    )
    return summary


def _compact_call_usage(orchestrator):
    usage = orchestrator.get("compact_call_usage")
    return dict(usage) if isinstance(usage, dict) else None


def _cache_projection_fields(metadata):
    return {
        "cache_projection_reused": bool(metadata.get("cache_projection_reused", False)),
        "cache_projection_reason": str(metadata.get("cache_projection_reason", "")),
        "cache_projection_generation": int(
            metadata.get("cache_projection_generation", 0) or 0
        ),
        "cache_projection_message_count": int(
            metadata.get("cache_projection_message_count", 0) or 0
        ),
        "cache_projection_chars": int(metadata.get("cache_projection_chars", 0) or 0),
        "cache_projection_context_refreshed": bool(
            metadata.get("cache_projection_context_refreshed", False)
        ),
        "provider_prompt_chars": int(metadata.get("provider_prompt_chars", 0) or 0),
    }


def _compact_net_benefit(orchestrator, compact_call_usage):
    if not compact_call_usage:
        return None
    pre_tokens = int(orchestrator.get("pre_compact_estimated_tokens", 0) or 0)
    post_tokens = int(orchestrator.get("post_compact_estimated_tokens", 0) or 0)
    compact_tokens = int(compact_call_usage.get("total_tokens", 0) or 0)
    return pre_tokens - post_tokens - compact_tokens


def _saved_chars(metadata, history, orchestrator):
    section_saved = sum(
        _section_reduction(item)["saved_chars"]
        for item in metadata.get("budget_reductions", []) or []
    )
    return (
        section_saved
        + int(history.get("microcompact_saved_chars", 0) or 0)
        + int(orchestrator.get("replacement_saved_chars", 0) or 0)
    )


def _section_reduction(item):
    before = int(item.get("before_chars", 0) or 0)
    after = int(item.get("after_chars", 0) or 0)
    return {
        "source": "section_reduction",
        "section": str(item.get("section", "")),
        "saved_chars": max(0, before - after),
    }


def _microcompact_reductions(metadata):
    history = dict(metadata.get("history", {}) or {})
    saved = int(history.get("microcompact_saved_chars", 0) or 0)
    refs = list(history.get("microcompact_artifact_refs", []) or [])
    if not saved and not refs:
        return []
    return [
        {
            "source": "microcompact",
            "section": "history",
            "saved_chars": saved,
            "artifact_refs": refs,
        }
    ]
