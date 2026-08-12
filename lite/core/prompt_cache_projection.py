"""Append-only provider context used for prompt-cache reuse across turns."""

from __future__ import annotations

import copy
import hashlib
import json

from ..providers.errors import sanitize_url


PROJECTION_VERSION = "lite.prompt_cache_projection.v2"


def prepare_prompt_cache_turn(
    agent, *, full_prompt, append_delta, context_refresh=None, metadata, history
):
    """Select a full prompt or append-only delta for the next provider turn."""

    identity = _projection_identity(agent, metadata)
    context_key = str(metadata.get("prompt_cache_key", ""))
    state = agent.session.get("prompt_cache_projection")
    reason = _invalidation_reason(
        agent,
        state,
        identity=identity,
        history=history,
    )
    if metadata.get("auto_compacted"):
        reason = "compaction"
    context_refreshed = bool(
        not reason and str(state.get("context_key", "")) != context_key
    )
    next_prompt = str(
        (context_refresh or full_prompt)
        if context_refreshed
        else append_delta or full_prompt
    )
    if not reason and _projection_exceeds_budget(agent, state, next_prompt):
        reason = "budget"
        context_refreshed = False
    reused = not reason
    if reused:
        messages = copy.deepcopy(list(state.get("messages", [])))
        prompt = next_prompt
        generation = int(state.get("generation", 1) or 1)
        routing_cache_key = str(state.get("routing_cache_key", "") or context_key)
        prefix_chars = int(
            state.get("prompt_cache_prefix_chars", 0)
            or metadata.get("prompt_cache_prefix_chars", 0)
            or 0
        )
    else:
        messages = []
        prompt = str(full_prompt)
        generation = int((state or {}).get("generation", 0) or 0) + 1
        routing_cache_key = _routing_cache_key(identity, context_key)
        prefix_chars = int(metadata.get("prompt_cache_prefix_chars", 0) or 0)

    metadata["prompt_cache_context_key"] = context_key
    metadata["prompt_cache_key"] = routing_cache_key
    metadata.update(
        {
            "cache_projection_version": PROJECTION_VERSION,
            "cache_projection_reused": reused,
            "cache_projection_reason": reason
            or ("context_refresh" if context_refreshed else "append"),
            "cache_projection_context_refreshed": context_refreshed,
            "cache_projection_generation": generation,
            "cache_projection_message_count": len(messages),
            "cache_projection_chars": _messages_chars(messages),
            "provider_prompt_chars": len(prompt),
        }
    )
    return {
        "prompt": prompt,
        "base_messages": messages,
        "identity": identity,
        "context_key": context_key,
        "routing_cache_key": routing_cache_key,
        "generation": generation,
        "prompt_cache_prefix_chars": prefix_chars,
        "enabled": _projection_enabled(agent),
    }


def commit_prompt_cache_turn(agent, conversation, final_text):
    """Persist the exact request chain plus the accepted assistant response."""

    turn = getattr(agent, "_prompt_cache_turn", None)
    if not turn or not turn.get("enabled") or conversation is None:
        return False
    messages = copy.deepcopy(list(getattr(conversation, "request_messages", ()) or ()))
    messages.append(
        {
            "role": "assistant",
            "content": str(final_text),
            "continuation": [],
            "tool_calls": [],
        }
    )
    agent.session["prompt_cache_projection"] = {
        "version": PROJECTION_VERSION,
        "generation": int(turn.get("generation", 1) or 1),
        "identity": copy.deepcopy(turn.get("identity", {})),
        "context_key": str(turn.get("context_key", "")),
        "routing_cache_key": str(turn.get("routing_cache_key", "")),
        "history_fingerprint": history_fingerprint(agent.session.get("history", [])),
        "prompt_cache_prefix_chars": int(
            turn.get("prompt_cache_prefix_chars", 0) or 0
        ),
        "messages": agent.redact_artifact(messages),
    }
    return True


def history_fingerprint(history):
    payload = json.dumps(
        list(history or []),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _projection_identity(agent, metadata):
    client = getattr(agent, "model_client", None)
    return {
        "provider": str(getattr(client, "provider", "") or client.__class__.__name__),
        "base_url": sanitize_url(getattr(client, "base_url", "")),
        "model": str(getattr(client, "model", "")),
        "tool_signature": str(
            getattr(getattr(agent, "prefix_state", None), "tool_signature", "")
            or ""
        ),
    }


def _routing_cache_key(identity, context_key):
    payload = json.dumps(
        {"identity": identity, "context_key": context_key},
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _invalidation_reason(agent, state, *, identity, history):
    if not _projection_enabled(agent):
        return "unsupported"
    if not isinstance(state, dict):
        return "missing"
    if state.get("version") != PROJECTION_VERSION:
        return "version"
    if state.get("identity") != identity:
        return "identity"
    if state.get("history_fingerprint") != history_fingerprint(history):
        return "history"
    messages = state.get("messages")
    if not isinstance(messages, list) or not messages:
        return "messages"
    return ""


def _projection_exceeds_budget(agent, state, next_prompt):
    projected_chars = _messages_chars(state.get("messages", [])) + len(next_prompt)
    budget_chars = int(
        getattr(getattr(agent, "context_manager", None), "total_budget", 0) or 0
    )
    return bool(budget_chars and projected_chars > budget_chars)


def _projection_enabled(agent):
    client = getattr(agent, "model_client", None)
    feature_enabled = getattr(agent, "feature_enabled", None)
    return bool(
        getattr(client, "supports_prompt_cache", False)
        and getattr(client, "supports_append_prompt_cache", False)
        and not callable(getattr(agent, "context_transform", None))
        and (not callable(feature_enabled) or feature_enabled("prompt_cache"))
    )


def _messages_chars(messages):
    if not messages:
        return 0
    return len(
        json.dumps(
            messages,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
