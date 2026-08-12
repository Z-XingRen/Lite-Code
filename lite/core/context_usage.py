"""Context usage estimation for prompt transparency."""

import hashlib

from ..providers.errors import sanitize_url
from .context_pressure import ContextPressureController
from .context_sections import compute_budget_tokens
from .context_tools import native_tool_chars

DEFAULT_CONTEXT_WINDOW = 200_000
TOKEN_ESTIMATION_METHOD = "typed_content_heuristic_v1"

def estimate_tokens(chars):
    return max(0, (int(chars) + 3) // 4)

def detect_content_type(text: str) -> str:
    if not text:
        return "mixed"
    sample = str(text)[:2000]
    cjk_count = sum(1 for ch in sample if "\u4e00" <= ch <= "\u9fff")
    if sample and cjk_count > len(sample) * 0.3:
        return "cjk_heavy"
    code_indicators = sample.count("{") + sample.count("}") + sample.count("/")
    if sample and code_indicators > len(sample) * 0.05:
        return "code"
    return "mixed"

def estimate_tokens_typed(text: str, content_type: str = "mixed") -> int:
    chars = len(str(text))
    if content_type == "code":
        return max(0, (chars * 10 + 31) // 32)
    if content_type == "cjk_heavy":
        return max(0, (chars * 10 + 17) // 18)
    return estimate_tokens(chars)

class ContextUsageAnalyzer:
    def __init__(self, agent):
        self.agent = agent

    def analyze(self, rendered, *, prompt_cache_key=None):
        tools_chars = native_tool_chars(self.agent)
        sections = {}
        raw_total = 0
        for name, section in rendered.items():
            key = "current_request" if name == "current_request" else name
            text = str(section.rendered)
            chars = int(section.rendered_chars)
            tokens = estimate_tokens_typed(text, detect_content_type(text))
            raw_tokens = estimate_tokens_typed(str(section.raw), detect_content_type(str(section.raw)))
            sections[key] = {"chars": chars, "tokens": tokens}
            raw_total += raw_tokens
        sections["tools"] = {"chars": tools_chars, "tokens": estimate_tokens(tools_chars)}
        raw_total += estimate_tokens(tools_chars)
        total = sum(section["tokens"] for section in sections.values())
        window = self._context_window()
        budget = compute_budget_tokens(window)
        reserved = int(getattr(self.agent, "max_new_tokens", 0) or 0)
        prompt_hash = self._prompt_hash(rendered)
        current_identity = {
            "provider": self._provider(),
            "provider_base_url": self._provider_base_url(),
            "model": str(getattr(getattr(self.agent, "model_client", None), "model", "")),
            "context_window": window,
            "prompt_cache_key": str(
                prompt_cache_key
                or getattr(getattr(self.agent, "prefix_state", None), "hash", "")
                or ""
            ),
            "prompt_hash": prompt_hash,
        }
        pressure = ContextPressureController().evaluate(
            estimated_input_tokens=max(total, raw_total),
            context_window=window,
            budget_tokens=budget,
            current_identity=current_identity,
            last_completion_metadata=getattr(self.agent, "last_completion_metadata", {}) or {},
            last_identity=self._last_identity(),
        )
        return {
            "estimation_method": TOKEN_ESTIMATION_METHOD,
            "model": current_identity["model"],
            "context_window": window,
            "budget_tokens": budget,
            "reserved_output_tokens": reserved,
            "total_estimated_tokens": total,
            "sections": sections,
            "free_tokens": budget - total,
            "current_identity": current_identity,
            **pressure.to_context_usage_fields(),
        }

    def _context_window(self):
        client_window = int(getattr(getattr(self.agent, "model_client", None), "context_window", 0) or 0)
        if client_window:
            return client_window
        model = str(getattr(getattr(self.agent, "model_client", None), "model", "")).lower()
        if "1m" in model or "1000000" in model:
            return 1_000_000
        return DEFAULT_CONTEXT_WINDOW

    def _prompt_hash(self, rendered):
        text = "\n\n".join(section.rendered for section in rendered.values()).strip()
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _provider(self):
        client = getattr(self.agent, "model_client", None)
        return str(getattr(client, "provider", "") or client.__class__.__name__ if client else "")

    def _provider_base_url(self):
        return sanitize_url(getattr(getattr(self.agent, "model_client", None), "base_url", ""))

    def _last_identity(self):
        metadata = dict(getattr(self.agent, "last_prompt_metadata", {}) or {})
        usage = dict(metadata.get("context_usage", {}) or {})
        identity = dict(usage.get("current_identity", {}) or {})
        return identity or {
            "provider": usage.get("provider") or metadata.get("provider"),
            "provider_base_url": usage.get("provider_base_url") or metadata.get("provider_base_url"),
            "model": usage.get("model") or metadata.get("model"),
            "context_window": usage.get("context_window"),
            "prompt_cache_key": metadata.get("prompt_cache_key"),
            "prompt_hash": metadata.get("prompt_hash") or usage.get("prompt_hash"),
        }


def apply_completion_usage(context_usage, completion_metadata):
    """Attach current-call provider usage to pre-request context metadata."""

    usage = dict(context_usage or {})
    completion = dict(completion_metadata or {})
    try:
        input_tokens = int(completion["input_tokens"])
    except (KeyError, TypeError, ValueError):
        return usage

    cached_tokens = _non_negative_optional_int(completion.get("cached_tokens"))
    cache_write_tokens = _non_negative_optional_int(
        completion.get("cache_write_tokens")
    )
    budget = max(1, int(usage.get("budget_tokens", 0) or 0))
    window = max(1, int(usage.get("context_window", 0) or 0))
    pressure_ratio = round(max(0, input_tokens) / budget, 6)
    usage.update(
        {
            "actual_input_tokens": input_tokens,
            "last_actual_input_tokens": input_tokens,
            "usage_source": "actual",
            "calibration_source": "current_completion",
            "cached_tokens": cached_tokens,
            "cache_write_tokens": cache_write_tokens,
            "pressure_ratio": pressure_ratio,
            "window_ratio": round(max(0, input_tokens) / window, 6),
            "pressure_tier": _pressure_tier(pressure_ratio),
        }
    )
    return usage


def _non_negative_optional_int(value):
    if value is None:
        return None
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return None


def _pressure_tier(ratio):
    if ratio >= 0.95:
        return "tier3_summary"
    if ratio >= 0.8:
        return "tier2_prune"
    if ratio >= 0.6:
        return "tier1_snip"
    return "tier0_observe"
