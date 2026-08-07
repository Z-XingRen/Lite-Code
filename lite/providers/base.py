"""Provider-neutral model conversation and tool-call types."""

from dataclasses import dataclass, field, replace
from typing import Any

TRUNCATED_STOP_REASON = "length"
TRUNCATED_STOP_REASON_ALIASES = frozenset(
    {"length", "max_tokens", "max_output_tokens", "incomplete"}
)


def normalize_stop_reason(value):
    raw = str(value or "").strip()
    if raw.lower() in TRUNCATED_STOP_REASON_ALIASES:
        return TRUNCATED_STOP_REASON
    return raw


def is_truncated_stop_reason(value):
    return normalize_stop_reason(value) == TRUNCATED_STOP_REASON


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    input_schema: dict
    strict: bool = False


@dataclass(frozen=True)
class ToolCall:
    call_id: str
    name: str
    arguments: dict


@dataclass(frozen=True)
class ToolOutput:
    call_id: str
    name: str
    content: str
    is_error: bool = False


@dataclass(frozen=True)
class ModelResult:
    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    stop_reason: str = ""
    continuation: tuple[dict, ...] = ()
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ConversationTurn:
    """One provider response plus the local observations that follow it."""

    continuation: tuple[dict, ...]
    tool_outputs: tuple[ToolOutput, ...] = ()
    feedback: tuple[str, ...] = ()


@dataclass
class ModelConversation:
    """A single in-process turn, replayed without remote conversation state."""

    initial_input: Any
    tools: tuple[ToolDefinition, ...] = ()
    turns: list[ConversationTurn] = field(default_factory=list)

    def append_result(self, result, *, tool_outputs=(), feedback=()):
        self.turns.append(
            ConversationTurn(
                continuation=tuple(result.continuation or ()),
                tool_outputs=tuple(tool_outputs or ()),
                feedback=tuple(str(item) for item in (feedback or ()) if str(item).strip()),
            )
        )

    def add_feedback(self, *messages):
        values = tuple(str(item) for item in messages if str(item).strip())
        if not values or not self.turns:
            return
        self.turns[-1] = replace(
            self.turns[-1], feedback=self.turns[-1].feedback + values
        )


def ensure_conversation(value, *, tools=()):
    if isinstance(value, ModelConversation):
        return value
    return ModelConversation(initial_input=value, tools=tuple(tools or ()))


def complete_model(model_client, request, max_new_tokens, **kwargs):
    """Complete a provider-neutral request while supporting text-only clients."""

    if hasattr(model_client, "complete_result"):
        result = model_client.complete_result(request, max_new_tokens, **kwargs)
        if isinstance(result, ModelResult):
            return result
        return ModelResult(text=str(result))

    prompt = request.initial_input if isinstance(request, ModelConversation) else request
    text = model_client.complete(prompt, max_new_tokens, **kwargs)
    metadata = dict(getattr(model_client, "last_completion_metadata", {}) or {})
    return ModelResult(text=str(text), metadata=metadata)
