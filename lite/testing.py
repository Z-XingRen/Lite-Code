"""Testing helpers for deterministic native-tool runtime checks."""

import os
import shlex
import subprocess
from dataclasses import replace

from .core import model_output as legacy_model_output
from .providers.base import ModelConversation, ModelResult, ToolCall


class ScriptedModelClient:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.prompts = []
        self.requests = []
        self._call_seq = 0
        self.supports_prompt_cache = False
        self.last_completion_metadata = {}

    def complete(self, request, max_new_tokens, **kwargs):
        if getattr(self, "_delegating_complete_override", False):
            return self._take_output()
        return self.complete_result(request, max_new_tokens, **kwargs).text

    def complete_result(self, request, max_new_tokens, **kwargs):
        prompt = request.initial_input if isinstance(request, ModelConversation) else request
        if type(self).complete is not ScriptedModelClient.complete:
            self.prompts.append(prompt)
            self.requests.append(request)
            self._delegating_complete_override = True
            try:
                output = self.complete(prompt, max_new_tokens, **kwargs)
            finally:
                self._delegating_complete_override = False
            result = self._normalize_output(output)
            if result.metadata:
                self.last_completion_metadata = dict(result.metadata)
                return result
            return replace(result, metadata=dict(self.last_completion_metadata))
        del max_new_tokens, kwargs
        return self._next_result(request)

    def _next_result(self, request, *, record=True):
        prompt = request.initial_input if isinstance(request, ModelConversation) else request
        if record:
            self.prompts.append(prompt)
            self.requests.append(request)
        if not getattr(self, "last_completion_metadata", None):
            self.last_completion_metadata = {}
        output = self._take_output()
        result = self._normalize_output(output)
        if result.metadata:
            self.last_completion_metadata = dict(result.metadata)
            return result
        return replace(result, metadata=dict(self.last_completion_metadata))

    def _take_output(self):
        if not self.outputs:
            raise RuntimeError("scripted model ran out of outputs")
        output = self.outputs.pop(0)
        if isinstance(output, BaseException):
            raise output
        return output

    def _normalize_output(self, output):
        if isinstance(output, ModelResult):
            return self._ensure_call_ids(output)
        if isinstance(output, ToolCall):
            return self._ensure_call_ids(ModelResult(tool_calls=(output,)))
        # Keep old deterministic fixtures readable while production execution
        # uses native ToolCall objects exclusively. Plain strings are normal
        # final answers; only explicitly tagged legacy fixtures are translated.
        if isinstance(output, str) and ("<tool" in output or "<final>" in output):
            kind, payload = legacy_model_output.parse(output)
            if kind in {"tool", "tools"}:
                values = [payload] if kind == "tool" else list(payload)
                return self._ensure_call_ids(
                    ModelResult(tool_calls=tuple(self._tool_call(item) for item in values))
                )
            if kind == "final":
                return ModelResult(text=str(payload))
            return ModelResult()
        if isinstance(output, dict):
            if "tool_calls" in output or "text" in output:
                calls = tuple(self._tool_call(item) for item in output.get("tool_calls", ()))
                return self._ensure_call_ids(
                    ModelResult(
                        text=str(output.get("text", "")),
                        tool_calls=calls,
                        stop_reason=str(output.get("stop_reason", "")),
                    )
                )
            if "name" in output:
                return self._ensure_call_ids(
                    ModelResult(tool_calls=(self._tool_call(output),))
                )
        if isinstance(output, (list, tuple)) and all(
            isinstance(item, (dict, ToolCall)) for item in output
        ):
            return self._ensure_call_ids(
                ModelResult(tool_calls=tuple(self._tool_call(item) for item in output))
            )
        return ModelResult(text=str(output))

    def _tool_call(self, value):
        if isinstance(value, ToolCall):
            return value
        return ToolCall(
            call_id=str(value.get("call_id", "")),
            name=str(value.get("name", "")),
            arguments=dict(value.get("args", value.get("arguments", {})) or {}),
        )

    def _ensure_call_ids(self, result):
        calls = []
        for call in result.tool_calls:
            if call.call_id:
                calls.append(call)
                continue
            self._call_seq += 1
            calls.append(replace(call, call_id=f"call_scripted_{self._call_seq:06d}"))
        return replace(result, tool_calls=tuple(calls))


class ScriptedStreamingModelClient(ScriptedModelClient):
    def __init__(self, streams):
        super().__init__([])
        self.streams = list(streams)
        self.cancellation_tokens = []
        self.abort_count = 0

    def stream_result(
        self, request, max_new_tokens, *, cancellation_token=None, **kwargs
    ):
        del max_new_tokens, kwargs
        prompt = request.initial_input if isinstance(request, ModelConversation) else request
        self.prompts.append(prompt)
        self.requests.append(request)
        self.cancellation_tokens.append(cancellation_token)
        if not self.streams:
            raise RuntimeError("scripted streaming model ran out of streams")
        stream = self.streams.pop(0)
        for item in stream:
            if cancellation_token is not None:
                cancellation_token.raise_if_cancelled()
            if callable(item):
                item(cancellation_token)
                continue
            yield item
        if cancellation_token is not None:
            cancellation_token.raise_if_cancelled()

    def abort(self):
        self.abort_count += 1


def scripted_tool(name, args=None, *, call_id=""):
    return {
        "call_id": str(call_id),
        "name": str(name),
        "args": dict(args or {}),
    }


def shell_join(arguments):
    """Render argv for the platform shell used by ``run_shell``."""

    values = [str(value) for value in arguments]
    if os.name == "nt":
        return subprocess.list2cmdline(values)
    return shlex.join(values)
