"""Tool abstraction shared by the runtime and prompt builder."""

from dataclasses import dataclass
from typing import Callable


_PARALLEL_READ_TOOLS = {"list_files", "read_file", "search", "todo_list"}
_MUTATING_TOOLS = {
    "write_file",
    "patch_file",
    "todo_add",
    "todo_update",
    "enter_plan_mode",
    "exit_plan_mode",
}


@dataclass(frozen=True)
class ToolResult:
    content: str
    is_error: bool = False


@dataclass(frozen=True)
class RegisteredTool:
    name: str
    schema: dict
    description: str
    risky: bool
    runner: Callable[[dict], str]
    execution_mode: str = ""
    effect_class: str = ""

    def __post_init__(self):
        effect_class = self.effect_class or (
            "read_only"
            if self.name in _PARALLEL_READ_TOOLS
            else "mutating"
            if self.name in _MUTATING_TOOLS
            else "opaque"
        )
        execution_mode = self.execution_mode or (
            "parallel" if effect_class == "read_only" else "sequential"
        )
        if effect_class not in {"read_only", "mutating", "opaque"}:
            raise ValueError(f"unsupported tool effect class: {effect_class}")
        if execution_mode not in {"parallel", "sequential"}:
            raise ValueError(f"unsupported tool execution mode: {execution_mode}")
        if execution_mode == "parallel" and effect_class != "read_only":
            raise ValueError("only read-only tools may execute in parallel")
        object.__setattr__(self, "effect_class", effect_class)
        object.__setattr__(self, "execution_mode", execution_mode)

    @property
    def read_only(self):
        return not self.risky

    @property
    def input_schema(self):
        return self.schema

    def execute(self, args):
        result = self.runner(args)
        if isinstance(result, ToolResult):
            return result
        return ToolResult(content=str(result))

    def __getitem__(self, key):
        if key == "run":
            return self.runner
        return getattr(self, key)
