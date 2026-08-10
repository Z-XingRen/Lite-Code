"""Pydantic argument models for all registered tools.

Each model is the single source of truth for a tool's argument structure,
defaults, and pure value-level constraints (type, range, non-empty).
Workspace-aware checks (path safety, file existence, patch uniqueness) still
live in validate_tool() since they require access to the agent.
"""

from __future__ import annotations

import copy
from typing import List, Optional, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)


def provider_input_schema(model):
    """Return a closed JSON Schema generated from a Pydantic tool model."""
    schema = copy.deepcopy(model.model_json_schema())

    def close_objects(value):
        if isinstance(value, dict):
            if value.get("type") == "object" or "properties" in value:
                value["additionalProperties"] = False
            for child in value.values():
                close_objects(child)
        elif isinstance(value, list):
            for child in value:
                close_objects(child)

    close_objects(schema)
    return schema


class ListFilesArgs(BaseModel):
    path: str = "."


class ReadFileArgs(BaseModel):
    path: str
    start: int = 1
    end: int = 200

    @field_validator("start")
    @classmethod
    def start_ge_one(cls, v: int) -> int:
        if v < 1:
            raise ValueError("start must be >= 1")
        return v

    @model_validator(mode="after")
    def end_ge_start(self) -> "ReadFileArgs":
        if self.end < self.start:
            raise ValueError("invalid line range")
        return self


class SearchArgs(BaseModel):
    pattern: str
    path: str = "."

    @field_validator("pattern")
    @classmethod
    def pattern_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("pattern must not be empty")
        return v


class InspectImageArgs(BaseModel):
    path: str
    question: str
    profile: str = "general"
    output_schema: str = ""

    @field_validator("path")
    @classmethod
    def path_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("path must not be empty")
        return v

    @field_validator("question")
    @classmethod
    def question_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("question must not be empty")
        return v


class RunShellArgs(BaseModel):
    command: str
    timeout: int = 20
    network_access: Optional[bool] = Field(
        default=None,
        description="Request network access for this command only.",
    )
    additional_readonly_paths: List[str] = Field(
        default_factory=list,
        description="Absolute host paths to mount read-only for this command only.",
    )
    additional_writable_paths: List[str] = Field(
        default_factory=list,
        description="Absolute host paths to mount writable for this command only.",
    )

    @field_validator("command")
    @classmethod
    def command_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("command must not be empty")
        return v

    @field_validator("timeout")
    @classmethod
    def timeout_in_range(cls, v: int) -> int:
        if v < 1 or v > 120:
            raise ValueError("timeout must be in [1, 120]")
        return v

    @field_validator("additional_readonly_paths", "additional_writable_paths")
    @classmethod
    def paths_not_empty(cls, values: List[str]) -> List[str]:
        if any(not str(value).strip() for value in values):
            raise ValueError("sandbox expansion paths must not be empty")
        return values


class WriteFileArgs(BaseModel):
    path: str
    content: str


class PatchFileArgs(BaseModel):
    path: str
    old_text: str
    new_text: str

    @field_validator("old_text")
    @classmethod
    def old_text_not_empty(cls, v: str) -> str:
        if not v:
            raise ValueError("old_text must not be empty")
        return v


class TodoAddArgs(BaseModel):
    content: str
    status: str = "pending"
    priority: str = "normal"
    note: str = ""

    @field_validator("content")
    @classmethod
    def content_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("content must not be empty")
        return v


class TodoUpdateArgs(BaseModel):
    model_config = ConfigDict(extra="allow")
    todo_id: str
    status: Optional[str] = None
    content: Optional[str] = None
    priority: Optional[str] = None
    note: Optional[str] = None

    @field_validator("todo_id")
    @classmethod
    def todo_id_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("todo_id must not be empty")
        return v


class TodoListArgs(BaseModel):
    pass


class AgentArgs(BaseModel):
    description: str
    prompt: str
    subagent_type: str = "worker"
    write_scope: Union[List[str], str, None] = None

    @field_validator("description")
    @classmethod
    def description_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("description must not be empty")
        return v

    @field_validator("prompt")
    @classmethod
    def prompt_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("prompt must not be empty")
        return v

    @field_validator("subagent_type")
    @classmethod
    def valid_subagent_type(cls, v: str) -> str:
        if v not in {"worker", "Explore"}:
            raise ValueError("subagent_type must be worker or Explore")
        return v

    @field_validator("write_scope", mode="before")
    @classmethod
    def valid_write_scope(cls, v: object) -> object:
        if v is not None and not isinstance(v, (list, str)):
            raise ValueError("write_scope must be a list of workspace paths")
        return v


class SendMessageArgs(BaseModel):
    to: str
    message: str

    @field_validator("to")
    @classmethod
    def to_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("to must not be empty")
        return v

    @field_validator("message")
    @classmethod
    def message_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("message must not be empty")
        return v


class TaskStopArgs(BaseModel):
    task_id: str

    @field_validator("task_id")
    @classmethod
    def task_id_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("task_id must not be empty")
        return v


class EnterPlanModeArgs(BaseModel):
    topic: str
    path: Optional[str] = None

    @field_validator("topic")
    @classmethod
    def topic_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("topic must not be empty")
        return v


class ExitPlanModeArgs(BaseModel):
    pass


class AskUserArgs(BaseModel):
    question: str
    choices: Optional[List[str]] = None

    @field_validator("question")
    @classmethod
    def question_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("question must not be empty")
        return v

    @field_validator("choices", mode="before")
    @classmethod
    def choices_must_be_list(cls, v: object) -> object:
        if v is not None and not isinstance(v, list):
            raise ValueError("choices must be a list")
        return v


def first_error_message(exc: "ValidationError") -> str:  # type: ignore[name-defined]
    """Extract a clean single-line message from a Pydantic ValidationError."""
    errors = exc.errors(include_url=False)
    if not errors:
        return str(exc)
    err = errors[0]
    msg = str(err.get("msg", "")).removeprefix("Value error, ")
    # For missing required fields, reproduce the old KeyError repr: "'fieldname'"
    # so callers that checked for that format continue to work.
    if err.get("type") == "missing":
        loc = err.get("loc", ())
        field = loc[-1] if loc else ""
        if field:
            return f"'{field}'"
    return msg
