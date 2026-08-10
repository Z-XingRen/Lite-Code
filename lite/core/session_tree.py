"""Append-only session tree projection backed by the session journal.

The journal is the source of truth.  This module only contains the deterministic
projection used to answer two questions: which nodes exist, and which history is
visible from the active head.
"""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field

from .session_journal_schema import (
    JournalCorruption,
    JournalSchemaError,
    canonical_json,
    json_copy,
    required_text,
)


SESSION_TREE_SCHEMA_VERSION = "lite.session_tree.v1"
TREE_ENTRY_TYPES = frozenset(
    {
        "message",
        "tool_exchange",
        "compaction",
        "branch_summary",
        "task_checkpoint",
        "plan_delta",
        "todo_delta",
        "working_state",
        "context_replacement",
    }
)


@dataclass(frozen=True)
class SessionTreeEntry:
    entry_id: str
    parent_id: str | None
    entry_type: str
    turn_id: str
    run_id: str
    created_at: str
    data: dict

    @classmethod
    def from_dict(cls, value):
        required = {
            "entry_id",
            "parent_id",
            "entry_type",
            "turn_id",
            "run_id",
            "created_at",
            "data",
        }
        if not isinstance(value, Mapping) or set(value) != required:
            raise JournalSchemaError("tree entry fields do not match the contract")
        entry_id = required_text(value["entry_id"], "entry_id")
        if entry_id == "<root>":
            raise JournalSchemaError("entry_id uses a reserved value")
        parent_id = value["parent_id"]
        if parent_id is not None:
            parent_id = required_text(parent_id, "parent_id")
        entry_type = required_text(value["entry_type"], "entry_type")
        if entry_type not in TREE_ENTRY_TYPES:
            raise JournalSchemaError(f"unsupported tree entry type: {entry_type}")
        if not isinstance(value["turn_id"], str):
            raise JournalSchemaError("turn_id must be a string")
        if not isinstance(value["run_id"], str):
            raise JournalSchemaError("run_id must be a string")
        if not isinstance(value["created_at"], str):
            raise JournalSchemaError("created_at must be a string")
        if not isinstance(value["data"], Mapping):
            raise JournalSchemaError("tree entry data must be an object")
        entry = cls(
            entry_id=entry_id,
            parent_id=parent_id,
            entry_type=entry_type,
            turn_id=value["turn_id"],
            run_id=value["run_id"],
            created_at=value["created_at"],
            data=json_copy(dict(value["data"])),
        )
        _validate_entry_data(entry)
        return entry

    def to_dict(self):
        return {
            "entry_id": self.entry_id,
            "parent_id": self.parent_id,
            "entry_type": self.entry_type,
            "turn_id": self.turn_id,
            "run_id": self.run_id,
            "created_at": self.created_at,
            "data": copy.deepcopy(self.data),
        }


@dataclass(frozen=True)
class SessionTreeState:
    schema_version: str = SESSION_TREE_SCHEMA_VERSION
    entries: dict[str, SessionTreeEntry] = field(default_factory=dict)
    children: dict[str | None, list[str]] = field(default_factory=dict)
    active_head: str | None = None
    labels: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_history(cls, history):
        state = cls()
        parent_id = None
        for index, item in enumerate(history):
            digest = hashlib.sha256(
                f"{index}\0{canonical_json(item)}".encode("utf-8")
            ).hexdigest()[:24]
            entry = message_entry(
                item,
                entry_id=f"seed_{index + 1:06d}_{digest}",
                parent_id=parent_id,
            )
            append_entry_in_place(state, entry)
            parent_id = entry.entry_id
        return state

    @classmethod
    def from_dict(cls, value):
        required = {"schema_version", "entries", "children", "active_head", "labels"}
        if not isinstance(value, Mapping) or set(value) != required:
            raise ValueError("invalid session tree snapshot fields")
        if value["schema_version"] != SESSION_TREE_SCHEMA_VERSION:
            raise ValueError("unsupported session tree snapshot schema")
        if not isinstance(value["entries"], Mapping):
            raise ValueError("session tree entries must be an object")
        entries = {
            str(entry_id): SessionTreeEntry.from_dict(entry)
            for entry_id, entry in value["entries"].items()
        }
        if any(entry_id != entry.entry_id for entry_id, entry in entries.items()):
            raise ValueError("session tree entry id mismatch")
        expected_children = {
            _decode_parent_key(key): list(children)
            for key, children in value["children"].items()
        } if isinstance(value["children"], Mapping) else None
        indexed_ids = [
            entry_id
            for children in (expected_children or {}).values()
            for entry_id in children
        ]
        if (
            expected_children is None
            or len(indexed_ids) != len(set(indexed_ids))
            or set(indexed_ids) != set(entries)
            or any(
                entry_id not in entries or entries[entry_id].parent_id != parent_id
                for parent_id, children in expected_children.items()
                for entry_id in children
            )
        ):
            raise ValueError("session tree children index mismatch")
        state = cls(entries=entries, children=expected_children)
        active_head = value["active_head"]
        if active_head is not None and active_head not in entries:
            raise ValueError("session tree active head is missing")
        labels = value["labels"]
        if not isinstance(labels, Mapping) or not all(
            isinstance(label, str)
            and label
            and isinstance(entry_id, str)
            and entry_id in entries
            for label, entry_id in labels.items()
        ):
            raise ValueError("invalid session tree labels")
        object.__setattr__(state, "active_head", active_head)
        object.__setattr__(state, "labels", dict(labels))
        return state

    def to_dict(self):
        return {
            "schema_version": self.schema_version,
            "entries": {
                entry_id: entry.to_dict() for entry_id, entry in self.entries.items()
            },
            "children": {
                _encode_parent_key(parent_id): list(children)
                for parent_id, children in self.children.items()
            },
            "active_head": self.active_head,
            "labels": dict(self.labels),
        }


def message_entry(item, *, entry_id, parent_id):
    if not isinstance(item, Mapping):
        raise JournalSchemaError("message history item must be an object")
    message = json_copy(dict(item))
    return SessionTreeEntry(
        entry_id=str(entry_id),
        parent_id=parent_id,
        entry_type="message",
        turn_id=str(message.get("turn_id", "")),
        run_id=str(message.get("run_id", "")),
        created_at=str(message.get("created_at", "")),
        data={"message": message},
    )


def append_entry_in_place(state, entry, *, require_active_parent=True):
    if entry.entry_id in state.entries:
        if state.entries[entry.entry_id] != entry:
            raise JournalCorruption(f"tree entry id conflict: {entry.entry_id}")
        return False
    if entry.parent_id is not None and entry.parent_id not in state.entries:
        raise JournalCorruption(f"tree parent does not exist: {entry.parent_id}")
    if require_active_parent and entry.parent_id != state.active_head:
        raise JournalCorruption(
            "tree append parent is not the active head: "
            f"expected {state.active_head}, got {entry.parent_id}"
        )
    state.entries[entry.entry_id] = entry
    state.children.setdefault(entry.parent_id, []).append(entry.entry_id)
    object.__setattr__(state, "active_head", entry.entry_id)
    return True


def move_head_in_place(state, target_entry_id):
    if target_entry_id is not None and target_entry_id not in state.entries:
        raise JournalCorruption(f"tree head target does not exist: {target_entry_id}")
    object.__setattr__(state, "active_head", target_entry_id)


def label_entry_in_place(state, label, entry_id):
    label = required_text(label, "label")
    if entry_id not in state.entries:
        raise JournalCorruption(f"tree label target does not exist: {entry_id}")
    state.labels[label] = entry_id


def active_path(state, head=None):
    entry_id = state.active_head if head is None else head
    path = []
    seen = set()
    while entry_id is not None:
        if entry_id in seen:
            raise JournalCorruption("session tree contains a cycle")
        seen.add(entry_id)
        try:
            entry = state.entries[entry_id]
        except KeyError as exc:
            raise JournalCorruption(f"session tree entry is missing: {entry_id}") from exc
        path.append(entry)
        entry_id = entry.parent_id
    path.reverse()
    return path


def project_history(state):
    history = []
    for entry in active_path(state):
        if entry.entry_type == "message":
            history.append(copy.deepcopy(entry.data["message"]))
        elif entry.entry_type == "tool_exchange":
            # Lite's compatibility transcript renders tool calls from the
            # result item. The paired assistant call remains canonical in the
            # node for provider-specific protocol projections.
            history.extend(copy.deepcopy(entry.data.get("results", [])))
        elif entry.entry_type in {"compaction", "context_replacement"}:
            history = copy.deepcopy(entry.data["history"])
    return history


def project_branch_state(state):
    """Derive branch-local resumable state from nodes on the active path."""

    projected = {
        "history": project_history(state),
        "plan": {},
        "todo": {},
        "working": {},
        "checkpoint": None,
        "summaries": [],
    }
    for entry in active_path(state):
        data = entry.data
        if entry.entry_type == "plan_delta":
            _apply_delta(projected["plan"], data.get("updates", data))
        elif entry.entry_type == "todo_delta":
            _apply_delta(projected["todo"], data.get("updates", data))
        elif entry.entry_type == "working_state":
            projected["working"] = copy.deepcopy(data.get("state", data))
        elif entry.entry_type == "task_checkpoint":
            projected["checkpoint"] = copy.deepcopy(data.get("checkpoint", data))
        elif entry.entry_type == "branch_summary":
            projected["summaries"].append(copy.deepcopy(data))
    return projected


def tree_rows(state):
    active_ids = {entry.entry_id for entry in active_path(state)}
    rows = []
    for entry in state.entries.values():
        rows.append(
            {
                **entry.to_dict(),
                "active": entry.entry_id in active_ids,
                "head": entry.entry_id == state.active_head,
                "children": list(state.children.get(entry.entry_id, [])),
                "labels": sorted(
                    label for label, entry_id in state.labels.items() if entry_id == entry.entry_id
                ),
            }
        )
    return rows


def _validate_entry_data(entry):
    if entry.entry_type == "message":
        if set(entry.data) != {"message"} or not isinstance(entry.data["message"], Mapping):
            raise JournalSchemaError("message tree entry must contain one message object")
    elif entry.entry_type == "tool_exchange":
        if set(entry.data) != {"assistant", "results"}:
            raise JournalSchemaError("tool_exchange must contain assistant and results")
        if not isinstance(entry.data["assistant"], Mapping):
            raise JournalSchemaError("tool_exchange assistant must be an object")
        results = entry.data["results"]
        if not isinstance(results, list) or not all(isinstance(item, Mapping) for item in results):
            raise JournalSchemaError("tool_exchange results must be a list of objects")
        call_ids = [str(item.get("call_id", "")) for item in results]
        if any(not call_id for call_id in call_ids) or len(call_ids) != len(set(call_ids)):
            raise JournalSchemaError("tool_exchange results require unique call_id values")
        calls = entry.data["assistant"].get("tool_calls")
        if not isinstance(calls, list) or not all(isinstance(item, Mapping) for item in calls):
            raise JournalSchemaError("tool_exchange assistant requires tool_calls")
        assistant_call_ids = [str(item.get("call_id", "")) for item in calls]
        if assistant_call_ids != call_ids:
            raise JournalSchemaError(
                "tool_exchange result order must exactly match assistant tool_calls"
            )
    elif entry.entry_type in {"compaction", "context_replacement"}:
        if "history" not in entry.data:
            raise JournalSchemaError(f"{entry.entry_type} must contain history")
        history = entry.data["history"]
        if not isinstance(history, list) or not all(isinstance(item, Mapping) for item in history):
            raise JournalSchemaError(f"{entry.entry_type} history must be a list of objects")


def _encode_parent_key(parent_id):
    return "<root>" if parent_id is None else parent_id


def _decode_parent_key(value):
    return None if value == "<root>" else value


def _apply_delta(target, updates):
    if not isinstance(updates, Mapping):
        raise JournalSchemaError("branch-state delta must be an object")
    for key, value in updates.items():
        if value is None:
            target.pop(str(key), None)
        else:
            target[str(key)] = copy.deepcopy(value)
