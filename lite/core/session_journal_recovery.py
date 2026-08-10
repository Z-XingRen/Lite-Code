"""Atomic snapshots and deterministic restoration for session journals."""

import hashlib
import json
import os
import threading
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from .session_journal_reducer import (
    CompletedOperation,
    JournalState,
    OpenOperation,
    apply_journal_record_in_place,
    replay_journal,
)
from .session_journal_schema import (
    JOURNAL_SCHEMA_VERSION,
    JournalCorruption,
    JournalRecord,
    canonical_json,
    json_copy,
)
from .session_tree import SessionTreeState, project_history


SNAPSHOT_SCHEMA_VERSION = "lite.session_journal.snapshot.v1"
_SNAPSHOT_FIELDS = frozenset(
    {
        "snapshot_schema_version",
        "journal_schema_version",
        "journal_offset",
        "journal_prefix_sha256",
        "state",
        "checksum",
    }
)


@dataclass(frozen=True)
class JournalRestore:
    state: JournalState
    complete_size: int
    discarded_tail: bytes
    used_snapshot: bool


@dataclass(frozen=True)
class JournalRecoveryAction:
    operation_id: str
    effect_type: str
    call_id: str
    action: str
    request: dict
    result_sequence: int


def snapshot_path_for(journal_path):
    path = Path(journal_path)
    return path.with_name(f"{path.name}.snapshot.json")


def restore_session_journal(path, *, snapshot_path=None):
    """Restore a journal, ignoring only a final unterminated crash fragment."""

    journal_path = Path(path)
    raw = journal_path.read_bytes()
    complete, discarded_tail = _split_complete_records(raw)
    if not complete:
        raise JournalCorruption("journal contains no complete records")
    records, offsets = _parse_complete_records(complete)

    candidate = _load_snapshot(
        Path(snapshot_path) if snapshot_path else snapshot_path_for(journal_path),
        complete,
        offsets,
    )
    if candidate is None:
        state = replay_journal(records)
        used_snapshot = False
    else:
        state, journal_offset = candidate
        tail_index = offsets.index(journal_offset) + 1
        for record in records[tail_index:]:
            apply_journal_record_in_place(state, record)
        used_snapshot = True
    return JournalRestore(
        state=state,
        complete_size=len(complete),
        discarded_tail=discarded_tail,
        used_snapshot=used_snapshot,
    )


def write_atomic_snapshot(snapshot_path, journal_path, state, *, sync=True):
    """Atomically persist a reducer state tied to an exact journal prefix."""

    target = Path(snapshot_path)
    journal = Path(journal_path)
    prefix = journal.read_bytes()
    if not prefix.endswith(b"\n"):
        raise JournalCorruption("cannot snapshot an unterminated journal")
    records, offsets = _parse_complete_records(prefix)
    if not records or records[-1].sequence != state.last_sequence:
        raise JournalCorruption("snapshot state does not match journal tail")
    if offsets[-1] != len(prefix):
        raise JournalCorruption("snapshot journal offset is not a record boundary")

    body = {
        "snapshot_schema_version": SNAPSHOT_SCHEMA_VERSION,
        "journal_schema_version": JOURNAL_SCHEMA_VERSION,
        "journal_offset": len(prefix),
        "journal_prefix_sha256": hashlib.sha256(prefix).hexdigest(),
        "state": state.to_dict(),
    }
    document = {
        **body,
        "checksum": hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest(),
    }
    payload = (canonical_json(document) + "\n").encode("utf-8")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(
        f".{target.name}.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with temporary.open("xb") as output:
            output.write(payload)
            output.flush()
            if sync:
                os.fsync(output.fileno())
        os.replace(temporary, target)
        if sync:
            _sync_directory(target.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return target


def recovery_actions_from_state(state):
    actions = []
    for operation in sorted(
        state.completed_operations.values(), key=lambda item: item.result_sequence
    ):
        result = operation.result
        if (
            operation.outcome != "interrupted"
            or result.get("synthetic") is not True
            or result.get("reason") != "process_interrupted"
        ):
            continue
        action = result.get("recovery_action")
        if action not in {"retry", "interrupt"}:
            continue
        actions.append(
            JournalRecoveryAction(
                operation_id=operation.operation_id,
                effect_type=operation.effect_type,
                call_id=operation.call_id,
                action=action,
                request=json_copy(operation.request),
                result_sequence=operation.result_sequence,
            )
        )
    return tuple(actions)


def _split_complete_records(raw):
    if raw.endswith(b"\n"):
        return raw, b""
    boundary = raw.rfind(b"\n") + 1
    return raw[:boundary], raw[boundary:]


def _parse_complete_records(raw):
    records = []
    offsets = []
    offset = 0
    for line_number, line in enumerate(raw.splitlines(keepends=True), start=1):
        offset += len(line)
        if not line.endswith(b"\n"):
            raise JournalCorruption(
                f"journal line {line_number} is not newline terminated"
            )
        try:
            value = json.loads(line.decode("utf-8"))
            record = JournalRecord.from_dict(value)
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
            raise JournalCorruption(
                f"invalid journal record at line {line_number}: {exc}"
            ) from exc
        records.append(record)
        offsets.append(offset)
    return records, offsets


def _load_snapshot(path, journal, offsets):
    try:
        raw = path.read_text(encoding="utf-8")
        document = json.loads(raw)
        body, checksum = _validate_snapshot_document(document)
        if hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest() != checksum:
            return None
        journal_offset = body["journal_offset"]
        if journal_offset not in offsets:
            raise JournalCorruption("snapshotted journal prefix is missing")
        prefix = journal[:journal_offset]
        if hashlib.sha256(prefix).hexdigest() != body["journal_prefix_sha256"]:
            raise JournalCorruption("snapshotted journal prefix changed")
        state = _journal_state_from_dict(body["state"])
        if state.last_sequence != offsets.index(journal_offset) + 1:
            return None
        return state, journal_offset
    except JournalCorruption:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
        return None


def _validate_snapshot_document(document):
    if not isinstance(document, Mapping) or set(document) != _SNAPSHOT_FIELDS:
        raise ValueError("invalid snapshot fields")
    if document["snapshot_schema_version"] != SNAPSHOT_SCHEMA_VERSION:
        raise ValueError("unsupported snapshot schema")
    if document["journal_schema_version"] != JOURNAL_SCHEMA_VERSION:
        raise ValueError("unsupported journal schema")
    offset = document["journal_offset"]
    if isinstance(offset, bool) or not isinstance(offset, int) or offset <= 0:
        raise ValueError("invalid journal offset")
    prefix_hash = document["journal_prefix_sha256"]
    checksum = document["checksum"]
    if not _is_sha256(prefix_hash) or not _is_sha256(checksum):
        raise ValueError("invalid snapshot hash")
    body = {key: document[key] for key in document if key != "checksum"}
    return body, checksum


def _journal_state_from_dict(value):
    required = {
        "schema_version",
        "last_sequence",
        "session",
        "open_operation",
        "completed_operations",
        "record_fingerprints",
    }
    if not isinstance(value, Mapping) or frozenset(value) not in {
        frozenset(required),
        frozenset(required | {"tree"}),
    }:
        raise ValueError("invalid snapshot state fields")
    if value["schema_version"] != JOURNAL_SCHEMA_VERSION:
        raise ValueError("unsupported snapshot state schema")
    last_sequence = value["last_sequence"]
    if (
        isinstance(last_sequence, bool)
        or not isinstance(last_sequence, int)
        or last_sequence <= 0
    ):
        raise ValueError("invalid snapshot sequence")
    if not isinstance(value["session"], Mapping):
        raise ValueError("snapshot session must be an object")
    session = json_copy(dict(value["session"]))
    tree_value = value.get("tree")
    if tree_value is None:
        # A pre-tree snapshot cannot prove that nodes hidden by an earlier
        # history_replaced record were retained. Fall back to full replay.
        raise ValueError("snapshot predates the session tree projection")
    tree = SessionTreeState.from_dict(tree_value)
    if session.get("history", []) != project_history(tree):
        raise ValueError("snapshot session history does not match active tree")
    open_operation = _open_operation_from_dict(value["open_operation"])
    completed = value["completed_operations"]
    if not isinstance(completed, Mapping):
        raise ValueError("completed_operations must be an object")
    completed_operations = {
        operation_id: _completed_operation_from_dict(operation_id, operation)
        for operation_id, operation in completed.items()
    }
    if open_operation and open_operation.operation_id in completed_operations:
        raise ValueError("open operation is already completed")
    fingerprints = value["record_fingerprints"]
    if not isinstance(fingerprints, Mapping) or not all(
        isinstance(record_id, str)
        and record_id
        and _is_sha256(fingerprint)
        for record_id, fingerprint in fingerprints.items()
    ):
        raise ValueError("invalid record fingerprints")
    if open_operation and open_operation.intent_sequence > last_sequence:
        raise ValueError("open operation exceeds snapshot sequence")
    if any(
        operation.result_sequence > last_sequence
        for operation in completed_operations.values()
    ):
        raise ValueError("completed operation exceeds snapshot sequence")
    return JournalState(
        schema_version=JOURNAL_SCHEMA_VERSION,
        last_sequence=last_sequence,
        session=session,
        tree=tree,
        open_operation=open_operation,
        completed_operations=completed_operations,
        record_fingerprints=dict(fingerprints),
    )


def _open_operation_from_dict(value):
    if value is None:
        return None
    required = {
        "operation_id",
        "effect_type",
        "call_id",
        "replay_policy",
        "request",
        "intent_sequence",
    }
    _validate_operation_fields(value, required)
    return OpenOperation(
        operation_id=value["operation_id"],
        effect_type=value["effect_type"],
        call_id=value["call_id"],
        replay_policy=value["replay_policy"],
        request=json_copy(dict(value["request"])),
        intent_sequence=value["intent_sequence"],
    )


def _completed_operation_from_dict(operation_id, value):
    required = {
        "operation_id",
        "effect_type",
        "call_id",
        "replay_policy",
        "request",
        "outcome",
        "result",
        "intent_sequence",
        "result_sequence",
    }
    if not isinstance(value, Mapping) or frozenset(value) not in {
        frozenset(required),
        frozenset(required | {"tree_entry_ids"}),
    }:
        raise ValueError("invalid operation fields")
    _validate_operation_fields({key: value[key] for key in required}, required)
    if not isinstance(operation_id, str) or value["operation_id"] != operation_id:
        raise ValueError("completed operation id mismatch")
    if value["outcome"] not in {"ok", "error", "interrupted"}:
        raise ValueError("invalid completed operation outcome")
    if not isinstance(value["result"], Mapping):
        raise ValueError("completed operation result must be an object")
    result_sequence = value["result_sequence"]
    if (
        isinstance(result_sequence, bool)
        or not isinstance(result_sequence, int)
        or result_sequence <= value["intent_sequence"]
    ):
        raise ValueError("invalid completed operation sequence")
    tree_entry_ids = value.get("tree_entry_ids", [])
    if not isinstance(tree_entry_ids, list) or not all(
        isinstance(entry_id, str) and entry_id for entry_id in tree_entry_ids
    ):
        raise ValueError("invalid completed operation tree entries")
    return CompletedOperation(
        operation_id=operation_id,
        effect_type=value["effect_type"],
        call_id=value["call_id"],
        replay_policy=value["replay_policy"],
        request=json_copy(dict(value["request"])),
        outcome=value["outcome"],
        result=json_copy(dict(value["result"])),
        intent_sequence=value["intent_sequence"],
        result_sequence=result_sequence,
        tree_entry_ids=tuple(tree_entry_ids),
    )


def _validate_operation_fields(value, required):
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("invalid operation fields")
    if not all(
        isinstance(value[field], str) and value[field]
        for field in ("operation_id", "effect_type", "call_id", "replay_policy")
    ):
        raise ValueError("invalid operation identity")
    if value["effect_type"] not in {
        "provider",
        "tool",
        "permission",
        "cancel",
        "retry",
        "snapshot",
    }:
        raise ValueError("invalid effect type")
    if value["replay_policy"] not in {"replay_safe", "interrupt"}:
        raise ValueError("invalid replay policy")
    if not isinstance(value["request"], Mapping):
        raise ValueError("operation request must be an object")
    intent_sequence = value["intent_sequence"]
    if (
        isinstance(intent_sequence, bool)
        or not isinstance(intent_sequence, int)
        or intent_sequence <= 0
    ):
        raise ValueError("invalid intent sequence")


def _is_sha256(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _sync_directory(path):
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
