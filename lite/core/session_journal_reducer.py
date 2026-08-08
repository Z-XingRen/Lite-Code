"""Pure state transitions and invariants for session journal replay."""

import copy
from collections.abc import Mapping
from dataclasses import dataclass, field

from .session_journal_schema import (
    EFFECT_INTENT,
    EFFECT_OUTCOMES,
    EFFECT_RESULT,
    EFFECT_TYPES,
    HISTORY_APPENDED,
    HISTORY_REPLACED,
    JOURNAL_SCHEMA_VERSION,
    REPLAY_POLICIES,
    SESSION_CREATED,
    SESSION_UPDATED,
    JournalCorruption,
    JournalSchemaError,
    coerce_journal_record,
    json_copy,
    payload_fields,
    required_text,
)


@dataclass(frozen=True)
class OpenOperation:
    operation_id: str
    effect_type: str
    call_id: str
    replay_policy: str
    request: dict
    intent_sequence: int

    def to_dict(self):
        return {
            "operation_id": self.operation_id,
            "effect_type": self.effect_type,
            "call_id": self.call_id,
            "replay_policy": self.replay_policy,
            "request": copy.deepcopy(self.request),
            "intent_sequence": self.intent_sequence,
        }


@dataclass(frozen=True)
class CompletedOperation:
    operation_id: str
    effect_type: str
    call_id: str
    replay_policy: str
    request: dict
    outcome: str
    result: dict
    intent_sequence: int
    result_sequence: int

    def to_dict(self):
        return {
            "operation_id": self.operation_id,
            "effect_type": self.effect_type,
            "call_id": self.call_id,
            "replay_policy": self.replay_policy,
            "request": copy.deepcopy(self.request),
            "outcome": self.outcome,
            "result": copy.deepcopy(self.result),
            "intent_sequence": self.intent_sequence,
            "result_sequence": self.result_sequence,
        }


@dataclass(frozen=True)
class JournalState:
    schema_version: str = JOURNAL_SCHEMA_VERSION
    last_sequence: int = 0
    session: dict | None = None
    open_operation: OpenOperation | None = None
    completed_operations: dict[str, CompletedOperation] = field(default_factory=dict)
    record_fingerprints: dict[str, str] = field(default_factory=dict)

    @classmethod
    def empty(cls):
        return cls()

    def to_dict(self):
        return {
            "schema_version": self.schema_version,
            "last_sequence": self.last_sequence,
            "session": copy.deepcopy(self.session),
            "open_operation": (
                self.open_operation.to_dict() if self.open_operation else None
            ),
            "completed_operations": {
                operation_id: operation.to_dict()
                for operation_id, operation in self.completed_operations.items()
            },
            "record_fingerprints": dict(self.record_fingerprints),
        }


def reduce_journal_record(state, record):
    """Return a new state after one record without mutating either input."""

    if not isinstance(state, JournalState):
        raise TypeError("state must be a JournalState")
    if state.schema_version != JOURNAL_SCHEMA_VERSION:
        raise JournalSchemaError(
            f"unsupported journal state schema: {state.schema_version}"
        )
    parsed = coerce_journal_record(record)
    expected = state.last_sequence + 1
    if parsed.sequence != expected:
        raise JournalCorruption(
            f"expected sequence {expected}, got {parsed.sequence}"
        )

    fingerprint = parsed.content_fingerprint()
    prior_fingerprint = state.record_fingerprints.get(parsed.record_id)
    if prior_fingerprint is not None:
        if prior_fingerprint != fingerprint:
            raise JournalCorruption(
                f"record_id content conflict: {parsed.record_id}"
            )
        return _copy_state(state, last_sequence=parsed.sequence)

    session = copy.deepcopy(state.session)
    open_operation = copy.deepcopy(state.open_operation)
    completed_operations = copy.deepcopy(state.completed_operations)

    if parsed.kind == SESSION_CREATED:
        session = _reduce_session_created(session, parsed)
    else:
        if session is None:
            raise JournalCorruption("journal mutation appears before session_created")
        if parsed.kind == EFFECT_INTENT:
            open_operation = _reduce_effect_intent(
                open_operation, completed_operations, parsed
            )
        elif open_operation is not None and parsed.kind != EFFECT_RESULT:
            raise JournalCorruption(
                f"cannot apply {parsed.kind} while open operation "
                f"{open_operation.operation_id} is unresolved"
            )
        elif parsed.kind == EFFECT_RESULT:
            completed = _reduce_effect_result(open_operation, parsed)
            completed_operations[completed.operation_id] = completed
            open_operation = None
        elif parsed.kind == HISTORY_APPENDED:
            session = _reduce_history_appended(session, parsed)
        elif parsed.kind == HISTORY_REPLACED:
            session = _reduce_history_replaced(session, parsed)
        elif parsed.kind == SESSION_UPDATED:
            session = _reduce_session_updated(session, parsed)

    fingerprints = dict(state.record_fingerprints)
    fingerprints[parsed.record_id] = fingerprint
    return JournalState(
        schema_version=JOURNAL_SCHEMA_VERSION,
        last_sequence=parsed.sequence,
        session=session,
        open_operation=open_operation,
        completed_operations=completed_operations,
        record_fingerprints=fingerprints,
    )


def replay_journal(records):
    state = JournalState.empty()
    for record in records:
        state = reduce_journal_record(state, record)
    return state


def _reduce_session_created(session, record):
    if session is not None:
        raise JournalCorruption("journal contains more than one session_created record")
    payload = payload_fields(record, required={"session"})
    value = payload["session"]
    if not isinstance(value, Mapping):
        raise JournalSchemaError("session_created session must be an object")
    canonical = json_copy(dict(value))
    if not str(canonical.get("id", "")).strip():
        raise JournalSchemaError("session_created session.id must be non-empty")
    history = canonical.get("history")
    if not isinstance(history, list) or not all(
        isinstance(item, Mapping) for item in history
    ):
        raise JournalSchemaError("session_created session.history must be a list of objects")
    return canonical


def _reduce_history_appended(session, record):
    payload = payload_fields(record, required={"item"})
    item = payload["item"]
    if not isinstance(item, Mapping):
        raise JournalSchemaError("history_appended item must be an object")
    updated = copy.deepcopy(session)
    updated["history"] = [
        *copy.deepcopy(updated.get("history", [])),
        json_copy(dict(item)),
    ]
    return updated


def _reduce_history_replaced(session, record):
    payload = payload_fields(record, required={"items"})
    items = payload["items"]
    if not isinstance(items, list) or not all(
        isinstance(item, Mapping) for item in items
    ):
        raise JournalSchemaError("history_replaced items must be a list of objects")
    updated = copy.deepcopy(session)
    updated["history"] = json_copy(items)
    return updated


def _reduce_session_updated(session, record):
    payload = payload_fields(record, required={"updates"})
    updates = payload["updates"]
    if not isinstance(updates, Mapping):
        raise JournalSchemaError("session_updated updates must be an object")
    forbidden = sorted({"id", "history"} & set(updates))
    if forbidden:
        raise JournalSchemaError(
            f"session_updated cannot replace protected fields: {forbidden}"
        )
    updated = copy.deepcopy(session)
    updated.update(json_copy(dict(updates)))
    return updated


def _reduce_effect_intent(open_operation, completed_operations, record):
    if open_operation is not None:
        raise JournalCorruption(
            f"operation {open_operation.operation_id} is already open"
        )
    if record.operation_id in completed_operations:
        raise JournalCorruption(
            f"operation_id was already completed: {record.operation_id}"
        )
    payload = payload_fields(
        record,
        required={"effect_type", "call_id", "replay_policy", "request"},
    )
    effect_type = required_text(payload["effect_type"], "effect_type")
    call_id = required_text(payload["call_id"], "call_id")
    replay_policy = required_text(payload["replay_policy"], "replay_policy")
    if effect_type not in EFFECT_TYPES:
        raise JournalSchemaError(f"unsupported effect_type: {effect_type}")
    if replay_policy not in REPLAY_POLICIES:
        raise JournalSchemaError(f"unsupported replay_policy: {replay_policy}")
    if not isinstance(payload["request"], Mapping):
        raise JournalSchemaError("effect_intent request must be an object")
    return OpenOperation(
        operation_id=record.operation_id,
        effect_type=effect_type,
        call_id=call_id,
        replay_policy=replay_policy,
        request=json_copy(dict(payload["request"])),
        intent_sequence=record.sequence,
    )


def _reduce_effect_result(open_operation, record):
    if open_operation is None:
        raise JournalCorruption("effect_result appears without an open operation")
    payload = payload_fields(
        record, required={"effect_type", "call_id", "outcome", "result"}
    )
    effect_type = required_text(payload["effect_type"], "effect_type")
    call_id = required_text(payload["call_id"], "call_id")
    outcome = required_text(payload["outcome"], "outcome")
    if record.operation_id != open_operation.operation_id:
        raise JournalCorruption(
            "effect result operation_id mismatch: "
            f"expected {open_operation.operation_id}, got {record.operation_id}"
        )
    if effect_type != open_operation.effect_type:
        raise JournalCorruption(
            "effect result effect_type mismatch: "
            f"expected {open_operation.effect_type}, got {effect_type}"
        )
    if call_id != open_operation.call_id:
        raise JournalCorruption(
            "effect result call_id mismatch: "
            f"expected {open_operation.call_id}, got {call_id}"
        )
    if outcome not in EFFECT_OUTCOMES:
        raise JournalSchemaError(f"unsupported effect outcome: {outcome}")
    if not isinstance(payload["result"], Mapping):
        raise JournalSchemaError("effect_result result must be an object")
    return CompletedOperation(
        operation_id=open_operation.operation_id,
        effect_type=open_operation.effect_type,
        call_id=open_operation.call_id,
        replay_policy=open_operation.replay_policy,
        request=copy.deepcopy(open_operation.request),
        outcome=outcome,
        result=json_copy(dict(payload["result"])),
        intent_sequence=open_operation.intent_sequence,
        result_sequence=record.sequence,
    )


def _copy_state(state, *, last_sequence):
    return JournalState(
        schema_version=state.schema_version,
        last_sequence=last_sequence,
        session=copy.deepcopy(state.session),
        open_operation=copy.deepcopy(state.open_operation),
        completed_operations=copy.deepcopy(state.completed_operations),
        record_fingerprints=dict(state.record_fingerprints),
    )
