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
    HEAD_MOVED,
    JOURNAL_SCHEMA_VERSION,
    REPLAY_POLICIES,
    SESSION_CREATED,
    SESSION_UPDATED,
    TREE_ENTRY_APPENDED,
    TREE_LABEL_UPDATED,
    JournalCorruption,
    JournalSchemaError,
    coerce_journal_record,
    json_copy,
    payload_fields,
    required_text,
)
from .session_tree import (
    SessionTreeEntry,
    SessionTreeState,
    append_entry_in_place,
    label_entry_in_place,
    message_entry,
    move_head_in_place,
    project_history,
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
    tree_entry_ids: tuple[str, ...] = ()

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
            "tree_entry_ids": list(self.tree_entry_ids),
        }


@dataclass(frozen=True)
class EffectCompletion:
    operation: CompletedOperation
    entries: tuple[SessionTreeEntry, ...] = ()


@dataclass(frozen=True)
class JournalState:
    schema_version: str = JOURNAL_SCHEMA_VERSION
    last_sequence: int = 0
    session: dict | None = None
    tree: SessionTreeState | None = None
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
            "tree": self.tree.to_dict() if self.tree is not None else None,
            "open_operation": (
                self.open_operation.to_dict() if self.open_operation else None
            ),
            "completed_operations": {
                operation_id: operation.to_dict()
                for operation_id, operation in self.completed_operations.items()
            },
            "record_fingerprints": dict(self.record_fingerprints),
        }


@dataclass(frozen=True)
class PreparedJournalTransition:
    """Validated record plus the bounded mutation needed by an online projection."""

    record: object
    fingerprint: str
    value: object = None
    duplicate: bool = False


def prepare_journal_record(state, record):
    """Validate one record without copying or mutating the existing state."""

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
        return PreparedJournalTransition(
            record=parsed,
            fingerprint=fingerprint,
            duplicate=True,
        )

    if parsed.kind == SESSION_CREATED:
        value = _session_created_value(state.session, parsed)
    else:
        if state.session is None:
            raise JournalCorruption("journal mutation appears before session_created")
        if parsed.kind == EFFECT_INTENT:
            value = _effect_intent_value(
                state.open_operation, state.completed_operations, parsed
            )
        elif state.open_operation is not None and parsed.kind != EFFECT_RESULT:
            raise JournalCorruption(
                f"cannot apply {parsed.kind} while open operation "
                f"{state.open_operation.operation_id} is unresolved"
            )
        elif parsed.kind == EFFECT_RESULT:
            value = _effect_result_value(state.open_operation, state.tree, parsed)
        elif parsed.kind == HISTORY_APPENDED:
            value = _history_appended_value(parsed)
        elif parsed.kind == HISTORY_REPLACED:
            value = _history_replaced_value(parsed)
        elif parsed.kind == SESSION_UPDATED:
            value = _session_updated_value(parsed)
        elif parsed.kind == TREE_ENTRY_APPENDED:
            value = _tree_entry_appended_value(state.tree, parsed)
        elif parsed.kind == HEAD_MOVED:
            value = _head_moved_value(state.tree, parsed)
        elif parsed.kind == TREE_LABEL_UPDATED:
            value = _tree_label_updated_value(state.tree, parsed)

    return PreparedJournalTransition(
        record=parsed,
        fingerprint=fingerprint,
        value=value,
    )


def apply_prepared_journal_record(state, transition):
    """Advance a writer-owned projection after its record is durable.

    This mutates only the private projection owned by a writer or recovery
    pass. The public ``reduce_journal_record`` function below remains pure.
    """

    parsed = transition.record
    if not transition.duplicate:
        if parsed.kind == SESSION_CREATED:
            session, tree = transition.value
            object.__setattr__(state, "session", session)
            object.__setattr__(state, "tree", tree)
        elif parsed.kind == EFFECT_INTENT:
            object.__setattr__(state, "open_operation", transition.value)
        elif parsed.kind == EFFECT_RESULT:
            completion = transition.value
            state.completed_operations[completion.operation.operation_id] = (
                completion.operation
            )
            for entry in completion.entries:
                append_entry_in_place(state.tree, entry)
            if completion.entries:
                state.session["history"] = project_history(state.tree)
                _advance_runtime_counters(state.session)
            object.__setattr__(state, "open_operation", None)
        elif parsed.kind == HISTORY_APPENDED:
            state.session["history"].append(transition.value)
            append_entry_in_place(
                state.tree,
                message_entry(
                    transition.value,
                    entry_id=_legacy_entry_id(parsed),
                    parent_id=state.tree.active_head,
                ),
            )
        elif parsed.kind == HISTORY_REPLACED:
            state.session["history"] = transition.value
            append_entry_in_place(
                state.tree,
                _replacement_entry(parsed, state.tree.active_head, transition.value),
            )
        elif parsed.kind == SESSION_UPDATED:
            state.session.update(transition.value)
        elif parsed.kind == TREE_ENTRY_APPENDED:
            append_entry_in_place(state.tree, transition.value)
            state.session["history"] = project_history(state.tree)
            _advance_runtime_counters(state.session)
        elif parsed.kind == HEAD_MOVED:
            move_head_in_place(state.tree, transition.value)
            state.session["history"] = project_history(state.tree)
        elif parsed.kind == TREE_LABEL_UPDATED:
            label, entry_id = transition.value
            label_entry_in_place(state.tree, label, entry_id)
        state.record_fingerprints[parsed.record_id] = transition.fingerprint
    object.__setattr__(state, "last_sequence", parsed.sequence)
    return state


def apply_journal_record_in_place(state, record):
    """Validate and advance a private projection in amortized O(record size)."""

    transition = prepare_journal_record(state, record)
    return apply_prepared_journal_record(state, transition)


def reduce_journal_record(state, record):
    """Return a new state after one record without mutating either input."""

    transition = prepare_journal_record(state, record)
    parsed = transition.record
    if transition.duplicate:
        return _copy_state(state, last_sequence=parsed.sequence)

    session = copy.deepcopy(state.session)
    tree = copy.deepcopy(state.tree)
    open_operation = copy.deepcopy(state.open_operation)
    completed_operations = copy.deepcopy(state.completed_operations)

    if parsed.kind == SESSION_CREATED:
        session, tree = copy.deepcopy(transition.value)
    elif parsed.kind == EFFECT_INTENT:
        open_operation = transition.value
    elif parsed.kind == EFFECT_RESULT:
        completion = transition.value
        completed_operations[completion.operation.operation_id] = completion.operation
        for entry in completion.entries:
            append_entry_in_place(tree, entry)
        if completion.entries:
            session["history"] = project_history(tree)
            _advance_runtime_counters(session)
        open_operation = None
    elif parsed.kind == HISTORY_APPENDED:
        session["history"].append(copy.deepcopy(transition.value))
        append_entry_in_place(
            tree,
            message_entry(
                transition.value,
                entry_id=_legacy_entry_id(parsed),
                parent_id=tree.active_head,
            ),
        )
    elif parsed.kind == HISTORY_REPLACED:
        session["history"] = copy.deepcopy(transition.value)
        append_entry_in_place(
            tree,
            _replacement_entry(parsed, tree.active_head, transition.value),
        )
    elif parsed.kind == SESSION_UPDATED:
        session.update(copy.deepcopy(transition.value))
    elif parsed.kind == TREE_ENTRY_APPENDED:
        append_entry_in_place(tree, transition.value)
        session["history"] = project_history(tree)
        _advance_runtime_counters(session)
    elif parsed.kind == HEAD_MOVED:
        move_head_in_place(tree, transition.value)
        session["history"] = project_history(tree)
    elif parsed.kind == TREE_LABEL_UPDATED:
        label, entry_id = transition.value
        label_entry_in_place(tree, label, entry_id)

    fingerprints = dict(state.record_fingerprints)
    fingerprints[parsed.record_id] = transition.fingerprint
    return JournalState(
        schema_version=JOURNAL_SCHEMA_VERSION,
        last_sequence=parsed.sequence,
        session=session,
        tree=tree,
        open_operation=open_operation,
        completed_operations=completed_operations,
        record_fingerprints=fingerprints,
    )


def replay_journal(records):
    state = JournalState.empty()
    for record in records:
        apply_journal_record_in_place(state, record)
    return state


def _session_created_value(session, record):
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
    return canonical, SessionTreeState.from_history(canonical["history"])


def _history_appended_value(record):
    payload = payload_fields(record, required={"item"})
    item = payload["item"]
    if not isinstance(item, Mapping):
        raise JournalSchemaError("history_appended item must be an object")
    return json_copy(dict(item))


def _history_replaced_value(record):
    payload = payload_fields(record, required={"items"})
    items = payload["items"]
    if not isinstance(items, list) or not all(
        isinstance(item, Mapping) for item in items
    ):
        raise JournalSchemaError("history_replaced items must be a list of objects")
    return json_copy(items)


def _session_updated_value(record):
    payload = payload_fields(record, required={"updates"})
    updates = payload["updates"]
    if not isinstance(updates, Mapping):
        raise JournalSchemaError("session_updated updates must be an object")
    forbidden = sorted({"id", "history"} & set(updates))
    if forbidden:
        raise JournalSchemaError(
            f"session_updated cannot replace protected fields: {forbidden}"
        )
    return json_copy(dict(updates))


def _effect_intent_value(open_operation, completed_operations, record):
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


def _effect_result_value(open_operation, tree, record):
    if open_operation is None:
        raise JournalCorruption("effect_result appears without an open operation")
    required = {"effect_type", "call_id", "outcome", "result"}
    optional = {"tree_delta"}
    missing = sorted(required - set(record.payload))
    unexpected = sorted(set(record.payload) - required - optional)
    if missing:
        raise JournalSchemaError(f"effect_result payload missing fields: {missing}")
    if unexpected:
        raise JournalSchemaError(
            f"effect_result payload has unexpected fields: {unexpected}"
        )
    payload = record.payload
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
    entries = _effect_tree_delta_value(tree, payload.get("tree_delta"))
    operation = CompletedOperation(
        operation_id=open_operation.operation_id,
        effect_type=open_operation.effect_type,
        call_id=open_operation.call_id,
        replay_policy=open_operation.replay_policy,
        request=copy.deepcopy(open_operation.request),
        outcome=outcome,
        result=json_copy(dict(payload["result"])),
        intent_sequence=open_operation.intent_sequence,
        result_sequence=record.sequence,
        tree_entry_ids=tuple(entry.entry_id for entry in entries),
    )
    return EffectCompletion(operation=operation, entries=entries)


def _tree_entry_appended_value(tree, record):
    payload = payload_fields(record, required={"entry"})
    if tree is None:
        raise JournalCorruption("tree entry appears before session tree initialization")
    entry = SessionTreeEntry.from_dict(payload["entry"])
    _validate_tree_append(tree, entry)
    return entry


def _head_moved_value(tree, record):
    payload = payload_fields(record, required={"target_entry_id", "reason"})
    target = payload["target_entry_id"]
    if target is not None and (not isinstance(target, str) or not target.strip()):
        raise JournalSchemaError("target_entry_id must be a non-empty string or null")
    if not isinstance(payload["reason"], str):
        raise JournalSchemaError("head move reason must be a string")
    if target is not None and target not in tree.entries:
        raise JournalCorruption(f"tree head target does not exist: {target}")
    return target


def _tree_label_updated_value(tree, record):
    payload = payload_fields(record, required={"label", "entry_id"})
    label = required_text(payload["label"], "label")
    entry_id = required_text(payload["entry_id"], "entry_id")
    if entry_id not in tree.entries:
        raise JournalCorruption(f"tree label target does not exist: {entry_id}")
    return label, entry_id


def _effect_tree_delta_value(tree, value):
    if value is None:
        return ()
    if not isinstance(value, Mapping) or set(value) != {"expected_head", "entries"}:
        raise JournalSchemaError(
            "effect_result tree_delta must contain expected_head and entries"
        )
    expected_head = value["expected_head"]
    if expected_head != tree.active_head:
        raise JournalCorruption(
            "effect tree delta head mismatch: "
            f"expected {tree.active_head}, got {expected_head}"
        )
    raw_entries = value["entries"]
    if not isinstance(raw_entries, list) or not raw_entries:
        raise JournalSchemaError("effect_result tree_delta entries must be non-empty")
    entries = tuple(SessionTreeEntry.from_dict(item) for item in raw_entries)
    known = set(tree.entries)
    parent = tree.active_head
    for entry in entries:
        if entry.entry_id in known:
            raise JournalCorruption(f"tree entry id conflict: {entry.entry_id}")
        if entry.parent_id != parent:
            raise JournalCorruption(
                "effect tree delta must be a contiguous append from expected_head"
            )
        known.add(entry.entry_id)
        parent = entry.entry_id
    return entries


def _validate_tree_append(tree, entry):
    if entry.entry_id in tree.entries:
        if tree.entries[entry.entry_id] != entry:
            raise JournalCorruption(f"tree entry id conflict: {entry.entry_id}")
        return
    if entry.parent_id is not None and entry.parent_id not in tree.entries:
        raise JournalCorruption(f"tree parent does not exist: {entry.parent_id}")
    if entry.parent_id != tree.active_head:
        raise JournalCorruption(
            "tree append parent is not the active head: "
            f"expected {tree.active_head}, got {entry.parent_id}"
        )


def _legacy_entry_id(record):
    return f"entry_{record.record_id}"


def _replacement_entry(record, parent_id, history):
    return SessionTreeEntry(
        entry_id=_legacy_entry_id(record),
        parent_id=parent_id,
        entry_type="context_replacement",
        turn_id="",
        run_id="",
        created_at="",
        data={"history": json_copy(history), "source": "legacy_history_replaced"},
    )


def _copy_state(state, *, last_sequence):
    return JournalState(
        schema_version=state.schema_version,
        last_sequence=last_sequence,
        session=copy.deepcopy(state.session),
        tree=copy.deepcopy(state.tree),
        open_operation=copy.deepcopy(state.open_operation),
        completed_operations=copy.deepcopy(state.completed_operations),
        record_fingerprints=dict(state.record_fingerprints),
    )


def _advance_runtime_counters(session):
    event_seq = int(session.get("_event_seq", 0) or 0)
    manual_turn_seq = int(session.get("_manual_turn_seq", 0) or 0)
    for item in session.get("history", []):
        event_id = str(item.get("event_id", ""))
        if event_id.startswith("event_") and event_id[6:].isdigit():
            event_seq = max(event_seq, int(event_id[6:]))
        turn_id = str(item.get("turn_id", ""))
        if turn_id.startswith("manual_") and turn_id[7:].isdigit():
            manual_turn_seq = max(manual_turn_seq, int(turn_id[7:]))
    if event_seq:
        session["_event_seq"] = event_seq
    if manual_turn_seq:
        session["_manual_turn_seq"] = manual_turn_seq
