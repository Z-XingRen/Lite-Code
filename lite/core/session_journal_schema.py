"""Versioned record schema for the append-only session journal."""

import copy
import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass


JOURNAL_SCHEMA_VERSION = "lite.session_journal.v1"

SESSION_CREATED = "session_created"
HISTORY_APPENDED = "history_appended"
HISTORY_REPLACED = "history_replaced"
SESSION_UPDATED = "session_updated"
EFFECT_INTENT = "effect_intent"
EFFECT_RESULT = "effect_result"

JOURNAL_RECORD_KINDS = frozenset(
    {
        SESSION_CREATED,
        HISTORY_APPENDED,
        HISTORY_REPLACED,
        SESSION_UPDATED,
        EFFECT_INTENT,
        EFFECT_RESULT,
    }
)
EFFECT_TYPES = frozenset(
    {"provider", "tool", "permission", "cancel", "retry", "snapshot"}
)
REPLAY_POLICIES = frozenset({"replay_safe", "interrupt"})
EFFECT_OUTCOMES = frozenset({"ok", "error", "interrupted"})

_RECORD_FIELDS = frozenset(
    {"schema_version", "sequence", "record_id", "operation_id", "kind", "payload"}
)


class JournalSchemaError(ValueError):
    """A journal record does not conform to its declared schema version."""


class JournalCorruption(ValueError):
    """A valid record would move the journal into an impossible state."""


@dataclass(frozen=True)
class JournalRecord:
    schema_version: str
    sequence: int
    record_id: str
    operation_id: str
    kind: str
    payload: dict

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, Mapping):
            raise JournalSchemaError("journal record must be an object")
        missing = sorted(_RECORD_FIELDS - set(value))
        unexpected = sorted(set(value) - _RECORD_FIELDS)
        if missing:
            raise JournalSchemaError(f"journal record missing fields: {missing}")
        if unexpected:
            raise JournalSchemaError(f"journal record has unexpected fields: {unexpected}")

        schema_version = required_text(value["schema_version"], "schema_version")
        if schema_version != JOURNAL_SCHEMA_VERSION:
            raise JournalSchemaError(
                f"unsupported journal schema: {schema_version}"
            )
        sequence = value["sequence"]
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence <= 0:
            raise JournalSchemaError("sequence must be a positive integer")
        record_id = required_text(value["record_id"], "record_id")
        operation_id = required_text(value["operation_id"], "operation_id")
        kind = required_text(value["kind"], "kind")
        if kind not in JOURNAL_RECORD_KINDS:
            raise JournalSchemaError(f"unsupported journal record kind: {kind}")
        if not isinstance(value["payload"], Mapping):
            raise JournalSchemaError("payload must be an object")

        return cls(
            schema_version=schema_version,
            sequence=sequence,
            record_id=record_id,
            operation_id=operation_id,
            kind=kind,
            payload=json_copy(dict(value["payload"])),
        )

    def to_dict(self):
        return {
            "schema_version": self.schema_version,
            "sequence": self.sequence,
            "record_id": self.record_id,
            "operation_id": self.operation_id,
            "kind": self.kind,
            "payload": copy.deepcopy(self.payload),
        }

    def content_fingerprint(self):
        content = canonical_json(
            {
                "schema_version": self.schema_version,
                "operation_id": self.operation_id,
                "kind": self.kind,
                "payload": self.payload,
            }
        ).encode("utf-8")
        return hashlib.sha256(content).hexdigest()


def coerce_journal_record(record):
    if isinstance(record, JournalRecord):
        return JournalRecord.from_dict(record.to_dict())
    return JournalRecord.from_dict(record)


def payload_fields(record, *, required):
    missing = sorted(required - set(record.payload))
    unexpected = sorted(set(record.payload) - required)
    if missing:
        raise JournalSchemaError(f"{record.kind} payload missing fields: {missing}")
    if unexpected:
        raise JournalSchemaError(
            f"{record.kind} payload has unexpected fields: {unexpected}"
        )
    return record.payload


def required_text(value, field_name):
    if not isinstance(value, str) or not value.strip():
        raise JournalSchemaError(f"{field_name} must be a non-empty string")
    return value


def json_copy(value):
    try:
        _validate_json_value(value)
        return copy.deepcopy(value)
    except (TypeError, ValueError) as exc:
        raise JournalSchemaError("journal payload must be JSON-compatible") from exc


def canonical_json(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _validate_json_value(value):
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite float")
        return
    if isinstance(value, list):
        for item in value:
            _validate_json_value(item)
        return
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("JSON object keys must be strings")
        for item in value.values():
            _validate_json_value(item)
        return
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")
