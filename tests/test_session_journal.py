import copy
import json
from pathlib import Path

import pytest

from lite.core.session_journal import (
    JOURNAL_SCHEMA_VERSION,
    JournalCorruption,
    JournalRecord,
    JournalSchemaError,
    JournalState,
    reduce_journal_record,
    replay_journal,
)


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "session_journal_v1.jsonl"


def record(sequence, kind, *, record_id=None, operation_id=None, payload=None):
    return {
        "schema_version": JOURNAL_SCHEMA_VERSION,
        "sequence": sequence,
        "record_id": record_id or f"record-{sequence}",
        "operation_id": operation_id or f"operation-{sequence}",
        "kind": kind,
        "payload": payload or {},
    }


def session_created(sequence=1, **overrides):
    session = {
        "id": "session-1",
        "created_at": "2026-08-08T00:00:00+00:00",
        "workspace_root": "C:/workspace",
        "history": [],
    }
    session.update(overrides)
    return record(
        sequence,
        "session_created",
        record_id="record-session",
        operation_id="session-1",
        payload={"session": session},
    )


def effect_intent(sequence=2, **overrides):
    payload = {
        "effect_type": "tool",
        "call_id": "call-1",
        "replay_policy": "interrupt",
        "request": {"name": "write_file", "args": {"path": "result.txt"}},
    }
    payload.update(overrides.pop("payload", {}))
    return record(
        sequence,
        "effect_intent",
        record_id=overrides.pop("record_id", f"record-{sequence}"),
        operation_id=overrides.pop("operation_id", "operation-effect-1"),
        payload=payload,
        **overrides,
    )


def effect_result(sequence=3, **overrides):
    payload = {
        "effect_type": "tool",
        "call_id": "call-1",
        "outcome": "ok",
        "result": {"content": "wrote result.txt"},
    }
    payload.update(overrides.pop("payload", {}))
    return record(
        sequence,
        "effect_result",
        record_id=overrides.pop("record_id", f"record-{sequence}"),
        operation_id=overrides.pop("operation_id", "operation-effect-1"),
        payload=payload,
        **overrides,
    )


def test_v1_fixture_replays_to_deterministic_canonical_state():
    records = [
        json.loads(line)
        for line in FIXTURE_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    state = replay_journal(records)

    assert state.schema_version == JOURNAL_SCHEMA_VERSION
    assert state.last_sequence == 4
    assert state.open_operation is None
    assert state.session["id"] == "fixture-session"
    assert state.session["history"] == [
        {"role": "assistant", "content": "Done.", "event_id": "event_000001"}
    ]
    assert state.completed_operations["op-provider-1"].outcome == "ok"
    assert replay_journal(records) == state


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", "lite.session_journal.v2", "unsupported journal schema"),
        ("sequence", True, "sequence must be a positive integer"),
        ("sequence", 0, "sequence must be a positive integer"),
        ("record_id", "", "record_id must be a non-empty string"),
        ("operation_id", "", "operation_id must be a non-empty string"),
        ("kind", "unknown", "unsupported journal record kind"),
        ("payload", [], "payload must be an object"),
    ],
)
def test_record_schema_rejects_invalid_contract(field, value, message):
    data = session_created()
    data[field] = value

    with pytest.raises(JournalSchemaError, match=message):
        JournalRecord.from_dict(data)


def test_record_schema_rejects_unknown_fields_and_non_json_payloads():
    data = session_created()
    data["unexpected"] = True
    with pytest.raises(JournalSchemaError, match="unexpected fields"):
        JournalRecord.from_dict(data)

    data = session_created()
    data["payload"]["session"]["bad"] = float("nan")
    with pytest.raises(JournalSchemaError, match="JSON-compatible"):
        JournalRecord.from_dict(data)


def test_reducer_is_pure_and_does_not_alias_record_payloads():
    initial = reduce_journal_record(JournalState.empty(), session_created())
    before = copy.deepcopy(initial.to_dict())
    append = record(
        2,
        "history_appended",
        payload={"item": {"role": "user", "content": "hello"}},
    )

    updated = reduce_journal_record(initial, append)
    append["payload"]["item"]["content"] = "mutated"

    assert initial.to_dict() == before
    assert initial.session["history"] == []
    assert updated.session["history"] == [{"role": "user", "content": "hello"}]


@pytest.mark.parametrize("sequence", [1, 3])
def test_reducer_rejects_duplicate_or_gapped_sequences(sequence):
    state = reduce_journal_record(JournalState.empty(), session_created())
    append = record(
        sequence,
        "history_appended",
        payload={"item": {"role": "user", "content": "hello"}},
    )

    with pytest.raises(JournalCorruption, match="expected sequence 2"):
        reduce_journal_record(state, append)


def test_identical_provisioned_record_id_is_idempotent_but_conflict_is_corrupt():
    records = [
        session_created(),
        record(
            2,
            "history_appended",
            record_id="provisioned-history",
            operation_id="history-1",
            payload={"item": {"role": "user", "content": "hello"}},
        ),
        record(
            3,
            "history_appended",
            record_id="provisioned-history",
            operation_id="history-1",
            payload={"item": {"role": "user", "content": "hello"}},
        ),
    ]

    state = replay_journal(records)

    assert state.last_sequence == 3
    assert state.session["history"] == [{"role": "user", "content": "hello"}]
    conflict = record(
        4,
        "history_appended",
        record_id="provisioned-history",
        operation_id="history-1",
        payload={"item": {"role": "user", "content": "different"}},
    )
    with pytest.raises(JournalCorruption, match="record_id content conflict"):
        reduce_journal_record(state, conflict)


def test_reducer_rejects_two_open_operations_and_mutation_during_effect():
    state = replay_journal([session_created(), effect_intent()])

    with pytest.raises(JournalCorruption, match="already open"):
        reduce_journal_record(
            state,
            effect_intent(
                3,
                operation_id="operation-effect-2",
                payload={"call_id": "call-2"},
            ),
        )

    with pytest.raises(JournalCorruption, match="open operation"):
        reduce_journal_record(
            state,
            record(
                3,
                "history_appended",
                payload={"item": {"role": "tool", "content": "too early"}},
            ),
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"operation_id": "wrong-operation"}, "operation_id mismatch"),
        ({"payload": {"effect_type": "provider"}}, "effect_type mismatch"),
        ({"payload": {"call_id": "wrong-call"}}, "call_id mismatch"),
    ],
)
def test_effect_result_must_match_the_open_intent(overrides, message):
    state = replay_journal([session_created(), effect_intent()])

    with pytest.raises(JournalCorruption, match=message):
        reduce_journal_record(state, effect_result(**overrides))


def test_effect_result_requires_intent_and_operation_ids_cannot_be_reused():
    state = reduce_journal_record(JournalState.empty(), session_created())
    with pytest.raises(JournalCorruption, match="without an open operation"):
        reduce_journal_record(state, effect_result(sequence=2))

    completed = replay_journal([session_created(), effect_intent(), effect_result()])
    with pytest.raises(JournalCorruption, match="operation_id was already completed"):
        reduce_journal_record(completed, effect_intent(sequence=4))


@pytest.mark.parametrize(
    "effect_type", ["provider", "tool", "permission", "cancel", "retry", "snapshot"]
)
def test_schema_covers_each_phase_three_effect_boundary(effect_type):
    state = replay_journal(
        [
            session_created(),
            effect_intent(payload={"effect_type": effect_type}),
            effect_result(payload={"effect_type": effect_type}),
        ]
    )

    assert state.completed_operations["operation-effect-1"].effect_type == effect_type


@pytest.mark.parametrize("replay_policy", ["replay_safe", "interrupt"])
def test_schema_records_recovery_policy(replay_policy):
    state = replay_journal(
        [session_created(), effect_intent(payload={"replay_policy": replay_policy})]
    )

    assert state.open_operation.replay_policy == replay_policy


@pytest.mark.parametrize("outcome", ["ok", "error", "interrupted"])
def test_schema_records_each_effect_outcome(outcome):
    state = replay_journal(
        [
            session_created(),
            effect_intent(),
            effect_result(payload={"outcome": outcome}),
        ]
    )

    assert state.completed_operations["operation-effect-1"].outcome == outcome


def test_session_updates_and_history_replacement_rebuild_canonical_state():
    state = replay_journal(
        [
            session_created(),
            record(
                2,
                "history_appended",
                payload={"item": {"role": "user", "content": "old"}},
            ),
            record(
                3,
                "session_updated",
                payload={"updates": {"runtime_mode": {"mode": "plan"}}},
            ),
            record(
                4,
                "history_replaced",
                payload={
                    "items": [
                        {
                            "role": "assistant",
                            "content": "compacted",
                            "kind": "compact_summary",
                        }
                    ]
                },
            ),
        ]
    )

    assert state.session["runtime_mode"] == {"mode": "plan"}
    assert state.session["history"] == [
        {
            "role": "assistant",
            "content": "compacted",
            "kind": "compact_summary",
        }
    ]
