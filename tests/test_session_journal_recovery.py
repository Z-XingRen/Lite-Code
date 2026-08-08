import json

import pytest

from lite.core.session_journal import (
    JournalCorruption,
    JournalWriterError,
    SessionJournalWriter,
    restore_session_journal,
)


EFFECT_TYPES = ("provider", "tool", "permission", "cancel", "retry", "snapshot")


def session(history=None):
    return {
        "id": "session-1",
        "created_at": "2026-08-08T00:00:00+00:00",
        "workspace_root": "C:/workspace",
        "history": list(history or []),
    }


def read_records(path):
    return [json.loads(line) for line in path.read_bytes().splitlines() if line]


def test_atomic_snapshot_restores_state_then_replays_journal_tail(tmp_path):
    path = tmp_path / "session.journal.jsonl"
    writer = SessionJournalWriter.create(path, session())
    try:
        writer.append_history({"role": "user", "content": "before snapshot"})
        snapshot_path = writer.write_snapshot()
        writer.append_history({"role": "assistant", "content": "after snapshot"})
    finally:
        writer.close()

    restored = restore_session_journal(path)
    repeated = restore_session_journal(path)

    assert snapshot_path == path.with_name(f"{path.name}.snapshot.json")
    assert restored.used_snapshot is True
    assert restored.discarded_tail == b""
    assert restored.state == repeated.state
    assert [item["content"] for item in restored.state.session["history"]] == [
        "before snapshot",
        "after snapshot",
    ]
    assert restored.state.open_operation is None


def test_failed_snapshot_replace_preserves_the_previous_snapshot(
    tmp_path, monkeypatch
):
    path = tmp_path / "session.journal.jsonl"
    writer = SessionJournalWriter.create(path, session())
    try:
        snapshot_path = writer.write_snapshot()
        before = snapshot_path.read_bytes()
        writer.append_history({"role": "user", "content": "new state"})

        monkeypatch.setattr(
            "lite.core.session_journal_recovery.os.replace",
            lambda _source, _target: (_ for _ in ()).throw(OSError("replace failed")),
        )
        with pytest.raises(OSError, match="replace failed"):
            writer.write_snapshot()

        assert snapshot_path.read_bytes() == before
        assert not list(tmp_path.glob(f".{snapshot_path.name}.*.tmp"))
    finally:
        writer.close()


def test_invalid_snapshot_falls_back_to_the_complete_journal(tmp_path):
    path = tmp_path / "session.journal.jsonl"
    writer = SessionJournalWriter.create(path, session())
    try:
        snapshot_path = writer.write_snapshot()
        writer.append_history({"role": "user", "content": "journal authority"})
    finally:
        writer.close()
    snapshot_path.write_text("{not json", encoding="utf-8")

    restored = restore_session_journal(path)

    assert restored.used_snapshot is False
    assert restored.state.session["history"][-1]["content"] == "journal authority"


def test_reopen_discards_only_an_unterminated_tail_before_appending(tmp_path):
    path = tmp_path / "session.journal.jsonl"
    writer = SessionJournalWriter.create(path, session())
    writer.close()
    partial = b'{"schema_version":"lite.session_journal.v1"'
    with path.open("ab") as journal:
        journal.write(partial)

    reopened = SessionJournalWriter.open(path)
    try:
        assert reopened.discarded_tail == partial
        reopened.append_history({"role": "user", "content": "safe append"})
    finally:
        reopened.close()

    assert path.read_bytes().endswith(b"\n")
    assert len(read_records(path)) == 2


def test_restore_rejects_a_malformed_complete_record_in_the_middle(tmp_path):
    path = tmp_path / "session.journal.jsonl"
    writer = SessionJournalWriter.create(path, session())
    try:
        writer.append_history({"role": "user", "content": "first"})
        writer.append_history({"role": "assistant", "content": "second"})
    finally:
        writer.close()
    lines = path.read_bytes().splitlines(keepends=True)
    path.write_bytes(lines[0] + b"{broken}\n" + lines[2])

    with pytest.raises(JournalCorruption, match="line 2"):
        restore_session_journal(path)


def test_restore_rejects_a_valid_json_rewrite_inside_a_snapshotted_prefix(tmp_path):
    path = tmp_path / "session.journal.jsonl"
    writer = SessionJournalWriter.create(path, session())
    try:
        writer.append_history({"role": "user", "content": "original"})
        writer.write_snapshot()
    finally:
        writer.close()
    records = read_records(path)
    records[1]["payload"]["item"]["content"] = "tampered"
    path.write_bytes(
        b"".join(
            json.dumps(record, separators=(",", ":")).encode("utf-8") + b"\n"
            for record in records
        )
    )

    with pytest.raises(JournalCorruption, match="snapshotted journal prefix changed"):
        restore_session_journal(path)


@pytest.mark.parametrize("effect_type", EFFECT_TYPES)
@pytest.mark.parametrize(
    ("prefix", "replay_policy", "expected_action"),
    [
        ("before_intent", "interrupt", None),
        ("after_intent", "interrupt", "interrupt"),
        ("after_intent", "replay_safe", "retry"),
        ("after_result", "interrupt", None),
    ],
)
def test_each_effect_crash_prefix_recovers_deterministically(
    tmp_path, effect_type, prefix, replay_policy, expected_action
):
    path = tmp_path / f"{effect_type}-{prefix}-{replay_policy}.jsonl"
    writer = SessionJournalWriter.create(path, session())
    operation_id = f"operation-{effect_type}"
    if prefix != "before_intent":
        intent = writer.begin_effect(
            effect_type,
            call_id=f"call-{effect_type}",
            request={"effect": effect_type},
            replay_policy=replay_policy,
            operation_id=operation_id,
        )
        if prefix == "after_result":
            writer.finish_effect(intent, outcome="ok", result={"done": True})
    writer.close()

    reopened = SessionJournalWriter.open(path)
    try:
        first_state = reopened.state
        actions = [
            action
            for action in reopened.recovery_actions
            if action.operation_id == operation_id
        ]
        if expected_action is None:
            assert actions == []
        else:
            assert [action.action for action in actions] == [expected_action]
            assert actions[0].request == {"effect": effect_type}
            completed = first_state.completed_operations[operation_id]
            assert completed.outcome == "interrupted"
            assert completed.result == {
                "reason": "process_interrupted",
                "recovery_action": expected_action,
                "synthetic": True,
            }
        assert first_state.open_operation is None
    finally:
        reopened.close()

    records_after_first_recovery = path.read_bytes()
    repeated = SessionJournalWriter.open(path)
    try:
        assert repeated.state == first_state
        assert path.read_bytes() == records_after_first_recovery
    finally:
        repeated.close()


def test_snapshot_crash_after_atomic_replace_recovers_from_its_open_intent(
    tmp_path, monkeypatch
):
    path = tmp_path / "session.journal.jsonl"
    writer = SessionJournalWriter.create(path, session())
    writer.append_history({"role": "user", "content": "durable"})

    def crash_before_result(_intent, *, outcome, result):
        raise SystemExit("crash after snapshot replace")

    monkeypatch.setattr(writer, "finish_effect", crash_before_result)
    with pytest.raises(SystemExit, match="crash after snapshot replace"):
        writer.write_snapshot()
    assert writer.snapshot_path.exists()
    writer.close()

    reopened = SessionJournalWriter.open(path)
    try:
        assert reopened.state.session["history"][-1]["content"] == "durable"
        snapshot_actions = [
            item for item in reopened.recovery_actions if item.effect_type == "snapshot"
        ]
        assert [item.action for item in snapshot_actions] == ["interrupt"]
    finally:
        reopened.close()


def test_open_recovers_a_lock_owned_by_a_dead_process(tmp_path):
    path = tmp_path / "session.journal.jsonl"
    writer = SessionJournalWriter.create(path, session())
    writer.close()
    lock_path = path.with_name(f"{path.name}.lock")
    lock_path.write_text("2147483647:1:crashed-owner", encoding="ascii")

    reopened = SessionJournalWriter.open(path)
    try:
        assert lock_path.exists()
        assert "crashed-owner" not in lock_path.read_text(encoding="ascii")
    finally:
        reopened.close()
    assert not lock_path.exists()


def test_open_does_not_steal_an_active_writer_lock(tmp_path):
    path = tmp_path / "session.journal.jsonl"
    writer = SessionJournalWriter.create(path, session())
    try:
        with pytest.raises(JournalWriterError, match="already active"):
            SessionJournalWriter.open(path)
    finally:
        writer.close()
