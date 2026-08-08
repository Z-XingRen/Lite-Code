import pytest

from lite.core.session_journal import SessionJournalWriter
from lite.core.session_migration import (
    AUTHORITY_JOURNAL,
    AUTHORITY_LEGACY,
    MigrationRollbackError,
    ShadowValidationError,
    authority_marker_path,
    migrate_legacy_session,
    migration_manifest_path,
    read_authority_marker,
    recover_pending_migration,
    rollback_session,
)
from lite.core.session_store import SessionStore


def legacy_session(session_id="migration-session", history=None):
    return {
        "id": session_id,
        "created_at": "2026-08-08T00:00:00+00:00",
        "workspace_root": "C:/workspace",
        "history": list(history or []),
        "memory": {"working": [], "durable": []},
    }


def build_store(tmp_path, session=None):
    store = SessionStore(tmp_path / ".lite" / "sessions")
    value = session or legacy_session()
    store.save(value)
    return store, value


def test_migration_shadow_validates_then_cutover_makes_journal_authoritative(tmp_path):
    store, value = build_store(
        tmp_path,
        legacy_session(history=[{"role": "user", "content": "before"}]),
    )
    legacy_before = store.path(value["id"]).read_bytes()

    result = migrate_legacy_session(store, value["id"])

    marker = read_authority_marker(store, value["id"])
    assert result.authority == AUTHORITY_JOURNAL
    assert marker["authority"] == AUTHORITY_JOURNAL
    assert marker["baseline_sequence"] == 1
    assert store.path(value["id"]).read_bytes() == legacy_before
    assert store.load(value["id"]) == value
    assert store.journal_path(value["id"]).exists()
    assert not migration_manifest_path(store, value["id"]).exists()


def test_shadow_mismatch_blocks_cutover_without_modifying_legacy(tmp_path):
    store, value = build_store(tmp_path)
    legacy_before = store.path(value["id"]).read_bytes()

    with pytest.raises(ShadowValidationError, match="shadow validation"):
        migrate_legacy_session(
            store,
            value["id"],
            shadow_validator=lambda _legacy, _state: {"id": "different"},
        )

    assert store.path(value["id"]).read_bytes() == legacy_before
    assert store.load(value["id"]) == value
    assert not authority_marker_path(store, value["id"]).exists()
    assert not store.journal_path(value["id"]).exists()
    assert not migration_manifest_path(store, value["id"]).exists()


@pytest.mark.parametrize("crash_stage", ["temp_write", "validation", "cutover_marker"])
def test_crash_prefix_recovers_with_legacy_as_the_only_authority(
    tmp_path, monkeypatch, crash_stage
):
    store, value = build_store(tmp_path)
    legacy_before = store.path(value["id"]).read_bytes()

    if crash_stage == "temp_write":
        original = SessionJournalWriter._write_line

        def crash_write(writer, record):
            if record.kind == "session_created":
                raise SystemExit("crash while writing temp journal")
            return original(writer, record)

        monkeypatch.setattr(SessionJournalWriter, "_write_line", crash_write)
    elif crash_stage == "validation":
        def crash_validate(_legacy, _state):
            raise SystemExit("crash during shadow validation")

        with pytest.raises(SystemExit, match="shadow validation"):
            migrate_legacy_session(
                store, value["id"], shadow_validator=crash_validate
            )
        recover_pending_migration(store, value["id"])
        assert store.path(value["id"]).read_bytes() == legacy_before
        assert store.load(value["id"]) == value
        assert not authority_marker_path(store, value["id"]).exists()
        return
    else:
        import lite.core.session_migration as migration

        original = migration.os.replace

        def crash_marker(source, target):
            if target == str(authority_marker_path(store, value["id"])):
                raise SystemExit("crash during atomic cutover marker")
            return original(source, target)

        monkeypatch.setattr(migration.os, "replace", crash_marker)

    with pytest.raises(SystemExit):
        migrate_legacy_session(store, value["id"])
    recover_pending_migration(store, value["id"])

    assert store.path(value["id"]).read_bytes() == legacy_before
    assert store.load(value["id"]) == value
    assert not authority_marker_path(store, value["id"]).exists()
    assert not store.journal_path(value["id"]).exists()
    assert not migration_manifest_path(store, value["id"]).exists()


def test_rollback_before_cutover_discards_pending_migration(tmp_path):
    store, value = build_store(tmp_path)

    def crash_validate(_legacy, _state):
        raise SystemExit("migration interrupted")

    with pytest.raises(SystemExit):
        migrate_legacy_session(store, value["id"], shadow_validator=crash_validate)

    rollback_session(store, value["id"])

    assert store.load(value["id"]) == value
    assert not authority_marker_path(store, value["id"]).exists()
    assert not store.journal_path(value["id"]).exists()
    assert not migration_manifest_path(store, value["id"]).exists()


def test_rollback_after_cutover_without_new_data_restores_legacy_authority(tmp_path):
    store, value = build_store(tmp_path)
    migrate_legacy_session(store, value["id"])
    store.path(value["id"]).unlink()

    result = rollback_session(store, value["id"])

    assert result.authority == AUTHORITY_LEGACY
    assert read_authority_marker(store, value["id"])["authority"] == AUTHORITY_LEGACY
    assert store.load(value["id"]) == value
    assert not store.journal_path(value["id"]).exists()


def test_rollback_refuses_to_restore_stale_legacy_after_new_journal_data(tmp_path):
    store, value = build_store(tmp_path)
    migrate_legacy_session(store, value["id"])
    writer = SessionJournalWriter.open(store.journal_path(value["id"]))
    try:
        writer.append_history({"role": "user", "content": "journal-only"})
    finally:
        writer.close()

    with pytest.raises(MigrationRollbackError, match="exclusive"):
        rollback_session(store, value["id"])

    assert read_authority_marker(store, value["id"])["authority"] == AUTHORITY_JOURNAL
    assert store.load(value["id"])["history"][-1]["content"] == "journal-only"


def test_session_store_explicit_migration_and_rollback_facade(tmp_path):
    store, value = build_store(tmp_path)

    migrated = store.migrate_session(value["id"])
    assert migrated.authority == AUTHORITY_JOURNAL
    assert store.session_authority(value["id"]) == AUTHORITY_JOURNAL

    rolled_back = store.rollback_session(value["id"])
    assert rolled_back.authority == AUTHORITY_LEGACY
    assert store.session_authority(value["id"]) == AUTHORITY_LEGACY


def test_journal_cutover_blocks_legacy_save_and_sidecars_are_not_sessions(tmp_path):
    store, value = build_store(tmp_path)
    migrate_legacy_session(store, value["id"])

    with pytest.raises(RuntimeError, match="read-only"):
        store.save({**value, "history": [{"role": "user", "content": "unsafe"}]})

    assert store.latest() == value["id"]
    assert [row["id"] for row in store.list_sessions()] == [value["id"]]
