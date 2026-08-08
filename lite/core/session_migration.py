"""Shadow-validated migration between legacy JSON and journal authority.

Migration is deliberately explicit.  A legacy session remains the only
authority until the journal has been replayed and an authority marker has
been atomically committed.  The marker is also the durable cutover boundary:
an orphaned journal without a marker is discarded during recovery, while a
marker pointing at the journal prevents legacy writes.
"""

import hashlib
import json
import os
import threading
import uuid
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

from .session_journal import (
    JOURNAL_SCHEMA_VERSION,
    JournalCorruption,
    SessionJournalWriter,
    restore_session_journal,
)


MIGRATION_SCHEMA_VERSION = "lite.session_migration.v1"
AUTHORITY_JOURNAL = "journal"
AUTHORITY_LEGACY = "legacy"


class MigrationError(RuntimeError):
    """The session cannot be migrated without risking split authority."""


class ShadowValidationError(MigrationError):
    """The reducer replay did not match the legacy canonical session."""


class MigrationRollbackError(MigrationError):
    """Rollback would discard data that only the journal can represent."""


@dataclass(frozen=True)
class MigrationResult:
    session_id: str
    authority: str
    marker_path: Path
    journal_path: Path
    legacy_path: Path
    backup_path: Path
    baseline_sequence: int


_MARKER_FIELDS = frozenset(
    {
        "schema_version",
        "journal_schema_version",
        "migration_id",
        "session_id",
        "authority",
        "legacy_path",
        "journal_path",
        "backup_path",
        "legacy_sha256",
        "baseline_sequence",
    }
)
_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "journal_schema_version",
        "migration_id",
        "session_id",
        "phase",
        "legacy_path",
        "journal_path",
        "temp_journal_path",
        "backup_path",
        "legacy_sha256",
    }
)


def authority_marker_path(store, session_id):
    return Path(store.root) / f".{_safe_id(session_id)}.authority"


def migration_manifest_path(store, session_id):
    return Path(store.root) / f".{_safe_id(session_id)}.migration"


def migration_backup_path(store, session_id):
    return Path(store.root) / f"{_safe_id(session_id)}.legacy.bak"


def read_authority_marker(store, session_id):
    path = authority_marker_path(store, session_id)
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MigrationError(f"invalid session authority marker: {path}") from exc
    _validate_marker(value, store, session_id)
    return value


def recover_pending_migration(store, session_id):
    """Resolve a crashed migration while preserving exactly one authority."""

    with _store_lock(store):
        return _recover_pending_migration(store, session_id)


def _recover_pending_migration(store, session_id):
    session_id = _safe_id(session_id)

    marker = read_authority_marker(store, session_id)
    manifest_path = migration_manifest_path(store, session_id)
    if marker is not None:
        try:
            manifest = _read_manifest(manifest_path, store, session_id)
        except MigrationError:
            manifest = None
        paths = [Path(manifest["temp_journal_path"])] if manifest else None
        _remove_manifest_and_temp(manifest_path, paths)
        if marker["authority"] == AUTHORITY_LEGACY:
            _archive_journal(Path(marker["journal_path"]), sync=False)
        return _result_from_marker(store, session_id, marker)

    manifest = _read_manifest(manifest_path, store, session_id)
    if manifest is None:
        return None
    _remove_manifest_and_temp(
        manifest_path,
        [
            Path(manifest["temp_journal_path"]),
            Path(manifest["backup_path"]),
            Path(manifest["journal_path"])
            if manifest.get("phase") == "journal_installed"
            else None,
            Path(manifest["journal_path"] + ".snapshot.json")
            if manifest.get("phase") == "journal_installed"
            else None,
        ],
    )
    return None


def migrate_legacy_session(
    store,
    session_id,
    *,
    shadow_validator=None,
    sync=True,
):
    """Migrate one legacy JSON session after deterministic shadow validation."""

    with _store_lock(store):
        return _migrate_legacy_session(
            store,
            session_id,
            shadow_validator=shadow_validator,
            sync=sync,
        )


def _migrate_legacy_session(
    store,
    session_id,
    *,
    shadow_validator,
    sync,
):

    session_id = _safe_id(session_id)
    marker = read_authority_marker(store, session_id)
    if marker and marker["authority"] == AUTHORITY_JOURNAL:
        return _result_from_marker(store, session_id, marker)
    _recover_pending_migration(store, session_id)

    legacy_path = Path(store.path(session_id))
    journal_path = Path(store.journal_path(session_id))
    manifest_path = migration_manifest_path(store, session_id)
    backup_path = migration_backup_path(store, session_id)
    if not legacy_path.exists():
        raise MigrationError(f"legacy session does not exist: {session_id}")
    if journal_path.exists() and journal_path.stat().st_size:
        raise MigrationError(f"journal already exists without cutover marker: {session_id}")

    try:
        legacy_bytes = legacy_path.read_bytes()
        legacy = _load_legacy(legacy_bytes, session_id)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise MigrationError(f"invalid legacy session: {session_id}") from exc

    migration_id = uuid.uuid4().hex
    temp_journal = Path(
        store.root / f".{journal_path.name}.{migration_id}.migration.tmp"
    )
    manifest = {
        "schema_version": MIGRATION_SCHEMA_VERSION,
        "journal_schema_version": JOURNAL_SCHEMA_VERSION,
        "migration_id": migration_id,
        "session_id": session_id,
        "phase": "shadow",
        "legacy_path": str(legacy_path),
        "journal_path": str(journal_path),
        "temp_journal_path": str(temp_journal),
        "backup_path": str(backup_path),
        "legacy_sha256": _sha256(legacy_bytes),
    }
    _atomic_json(manifest_path, manifest, sync=sync)
    writer = None
    try:
        writer = SessionJournalWriter.create(temp_journal, legacy, sync=sync)
        state = writer.state
        shadow = (
            shadow_validator(legacy, state)
            if shadow_validator is not None
            else state.session
        )
        _assert_shadow_equal(legacy, state.session, shadow)
        manifest["phase"] = "validated"
        _atomic_json(manifest_path, manifest, sync=sync)
    except Exception:
        _cleanup_failed_migration(manifest_path, manifest, remove_journal=False)
        raise
    finally:
        if writer is not None:
            writer.close()

    try:
        _copy_backup(backup_path, legacy_bytes, sync=sync)
        manifest["phase"] = "backup_ready"
        _atomic_json(manifest_path, manifest, sync=sync)
        os.replace(str(temp_journal), str(journal_path))
        if sync:
            _sync_directory(journal_path.parent)
        manifest["phase"] = "journal_installed"
        _atomic_json(manifest_path, manifest, sync=sync)
        marker = {
            "schema_version": MIGRATION_SCHEMA_VERSION,
            "journal_schema_version": JOURNAL_SCHEMA_VERSION,
            "migration_id": migration_id,
            "session_id": session_id,
            "authority": AUTHORITY_JOURNAL,
            "legacy_path": str(legacy_path),
            "journal_path": str(journal_path),
            "backup_path": str(backup_path),
            "legacy_sha256": manifest["legacy_sha256"],
            "baseline_sequence": state.last_sequence,
        }
        _atomic_json(authority_marker_path(store, session_id), marker, sync=sync)
    except Exception:
        _cleanup_failed_migration(manifest_path, manifest, remove_journal=True)
        raise

    _unlink(manifest_path)
    _unlink(temp_journal)
    return _result_from_marker(store, session_id, marker)


def rollback_session(store, session_id, *, sync=True):
    """Rollback a migration only when no journal-exclusive state exists."""

    with _store_lock(store):
        return _rollback_session(store, session_id, sync=sync)


def _rollback_session(store, session_id, *, sync):

    session_id = _safe_id(session_id)
    marker = read_authority_marker(store, session_id)
    if marker is None:
        _recover_pending_migration(store, session_id)
        return _legacy_result(store, session_id)
    if marker["authority"] == AUTHORITY_LEGACY:
        return _result_from_marker(store, session_id, marker)

    legacy_path = Path(marker["legacy_path"])
    journal_path = Path(marker["journal_path"])
    backup_path = Path(marker["backup_path"])
    try:
        backup_bytes = backup_path.read_bytes()
    except OSError as exc:
        raise MigrationRollbackError("legacy backup is unavailable") from exc
    if _sha256(backup_bytes) != marker["legacy_sha256"]:
        raise MigrationRollbackError("legacy backup changed after cutover")
    if legacy_path.exists() and _sha256(legacy_path.read_bytes()) != marker["legacy_sha256"]:
        raise MigrationRollbackError("legacy session changed after cutover")
    try:
        restored = restore_session_journal(journal_path)
    except (OSError, JournalCorruption, ValueError) as exc:
        raise MigrationRollbackError("journal cannot be safely rolled back") from exc
    if restored.state.last_sequence != marker["baseline_sequence"]:
        raise MigrationRollbackError(
            "journal contains new-format exclusive data; refusing stale legacy restore"
        )
    legacy = _load_legacy(backup_bytes, session_id)
    if _canonical(legacy) != _canonical(restored.state.session):
        raise MigrationRollbackError(
            "journal contains new-format exclusive data; refusing stale legacy restore"
        )

    legacy_marker = dict(marker)
    legacy_marker["authority"] = AUTHORITY_LEGACY
    _atomic_bytes(legacy_path, backup_bytes, sync=sync)
    _atomic_json(authority_marker_path(store, session_id), legacy_marker, sync=sync)
    _archive_journal(journal_path, sync=sync)
    return _result_from_marker(store, session_id, legacy_marker)


def _result_from_marker(store, session_id, marker):
    return MigrationResult(
        session_id=str(session_id),
        authority=marker["authority"],
        marker_path=authority_marker_path(store, session_id),
        journal_path=Path(marker["journal_path"]),
        legacy_path=Path(marker["legacy_path"]),
        backup_path=Path(marker["backup_path"]),
        baseline_sequence=int(marker["baseline_sequence"]),
    )


def _legacy_result(store, session_id):
    legacy_path = Path(store.path(session_id))
    return MigrationResult(
        session_id=str(session_id),
        authority=AUTHORITY_LEGACY,
        marker_path=authority_marker_path(store, session_id),
        journal_path=Path(store.journal_path(session_id)),
        legacy_path=legacy_path,
        backup_path=migration_backup_path(store, session_id),
        baseline_sequence=0,
    )


def _load_legacy(raw, session_id):
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("legacy session must be an object")
    value = json.loads(json.dumps(dict(value), ensure_ascii=False, allow_nan=False))
    if str(value.get("id", "")).strip() != str(session_id):
        raise ValueError("legacy session id does not match requested id")
    value.setdefault("history", [])
    if not isinstance(value["history"], list):
        raise ValueError("legacy session history must be a list")
    if not all(isinstance(item, Mapping) for item in value["history"]):
        raise ValueError("legacy session history items must be objects")
    return value


def _assert_shadow_equal(legacy, replayed, shadow):
    if shadow is True:
        candidate = replayed
    else:
        candidate = shadow.session if hasattr(shadow, "session") else shadow
    if shadow is False or not isinstance(candidate, Mapping):
        raise ShadowValidationError("shadow validation did not produce a session")
    if _canonical(legacy) != _canonical(replayed) or _canonical(legacy) != _canonical(candidate):
        raise ShadowValidationError("shadow validation mismatch")


def _copy_backup(backup_path, content, *, sync):
    if backup_path.exists():
        if backup_path.read_bytes() != content:
            raise MigrationError("legacy backup already exists with different content")
        return
    _atomic_bytes(backup_path, content, sync=sync)


def _archive_journal(journal_path, *, sync):
    if not journal_path.exists():
        return
    archive = journal_path.with_name(f"{journal_path.name}.rolled-back")
    if archive.exists():
        archive = journal_path.with_name(
            f"{journal_path.name}.rolled-back.{uuid.uuid4().hex}"
        )
    os.replace(str(journal_path), str(archive))
    snapshot = Path(f"{journal_path}.snapshot.json")
    if snapshot.exists():
        _unlink(snapshot)
    if sync:
        _sync_directory(journal_path.parent)


def _cleanup_failed_migration(manifest_path, manifest, *, remove_journal):
    paths = [Path(manifest["temp_journal_path"]), Path(manifest["backup_path"])]
    if remove_journal and manifest.get("phase") == "journal_installed":
        paths.append(Path(manifest["journal_path"]))
        paths.append(Path(manifest["journal_path"] + ".snapshot.json"))
    _remove_manifest_and_temp(manifest_path, paths)


def _remove_manifest_and_temp(manifest_path, paths):
    for path in paths or ():
        if path is None:
            continue
        _unlink(path)
        _unlink(Path(f"{path}.lock"))
    _unlink(manifest_path)
    for temporary in Path(manifest_path).parent.glob(f".{Path(manifest_path).name}.*.tmp"):
        _unlink(temporary)


def _read_manifest(path, store, session_id):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MigrationError(f"invalid migration manifest: {path}") from exc
    if not isinstance(value, Mapping) or set(value) != _MANIFEST_FIELDS:
        raise MigrationError(f"invalid migration manifest: {path}")
    if value["schema_version"] != MIGRATION_SCHEMA_VERSION:
        raise MigrationError(f"unsupported migration manifest: {path}")
    manifest = dict(value)
    if manifest["journal_schema_version"] != JOURNAL_SCHEMA_VERSION:
        raise MigrationError(f"unsupported journal schema in manifest: {path}")
    if str(manifest["session_id"]) != str(session_id):
        raise MigrationError(f"migration manifest session id mismatch: {path}")
    if manifest["phase"] not in {
        "shadow",
        "validated",
        "backup_ready",
        "journal_installed",
    }:
        raise MigrationError(f"invalid migration phase: {path}")
    expected = {
        "legacy_path": Path(store.path(session_id)),
        "journal_path": Path(store.journal_path(session_id)),
        "backup_path": migration_backup_path(store, session_id),
    }
    for field, expected_path in expected.items():
        if not _same_path(manifest[field], expected_path):
            raise MigrationError(f"migration manifest path mismatch: {field}")
    temp = Path(manifest["temp_journal_path"])
    root = Path(store.root).resolve()
    if temp.resolve().parent != root or not temp.name.startswith(
        f".{expected['journal_path'].name}."
    ) or not temp.name.endswith(".migration.tmp"):
        raise MigrationError("migration manifest temp path mismatch")
    if not _is_sha256(manifest["legacy_sha256"]):
        raise MigrationError("invalid legacy session hash in migration manifest")
    if not str(manifest["migration_id"]).strip():
        raise MigrationError("invalid migration id in manifest")
    return manifest


def _validate_marker(value, store, session_id):
    if not isinstance(value, Mapping) or set(value) != _MARKER_FIELDS:
        raise MigrationError("invalid session authority marker")
    if value["schema_version"] != MIGRATION_SCHEMA_VERSION:
        raise MigrationError("unsupported session authority marker")
    if value["journal_schema_version"] != JOURNAL_SCHEMA_VERSION:
        raise MigrationError("unsupported journal schema in authority marker")
    if str(value["session_id"]) != str(session_id):
        raise MigrationError("authority marker session id mismatch")
    if value["authority"] not in {AUTHORITY_JOURNAL, AUTHORITY_LEGACY}:
        raise MigrationError("invalid session authority")
    if not _is_sha256(value["legacy_sha256"]):
        raise MigrationError("invalid legacy session hash in authority marker")
    expected = {
        "legacy_path": Path(store.path(session_id)),
        "journal_path": Path(store.journal_path(session_id)),
        "backup_path": migration_backup_path(store, session_id),
    }
    for field, expected_path in expected.items():
        if not _same_path(value[field], expected_path):
            raise MigrationError(f"authority marker path mismatch: {field}")
    if (
        isinstance(value["baseline_sequence"], bool)
        or not isinstance(value["baseline_sequence"], int)
        or value["baseline_sequence"] <= 0
    ):
        raise MigrationError("invalid migration baseline sequence")


def _canonical(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256(content):
    return hashlib.sha256(content).hexdigest()


def _atomic_json(path, value, *, sync):
    _atomic_bytes(path, (_canonical(value) + "\n").encode("utf-8"), sync=sync)


def _atomic_bytes(path, payload, *, sync):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with temporary.open("xb") as output:
            output.write(payload)
            output.flush()
            if sync:
                os.fsync(output.fileno())
        os.replace(str(temporary), str(path))
        if sync:
            _sync_directory(path.parent)
    finally:
        _unlink(temporary)


def _sync_directory(path):
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _unlink(path):
    try:
        Path(path).unlink()
    except FileNotFoundError:
        pass


def _is_sha256(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _same_path(left, right):
    return Path(left).resolve() == Path(right).resolve()


def _store_lock(store):
    return getattr(store, "_lock", nullcontext())


def _safe_id(value):
    value = str(value or "").strip()
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError("invalid session id")
    return value
