"""Session JSON storage."""

import json
import os
import threading
from datetime import datetime
from pathlib import Path

from .workspace import clip


class SessionStore:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._runtime_journal_owners = {}

    def path(self, session_id):
        return self.root / f"{_safe_session_id(session_id)}.json"

    def event_path(self, session_id):
        return self.root / f"{_safe_session_id(session_id)}.events.jsonl"

    def journal_path(self, session_id):
        return self.root / f"{_safe_session_id(session_id)}.journal.jsonl"

    def authority_path(self, session_id):
        from .session_migration import authority_marker_path

        return authority_marker_path(self, session_id)

    def session_authority(self, session_id):
        from .session_migration import AUTHORITY_LEGACY, read_authority_marker

        marker = read_authority_marker(self, session_id)
        return marker["authority"] if marker is not None else AUTHORITY_LEGACY

    def migrate_session(self, session_id, **kwargs):
        from .session_migration import migrate_legacy_session

        return migrate_legacy_session(self, session_id, **kwargs)

    def rollback_session(self, session_id, **kwargs):
        from .session_migration import rollback_session

        return rollback_session(self, session_id, **kwargs)

    def save(self, session):
        path = self.path(session["id"])
        payload = json.dumps(session, indent=2)
        with self._lock:
            from .session_migration import AUTHORITY_JOURNAL, read_authority_marker

            marker = read_authority_marker(self, session["id"])
            if marker is not None and marker["authority"] == AUTHORITY_JOURNAL:
                raise RuntimeError(
                    "legacy session is read-only after journal cutover; append to the journal"
                )
            tmp_path = path.with_name(
                f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
            )
            tmp_path.write_text(payload, encoding="utf-8")
            os.replace(tmp_path, path)
        return path

    def load(self, session_id):
        with self._lock:
            from .session_migration import (
                AUTHORITY_JOURNAL,
                recover_pending_migration,
                read_authority_marker,
            )

            recover_pending_migration(self, session_id)
            marker = read_authority_marker(self, session_id)
            if marker is not None and marker["authority"] == AUTHORITY_JOURNAL:
                from .session_journal import restore_session_journal

                restored = restore_session_journal(self.journal_path(session_id))
                return restored.state.session
            return json.loads(self.path(session_id).read_text(encoding="utf-8"))

    def latest(self):
        entries = sorted(
            self._session_entries(), key=lambda item: item[1].stat().st_mtime
        )
        return entries[-1][0] if entries else None

    def list_sessions(self):
        rows = []
        for index, (session_id, path) in enumerate(
            sorted(
                self._session_entries(),
                key=lambda item: item[1].stat().st_mtime,
                reverse=True,
            ),
            start=1,
        ):
            try:
                session = self.load(session_id)
            except (OSError, ValueError, RuntimeError):
                continue
            history = list(session.get("history", []))
            rows.append(
                {
                    "index": index,
                    "id": str(session.get("id", session_id)),
                    "created_at": str(session.get("created_at", "")),
                    "updated_at": datetime.fromtimestamp(
                        path.stat().st_mtime
                    ).isoformat(timespec="seconds"),
                    "history_count": len(history),
                    "runtime_mode": str(
                        session.get("runtime_mode", {}).get("mode", "default")
                        or "default"
                    ),
                    "workspace_root": str(session.get("workspace_root", "")),
                    "last_final_answer": _last_final_preview(history),
                }
            )
        return rows

    def _session_paths(self):
        sidecar_suffixes = (
            ".journal.jsonl.snapshot.json",
        )
        return [
            path
            for path in self.root.glob("*.json")
            if not path.name.endswith(sidecar_suffixes)
        ]

    def _session_entries(self):
        session_ids = {path.stem for path in self._session_paths()}
        suffix = ".journal.jsonl"
        for path in self.root.glob(f"*{suffix}"):
            session_id = path.name[: -len(suffix)]
            if self.authority_path(session_id).exists():
                session_ids.add(session_id)
        entries = []
        for session_id in session_ids:
            try:
                path = (
                    self.journal_path(session_id)
                    if self.session_authority(session_id) == "journal"
                    else self.path(session_id)
                )
            except (ValueError, RuntimeError):
                continue
            if path.exists():
                entries.append((session_id, path))
        return entries


def _last_final_preview(history):
    for item in reversed(history):
        if item.get("role") == "assistant":
            return clip(item.get("content", ""), 80)
    return ""


def _safe_session_id(session_id):
    value = str(session_id or "").strip()
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError("invalid session id")
    return value
