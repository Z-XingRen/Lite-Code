"""Path-scoped workspace change tracking for risky tools.

The legacy workspace snapshot hashes every file in the repository. A
transparent file tool already tells us its target path, so this tracker hashes
only those paths before and after the effect. Opaque tools maintain a verified
content cache: metadata finds candidates and only candidates are rehashed.
The cache is seeded or repaired by a conservative legacy snapshot when its
continuity cannot be proved.
"""

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path

from .workspace import IGNORED_PATH_NAMES

TRANSPARENT_TOOL_NAMES = frozenset(
    {"write_file", "patch_file", "delete_file", "rename", "rename_file"}
)
OPAQUE_TOOL_NAMES = frozenset({"run_shell", "verify"})
_PATH_ARGUMENT_KEYS = (
    "path",
    "paths",
    "source",
    "source_path",
    "old_path",
    "new_path",
    "destination",
    "destination_path",
    "dest",
    "to",
)


@dataclass(frozen=True)
class _FileState:
    """The content identity needed to preserve snapshot diff semantics."""

    exists: bool
    digest: str = ""

    @classmethod
    def capture(cls, path):
        if not path.is_file():
            return cls(False)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return cls(True, digest)


@dataclass(frozen=True)
class _PathMetadata:
    size: int
    mtime_ns: int
    ctime_ns: int
    inode: int
    mode: int


class WorkspaceTrackingError(RuntimeError):
    """Raised when workspace evidence cannot be completed safely."""


@dataclass(frozen=True)
class WorkspaceChangeToken:
    tool_name: str
    paths: tuple[str, ...]
    before: dict[str, _FileState]
    mode: str = "transparent"
    before_metadata: dict[str, _PathMetadata] = field(default_factory=dict)
    metadata_errors: tuple[str, ...] = ()
    requires_legacy_snapshot: bool = False


class WorkspaceChangeTracker:
    """Track known file targets without scanning unrelated workspace files.

    ``begin`` returns a metadata candidate token for ``run_shell`` and a
    path-scoped token for transparent tools. Explicit ``target_paths`` is
    provided for future transparent tools such as a rename operation and makes
    the contract independently testable without registering a product tool.
    """

    def __init__(self, root):
        self.root = Path(root).resolve()
        self.last_observation = {}
        self._fallback_count = 0
        self._content_digests = {}
        self._cached_metadata = {}
        self._cache_is_complete = False

    def begin(self, tool, args=None, *, target_paths=None):
        tool_name = str(getattr(tool, "name", tool))
        paths = self._candidate_paths(
            tool_name,
            args or {},
            target_paths=target_paths,
        )
        if tool_name in OPAQUE_TOOL_NAMES:
            before_metadata, errors = self._scan_metadata()
            return WorkspaceChangeToken(
                tool_name,
                (),
                {},
                mode="opaque",
                before_metadata=before_metadata,
                metadata_errors=tuple(errors),
                requires_legacy_snapshot=self._requires_legacy_snapshot(
                    before_metadata, errors
                ),
            )
        has_declared_targets = bool(target_paths)
        has_path_arguments = any(key in (args or {}) for key in _PATH_ARGUMENT_KEYS)
        if not paths and not (
            has_declared_targets
            or (tool_name in TRANSPARENT_TOOL_NAMES and has_path_arguments)
        ):
            return None
        before = {
            path: _FileState.capture(self.root / Path(path))
            for path in paths
        }
        return WorkspaceChangeToken(tool_name, tuple(paths), before)

    def finish(
        self,
        token,
        result=None,
        *,
        legacy_before=None,
        legacy_after=None,
        fallback_duration_ms=0,
    ):
        """Return changed paths and summaries in legacy snapshot order."""
        if token is None:
            return [], []
        if token.mode == "opaque":
            after_metadata, errors = self._scan_metadata()
            candidates, _ = self._diff_metadata(
                token.before_metadata, after_metadata
            )
            scan_errors = tuple(token.metadata_errors) + tuple(errors)
            if legacy_before is not None and legacy_after is not None:
                self._fallback_count += 1
                if scan_errors:
                    self._clear_content_cache()
                else:
                    self._seed_content_cache(legacy_after, after_metadata)
                self._set_observation(
                    mode="opaque",
                    candidates=candidates,
                    fallback=True,
                    fallback_reason=(
                        "metadata_scan_error" if scan_errors else "opaque_tool"
                    ),
                    fallback_duration_ms=fallback_duration_ms,
                    scan_errors=scan_errors,
                )
                return self._diff_snapshots(legacy_before, legacy_after)
            if scan_errors:
                self._clear_content_cache()
                raise WorkspaceTrackingError(
                    "opaque workspace metadata scan failed: "
                    + "; ".join(scan_errors)
                )
            if token.requires_legacy_snapshot:
                raise WorkspaceTrackingError(
                    "opaque workspace tracking requires a legacy snapshot"
                )
            changed, summaries = self._finish_cached_opaque(
                token, after_metadata, candidates
            )
            self._set_observation(
                mode="opaque",
                candidates=candidates,
                fallback=False,
                fallback_reason="",
                fallback_duration_ms=0,
                scan_errors=(),
            )
            return changed, summaries
        after = {
            path: _FileState.capture(self.root / Path(path))
            for path in token.paths
        }
        changed = []
        summaries = []
        for path in sorted(token.before):
            before = token.before[path]
            current = after[path]
            if before.exists == current.exists and (
                not before.exists or before.digest == current.digest
            ):
                continue
            changed.append(path)
            if not before.exists and current.exists:
                summaries.append(f"created:{path}")
            elif before.exists and not current.exists:
                summaries.append(f"deleted:{path}")
            else:
                summaries.append(f"modified:{path}")
        self._set_observation(
            mode="transparent",
            candidates=changed,
            fallback=False,
            fallback_reason="",
            fallback_duration_ms=0,
            scan_errors=(),
        )
        self._refresh_content_cache(token.paths, after)
        return changed, summaries

    def _finish_cached_opaque(self, token, after_metadata, candidates):
        changed = []
        summaries = []
        for path in candidates:
            before_metadata = token.before_metadata.get(path)
            after_file_metadata = after_metadata.get(path)
            before_digest = self._content_digests.get(path)
            if before_metadata is not None and before_digest is None:
                self._clear_content_cache()
                raise WorkspaceTrackingError(
                    f"opaque workspace content cache is missing: {path}"
                )
            if after_file_metadata is None:
                changed.append(path)
                summaries.append(f"deleted:{path}")
                self._content_digests.pop(path, None)
                continue
            current = _FileState.capture(self.root / Path(path))
            self._content_digests[path] = current.digest
            if before_metadata is None:
                changed.append(path)
                summaries.append(f"created:{path}")
            elif before_digest != current.digest:
                changed.append(path)
                summaries.append(f"modified:{path}")
        self._cached_metadata = dict(after_metadata)
        return changed, summaries

    def _requires_legacy_snapshot(self, metadata, errors):
        return (
            bool(errors)
            or not self._cache_is_complete
            or metadata != self._cached_metadata
            or set(metadata) != set(self._content_digests)
        )

    def _seed_content_cache(self, digests, metadata):
        self._content_digests = dict(digests)
        self._cached_metadata = dict(metadata)
        self._cache_is_complete = set(digests) == set(metadata)

    def _refresh_content_cache(self, paths, states):
        if not self._cache_is_complete:
            return
        for path in paths:
            state = states[path]
            if not state.exists:
                self._content_digests.pop(path, None)
                self._cached_metadata.pop(path, None)
                continue
            metadata = self._path_metadata(path)
            if metadata is None:
                self._clear_content_cache()
                return
            self._content_digests[path] = state.digest
            self._cached_metadata[path] = metadata

    def _clear_content_cache(self):
        self._content_digests = {}
        self._cached_metadata = {}
        self._cache_is_complete = False

    def _scan_metadata(self):
        metadata = {}
        errors = []
        for path in self.root.rglob("*"):
            try:
                relative = path.relative_to(self.root)
            except ValueError:
                continue
            if any(part in IGNORED_PATH_NAMES for part in relative.parts) or not path.is_file():
                continue
            relative_path = relative.as_posix()
            metadata_item, error = self._metadata_for_path(relative_path)
            if error:
                errors.append(f"{relative_path}: {error}")
                continue
            metadata[relative_path] = metadata_item
        return metadata, errors

    def _path_metadata(self, path):
        metadata, error = self._metadata_for_path(path)
        return None if error else metadata

    def _metadata_for_path(self, path):
        try:
            stat = (self.root / Path(path)).stat()
        except OSError as exc:
            return None, str(exc)
        return (
            _PathMetadata(
                size=stat.st_size,
                mtime_ns=stat.st_mtime_ns,
                ctime_ns=stat.st_ctime_ns,
                inode=getattr(stat, "st_ino", 0),
                mode=stat.st_mode,
            ),
            "",
        )

    @staticmethod
    def _diff_metadata(before, after):
        changed = []
        summaries = []
        for path in sorted(set(before) | set(after)):
            old = before.get(path)
            current = after.get(path)
            if old == current:
                continue
            changed.append(path)
            if old is None:
                summaries.append(f"created:{path}")
            elif current is None:
                summaries.append(f"deleted:{path}")
            else:
                summaries.append(f"modified:{path}")
        return changed, summaries

    @staticmethod
    def _diff_snapshots(before, after):
        changed = []
        summaries = []
        for path in sorted(set(before) | set(after)):
            if before.get(path) == after.get(path):
                continue
            changed.append(path)
            if path not in before:
                summaries.append(f"created:{path}")
            elif path not in after:
                summaries.append(f"deleted:{path}")
            else:
                summaries.append(f"modified:{path}")
        return changed, summaries

    def _set_observation(
        self,
        *,
        mode,
        candidates,
        fallback,
        fallback_reason,
        fallback_duration_ms,
        scan_errors,
    ):
        self.last_observation = {
            "workspace_tracker_mode": mode,
            "workspace_tracker_candidates": list(candidates),
            "workspace_tracker_fallback": bool(fallback),
            "workspace_tracker_fallback_reason": fallback_reason,
            "workspace_tracker_fallback_count": self._fallback_count,
            "workspace_tracker_fallback_ms": round(float(fallback_duration_ms or 0), 3),
            "workspace_tracker_scan_errors": list(scan_errors),
        }

    def _candidate_paths(self, tool_name, args, *, target_paths=None):
        if target_paths is None and tool_name not in TRANSPARENT_TOOL_NAMES:
            return []
        raw_paths = list(target_paths or ())
        if target_paths is None:
            for key in _PATH_ARGUMENT_KEYS:
                value = args.get(key)
                if isinstance(value, (list, tuple, set)):
                    raw_paths.extend(value)
                elif value not in (None, ""):
                    raw_paths.append(value)
        normalized = []
        seen = set()
        for raw_path in raw_paths:
            if not isinstance(raw_path, (str, os.PathLike)):
                continue
            path = self._normalize_path(raw_path)
            if path is None or path in seen:
                continue
            seen.add(path)
            normalized.append(path)
        return normalized

    def _normalize_path(self, raw_path):
        path = Path(raw_path)
        resolved = (path if path.is_absolute() else self.root / path).resolve(
            strict=False
        )
        try:
            relative = resolved.relative_to(self.root).as_posix()
        except ValueError as exc:
            raise ValueError(f"path escapes workspace: {raw_path}") from exc
        if any(part in IGNORED_PATH_NAMES for part in Path(relative).parts):
            return None
        return relative
