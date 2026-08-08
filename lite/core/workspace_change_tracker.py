"""Path-scoped workspace change tracking for transparent tools.

The legacy workspace snapshot hashes every file in the repository.  A
transparent file tool already tells us its target path, so this tracker hashes
only those paths before and after the effect.  Opaque tools intentionally do
not produce a token here; their conservative fallback remains the caller's
responsibility until the opaque-tool work package is implemented.
"""

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

from .workspace import IGNORED_PATH_NAMES

TRANSPARENT_TOOL_NAMES = frozenset(
    {"write_file", "patch_file", "delete_file", "rename", "rename_file"}
)
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
class WorkspaceChangeToken:
    tool_name: str
    paths: tuple[str, ...]
    before: dict[str, _FileState]


class WorkspaceChangeTracker:
    """Track known file targets without scanning unrelated workspace files.

    ``begin`` returns ``None`` for an opaque tool.  Callers must then use an
    appropriate conservative tracker.  Explicit ``target_paths`` is provided
    for future transparent tools such as a rename operation and makes the
    contract independently testable without registering a product tool.
    """

    def __init__(self, root):
        self.root = Path(root).resolve()

    def begin(self, tool, args=None, *, target_paths=None):
        tool_name = str(getattr(tool, "name", tool))
        paths = self._candidate_paths(
            tool_name,
            args or {},
            target_paths=target_paths,
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

    def finish(self, token, result=None):
        """Return changed paths and summaries in legacy snapshot order."""
        if token is None:
            return [], []
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
        return changed, summaries

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
