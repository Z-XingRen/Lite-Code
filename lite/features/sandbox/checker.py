"""Sandbox backend availability checks."""

import sys
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class SandboxBackend:
    name: str
    path: str


class SandboxChecker:
    def __init__(
        self,
        which: Callable[[str], str | None],
        *,
        platform: str | None = None,
    ) -> None:
        self.which = which
        self.platform = platform or sys.platform

    def resolve_backend(self, backend: str) -> SandboxBackend:
        if backend in {"none", "off"}:
            return SandboxBackend("none", "")
        candidates = self._auto_candidates() if backend == "auto" else (backend,)
        for candidate in candidates:
            executable = {
                "bubblewrap": "bwrap",
                "sandbox-exec": "sandbox-exec",
                "docker": "docker",
                "podman": "podman",
            }.get(candidate)
            path = self.which(executable) if executable else None
            if path:
                return SandboxBackend(candidate, str(path))
        return SandboxBackend(str(backend), "")

    def backend_path(self, backend: str) -> str:
        return self.resolve_backend(backend).path

    def _auto_candidates(self) -> tuple[str, ...]:
        if self.platform.startswith("linux"):
            return ("bubblewrap", "docker", "podman")
        if self.platform == "darwin":
            return ("sandbox-exec", "docker", "podman")
        if self.platform.startswith("win"):
            return ("docker", "podman")
        return ("docker", "podman")
