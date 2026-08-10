"""Sandbox configuration for shell execution."""

from dataclasses import dataclass
from typing import Any, Mapping

SANDBOX_MODES = {"off", "best_effort", "required"}
SANDBOX_BACKENDS = {
    "auto",
    "bubblewrap",
    "sandbox-exec",
    "docker",
    "podman",
    "none",
}


@dataclass(frozen=True)
class SandboxConfig:
    mode: str = "off"
    backend: str = "auto"
    workspace_write: bool = True
    network_access: bool = False
    container_image: str = "python:3.12-slim"
    excluded_commands: tuple[str, ...] = ()
    extra_readonly_paths: tuple[str, ...] = ()
    extra_writable_paths: tuple[str, ...] = ()
    deny_read: tuple[str, ...] = ()
    deny_write: tuple[str, ...] = ()

    @property
    def enabled(self) -> bool:
        return self.mode != "off"


def resolve_sandbox_config(
    values: Mapping[str, Any] | None,
) -> SandboxConfig:
    sandbox = dict((values or {}).get("sandbox", {}) or {})
    filesystem = dict(sandbox.get("filesystem", {}) or {})
    mode = str(sandbox.get("mode", "off") or "off")
    backend = str(sandbox.get("backend", "auto") or "auto")
    if mode not in SANDBOX_MODES:
        raise ValueError(f"sandbox.mode must be one of {sorted(SANDBOX_MODES)}")
    if backend not in SANDBOX_BACKENDS:
        raise ValueError(f"sandbox.backend must be one of {sorted(SANDBOX_BACKENDS)}")
    return SandboxConfig(
        mode=mode,
        backend=backend,
        workspace_write=bool(sandbox.get("workspace_write", True)),
        network_access=bool(sandbox.get("network_access", False)),
        container_image=str(
            sandbox.get("container_image", "python:3.12-slim")
            or "python:3.12-slim"
        ),
        excluded_commands=tuple(
            str(item) for item in sandbox.get("excluded_commands", []) or []
        ),
        extra_readonly_paths=tuple(
            str(item) for item in filesystem.get("extra_readonly_paths", []) or []
        ),
        extra_writable_paths=tuple(
            str(item) for item in filesystem.get("extra_writable_paths", []) or []
        ),
        deny_read=tuple(str(item) for item in filesystem.get("deny_read", []) or []),
        deny_write=tuple(str(item) for item in filesystem.get("deny_write", []) or []),
    )
