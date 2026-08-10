"""Cross-platform shell sandbox runner with per-command permission expansion."""

import json
import os
import tempfile
from dataclasses import replace
from pathlib import Path
from shutil import which as default_which

from .checker import SandboxChecker
from .command_matcher import command_is_excluded
from .config import SandboxConfig
from .process import run_cancellable_process


if os.name == "nt":
    _DEFAULT_SHELL = os.environ.get("COMSPEC") or str(
        Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "cmd.exe"
    )
else:
    _DEFAULT_SHELL = None


class SandboxRunner:
    def __init__(self, config=None, *, which=None, run=None, emit_event=None):
        self.config = config or SandboxConfig()
        self.which = which or default_which
        self.run_process = run
        self.emit_event = emit_event or (lambda event, payload: None)

    def run(
        self,
        command,
        *,
        cwd,
        env,
        timeout,
        cancellation_token=None,
        network_access=None,
        additional_readonly_paths=(),
        additional_writable_paths=(),
    ):
        config = self._command_config(
            Path(cwd),
            network_access=network_access,
            additional_readonly_paths=additional_readonly_paths,
            additional_writable_paths=additional_writable_paths,
        )
        if config.mode == "off" or (
            config.mode != "required"
            and command_is_excluded(command, config.excluded_commands)
        ):
            return self._plain(
                command,
                cwd=cwd,
                env=env,
                timeout=timeout,
                cancellation_token=cancellation_token,
            )

        backend = SandboxChecker(self.which).resolve_backend(config.backend)
        if not backend.path:
            self.emit_event(
                "sandbox_unavailable",
                {
                    "mode": config.mode,
                    "backend": config.backend,
                    "command": str(command or "")[:200],
                },
            )
            if config.mode == "required":
                raise RuntimeError("sandbox required but unavailable")
            return self._plain(
                command,
                cwd=cwd,
                env=env,
                timeout=timeout,
                cancellation_token=cancellation_token,
            )

        self.emit_event(
            "sandbox_selected",
            {
                "backend": backend.name,
                "network_access": config.network_access,
                "workspace_write": config.workspace_write,
            },
        )
        if backend.name == "bubblewrap":
            argv = self._bubblewrap_argv(backend.path, command, Path(cwd), config)
            return self._run_process(
                argv,
                cwd=cwd,
                env=env,
                timeout=timeout,
                cancellation_token=cancellation_token,
            )
        if backend.name == "sandbox-exec":
            return self._run_sandbox_exec(
                backend.path,
                command,
                Path(cwd),
                config,
                env=env,
                timeout=timeout,
                cancellation_token=cancellation_token,
            )
        if backend.name in {"docker", "podman"}:
            argv = self._container_argv(
                backend.path, backend.name, command, Path(cwd), config
            )
            return self._run_process(
                argv,
                cwd=cwd,
                env=env,
                timeout=timeout,
                cancellation_token=cancellation_token,
            )
        raise RuntimeError(f"unsupported sandbox backend: {backend.name}")

    def _command_config(
        self,
        cwd,
        *,
        network_access,
        additional_readonly_paths,
        additional_writable_paths,
    ):
        readonly = self._validate_expansion_paths(
            cwd,
            additional_readonly_paths,
            denied=self.config.deny_read,
            label="read-only",
        )
        writable = self._validate_expansion_paths(
            cwd,
            additional_writable_paths,
            denied=(*self.config.deny_read, *self.config.deny_write),
            label="writable",
        )
        effective_network = (
            self.config.network_access
            if network_access is None
            else bool(network_access)
        )
        expanded = bool(readonly or writable) or (
            effective_network and not self.config.network_access
        )
        if expanded:
            self.emit_event(
                "sandbox_permission_expansion",
                {
                    "network_access": effective_network,
                    "additional_readonly_paths": list(readonly),
                    "additional_writable_paths": list(writable),
                    "scope": "single_command",
                },
            )
        return replace(
            self.config,
            network_access=effective_network,
            extra_readonly_paths=tuple(
                dict.fromkeys((*self.config.extra_readonly_paths, *readonly))
            ),
            extra_writable_paths=tuple(
                dict.fromkeys((*self.config.extra_writable_paths, *writable))
            ),
        )

    @staticmethod
    def _validate_expansion_paths(cwd, values, *, denied, label):
        resolved = []
        denied_paths = [Path(value).expanduser().resolve() for value in denied]
        for raw in values or ():
            path = Path(str(raw)).expanduser()
            if not path.is_absolute():
                raise ValueError(f"additional {label} path must be absolute: {raw}")
            path = path.resolve()
            if not path.exists():
                raise ValueError(f"additional {label} path does not exist: {path}")
            if any(_paths_overlap(path, denied_path) for denied_path in denied_paths):
                raise PermissionError(
                    f"additional {label} path conflicts with a hard deny: {path}"
                )
            if path == cwd.resolve():
                continue
            resolved.append(str(path))
        return tuple(resolved)

    def _plain(self, command, *, cwd, env, timeout, cancellation_token=None):
        kwargs = {
            "cwd": cwd,
            "shell": True,
            "timeout": timeout,
            "env": env,
            "cancellation_token": cancellation_token,
        }
        if _DEFAULT_SHELL:
            # Python resolves ``shell=True`` through %ComSpec% on Windows.
            # Keep an absolute launcher even when the child environment is
            # deliberately reduced to a strict allowlist.
            kwargs["executable"] = _DEFAULT_SHELL
        return self._run_process(command, **kwargs)

    def _run_process(self, command, **kwargs):
        cancellation_token = kwargs.pop("cancellation_token", None)
        if self.run_process is not None:
            if cancellation_token is not None:
                cancellation_token.raise_if_cancelled()
            return self.run_process(
                command,
                capture_output=True,
                text=True,
                **kwargs,
            )
        return run_cancellable_process(
            command,
            cancellation_token=cancellation_token,
            **kwargs,
        )

    def _bubblewrap_argv(self, backend_path, command, cwd, config):
        argv = [
            backend_path,
            "--die-with-parent",
            "--proc",
            "/proc",
            "--dev",
            "/dev",
        ]
        if not config.network_access:
            argv.append("--unshare-net")
        for system_path in ("/usr", "/bin", "/sbin", "/lib", "/lib64", "/etc"):
            if Path(system_path).exists():
                argv.extend(["--ro-bind", system_path, system_path])
        argv.extend(["--tmpfs", "/tmp"])
        bind_mode = "--bind" if config.workspace_write else "--ro-bind"
        argv.extend([bind_mode, str(cwd), str(cwd)])
        for path in config.extra_readonly_paths:
            argv.extend(["--ro-bind", path, path])
        for path in config.extra_writable_paths:
            argv.extend(["--bind", path, path])
        for path in (*config.deny_read, *config.deny_write):
            argv.extend(["--tmpfs", path])
        argv.extend(["--chdir", str(cwd), "--", "/bin/sh", "-lc", str(command)])
        return argv

    def _run_sandbox_exec(
        self,
        backend_path,
        command,
        cwd,
        config,
        *,
        env,
        timeout,
        cancellation_token,
    ):
        profile = self._sandbox_exec_profile(cwd, config)
        profile_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", suffix=".sb", delete=False
            ) as handle:
                handle.write(profile)
                profile_path = handle.name
            argv = [
                backend_path,
                "-f",
                profile_path,
                "/bin/sh",
                "-lc",
                str(command),
            ]
            return self._run_process(
                argv,
                cwd=cwd,
                env=env,
                timeout=timeout,
                cancellation_token=cancellation_token,
            )
        finally:
            if profile_path:
                Path(profile_path).unlink(missing_ok=True)

    @staticmethod
    def _sandbox_exec_profile(cwd, config):
        lines = [
            "(version 1)",
            "(deny default)",
            "(allow process*)",
            "(allow sysctl-read)",
            "(allow file-read*)",
            '(allow file-write* (subpath "/tmp") (subpath "/private/tmp"))',
        ]
        writable = list(config.extra_writable_paths)
        if config.workspace_write:
            writable.insert(0, str(cwd))
        for path in writable:
            lines.append(f"(allow file-write* (subpath {json.dumps(str(path))}))")
        if config.network_access:
            lines.append("(allow network*)")
        for path in config.deny_read:
            lines.append(f"(deny file-read* (subpath {json.dumps(str(path))}))")
        for path in config.deny_write:
            lines.append(f"(deny file-write* (subpath {json.dumps(str(path))}))")
        return "\n".join(lines) + "\n"

    def _container_argv(self, backend_path, backend_name, command, cwd, config):
        if config.deny_read or config.deny_write:
            raise RuntimeError(
                f"{backend_name} backend cannot safely express deny_read/deny_write; "
                "use mount allowlists instead"
            )
        argv = [
            backend_path,
            "run",
            "--rm",
            "--read-only",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            "--workdir=/workspace",
        ]
        if not config.network_access:
            argv.append("--network=none")
        workspace_mount = _container_mount(cwd, "/workspace", config.workspace_write)
        argv.extend(["--mount", workspace_mount])
        for index, path in enumerate(config.extra_readonly_paths):
            target = _container_target(path, kind="ro", index=index)
            argv.extend(["--mount", _container_mount(path, target, False)])
        for index, path in enumerate(config.extra_writable_paths):
            target = _container_target(path, kind="rw", index=index)
            argv.extend(["--mount", _container_mount(path, target, True)])
        argv.extend(
            [
                "--tmpfs=/tmp:rw,noexec,nosuid,size=64m",
                config.container_image,
                "/bin/sh",
                "-lc",
                str(command),
            ]
        )
        return argv


def _paths_overlap(left, right):
    try:
        left.relative_to(right)
        return True
    except ValueError:
        pass
    try:
        right.relative_to(left)
        return True
    except ValueError:
        return False


def _container_mount(source, target, writable):
    source_text = str(Path(source).resolve())
    if "," in source_text:
        raise ValueError("container mount paths cannot contain commas")
    mode = "" if writable else ",readonly"
    return f"type=bind,source={source_text},target={target}{mode}"


def _container_target(path, *, kind, index):
    if os.name != "nt":
        return str(Path(path).resolve())
    return f"/mnt/lite-{kind}-{index}"
