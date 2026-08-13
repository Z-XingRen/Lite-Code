import json
import os
import subprocess
import sys

import pytest

from lite.features.sandbox.config import SandboxConfig
from lite.features.sandbox.checker import SandboxChecker
from lite.features.sandbox.runner import SandboxRunner
from lite.testing import shell_join


def test_required_sandbox_rejects_when_backend_is_unavailable(tmp_path):
    runner = SandboxRunner(
        SandboxConfig(mode="required", backend="bubblewrap"), which=lambda name: None
    )

    with pytest.raises(RuntimeError, match="sandbox required but unavailable"):
        runner.run("echo hi", cwd=tmp_path, env={}, timeout=5)


def test_required_sandbox_does_not_honor_excluded_commands(tmp_path):
    runner = SandboxRunner(
        SandboxConfig(mode="required", backend="bubblewrap", excluded_commands=("*",)),
        which=lambda name: None,
    )

    with pytest.raises(RuntimeError, match="sandbox required but unavailable"):
        runner.run("echo hi", cwd=tmp_path, env={}, timeout=5)


def test_best_effort_sandbox_records_degrade_and_runs_without_backend(tmp_path):
    events = []
    runner = SandboxRunner(
        SandboxConfig(mode="best_effort", backend="bubblewrap"),
        which=lambda name: None,
        emit_event=lambda event, payload: events.append((event, payload)),
    )

    result = runner.run(
        shell_join([sys.executable, "-c", "print(42)"]),
        cwd=tmp_path,
        env=os.environ.copy(),
        timeout=5,
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "42"
    assert events[0][0] == "sandbox_unavailable"


def test_off_sandbox_keeps_plain_subprocess_behavior(tmp_path):
    runner = SandboxRunner(SandboxConfig(mode="off"), run=subprocess.run)

    result = runner.run("echo lite", cwd=tmp_path, env=os.environ.copy(), timeout=5)

    assert result.stdout.strip() == "lite"


def test_off_sandbox_can_run_argv_without_shell_interpretation(tmp_path):
    runner = SandboxRunner(SandboxConfig(mode="off"), run=subprocess.run)

    result = runner.run(
        [sys.executable, "-c", "print('verified')", "&&", "echo", "bypass"],
        cwd=tmp_path,
        env=os.environ.copy(),
        timeout=5,
        shell=False,
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "verified"


def test_auto_backend_is_platform_aware():
    available = {
        "bwrap": "/bin/bwrap",
        "sandbox-exec": "/usr/bin/sandbox-exec",
        "docker": "/usr/bin/docker",
    }
    which = available.get

    assert SandboxChecker(which, platform="linux").resolve_backend("auto").name == "bubblewrap"
    assert SandboxChecker(which, platform="darwin").resolve_backend("auto").name == "sandbox-exec"
    assert SandboxChecker(which, platform="win32").resolve_backend("auto").name == "docker"


def test_bubblewrap_denies_network_by_default(tmp_path):
    calls = []
    runner = SandboxRunner(
        SandboxConfig(mode="required", backend="bubblewrap"),
        which=lambda name: "/usr/bin/bwrap" if name == "bwrap" else None,
        run=lambda command, **kwargs: calls.append((command, kwargs))
        or subprocess.CompletedProcess(command, 0, "ok", ""),
    )

    runner.run("echo hi", cwd=tmp_path, env={}, timeout=5)

    assert "--unshare-net" in calls[0][0]
    assert ["--tmpfs", "/tmp"] == calls[0][0][
        calls[0][0].index("--tmpfs") : calls[0][0].index("--tmpfs") + 2
    ]


def test_single_command_network_expansion_is_observable(tmp_path):
    calls = []
    events = []
    runner = SandboxRunner(
        SandboxConfig(mode="required", backend="bubblewrap"),
        which=lambda name: "/usr/bin/bwrap" if name == "bwrap" else None,
        run=lambda command, **kwargs: calls.append((command, kwargs))
        or subprocess.CompletedProcess(command, 0, "ok", ""),
        emit_event=lambda event, payload: events.append((event, payload)),
    )

    runner.run(
        "echo hi", cwd=tmp_path, env={}, timeout=5, network_access=True
    )

    assert "--unshare-net" not in calls[0][0]
    expansion = next(payload for event, payload in events if event == "sandbox_permission_expansion")
    assert expansion["network_access"] is True
    assert expansion["scope"] == "single_command"


def test_dynamic_path_expansion_cannot_override_hard_deny(tmp_path):
    denied = tmp_path / "private"
    denied.mkdir()
    runner = SandboxRunner(
        SandboxConfig(
            mode="required",
            backend="bubblewrap",
            deny_read=(str(denied),),
        )
    )

    with pytest.raises(PermissionError, match="conflicts with a hard deny"):
        runner.run(
            "echo hi",
            cwd=tmp_path,
            env={},
            timeout=5,
            additional_readonly_paths=[str(denied)],
        )


def test_container_backend_uses_hardened_defaults(tmp_path):
    calls = []
    runner = SandboxRunner(
        SandboxConfig(mode="required", backend="docker"),
        which=lambda name: "docker.exe" if name == "docker" else None,
        run=lambda command, **kwargs: calls.append((command, kwargs))
        or subprocess.CompletedProcess(command, 0, "ok", ""),
    )

    runner.run("python -V", cwd=tmp_path, env={}, timeout=5)

    argv = calls[0][0]
    assert "--read-only" in argv
    assert "--cap-drop=ALL" in argv
    assert "--security-opt=no-new-privileges" in argv
    assert "--network=none" in argv
    assert "--workdir=/workspace" in argv
    assert "python:3.12-slim" in argv


def test_sandbox_exec_profile_allows_only_requested_writes_and_network(tmp_path):
    runner = SandboxRunner(SandboxConfig())
    config = SandboxConfig(
        mode="required",
        backend="sandbox-exec",
        workspace_write=True,
        network_access=False,
        deny_read=("/Users/example/.ssh",),
    )

    profile = runner._sandbox_exec_profile(tmp_path, config)

    assert "(deny default)" in profile
    assert "(allow network*)" not in profile
    assert f"(allow file-write* (subpath {json.dumps(str(tmp_path))}))" in profile
    assert '(deny file-read* (subpath "/Users/example/.ssh"))' in profile
