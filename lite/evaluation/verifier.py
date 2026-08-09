"""Portable execution for benchmark-owned verifier commands."""

import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


def find_bash():
    discovered = shutil.which("bash")
    windows_candidates = (
        Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Git" / "bin" / "bash.exe",
        Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Git" / "usr" / "bin" / "bash.exe",
    )
    return _select_bash_path(
        platform_name=os.name,
        discovered=discovered,
        windows_candidates=windows_candidates,
    )


def _select_bash_path(*, platform_name, discovered, windows_candidates):
    if platform_name == "nt":
        for candidate in windows_candidates:
            if candidate.is_file():
                return str(candidate)
    return discovered


def run_verifier(command, *, cwd, timeout=None):
    command = str(command)
    bash = find_bash() if os.name == "nt" else None
    if bash:
        command = re.sub(r"\bpython3\b", _bash_python(), command)
        return subprocess.run(
            [bash, "-lc", command],
            cwd=cwd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    return subprocess.run(
        command,
        cwd=cwd,
        shell=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )


def _bash_python():
    value = Path(sys.executable).as_posix()
    if re.match(r"^[A-Za-z]:/", value):
        value = f"/{value[0].lower()}{value[2:]}"
    return shlex.quote(value)
