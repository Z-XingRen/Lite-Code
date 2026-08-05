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
    if discovered:
        return discovered
    if os.name != "nt":
        return None
    for candidate in (
        Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Git" / "bin" / "bash.exe",
        Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Git" / "usr" / "bin" / "bash.exe",
    ):
        if candidate.is_file():
            return str(candidate)
    return None


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
            timeout=timeout,
        )
    return subprocess.run(
        command,
        cwd=cwd,
        shell=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _bash_python():
    value = Path(sys.executable).as_posix()
    if re.match(r"^[A-Za-z]:/", value):
        value = f"/{value[0].lower()}{value[2:]}"
    return shlex.quote(value)
