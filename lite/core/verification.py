"""Verification evidence reducer for shell-command tool traces."""

import hashlib
import os
import re
import shlex
import subprocess
from pathlib import Path

VERIFICATION_SIGNAL_SCHEMA = "lite.verification_signal.v1"
VERIFICATION_RECEIPT_SCHEMA = "lite.verification_receipt.v1"
_REVISION_IGNORED_PARTS = frozenset(
    {
        ".git",
        ".lite",
        ".mypy_cache",
        ".nox",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "node_modules",
        "target",
        "venv",
    }
)


def reduce_verification_signal(previous, event, changed_paths):
    signal = dict(previous or {})
    if event.get("event") != "tool_executed":
        return signal
    if event.get("workspace_changed"):
        previous_mutation = int(signal.get("last_mutation_sequence", 0) or 0)
        signal = {
            "schema_version": VERIFICATION_SIGNAL_SCHEMA,
            "state": "missing",
            "last_workspace_change_span_id": str(event.get("span_id", "")),
            "changed_paths": list(changed_paths or []),
            "test_state": "missing",
            "last_mutation_sequence": _event_sequence(
                event, previous_mutation + 1
            ),
            "last_successful_verification_sequence": 0,
        }
    receipt = dict(event.get("verification_receipt", {}) or {})
    command = str(
        receipt.get("command")
        or (event.get("args", {}) or {}).get("command", "")
    ).strip()
    command_class = str(receipt.get("command_class", "")).strip()
    command_class = command_class or classify_verification_command(command)
    is_shell_verification = event.get("name") == "run_shell" and bool(command_class)
    is_receipt_verification = (
        event.get("name") == "verify"
        and receipt.get("schema_version") == VERIFICATION_RECEIPT_SCHEMA
    )
    if not is_shell_verification and not is_receipt_verification:
        return signal
    exit_code = int(receipt.get("exit_code", 0) or 0)
    passed = str(event.get("status", "")) in {"", "ok"} and exit_code == 0
    covered_paths = list(receipt.get("covered_paths", []) or [])
    covers_changed_paths = bool(changed_paths) and all(
        _path_is_covered(path, covered_paths) for path in changed_paths
    )
    signal.update(
        {
            "schema_version": VERIFICATION_SIGNAL_SCHEMA,
            "state": "passed" if passed else "failed",
            "source_span_id": str(event.get("span_id", "")),
            "command": command,
            "command_class": command_class,
            "after_last_workspace_change": bool(
                signal.get("last_workspace_change_span_id") or changed_paths
            ),
            "changed_paths_present": bool(changed_paths),
            "covers_changed_paths": covers_changed_paths,
            "coverage_confidence": str(
                receipt.get("coverage_confidence", "unknown") or "unknown"
            ),
            "changed_paths": list(changed_paths or []),
        }
    )
    if passed:
        signal["last_successful_verification_sequence"] = _event_sequence(
            event,
            int(signal.get("last_mutation_sequence", 0) or 0) + 1,
        )
    if receipt:
        signal.update(
            {
                "verification_receipt": receipt,
                "workspace_revision": str(receipt.get("workspace_revision", "")),
                "after_mutation_sequence": int(
                    receipt.get("after_mutation_sequence", 0) or 0
                ),
            }
        )
    if command_class == "test":
        signal.update(
            {
                "test_state": "passed" if passed else "failed",
                "test_command": command,
                "test_span_id": str(event.get("span_id", "")),
                "test_after_last_workspace_change": bool(
                    signal.get("last_workspace_change_span_id") or changed_paths
                ),
            }
        )
    return signal


def _event_sequence(event, fallback):
    for value in (
        event.get("mutation_sequence"),
        event.get("verification_sequence"),
        str(event.get("span_id", "")).rsplit("_", 1)[-1],
    ):
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            return parsed
    return int(fallback)


def _path_is_covered(path, covered_paths):
    candidate = str(path).replace("\\", "/").strip("/")
    for covered in covered_paths:
        prefix = str(covered).replace("\\", "/").strip("/")
        if (
            prefix in {"", "."}
            or candidate == prefix
            or candidate.startswith(prefix + "/")
        ):
            return True
    return False


def workspace_revision(root):
    """Return a content revision for the current workspace, excluding runtime state."""

    root = Path(root).resolve()
    revision = _git_workspace_revision(root)
    if revision:
        return revision
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(part in _REVISION_IGNORED_PARTS for part in relative.parts):
            continue
        try:
            payload = path.read_bytes()
        except OSError:
            continue
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(payload).digest())
    return "sha256:" + digest.hexdigest()


def _git_workspace_revision(root):
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            check=True,
            timeout=10,
        ).stdout
        diff = subprocess.run(
            ["git", "diff", "--binary", "--no-ext-diff", "HEAD", "--"],
            cwd=root,
            capture_output=True,
            check=True,
            timeout=30,
        ).stdout
        untracked = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard", "-z"],
            cwd=root,
            capture_output=True,
            check=True,
            timeout=10,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return ""

    digest = hashlib.sha256()
    digest.update(head.strip())
    digest.update(b"\0")
    digest.update(diff)
    for raw_path in sorted(item for item in untracked.split(b"\0") if item):
        try:
            relative = raw_path.decode("utf-8", errors="surrogateescape")
            if any(part in _REVISION_IGNORED_PARTS for part in Path(relative).parts):
                continue
            path = root / relative
            payload = (
                os.readlink(path).encode("utf-8", errors="surrogateescape")
                if path.is_symlink()
                else path.read_bytes()
            )
        except OSError:
            continue
        digest.update(raw_path)
        digest.update(b"\0")
        digest.update(hashlib.sha256(payload).digest())
    return "sha256:" + digest.hexdigest()


def classify_verification_command(command):
    try:
        tokens = shlex.split(str(command), posix=os.name != "nt")
    except ValueError:
        tokens = str(command).split()
    tokens = [token.strip("\"'").lower() for token in tokens]
    if not tokens or tokens[0] in {"echo", "printf", "grep", "rg", "cat"}:
        return ""
    if tokens[0] == "uv" and len(tokens) > 2 and tokens[1] == "run":
        while len(tokens) > 2 and tokens[2].startswith("-"):
            tokens = tokens[:2] + tokens[3:]
        tokens = tokens[2:]
    python_cmd = re.split(r"[\\/]", tokens[0])[-1].removesuffix(".exe")
    if len(tokens) > 2 and _is_python_command(python_cmd) and tokens[1] == "-m":
        module = tokens[2]
        if module == "ruff" and len(tokens) > 3 and tokens[3] == "check":
            return "lint"
        return {
            "compileall": "compile",
            "mypy": "typecheck",
            "py_compile": "compile",
            "pytest": "test",
        }.get(module, "")
    if len(tokens) > 2 and _is_python_command(python_cmd) and tokens[1] == "-c":
        return "synthetic" if "assert" in " ".join(tokens[2:]) else ""
    if tokens[0] in {"pytest", "tox"}:
        return "test"
    if tokens[0] == "ruff" and len(tokens) > 1 and tokens[1] == "check":
        return "lint"
    if tokens[0] in {"mypy", "pyright"}:
        return "typecheck"
    if tokens[0] in {"npm", "pnpm"}:
        return _js_command_class(tokens)
    if tokens[:2] in (["yarn", "test"], ["go", "test"], ["cargo", "test"], ["make", "test"]):
        return "test"
    return ""


def _js_command_class(tokens):
    if len(tokens) < 2:
        return ""
    if tokens[1] == "test":
        return "test"
    if len(tokens) > 2 and tokens[1:3] in (["run", "test"], ["run", "build"]):
        return "test" if tokens[2] == "test" else "build"
    return "build" if tokens[1] == "build" else ""


def _is_python_command(command):
    suffix = command.removeprefix("python3.")
    return command in {"python", "python3"} or (
        suffix != command and suffix.replace(".", "").isdigit()
    )
