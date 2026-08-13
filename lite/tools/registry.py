"""工具定义与执行辅助逻辑。

可以把这个文件看成 agent 的能力白名单：模型能申请哪些动作、这些动作
如何做参数校验，以及最终如何执行，都是在这里定义的。
"""

import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import textwrap
from contextlib import contextmanager
from functools import partial
from pathlib import Path

from pydantic import ValidationError

from ..core.verification import (
    VERIFICATION_RECEIPT_SCHEMA,
    _contains_shell_control_operator,
    classify_verification_command,
    workspace_revision,
)
from ..core.workspace import IGNORED_PATH_NAMES
from ..features.sandbox.process import run_cancellable_process
from . import media as media_tools
from .base import RegisteredTool
from .agents import (
    AGENT_TOOL_EXAMPLES,
    AGENT_TOOL_NAMES,
    AGENT_TOOL_SPECS,
    tool_agent,
    tool_send_message,
    tool_task_stop,
    validate_agent_runtime,
)
from .ask_user import (
    ASK_USER_TOOL_EXAMPLES,
    ASK_USER_TOOL_SPECS,
    tool_ask_user,
)
from .plan import (
    PLAN_TOOL_EXAMPLES,
    PLAN_TOOL_SPECS,
    tool_enter_plan_mode,
    tool_exit_plan_mode,
)
from .todos import (
    TODO_TOOL_EXAMPLES,
    TODO_TOOL_SPECS,
    tool_todo_add,
    tool_todo_list,
    tool_todo_update,
)
from .schemas import (
    AgentArgs,
    AskUserArgs,
    EnterPlanModeArgs,
    ExitPlanModeArgs,
    InspectImageArgs,
    ListFilesArgs,
    PatchFileArgs,
    ReadFileArgs,
    RunShellArgs,
    VerifyArgs,
    SearchArgs,
    SendMessageArgs,
    TaskStopArgs,
    TodoAddArgs,
    TodoListArgs,
    TodoUpdateArgs,
    WriteFileArgs,
    first_error_message,
    provider_input_schema,
)

_TOOL_SCHEMAS = {
    "list_files": ListFilesArgs,
    "read_file": ReadFileArgs,
    "search": SearchArgs,
    "inspect_image": InspectImageArgs,
    "run_shell": RunShellArgs,
    "verify": VerifyArgs,
    "write_file": WriteFileArgs,
    "patch_file": PatchFileArgs,
    "todo_add": TodoAddArgs,
    "todo_update": TodoUpdateArgs,
    "todo_list": TodoListArgs,
    "agent": AgentArgs,
    "send_message": SendMessageArgs,
    "task_stop": TaskStopArgs,
    "enter_plan_mode": EnterPlanModeArgs,
    "exit_plan_mode": ExitPlanModeArgs,
    "ask_user": AskUserArgs,
}

BASE_TOOL_SPECS = {
    "list_files": {
        "risky": False,
        "description": "List files in the workspace.",
    },
    "read_file": {
        "risky": False,
        "description": "Read a UTF-8 file by line range.",
    },
    "search": {
        "risky": False,
        "description": "Search the workspace with rg or a simple fallback.",
    },
    "run_shell": {
        "risky": True,
        "description": (
            "Run a command in the workspace root using the host shell. On Windows "
            "this is cmd.exe, not a POSIX shell; do not use heredocs. Prefer "
            "patch_file/write_file for edits and python -m pytest for Python tests. "
            "Network and host paths are sandboxed by default; request one-command "
            "access explicitly when needed."
        ),
    },
    "verify": {
        "risky": True,
        "description": (
            "Run project verification in the workspace root and return a structured "
            "VerificationReceipt. On Windows the default Python verification uses "
            "the current interpreter (python -m pytest -q), avoiding pytest launcher "
            "path/import problems. A custom command must be one direct executable, "
            "without shell pipes, redirection, or command chaining. Provide command "
            "only when the project needs a different test, lint, build, or typecheck "
            "command."
        ),
    },
    "write_file": {
        "risky": True,
        "description": "Write a text file.",
    },
    "patch_file": {
        "risky": True,
        "description": "Replace one exact text block in a file.",
    },
    **media_tools.MEDIA_TOOL_SPECS,
    **TODO_TOOL_SPECS,
    **AGENT_TOOL_SPECS,
    **PLAN_TOOL_SPECS,
    **ASK_USER_TOOL_SPECS,
}

TOOL_EXAMPLES = {
    "list_files": {"path": "."},
    "read_file": {"path": "README.md", "start": 1, "end": 80},
    "search": {"pattern": "binary_search", "path": "."},
    "run_shell": {"command": "uv run --with pytest python -m pytest -q", "timeout": 20},
    "verify": {"command": "", "timeout": 120, "covered_paths": []},
    "write_file": {"path": "binary_search.py", "content": "def binary_search(nums, target):\n    return -1\n"},
    "patch_file": {"path": "binary_search.py", "old_text": "return -1", "new_text": "return mid"},
    **media_tools.MEDIA_TOOL_EXAMPLES,
    **TODO_TOOL_EXAMPLES,
    **AGENT_TOOL_EXAMPLES,
    **PLAN_TOOL_EXAMPLES,
    **ASK_USER_TOOL_EXAMPLES,
}


def build_tool_registry(agent):
    # 工具不是动态发现的，而是显式注册的。
    # 这样模型看到的是一个有边界、可审计的动作集合。
    tools = {
        name: RegisteredTool(
            name=name,
            schema=provider_input_schema(_TOOL_SCHEMAS[name]),
            description=spec["description"],
            risky=bool(spec["risky"]),
            runner=partial(_TOOL_RUNNERS[name], agent),
        )
        for name, spec in BASE_TOOL_SPECS.items()
    }
    return tools


def tool_example(name):
    example = TOOL_EXAMPLES.get(name, "")
    if isinstance(example, dict):
        return json.dumps(example, ensure_ascii=False, sort_keys=True)
    return str(example)
def validate_tool(agent, name, args):
    args = args or {}

    schema_cls = _TOOL_SCHEMAS.get(name)
    if schema_cls is not None:
        try:
            schema_cls.model_validate(args)
        except ValidationError as exc:
            raise ValueError(first_error_message(exc)) from exc

    # Workspace-aware checks that require the agent (path safety, file state).
    if name == "list_files":
        path = agent.path(args.get("path", "."))
        if not path.is_dir():
            raise ValueError("path is not a directory")

    elif name == "read_file":
        path = agent.path(args["path"])
        if not path.is_file():
            raise ValueError("path is not a file")

    elif name == "search":
        agent.path(args.get("path", "."))

    elif name == "verify":
        command = str(args.get("command", "")).strip()
        if command:
            _validate_verification_command(command)
        for path in args.get("covered_paths", []):
            agent.path(path)

    elif name in media_tools.MEDIA_TOOL_NAMES:
        media_tools.validate_media_runtime(agent, name, args)

    elif name == "write_file":
        path = agent.path(args["path"])
        if path.exists() and path.is_dir():
            raise ValueError("path is a directory")

    elif name == "patch_file":
        # patch_file 故意做得很严格：old_text 必须精确命中且只能出现一次，
        # 这样修改行为才是确定的，失败原因也更容易解释。
        path = agent.path(args["path"])
        if not path.is_file():
            raise ValueError("path is not a file")
        text = path.read_text(encoding="utf-8")
        count = text.count(str(args.get("old_text", "")))
        if count != 1:
            raise ValueError(f"old_text must occur exactly once, found {count}")

    elif name in AGENT_TOOL_NAMES:
        validate_agent_runtime(agent, name, args)


def tool_list_files(agent, args):
    path = agent.path(args.get("path", "."))
    if not path.is_dir():
        raise ValueError("path is not a directory")
    entries = _visible_entries(path)
    lines = []
    for entry in entries[:200]:
        kind = "[D]" if entry.is_dir() else "[F]"
        lines.append(f"{kind} {entry.relative_to(agent.root).as_posix()}")
        if entry.is_dir():
            for child in _visible_entries(entry)[:12]:
                child_kind = "[D]" if child.is_dir() else "[F]"
                lines.append(
                    f"  {child_kind} {child.relative_to(agent.root).as_posix()}"
                )
    return "\n".join(lines) or "(empty)"

def _visible_entries(path):
    return [item for item in sorted(path.iterdir(), key=lambda item: (item.is_file(), item.name.lower())) if item.name not in IGNORED_PATH_NAMES]


def tool_read_file(agent, args):
    path = agent.path(args["path"])
    if not path.is_file():
        raise ValueError("path is not a file")
    start = int(args.get("start", 1))
    end = int(args.get("end", 200))
    if start < 1 or end < start:
        raise ValueError("invalid line range")
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    body = "\n".join(
        f"{number:>4}: {line}"
        for number, line in enumerate(lines[start - 1 : end], start=start)
    )
    return f"# {path.relative_to(agent.root).as_posix()}\n{body}"


def tool_search(agent, args):
    pattern = str(args.get("pattern", "")).strip()
    if not pattern:
        raise ValueError("pattern must not be empty")
    path = agent.path(args.get("path", "."))

    if shutil.which("rg"):
        # 优先用 rg，因为搜索会非常频繁，搜索延迟会直接影响 agent 控制循环。
        result = run_cancellable_process(
            ["rg", "-n", "--smart-case", "--max-count", "200", pattern, str(path)],
            cwd=agent.root, env=agent.shell_env(), timeout=120,
            cancellation_token=agent.current_cancellation_token,
        )
        return result.stdout.strip() or result.stderr.strip() or "(no matches)"

    matches = []
    files = (
        [path]
        if path.is_file()
        else [
            item
            for item in path.rglob("*")
            if item.is_file()
            and not any(
                part in IGNORED_PATH_NAMES
                for part in item.relative_to(agent.root).parts
            )
        ]
    )
    for file_path in files:
        for number, line in enumerate(
            file_path.read_text(encoding="utf-8", errors="replace").splitlines(),
            start=1,
        ):
            if (number - 1) % 200 == 0:
                agent.current_cancellation_token.raise_if_cancelled()
            if pattern.lower() in line.lower():
                matches.append(f"{file_path.relative_to(agent.root).as_posix()}:{number}:{line}")
                if len(matches) >= 200:
                    return "\n".join(matches)
    return "\n".join(matches) or "(no matches)"


def tool_run_shell(agent, args):
    command = str(args.get("command", "")).strip()
    if not command:
        raise ValueError("command must not be empty")
    timeout = int(args.get("timeout", 20))
    if timeout < 1 or timeout > 120:
        raise ValueError("timeout must be in [1, 120]")
    result = _run_workspace_command(agent, command, args, timeout)
    return _format_command_result(result)


def tool_verify(agent, args):
    """Run a project check and attach a machine-readable verification receipt."""

    command = str(args.get("command", "")).strip() or discover_verification_command(
        agent.root
    )
    command_class = _validate_verification_command(command)
    timeout = int(args.get("timeout", 120))
    covered_paths = [
        agent.path(path).relative_to(agent.root).as_posix()
        for path in args.get("covered_paths", [])
    ]
    with _verification_workspace(agent) as (verification_root, backup_root, before):
        result = _run_workspace_command(
            agent,
            _verification_argv(command),
            args,
            timeout,
            shell=False,
            cwd=verification_root,
        )
        scope_violations = _verification_scope_violations(
            agent, verification_root, backup_root, before
        )
    exit_code = int(result.returncode)
    stderr = result.stderr
    if scope_violations:
        exit_code = exit_code or 126
        detail = "verify write_scope violation: " + ", ".join(scope_violations)
        stderr = f"{stderr.rstrip()}\n{detail}".strip()
        agent.session_event_bus.emit(
            "scope_violation",
            {"tool_name": "verify", "affected_paths": scope_violations},
        )
    changed_paths = list(
        getattr(getattr(agent, "current_task_state", None), "changed_paths", []) or []
    )
    if not covered_paths:
        covered_paths = list(changed_paths)
    coverage_confidence = (
        "declared"
        if args.get("covered_paths")
        else "inferred_changed_paths"
        if changed_paths
        else "unknown"
    )
    receipt = {
        "schema_version": VERIFICATION_RECEIPT_SCHEMA,
        "command": command,
        "command_class": command_class,
        "exit_code": exit_code,
        "workspace_revision": workspace_revision(agent.root),
        "covered_paths": covered_paths,
        "after_mutation_sequence": _last_mutation_sequence(agent),
        "coverage_confidence": coverage_confidence,
    }
    agent._pending_tool_result_metadata = {
        "verification_receipt": receipt,
        "security_event_type": "scope_violation" if scope_violations else "",
        "write_scope_violation_paths": scope_violations,
    }
    return textwrap.dedent(
        f"""\
        exit_code: {exit_code}
        verification_receipt:
        {json.dumps(receipt, ensure_ascii=False, sort_keys=True)}
        stdout:
        {result.stdout.strip() or "(empty)"}
        stderr:
        {stderr.strip() or "(empty)"}
        """
    ).strip()


def _run_workspace_command(agent, command, args, timeout, *, shell=True, cwd=None):
    cwd = Path(cwd or agent.root)
    env = agent.shell_env()
    env["PWD"] = str(cwd)
    runner = getattr(agent, "sandbox_runner", None)
    if runner is None:
        return run_cancellable_process(
            command,
            cwd=cwd,
            shell=shell,
            timeout=timeout,
            # 这里传入的是过滤后的环境变量，而不是直接继承整个父 shell 环境，
            # 目的是减少敏感信息被意外带进命令执行环境的风险。
            env=env,
            cancellation_token=getattr(agent, "current_cancellation_token", None),
        )
    return runner.run(
        command,
        cwd=cwd,
        env=env,
        timeout=timeout,
        cancellation_token=getattr(agent, "current_cancellation_token", None),
        network_access=args.get("network_access"),
        additional_readonly_paths=args.get("additional_readonly_paths", ()),
        additional_writable_paths=args.get("additional_writable_paths", ()),
        shell=shell,
    )


@contextmanager
def _verification_workspace(agent):
    if not getattr(agent, "write_scope", ()):
        yield agent.root, None, None
        return
    symlinks = _workspace_symlinks(agent.root)
    if symlinks:
        raise ValueError(
            "verify with write_scope does not support workspace symlinks: "
            + ", ".join(symlinks[:5])
        )
    with tempfile.TemporaryDirectory(prefix="lite-verify-") as temp_dir:
        backup_root = Path(temp_dir) / "backup"
        shutil.copytree(
            agent.root,
            backup_root,
            ignore=shutil.ignore_patterns(*IGNORED_PATH_NAMES),
        )
        isolated_root = Path(temp_dir) / "workspace"
        shutil.copytree(backup_root, isolated_root)
        before = _verification_snapshot(backup_root)
        yield isolated_root, backup_root, before


def _verification_scope_violations(agent, root, backup_root, before):
    if before is None:
        return []
    after = _verification_snapshot(root)
    isolated_changes = [
        path
        for path in sorted(set(before) | set(after))
        if before.get(path) != after.get(path)
    ]
    live_changes = _verification_live_changes(agent, before)
    live_violations = [
        path for path in live_changes if not _write_scope_allows(agent, path)
    ]
    if live_violations:
        _restore_verification_changes(agent.root, backup_root, live_violations)
    outside_scope = [
        path for path in isolated_changes if not _write_scope_allows(agent, path)
    ]
    return sorted(set(outside_scope) | set(live_violations))


def _verification_live_changes(agent, before):
    live_after = _verification_snapshot(agent.root)
    return [
        path
        for path in sorted(set(before) | set(live_after))
        if before.get(path) != live_after.get(path)
    ]


def _verification_snapshot(root):
    snapshot = {}
    root = Path(root)
    for current, dir_names, file_names in os.walk(root, topdown=True):
        current = Path(current)
        kept_dirs = []
        for name in sorted(dir_names):
            if name in IGNORED_PATH_NAMES:
                continue
            path = current / name
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                snapshot[relative] = "symlink:" + os.readlink(path)
                continue
            snapshot[relative] = "directory"
            kept_dirs.append(name)
        dir_names[:] = kept_dirs
        for name in sorted(file_names):
            if name in IGNORED_PATH_NAMES:
                continue
            path = current / name
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                snapshot[relative] = "symlink:" + os.readlink(path)
            else:
                snapshot[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return snapshot


def _restore_verification_changes(root, backup_root, paths):
    for relative_path in paths:
        target = root / Path(relative_path)
        source = backup_root / Path(relative_path)
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists() or target.is_symlink():
            target.unlink()
        if source.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        elif source.is_dir():
            target.mkdir(parents=True, exist_ok=True)


def _workspace_symlinks(root):
    root = Path(root)
    symlinks = []
    for current, dir_names, file_names in os.walk(root, topdown=True):
        current = Path(current)
        kept_dirs = []
        for name in sorted(dir_names):
            if name in IGNORED_PATH_NAMES:
                continue
            path = current / name
            if path.is_symlink():
                symlinks.append(path.relative_to(root).as_posix())
            else:
                kept_dirs.append(name)
        dir_names[:] = kept_dirs
        for name in sorted(file_names):
            path = current / name
            if name not in IGNORED_PATH_NAMES and path.is_symlink():
                symlinks.append(path.relative_to(root).as_posix())
    return symlinks


def _write_scope_allows(agent, relative_path):
    requested = agent.path(relative_path)
    for raw_scope in agent.write_scope:
        scope = agent.path(raw_scope)
        try:
            requested.relative_to(scope)
            return True
        except ValueError:
            continue
    return False


def _verification_argv(command):
    if _contains_shell_control_operator(command):
        raise ValueError(
            "verify.command must be one direct verification command without "
            "shell control operators"
        )
    try:
        argv = shlex.split(str(command), posix=os.name != "nt")
    except ValueError as exc:
        raise ValueError(f"invalid verify.command quoting: {exc}") from exc
    if not argv:
        raise ValueError("verify.command must not be empty")
    argv[0] = str(_resolve_verification_executable(argv[0]))
    if os.name == "nt":
        argv[1:] = [
            token[1:-1]
            if len(token) >= 2 and token[0] == token[-1] and token[0] in {'"', "'"}
            else token
            for token in argv[1:]
        ]
    return argv


def _resolve_verification_executable(raw):
    executable = str(raw).strip().strip('"\'')
    path = Path(executable)
    if path.is_absolute():
        return path
    resolved = shutil.which(executable)
    if resolved:
        return Path(resolved)
    if os.name == "nt" and not path.suffix:
        for suffix in (".exe", ".cmd", ".bat"):
            resolved = shutil.which(executable + suffix)
            if resolved:
                return Path(resolved)
    return path


def _format_command_result(result):
    return textwrap.dedent(
        f"""\
        exit_code: {result.returncode}
        stdout:
        {result.stdout.strip() or "(empty)"}
        stderr:
        {result.stderr.strip() or "(empty)"}
        """
    ).strip()


def discover_verification_command(root):
    """Choose a local verification command without invoking a platform launcher."""

    root = root.resolve()
    package_json = root / "package.json"
    if package_json.is_file():
        try:
            scripts = json.loads(package_json.read_text(encoding="utf-8")).get(
                "scripts", {}
            )
        except (OSError, ValueError):
            scripts = {}
        if isinstance(scripts, dict) and scripts.get("test"):
            manager = (
                "pnpm" if (root / "pnpm-lock.yaml").exists()
                else "yarn" if (root / "yarn.lock").exists()
                else "npm"
            )
            return f"{manager} test"
    if (root / "go.mod").is_file():
        return "go test ./..."
    if (root / "Cargo.toml").is_file():
        return "cargo test"
    if (
        (root / "pyproject.toml").is_file()
        or (root / "pytest.ini").is_file()
        or (root / "tox.ini").is_file()
        or (root / "tests").is_dir()
        or any(root.glob("test_*.py"))
    ):
        return f"{_quote_argument(sys.executable)} -m pytest -q"
    raise ValueError("no default verification command found; provide verify.command")


def _quote_argument(value):
    if os.name == "nt":
        return subprocess.list2cmdline([str(value)])
    return shlex.quote(str(value))


def _validate_verification_command(command):
    command_class = classify_verification_command(command)
    if not command_class:
        raise ValueError(
            "verify.command is not recognized as a test, lint, typecheck, "
            "compile, or build command; use run_shell for other commands"
        )
    return command_class


def _last_mutation_sequence(agent):
    signal = dict(
        getattr(getattr(agent, "current_task_state", None), "evidence_summaries", {})
        .get("verification_signal", {})
        or {}
    )
    span_id = str(signal.get("last_workspace_change_span_id", ""))
    try:
        return int(span_id.rsplit("_", 1)[-1])
    except (TypeError, ValueError):
        return 0


def tool_write_file(agent, args):
    path = agent.path(args["path"])
    content = str(args["content"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return f"wrote {path.relative_to(agent.root).as_posix()} ({len(content)} chars)"


def tool_patch_file(agent, args):
    path = agent.path(args["path"])
    if not path.is_file():
        raise ValueError("path is not a file")
    old_text = str(args.get("old_text", ""))
    if not old_text:
        raise ValueError("old_text must not be empty")
    if "new_text" not in args:
        raise ValueError("missing new_text")
    text = path.read_text(encoding="utf-8")
    count = text.count(old_text)
    if count != 1:
        raise ValueError(f"old_text must occur exactly once, found {count}")
    path.write_text(text.replace(old_text, str(args["new_text"]), 1), encoding="utf-8")
    return f"patched {path.relative_to(agent.root).as_posix()}"


_TOOL_RUNNERS = {
    "list_files": tool_list_files,
    "read_file": tool_read_file,
    "search": tool_search,
    "run_shell": tool_run_shell,
    "verify": tool_verify,
    "write_file": tool_write_file,
    "patch_file": tool_patch_file,
    "todo_add": tool_todo_add,
    "todo_update": tool_todo_update,
    "todo_list": tool_todo_list,
    "agent": tool_agent,
    "send_message": tool_send_message,
    "task_stop": tool_task_stop,
    "enter_plan_mode": tool_enter_plan_mode,
    "exit_plan_mode": tool_exit_plan_mode,
    "ask_user": tool_ask_user,
    **media_tools.MEDIA_TOOL_RUNNERS,
}
