"""Run a fixed formal-evaluation subset through the pinned Harbor checkout."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from urllib.parse import urlparse

from lite.config import load_project_env, resolve_provider_config


ROOT = Path(__file__).resolve().parents[2]
HARBOR = ROOT / "benchmarks" / "external" / "harbor"
REGISTRY = HARBOR / "registry.json"
MANIFEST = ROOT / "artifacts" / "formal-evaluation-20260806" / "harbor-subsets.json"
HARBOR_EXE = HARBOR / ".venv" / "Scripts" / "harbor.exe"
ARTIFACT_ROOT = ROOT / "artifacts" / "formal-evaluation-20260806"
SUBSET_DIRS = {
    "terminal-bench-20": "terminal-bench-20",
    "swebench-verified-10": "swebench-verified-10",
}


def _provider_config():
    load_project_env(ROOT, override=True)
    config = resolve_provider_config(
        "openai", start=ROOT, config_path=str(ROOT / ".lite.toml")
    )
    if config.protocol != "openai" or not config.api_key:
        raise RuntimeError("Harbor formal runs require an OpenAI-compatible API key")
    return config


def build_command(
    subset_name: str,
    agent: str,
    *,
    max_tasks: int = 0,
    artifact_root: Path = ARTIFACT_ROOT,
) -> tuple[list[str], Path]:
    if subset_name not in SUBSET_DIRS:
        raise ValueError(f"unknown subset: {subset_name}")
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    subset = payload["subsets"][subset_name]
    output = artifact_root / SUBSET_DIRS[subset_name]
    output.mkdir(parents=True, exist_ok=True)
    config = _provider_config() if agent == "lite" else None
    model_slug = (
        "-".join(part for part in config.model.lower().replace(".", "-").split("-") if part)
        if config
        else "oracle"
    )
    job_name = f"{subset_name}-{agent}-{model_slug}"
    command = [
        str(HARBOR_EXE),
        "run",
        "--dataset",
        subset["dataset"],
        "--registry-path",
        str(REGISTRY),
        "--agent",
        (
            "scripts.formal_eval.harbor_lite_agent:LiteHarborAgent"
            if agent == "lite"
            else "oracle"
        ),
        "--jobs-dir",
        str(output / "jobs"),
        "--job-name",
        job_name,
        "--n-attempts",
        "1",
        "--n-concurrent",
        "1",
        "--max-retries",
        "0",
        "--agent-setup-timeout-multiplier",
        "3",
    ]
    if agent == "lite":
        command.extend(["--model", f"openai/{config.model}"])
        hostname = urlparse(config.base_url).hostname
        if hostname:
            command.extend(["--allow-agent-host", hostname])
    tasks = list(subset["tasks"])
    if max_tasks > 0:
        tasks = tasks[:max_tasks]
    for task in tasks:
        command.extend(["--include-task-name", task["task_id"]])
    return command, output


def preflight():
    checks = {
        "harbor_executable": HARBOR_EXE.is_file(),
        "registry": REGISTRY.is_file(),
        "subset_manifest": MANIFEST.is_file(),
        "docker_cli": bool(shutil.which("docker")),
        "docker_daemon": False,
    }
    detail = ""
    if checks["docker_cli"]:
        try:
            completed = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            )
            checks["docker_daemon"] = completed.returncode == 0
            detail = (completed.stderr or completed.stdout)[-1000:]
        except (OSError, subprocess.SubprocessError) as exc:
            detail = f"{type(exc).__name__}: {exc}"
    return {
        "checks": checks,
        "ready": all(checks.values()),
        "detail": detail,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", choices=sorted(SUBSET_DIRS), required=True)
    parser.add_argument("--agent", choices=("oracle", "lite"), required=True)
    parser.add_argument("--max-tasks", type=int, default=0)
    parser.add_argument("--output-dir", default=str(ARTIFACT_ROOT))
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    artifact_root = Path(args.output_dir).resolve()
    output = artifact_root / SUBSET_DIRS[args.subset]
    output.mkdir(parents=True, exist_ok=True)
    readiness = preflight()
    (output / "preflight.json").write_text(
        json.dumps(readiness, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.preflight_only or not readiness["ready"]:
        print(json.dumps(readiness, ensure_ascii=True))
        return 0 if readiness["ready"] else 2
    command, output = build_command(
        args.subset,
        args.agent,
        max_tasks=args.max_tasks,
        artifact_root=artifact_root,
    )
    env = {
        **os.environ,
        "PYTHONPATH": str(ROOT),
        "PYTHONUTF8": "1",
        "PYTHONIOENCODING": "utf-8",
    }
    stdout_path = output / f"{args.agent}.stdout.log"
    stderr_path = output / f"{args.agent}.stderr.log"
    with (
        stdout_path.open("w", encoding="utf-8") as stdout,
        stderr_path.open("w", encoding="utf-8") as stderr,
    ):
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            stdout=stdout,
            stderr=stderr,
            text=True,
            check=False,
        )
    result = {
        "subset": args.subset,
        "agent": args.agent,
        "returncode": completed.returncode,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "command": command,
    }
    (output / f"{args.agent}.run.json").write_text(
        json.dumps(result, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=True))
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
