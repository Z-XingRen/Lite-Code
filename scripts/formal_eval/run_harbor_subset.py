"""Run a fixed formal-evaluation subset through the pinned Harbor checkout."""

from __future__ import annotations

import argparse
import json
import os
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
    if config.model != "gpt-5.5" or config.reasoning_effort != "medium":
        raise RuntimeError(
            "Harbor formal runs require .lite.toml model=gpt-5.5 and "
            "reasoning_effort=medium"
        )
    if config.protocol != "openai" or not config.api_key:
        raise RuntimeError("Harbor formal runs require an OpenAI-compatible API key")
    return config


def build_command(subset_name: str, agent: str) -> tuple[list[str], Path]:
    if subset_name not in SUBSET_DIRS:
        raise ValueError(f"unknown subset: {subset_name}")
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    subset = payload["subsets"][subset_name]
    output = ARTIFACT_ROOT / SUBSET_DIRS[subset_name]
    output.mkdir(parents=True, exist_ok=True)
    job_name = (
        f"{subset_name}-{agent}-gpt-5-5" if agent == "lite" else f"{subset_name}-oracle"
    )
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
        config = _provider_config()
        command.extend(["--model", f"openai/{config.model}"])
        hostname = urlparse(config.base_url).hostname
        if hostname:
            command.extend(["--allow-agent-host", hostname])
    for task in subset["tasks"]:
        command.extend(["--include-task-name", task["task_id"]])
    return command, output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", choices=sorted(SUBSET_DIRS), required=True)
    parser.add_argument("--agent", choices=("oracle", "lite"), required=True)
    args = parser.parse_args(argv)
    if not HARBOR_EXE.is_file():
        raise RuntimeError(f"Harbor executable not found: {HARBOR_EXE}")
    command, output = build_command(args.subset, args.agent)
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
