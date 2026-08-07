"""Harbor custom agent adapter for the project-local Lite runtime.

The adapter installs the current Lite source snapshot into the task container,
runs the normal non-interactive CLI, and copies Lite's native evidence into the
Harbor agent log directory.  Harbor remains responsible for task isolation and
verification; this module does not alter the benchmark task or verifier.
"""

from __future__ import annotations

import hashlib
import json
import shlex
import tomllib
from pathlib import Path
from typing import override

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.models.agent.context import AgentContext

from lite.config import load_project_env, resolve_provider_config
from lite.evaluation.context_cost import _usage_from_trace
from lite.evaluation.harnessbench import build_adapter_metadata


ROOT = Path(__file__).resolve().parents[2]
CONTAINER_SOURCE = "/opt/lite-source"
CONTAINER_RUNTIME = "/opt/lite-runtime"
CONTAINER_AGENT_LOGS = "/logs/agent"


class LiteHarborAgent(BaseAgent):
    """Run the project-local Lite CLI as a Harbor custom agent."""

    SUPPORTS_WINDOWS = False

    @staticmethod
    @override
    def name() -> str:
        return "lite"

    @override
    def version(self) -> str:
        with (ROOT / "pyproject.toml").open("rb") as handle:
            return str(tomllib.load(handle)["project"]["version"])

    def _provider_config(self):
        load_project_env(ROOT, override=True)
        config = resolve_provider_config(
            "openai", start=ROOT, config_path=str(ROOT / ".lite.toml")
        )
        if config.model != "gpt-5.5":
            raise RuntimeError(
                f"formal Harbor evaluation requires .lite.toml gpt-5.5, got {config.model}"
            )
        if config.reasoning_effort != "medium":
            raise RuntimeError(
                "formal Harbor evaluation requires reasoning_effort=medium, "
                f"got {config.reasoning_effort or 'unset'}"
            )
        if config.protocol != "openai" or not config.api_key:
            raise RuntimeError(
                "formal Harbor evaluation requires an OpenAI-compatible API key"
            )
        requested_model = (self.model_name or "").split("/", 1)[-1]
        if requested_model and requested_model != config.model:
            raise RuntimeError(
                f"Harbor requested model {self.model_name}, but .lite.toml requires {config.model}"
            )
        return config

    @override
    async def setup(self, environment: BaseEnvironment) -> None:
        self._provider_config()
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        prepare = await environment.exec(
            f"mkdir -p {CONTAINER_SOURCE} {CONTAINER_RUNTIME} {CONTAINER_AGENT_LOGS}",
            timeout_sec=60,
            user="root",
        )
        self._require_success("prepare Lite directories", prepare)
        await environment.upload_dir(ROOT / "lite", f"{CONTAINER_SOURCE}/lite")
        await environment.upload_file(
            ROOT / "pyproject.toml", f"{CONTAINER_SOURCE}/pyproject.toml"
        )
        await environment.upload_file(
            ROOT / ".lite.toml", f"{CONTAINER_SOURCE}/.lite.toml"
        )

        install = await environment.exec(
            _install_command(), timeout_sec=600, user="root"
        )
        (self.logs_dir / "setup.stdout.log").write_text(
            install.stdout or "", encoding="utf-8"
        )
        (self.logs_dir / "setup.stderr.log").write_text(
            install.stderr or "", encoding="utf-8"
        )
        self._require_success("install Lite runtime", install)

    @override
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        config = self._provider_config()
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        session_id = _session_id(self.session_id or str(self.context_id or "harbor"))
        instruction_path = self.logs_dir / "instruction.md"
        instruction_path.write_text(instruction, encoding="utf-8")
        remote_instruction = f"/tmp/{session_id}-instruction.md"
        await environment.upload_file(instruction_path, remote_instruction)

        pwd = await environment.exec("pwd", timeout_sec=30)
        self._require_success("resolve task workdir", pwd)
        workdir = (pwd.stdout or "/app").strip() or "/app"
        command = _run_command(
            workdir=workdir,
            instruction_path=remote_instruction,
            session_id=session_id,
            model=config.model,
            reasoning_effort=config.reasoning_effort,
        )
        result = await environment.exec(
            command,
            cwd=workdir,
            env={
                "PYTHONPATH": CONTAINER_RUNTIME,
                "PYTHONUTF8": "1",
                "PYTHONUNBUFFERED": "1",
                "LITE_PROVIDER": "openai",
                "LITE_API_KEY": config.api_key,
                "LITE_BASE_URL": config.base_url,
                "LITE_MODEL": config.model,
                "LITE_REASONING_EFFORT": config.reasoning_effort,
            },
            timeout_sec=None,
        )
        (self.logs_dir / "lite.stdout.log").write_text(
            result.stdout or "", encoding="utf-8"
        )
        (self.logs_dir / "lite.stderr.log").write_text(
            result.stderr or "", encoding="utf-8"
        )
        evidence = await environment.exec(
            _copy_evidence_command(workdir), timeout_sec=120
        )
        (self.logs_dir / "evidence-copy.stderr.log").write_text(
            evidence.stderr or "", encoding="utf-8"
        )
        (self.logs_dir / "lite-exit.json").write_text(
            json.dumps(
                {
                    "returncode": result.return_code,
                    "session_id": session_id,
                    "model": config.model,
                    "reasoning_effort": config.reasoning_effort,
                    "workdir": workdir,
                    "evidence_copy_returncode": evidence.return_code,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        self._require_success("copy Lite evidence", evidence)
        self._require_success("run Lite agent", result)

    @override
    def populate_context_post_run(self, context: AgentContext) -> None:
        workspace = self.logs_dir / "lite-workspace"
        exit_payload = _read_json(self.logs_dir / "lite-exit.json")
        session_id = str(exit_payload.get("session_id", ""))
        metadata = build_adapter_metadata(
            workspace,
            session_id=session_id,
            returncode=int(exit_payload.get("returncode", 1)),
        )
        (self.logs_dir / "lite-adapter-metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8",
        )

        usage = _aggregate_usage(workspace / ".lite" / "runs")
        if usage["usage_source"] == "actual":
            context.n_input_tokens = usage["input_tokens"]
            context.n_cache_tokens = usage["cached_tokens"]
            context.n_output_tokens = usage["output_tokens"]
        context.metadata = {
            **(context.metadata or {}),
            "lite_returncode": int(exit_payload.get("returncode", 1)),
            "lite_session_id": session_id,
            "lite_model": exit_payload.get("model"),
            "lite_reasoning_effort": exit_payload.get("reasoning_effort"),
            "lite_evidence_copy_returncode": int(
                exit_payload.get("evidence_copy_returncode", 1)
            ),
            "lite_evidence_available": metadata["lite_evidence_available"],
            "lite_evidence_missing": metadata["lite_evidence_missing"],
            "lite_usage": usage,
        }

    @staticmethod
    def _require_success(action: str, result: ExecResult) -> None:
        if result.return_code == 0:
            return
        detail = (result.stderr or result.stdout or "no output").strip()
        raise RuntimeError(f"failed to {action} (exit {result.return_code}): {detail}")


def _session_id(raw: str) -> str:
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f"harbor-{digest}"


def _install_command() -> str:
    return f"""
set -eu
if ! command -v python3 >/dev/null 2>&1; then
  if command -v apt-get >/dev/null 2>&1; then
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      python3 python3-pip ca-certificates
  elif command -v apk >/dev/null 2>&1; then
    apk add --no-cache python3 py3-pip ca-certificates
  elif command -v dnf >/dev/null 2>&1; then
    dnf install -y python3 python3-pip ca-certificates
  else
    echo 'python3 is unavailable and no supported package manager was found' >&2
    exit 127
  fi
fi
if ! python3 -m pip --version >/dev/null 2>&1; then
  python3 -m ensurepip --upgrade >/dev/null 2>&1 || true
fi
python3 -m pip --version
python3 -m pip install --disable-pip-version-check --no-warn-script-location \
  --no-cache-dir --upgrade --target {CONTAINER_RUNTIME} {CONTAINER_SOURCE}
PYTHONPATH={CONTAINER_RUNTIME} python3 -c \
  'from importlib.metadata import version; print(version("lite"))'
""".strip()


def _run_command(
    *,
    workdir: str,
    instruction_path: str,
    session_id: str,
    model: str,
    reasoning_effort: str,
) -> str:
    args = [
        "python3",
        "-m",
        "lite",
        "--cwd",
        workdir,
        "--repo-root",
        workdir,
        "--config",
        f"{CONTAINER_SOURCE}/.lite.toml",
        "--provider",
        "openai",
        "--model",
        model,
        "--reasoning-effort",
        reasoning_effort,
        "--prompt-file",
        instruction_path,
        "--session-id",
        session_id,
        "--approval",
        "auto",
        "--non-interactive",
        "--final-readiness",
        "warn",
        "--no-auto-dream",
        "--max-steps",
        "50",
    ]
    return " ".join(shlex.quote(arg) for arg in args)


def _copy_evidence_command(workdir: str) -> str:
    source = shlex.quote(f"{workdir.rstrip('/')}/.lite")
    target = shlex.quote(f"{CONTAINER_AGENT_LOGS}/lite-workspace")
    return f"test -d {source} && mkdir -p {target} && " f"cp -a {source} {target}/.lite"


def _aggregate_usage(runs_root: Path) -> dict[str, object]:
    totals: dict[str, object] = {
        "input_tokens": 0,
        "cached_tokens": 0,
        "output_tokens": 0,
        "model_call_count": 0,
        "usage_sources": [],
        "usage_source": "none",
    }
    sources: set[str] = set()
    if runs_root.is_dir():
        for trace in sorted(runs_root.glob("*/trace.jsonl")):
            item = _usage_from_trace(trace)
            usage = item["usage"]
            totals["input_tokens"] += int(usage.input_tokens)
            totals["cached_tokens"] += int(usage.cached_tokens)
            totals["output_tokens"] += int(usage.output_tokens)
            totals["model_call_count"] += int(usage.model_call_count)
            sources.add(str(usage.usage_source))
    totals["usage_sources"] = sorted(sources)
    totals["usage_source"] = (
        "actual" if sources == {"actual"} else ("mixed" if sources else "none")
    )
    return totals


def _read_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}
