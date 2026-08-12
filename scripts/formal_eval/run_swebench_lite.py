"""Run fixed SWE-bench Lite subsets with Lite and the official evaluator."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

from lite import Lite, SessionStore, WorkspaceContext
from lite.config import (
    default_max_tokens_for_provider,
    load_project_env,
    resolve_provider_config,
)
from lite.core.run_store import RunStore
from lite.core.task_state import STOP_REASON_FINAL_ANSWER_RETURNED
from lite.evaluation.context_cost import _usage_from_trace
from lite.providers.errors import ProviderError
from lite.providers.runtime import model_client_from_config

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 dependency
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / ".lite.toml"
MANIFEST = ROOT / "benchmarks" / "swebench_lite_subsets.json"
SWEBENCH = ROOT / "benchmarks" / "external" / "swe-bench"
EVALUATOR_MODULE = SWEBENCH / "swebench" / "harness" / "run_evaluation.py"
EVALUATOR_DOCKERFILE = ROOT / "scripts" / "formal_eval" / "swebench-evaluator.Dockerfile"
EVALUATOR_IMAGE_REPOSITORY = "lite-swebench-evaluator"
HARD_SAFETY_EVENTS = {
    "secret_exposure",
    "scope_violation",
    "workspace_escape",
    "unsafe_command_executed",
    "worker_scope_violation",
}
SAFE_INSTANCE_FIELDS = ("instance_id", "repo", "base_commit", "problem_statement")
ALLOWED_TOOLS = (
    "list_files",
    "read_file",
    "write_file",
    "patch_file",
    "search",
    "run_shell",
)


def _json_url(url: str) -> object:
    request = Request(url, headers={"User-Agent": "lite-evaluation/2"})
    with urlopen(request, timeout=60) as response:  # noqa: S310 - fixed HTTPS hosts
        return json.load(response)


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _run(
    command: list[str],
    *,
    cwd: Path = ROOT,
    timeout: int = 60,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )


def _temperature() -> float | None:
    with CONFIG.open("rb") as handle:
        payload = tomllib.load(handle)
    provider = str(payload.get("provider") or "openai")
    profile = payload.get("providers", {}).get(provider, {})
    value = profile.get("temperature") if isinstance(profile, dict) else None
    if value is None:
        return None
    if not isinstance(value, (int, float)):
        raise ValueError(f"provider {provider} temperature must be numeric")
    return float(value)


def _provider_config():
    load_project_env(ROOT, override=True)
    return resolve_provider_config(None, start=ROOT, config_path=str(CONFIG))


def _model_identity(config, temperature: float | None) -> dict[str, object]:
    return {
        "source": str(CONFIG),
        "provider": config.name,
        "protocol": config.protocol,
        "model": config.model,
        "reasoning_effort": config.reasoning_effort,
        "strict_tools": config.strict_tools,
        "temperature": temperature,
        "base_url_hostname": urlparse(config.base_url).hostname,
        "api_key_present": bool(config.api_key),
    }


def _model_client(config, temperature: float | None):
    args = SimpleNamespace(temperature=temperature, openai_timeout=300)
    return model_client_from_config(config, args, timeout=300)


def _probe_provider(config, temperature: float | None) -> dict[str, object]:
    started = time.monotonic()
    try:
        client = _model_client(config, temperature)
        client.complete("Reply with exactly OK.", 16)
        metadata = dict(getattr(client, "last_completion_metadata", {}) or {})
        usage_present = all(
            metadata.get(name) is not None
            for name in ("input_tokens", "output_tokens")
        )
        return {
            "available": True,
            "usage_present": usage_present,
            "wall_time_ms": int((time.monotonic() - started) * 1000),
            "error": "",
        }
    except Exception as exc:
        return {
            "available": False,
            "usage_present": False,
            "wall_time_ms": int((time.monotonic() - started) * 1000),
            "error": f"{type(exc).__name__}: {exc}",
        }


def _disk(path: Path) -> dict[str, object]:
    usage = shutil.disk_usage(path)
    return {
        "path": str(path),
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
    }


def _temp_check() -> dict[str, object]:
    temp_root = Path(tempfile.gettempdir()).resolve()
    result: dict[str, object] = {
        "path": str(temp_root),
        "exists": temp_root.is_dir(),
        "writable": False,
    }
    try:
        with tempfile.NamedTemporaryFile(dir=temp_root) as handle:
            handle.write(b"lite-preflight")
            handle.flush()
        result["writable"] = True
    except OSError as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["disk"] = _disk(temp_root)
    return result


def _remote_dataset_revision(dataset: str) -> tuple[str, str]:
    try:
        payload = _json_url(f"https://huggingface.co/api/datasets/{dataset}")
        revision = payload.get("sha") if isinstance(payload, dict) else None
        if not isinstance(revision, str):
            raise RuntimeError("dataset metadata did not include sha")
        return revision, ""
    except Exception as exc:
        return "", f"{type(exc).__name__}: {exc}"


def _evaluator_image_tag() -> str:
    revision = _git_commit(SWEBENCH) or _sha256(EVALUATOR_MODULE).split(":", 1)[1]
    dockerfile_digest = (
        _sha256(EVALUATOR_DOCKERFILE).split(":", 1)[1]
        if EVALUATOR_DOCKERFILE.is_file()
        else "missing"
    )
    image_digest = hashlib.sha256(
        f"{revision}:{dockerfile_digest}".encode()
    ).hexdigest()[:12]
    return f"{EVALUATOR_IMAGE_REPOSITORY}:{image_digest}"


def _ensure_evaluator_image() -> dict[str, object]:
    image = _evaluator_image_tag()
    inspect = _run(["docker", "image", "inspect", image], timeout=60)
    if inspect.returncode == 0:
        return {
            "available": True,
            "image": image,
            "built": False,
            "detail": "",
        }
    if not EVALUATOR_DOCKERFILE.is_file():
        return {
            "available": False,
            "image": image,
            "built": False,
            "detail": f"missing evaluator Dockerfile: {EVALUATOR_DOCKERFILE}",
        }
    build = _run(
        [
            "docker",
            "build",
            "--pull",
            "-f",
            str(EVALUATOR_DOCKERFILE),
            "-t",
            image,
            str(SWEBENCH),
        ],
        timeout=30 * 60,
    )
    return {
        "available": build.returncode == 0,
        "image": image,
        "built": build.returncode == 0,
        "detail": (build.stderr or build.stdout)[-4000:],
    }


def _evaluator_dependencies(*, docker_daemon: bool) -> dict[str, object]:
    if platform.system() == "Windows":
        if not docker_daemon:
            return {
                "available": False,
                "mode": "linux_container",
                "python": None,
                "detail": "Docker daemon is required for the Linux evaluator container.",
            }
        image = _ensure_evaluator_image()
        if not image["available"]:
            return {
                "available": False,
                "mode": "linux_container",
                "python": None,
                **image,
            }
        completed = _run(
            [
                "docker",
                "run",
                "--rm",
                "--entrypoint",
                "python",
                str(image["image"]),
                "-c",
                "import datasets, docker, resource, swebench; print('ok')",
            ],
            timeout=120,
        )
        return {
            "available": completed.returncode == 0,
            "mode": "linux_container",
            "python": f"docker:{image['image']}",
            "image": image["image"],
            "image_built": image["built"],
            "detail": (completed.stderr or completed.stdout)[-1000:],
        }
    env = {**os.environ, "PYTHONPATH": str(SWEBENCH)}
    completed = _run(
        [
            sys.executable,
            "-c",
            "import datasets, docker, swebench; print('ok')",
        ],
        timeout=60,
        env=env,
    )
    return {
        "available": completed.returncode == 0,
        "mode": "host_python",
        "python": sys.executable,
        "detail": (completed.stderr or completed.stdout)[-1000:],
    }


def preflight(*, probe_provider: bool = True) -> dict[str, object]:
    checks: dict[str, bool] = {
        "config": CONFIG.is_file(),
        "manifest": MANIFEST.is_file(),
        "evaluator_source": EVALUATOR_MODULE.is_file(),
        "docker_cli": bool(shutil.which("docker")),
        "docker_daemon": False,
        "temp_directory": False,
        "disk_minimum": False,
        "dataset_revision": False,
        "provider_config": False,
        "provider_available": False,
        "evaluator_dependencies": False,
    }
    errors: list[dict[str, str]] = []
    model_identity: dict[str, object] = {
        "source": str(CONFIG),
        "api_key_present": False,
    }
    provider_probe: dict[str, object] = {
        "available": False,
        "skipped": not probe_provider,
    }
    manifest_payload: dict[str, object] = {}
    if checks["manifest"]:
        try:
            manifest_payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append({"category": "harness_error", "detail": str(exc)})
    if checks["config"]:
        try:
            config = _provider_config()
            temperature = _temperature()
            model_identity = _model_identity(config, temperature)
            checks["provider_config"] = bool(
                config.model
                and config.api_key
                and config.protocol in {"openai", "anthropic"}
            )
            if probe_provider and checks["provider_config"]:
                provider_probe = _probe_provider(config, temperature)
            checks["provider_available"] = bool(provider_probe.get("available"))
            if not probe_provider:
                checks["provider_available"] = checks["provider_config"]
        except Exception as exc:
            errors.append({"category": "provider_error", "detail": str(exc)})
    if checks["docker_cli"]:
        try:
            completed = _run(["docker", "info"], timeout=30)
            checks["docker_daemon"] = completed.returncode == 0
            if completed.returncode:
                errors.append(
                    {
                        "category": "docker_error",
                        "detail": (completed.stderr or completed.stdout)[-1000:],
                    }
                )
        except (OSError, subprocess.SubprocessError) as exc:
            errors.append({"category": "docker_error", "detail": str(exc)})
    temp = _temp_check()
    checks["temp_directory"] = bool(temp["exists"] and temp["writable"])
    workspace_disk = _disk(ROOT)
    checks["disk_minimum"] = bool(
        workspace_disk["free_bytes"] >= 20 * 1024**3
        and temp["disk"]["free_bytes"] >= 10 * 1024**3
    )
    evaluator_dependencies = _evaluator_dependencies(
        docker_daemon=checks["docker_daemon"]
    )
    checks["evaluator_dependencies"] = bool(evaluator_dependencies["available"])
    dataset = str(manifest_payload.get("dataset") or "")
    pinned_revision = str(manifest_payload.get("dataset_revision") or "")
    remote_revision, dataset_error = _remote_dataset_revision(dataset) if dataset else ("", "missing dataset")
    checks["dataset_revision"] = bool(
        pinned_revision and remote_revision == pinned_revision
    )
    if dataset_error:
        errors.append({"category": "dataset_error", "detail": dataset_error})
    blocked_reasons = [name for name, passed in checks.items() if not passed]
    return {
        "schema_version": "lite.eval.preflight.v2",
        "status": "ready" if all(checks.values()) else "blocked",
        "ready": all(checks.values()),
        "checks": checks,
        "blocked_reasons": blocked_reasons,
        "model": model_identity,
        "provider_probe": provider_probe,
        "dataset": {
            "name": dataset,
            "manifest": str(MANIFEST),
            "manifest_sha256": _sha256(MANIFEST) if MANIFEST.is_file() else None,
            "pinned_revision": pinned_revision,
            "remote_revision": remote_revision,
        },
        "evaluator": {
            "source": str(EVALUATOR_MODULE),
            "dockerfile": str(EVALUATOR_DOCKERFILE),
            "checkout_commit": _git_commit(SWEBENCH),
            "dependencies": evaluator_dependencies,
        },
        "docker": {
            "cli": shutil.which("docker"),
            "daemon_available": checks["docker_daemon"],
        },
        "storage": {"workspace": workspace_disk, "temp": temp},
        "errors": errors,
    }


def _git_commit(repo: Path) -> str | None:
    if not (repo / ".git").exists():
        return None
    completed = _run(["git", "rev-parse", "HEAD"], cwd=repo)
    return completed.stdout.strip() if completed.returncode == 0 else None


def _safe_instances(split: str, expected_count: int) -> dict[str, dict[str, str]]:
    instances: dict[str, dict[str, str]] = {}
    for offset in range(0, expected_count, 100):
        query = urlencode(
            {
                "dataset": "SWE-bench/SWE-bench_Lite",
                "config": "default",
                "split": split,
                "offset": offset,
                "length": min(100, expected_count - offset),
            }
        )
        payload = _json_url(f"https://datasets-server.huggingface.co/rows?{query}")
        rows = payload.get("rows", []) if isinstance(payload, dict) else []
        for item in rows:
            row = item.get("row", {}) if isinstance(item, dict) else {}
            if not isinstance(row, dict):
                continue
            safe = {name: row.get(name) for name in SAFE_INSTANCE_FIELDS}
            if not all(isinstance(value, str) and value for value in safe.values()):
                raise RuntimeError("SWE-bench Lite row is missing required safe metadata")
            instances[safe["instance_id"]] = safe
    if len(instances) != expected_count:
        raise RuntimeError(f"expected {expected_count} safe instances, got {len(instances)}")
    return instances


def _fresh_workspace(instance: dict[str, str], workspace: Path) -> None:
    if workspace.exists():
        raise RuntimeError(f"workspace already exists: {workspace}")
    workspace.parent.mkdir(parents=True, exist_ok=True)
    clone_errors: list[str] = []
    for attempt in range(3):
        if attempt and workspace.exists():
            shutil.rmtree(workspace)
        clone = _run(
            [
                "git",
                "clone",
                "--no-tags",
                f"https://github.com/{instance['repo']}.git",
                str(workspace),
            ],
            timeout=900,
        )
        if clone.returncode == 0:
            break
        clone_errors.append((clone.stderr or clone.stdout)[-2000:])
        if attempt < 2:
            time.sleep(2**attempt)
    else:
        raise RuntimeError(f"git clone failed after 3 attempts: {clone_errors[-1]}")
    switch = _run(
        ["git", "-c", "advice.detachedHead=false", "switch", "--detach", instance["base_commit"]],
        cwd=workspace,
        timeout=120,
    )
    if switch.returncode:
        fetch = _run(
            ["git", "fetch", "--no-tags", "origin", instance["base_commit"]],
            cwd=workspace,
            timeout=300,
        )
        if fetch.returncode:
            raise RuntimeError(f"git fetch failed: {(fetch.stderr or fetch.stdout)[-2000:]}")
        switch = _run(
            ["git", "-c", "advice.detachedHead=false", "switch", "--detach", "FETCH_HEAD"],
            cwd=workspace,
            timeout=120,
        )
        if switch.returncode:
            raise RuntimeError(f"git switch failed: {(switch.stderr or switch.stdout)[-2000:]}")
    exclude = workspace / ".git" / "info" / "exclude"
    with exclude.open("a", encoding="utf-8") as handle:
        handle.write("\n.lite/\n")
    status = _run(["git", "status", "--porcelain"], cwd=workspace)
    if status.returncode or status.stdout.strip():
        raise RuntimeError(f"new workspace is not clean: {status.stdout[-2000:]}")


def _trace_events(workspace: Path) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    for path in sorted((workspace / ".lite" / "runs").glob("*/trace.jsonl")):
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                events.append(event)
    return events


def _usage(workspace: Path) -> dict[str, object]:
    totals: dict[str, object] = {
        "input_tokens": 0,
        "cached_tokens": 0,
        "output_tokens": 0,
        "model_call_count": 0,
        "usage_sources": [],
        "usage_source": "none",
    }
    sources: set[str] = set()
    for path in sorted((workspace / ".lite" / "runs").glob("*/trace.jsonl")):
        item = _usage_from_trace(path)
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


def _extract_patch(workspace: Path) -> tuple[str, list[str]]:
    intent = _run(["git", "add", "-N", "--", "."], cwd=workspace, timeout=120)
    if intent.returncode:
        raise RuntimeError(f"git add -N failed: {(intent.stderr or intent.stdout)[-2000:]}")
    diff = _run(
        ["git", "-c", "core.fileMode=false", "diff", "--binary", "--no-ext-diff"],
        cwd=workspace,
        timeout=120,
    )
    names = _run(["git", "diff", "--name-only"], cwd=workspace, timeout=120)
    if diff.returncode or names.returncode:
        raise RuntimeError("failed to extract git diff")
    changed = [line.strip().replace("\\", "/") for line in names.stdout.splitlines() if line.strip()]
    return diff.stdout, changed


def _failure_category(error: str, *, timed_out: bool, safety_pass: bool) -> str:
    if timed_out:
        return "timeout"
    if not safety_pass:
        return "workspace_scope_violation"
    if not error:
        return "pending_verifier"
    lowered = error.lower()
    if "provider" in lowered or "retryexhausted" in lowered:
        return "provider_error"
    if "tool" in lowered and "argument" in lowered:
        return "invalid_tool_args"
    return "model_failure"


def _run_agent(
    instance: dict[str, str],
    *,
    workspace: Path,
    task_dir: Path,
    config,
    temperature: float | None,
    wall_timeout: int,
) -> dict[str, object]:
    started = time.monotonic()
    _fresh_workspace(instance, workspace)
    task_dir.mkdir(parents=True, exist_ok=False)
    client = _model_client(config, temperature)
    agent = Lite(
        model_client=client,
        workspace=WorkspaceContext.build(workspace, repo_root_override=workspace),
        session_store=SessionStore(workspace / ".lite" / "sessions"),
        run_store=RunStore(workspace / ".lite" / "runs"),
        approval_policy="auto",
        max_steps=50,
        max_new_tokens=default_max_tokens_for_provider(config.name),
        allowed_tools=list(ALLOWED_TOOLS),
        feature_flags={"context_reduction": True},
        final_readiness_mode="verify",
    )
    prompt = (
        "Fix the following repository issue. Work only inside this repository, do not "
        "modify tests or benchmark files, run relevant tests, and finish with a concise "
        "final answer. You are not given any reference patch or hidden verifier details.\n\n"
        + instance["problem_statement"]
    )
    (task_dir / "issue.md").write_text(prompt, encoding="utf-8")
    error = ""
    timed_out = False
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(agent.ask, prompt)
    try:
        future.result(timeout=wall_timeout)
    except concurrent.futures.TimeoutError:
        timed_out = True
        error = f"wall_timeout:{wall_timeout}s"
        agent.abort_current_turn()
        executor.shutdown(wait=False, cancel_futures=True)
    except ProviderError as exc:
        error = f"ProviderError: {exc}"
        executor.shutdown(wait=False, cancel_futures=True)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        executor.shutdown(wait=False, cancel_futures=True)
    else:
        executor.shutdown(wait=True)
    patch, changed_paths = _extract_patch(workspace)
    (task_dir / "prediction.patch").write_text(patch, encoding="utf-8")
    events = _trace_events(workspace)
    event_names = [str(event.get("event", "")) for event in events]
    safety_pass = not any(name in HARD_SAFETY_EVENTS for name in event_names)
    tests_modified = any(
        path.startswith("tests/") or "/tests/" in path for path in changed_paths
    )
    scope_pass = bool(changed_paths) and not tests_modified and safety_pass
    task_state = getattr(agent, "current_task_state", None)
    finalization_pass = bool(
        task_state and task_state.stop_reason == STOP_REASON_FINAL_ANSWER_RETURNED
    )
    row: dict[str, object] = {
        "instance_id": instance["instance_id"],
        "status": "generated" if patch and not error else "failed",
        "resolved": None,
        "patch_apply": None,
        "target_pass": None,
        "regression_pass": None,
        "scope_pass": scope_pass,
        "finalization_pass": finalization_pass,
        "safety_pass": safety_pass,
        "scc": False,
        "changed_paths": changed_paths,
        "tests_modified": tests_modified,
        "failure_category": _failure_category(
            error, timed_out=timed_out, safety_pass=safety_pass
        ),
        "error": error,
        "usage": _usage(workspace),
        "tool_steps": int(getattr(task_state, "tool_steps", 0) or 0),
        "attempts": int(getattr(task_state, "attempts", 0) or 0),
        "wall_time_ms": int((time.monotonic() - started) * 1000),
        "workspace": str(workspace),
        "trace_paths": [
            str(path)
            for path in sorted((workspace / ".lite" / "runs").glob("*/trace.jsonl"))
        ],
    }
    (task_dir / "agent-report.json").write_text(
        json.dumps(row, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )
    return row


def _write_predictions(
    output: Path, rows: list[dict[str, object]], config
) -> Path:
    path = output / "prediction.jsonl"
    model_name = f"lite-{config.name}/{config.model}"
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            patch_path = output / "tasks" / str(row["instance_id"]) / "prediction.patch"
            patch = patch_path.read_text(encoding="utf-8") if patch_path.is_file() else ""
            handle.write(
                json.dumps(
                    {
                        "instance_id": row["instance_id"],
                        "model_name_or_path": model_name,
                        "model_patch": patch,
                    },
                    ensure_ascii=True,
                )
                + "\n"
            )
    return path


def _load_agent_rows(
    output: Path, task_records: list[dict[str, object]]
) -> list[dict[str, object]]:
    rows = []
    for record in task_records:
        instance_id = str(record["instance_id"])
        report_path = output / "tasks" / instance_id / "agent-report.json"
        if not report_path.is_file():
            raise RuntimeError(f"missing agent report for verifier retry: {instance_id}")
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("instance_id") != instance_id:
            raise RuntimeError(f"invalid agent report for verifier retry: {instance_id}")
        rows.append(payload)
    return rows


def _load_existing_agent_row(output: Path, instance_id: str) -> dict[str, object] | None:
    report_path = output / "tasks" / instance_id / "agent-report.json"
    if not report_path.is_file():
        return None
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("instance_id") != instance_id:
        raise RuntimeError(f"invalid agent report for resume: {instance_id}")
    patch_path = output / "tasks" / instance_id / "prediction.patch"
    if not patch_path.is_file():
        raise RuntimeError(f"missing prediction patch for resume: {instance_id}")
    return payload


def _prepare_incomplete_resume(output: Path, instance_id: str) -> None:
    for path in (
        output / "workspaces" / instance_id,
        output / "tasks" / instance_id,
    ):
        if path.exists():
            shutil.rmtree(path)


def _evaluation_identity(
    *, subset_name: str, task_records: list[dict[str, object]], model: dict[str, object]
) -> dict[str, object]:
    return {
        "schema_version": "lite.swebench_lite.evaluation_identity.v1",
        "subset": subset_name,
        "task_ids": [str(record["instance_id"]) for record in task_records],
        "manifest_sha256": _sha256(MANIFEST),
        "model": model,
    }


def _ensure_evaluation_identity(
    output: Path, expected: dict[str, object], *, resume: bool
) -> None:
    path = output / "evaluation-manifest.json"
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != expected:
            raise RuntimeError(
                "SWE-bench evaluation identity changed; use a fresh output directory"
            )
        return
    has_existing_rows = any((output / "tasks").glob("*/agent-report.json"))
    if has_existing_rows and not resume:
        raise RuntimeError(
            "existing SWE-bench rows require --resume or a fresh output directory"
        )
    path.write_text(
        json.dumps(expected, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )


def _run_evaluator(
    *,
    output: Path,
    prediction: Path,
    dataset: str,
    split: str,
    instance_ids: list[str],
    run_id: str,
) -> subprocess.CompletedProcess[str]:
    verifier = output / "verifier"
    verifier.mkdir(parents=True, exist_ok=True)
    if platform.system() == "Windows":
        image = _ensure_evaluator_image()
        if not image["available"]:
            raise RuntimeError(f"Linux evaluator image unavailable: {image['detail']}")
        command = [
            "docker",
            "run",
            "--rm",
            "--name",
            f"lite-swebench-{run_id}",
            "-v",
            "/var/run/docker.sock:/var/run/docker.sock",
            "-v",
            f"{output}:/evaluation",
            "-v",
            "lite-swebench-hf-cache:/root/.cache/huggingface",
            "-w",
            "/evaluation/verifier",
            str(image["image"]),
            "--dataset_name",
            dataset,
            "--split",
            split,
            "--predictions_path",
            "/evaluation/prediction.jsonl",
            "--run_id",
            run_id,
            "--max_workers",
            "1",
            "--report_dir",
            "/evaluation",
            "--instance_ids",
            *instance_ids,
        ]
        completed = _run(command, timeout=6 * 60 * 60)
    else:
        command = [
            sys.executable,
            "-m",
            "swebench.harness.run_evaluation",
            "--dataset_name",
            dataset,
            "--split",
            split,
            "--predictions_path",
            str(prediction),
            "--run_id",
            run_id,
            "--max_workers",
            "1",
            "--report_dir",
            str(output),
            "--instance_ids",
            *instance_ids,
        ]
        env = {**os.environ, "PYTHONPATH": str(SWEBENCH)}
        completed = _run(command, cwd=verifier, timeout=6 * 60 * 60, env=env)
    (output / "verifier.stdout.log").write_text(
        completed.stdout, encoding="utf-8"
    )
    (output / "verifier.stderr.log").write_text(
        completed.stderr, encoding="utf-8"
    )
    (output / "verifier-command.json").write_text(
        json.dumps(
            {"command": command, "returncode": completed.returncode}, indent=2
        )
        + "\n",
        encoding="utf-8",
    )
    return completed


def _test_group_pass(report: dict[str, object], group: str) -> bool | None:
    tests_status = report.get("tests_status")
    if not isinstance(tests_status, dict):
        return None
    group_status = tests_status.get(group)
    if not isinstance(group_status, dict):
        return None
    failed = group_status.get("failure", [])
    # Match the official grading helpers: an empty test group is vacuously passing.
    return not failed


def _merge_verifier(
    output: Path,
    rows: list[dict[str, object]],
    *,
    run_id: str,
    config,
    evaluator_returncode: int,
) -> None:
    model_slug = f"lite-{config.name}__{config.model}".replace("/", "__")
    for row in rows:
        instance_id = str(row["instance_id"])
        log_dir = (
            output
            / "verifier"
            / "logs"
            / "run_evaluation"
            / run_id
            / model_slug
            / instance_id
        )
        report_path = log_dir / "report.json"
        instance_log = log_dir / "run_instance.log"
        report_payload: dict[str, object] = {}
        if report_path.is_file():
            raw = json.loads(report_path.read_text(encoding="utf-8"))
            candidate = raw.get(instance_id, raw) if isinstance(raw, dict) else {}
            report_payload = candidate if isinstance(candidate, dict) else {}
        log_text = (
            instance_log.read_text(encoding="utf-8", errors="replace")
            if instance_log.is_file()
            else ""
        )
        row["patch_apply"] = ">>>>> Applied Patch" in log_text
        row["resolved"] = bool(report_payload.get("resolved")) if report_payload else False
        row["target_pass"] = _test_group_pass(report_payload, "FAIL_TO_PASS")
        row["regression_pass"] = _test_group_pass(report_payload, "PASS_TO_PASS")
        if row["resolved"]:
            row["target_pass"] = True if row["target_pass"] is None else row["target_pass"]
            row["regression_pass"] = True if row["regression_pass"] is None else row["regression_pass"]
        row["scc"] = bool(
            row["resolved"]
            and row["patch_apply"]
            and row["target_pass"]
            and row["regression_pass"]
            and row["scope_pass"]
            and row["finalization_pass"]
            and row["safety_pass"]
            and not row["error"]
        )
        if row["scc"]:
            row["status"] = "resolved"
            row["failure_category"] = "none"
        elif evaluator_returncode != 0 and not report_payload:
            row["status"] = "error"
            row["failure_category"] = "docker_error"
        elif not row["patch_apply"]:
            row["status"] = "failed"
            row["failure_category"] = "patch_apply_failed"
        elif not row["target_pass"]:
            row["status"] = "failed"
            row["failure_category"] = "target_verifier_failed"
        elif not row["regression_pass"]:
            row["status"] = "failed"
            row["failure_category"] = "regression_failed"
        elif not row["finalization_pass"]:
            row["status"] = "failed"
            row["failure_category"] = "finalization_missing"
        (output / "tasks" / instance_id / "report.json").write_text(
            json.dumps(
                {"runtime": row, "official_verifier": report_payload},
                ensure_ascii=True,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


def _aggregate(output: Path, rows: list[dict[str, object]], identity: dict[str, object]) -> None:
    denominator = len(rows)
    metrics = {}
    for key in (
        "resolved",
        "patch_apply",
        "target_pass",
        "regression_pass",
        "scope_pass",
        "finalization_pass",
        "safety_pass",
        "scc",
    ):
        passed = sum(value is True for value in (row.get(key) for row in rows))
        metrics[key] = {
            "passed": passed,
            "total": denominator,
            "rate": passed / denominator if denominator else None,
        }
    payload = {
        "schema_version": "lite.swebench_lite.report.v1",
        "status": "completed",
        "model": identity,
        "task_count": denominator,
        "metrics": metrics,
        "failure_by_category": {},
        "actual_usage_task_count": sum(
            row["usage"]["usage_source"] == "actual" for row in rows
        ),
        "rows": rows,
    }
    for row in rows:
        category = str(row["failure_category"])
        payload["failure_by_category"][category] = (
            payload["failure_by_category"].get(category, 0) + 1
        )
    (output / "report.json").write_text(
        json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )
    with (output / "trace.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            for trace_path in row["trace_paths"]:
                for line in Path(trace_path).read_text(
                    encoding="utf-8", errors="replace"
                ).splitlines():
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    event["swebench_instance_id"] = row["instance_id"]
                    handle.write(json.dumps(event, ensure_ascii=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", choices=("dev", "smoke-5", "calibration-50"), default="smoke-5")
    parser.add_argument("--output-dir", default="artifacts/eval-v2/swebench-lite")
    parser.add_argument("--preflight-output", default="artifacts/eval-v2/preflight.json")
    parser.add_argument("--max-tasks", type=int, default=0)
    parser.add_argument("--wall-timeout", type=int, default=900)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--verifier-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-provider-probe", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    preflight_path = Path(args.preflight_output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    preflight_path.parent.mkdir(parents=True, exist_ok=True)
    readiness = preflight(probe_provider=not args.skip_provider_probe)
    preflight_path.write_text(
        json.dumps(readiness, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )
    if args.preflight_only or not readiness["ready"]:
        print(
            json.dumps(
                {
                    "status": readiness["status"],
                    "preflight": str(preflight_path),
                    "blocked_reasons": readiness["blocked_reasons"],
                }
            )
        )
        return 0 if readiness["ready"] else 2

    config = _provider_config()
    temperature = _temperature()
    identity = _model_identity(config, temperature)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    subset = manifest["subsets"][args.subset]
    task_records = list(subset["tasks"])
    if args.max_tasks > 0:
        task_records = task_records[: args.max_tasks]
    _ensure_evaluation_identity(
        output,
        _evaluation_identity(
            subset_name=args.subset, task_records=task_records, model=identity
        ),
        resume=args.resume or args.verifier_only,
    )
    if args.verifier_only:
        rows = _load_agent_rows(output, task_records)
        prediction = output / "prediction.jsonl"
        if not prediction.is_file():
            raise RuntimeError("verifier retry requires an existing prediction.jsonl")
    else:
        safe_instances = _safe_instances(subset["split"], subset["source_task_count"])
        rows = []
        for record in task_records:
            instance_id = record["instance_id"]
            existing = (
                _load_existing_agent_row(output, instance_id) if args.resume else None
            )
            if existing is not None:
                rows.append(existing)
                continue
            if args.resume:
                _prepare_incomplete_resume(output, instance_id)
            rows.append(
                _run_agent(
                    safe_instances[instance_id],
                    workspace=output / "workspaces" / instance_id,
                    task_dir=output / "tasks" / instance_id,
                    config=config,
                    temperature=temperature,
                    wall_timeout=args.wall_timeout,
                )
            )
            (output / "agent-rows.partial.json").write_text(
                json.dumps(rows, ensure_ascii=True, indent=2) + "\n",
                encoding="utf-8",
            )
        prediction = _write_predictions(output, rows, config)
    run_id = f"lite-{args.subset}-{int(time.time())}"
    completed = _run_evaluator(
        output=output,
        prediction=prediction,
        dataset=manifest["dataset"],
        split=subset["split"],
        instance_ids=[str(row["instance_id"]) for row in rows],
        run_id=run_id,
    )
    _merge_verifier(
        output,
        rows,
        run_id=run_id,
        config=config,
        evaluator_returncode=completed.returncode,
    )
    _aggregate(output, rows, identity)
    print(
        json.dumps(
            {
                "status": "completed",
                "tasks": len(rows),
                "report": str(output / "report.json"),
                "verifier_returncode": completed.returncode,
            }
        )
    )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
