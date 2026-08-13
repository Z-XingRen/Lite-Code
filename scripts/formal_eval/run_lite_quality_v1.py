from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlparse

from lite import Lite, SessionStore, WorkspaceContext
from lite.config import (
    default_max_tokens_for_provider,
    load_project_env,
    resolve_provider_config,
)
from lite.core.run_store import RunStore
from lite.core.task_state import STOP_REASON_FINAL_ANSWER_RETURNED
from lite.evaluation.context_cost import _usage_from_trace
from lite.providers.runtime import model_client_from_config

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 dependency
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "benchmarks/formal_v1/manifest.json"
GRADER = ROOT / "benchmarks/formal_v1/graders/grade.py"
ARTIFACT_ROOT = ROOT / "artifacts/formal-evaluation-20260806/quality"
IGNORED_PARTS = {".lite", ".git", "__pycache__", ".pytest_cache"}
HARD_SAFETY_EVENTS = {
    "secret_exposure",
    "scope_violation",
    "workspace_escape",
    "unsafe_command_executed",
    "worker_scope_violation",
}


def configured_temperature(config):
    with (ROOT / ".lite.toml").open("rb") as handle:
        payload = tomllib.load(handle)
    profile = payload.get("providers", {}).get(config.name, {})
    return profile.get("temperature") if isinstance(profile, dict) else None


def digest_tree(path: Path):
    h = hashlib.sha256()
    for item in sorted(path.rglob("*")):
        if not item.is_file() or any(
            part in IGNORED_PARTS for part in item.relative_to(path).parts
        ):
            continue
        rel = item.relative_to(path).as_posix()
        h.update(rel.encode())
        h.update(b"\0")
        h.update(hashlib.sha256(item.read_bytes()).digest())
    return "sha256:" + h.hexdigest()


def file_snapshot(path: Path):
    result = {}
    for item in path.rglob("*"):
        if not item.is_file() or any(
            part in IGNORED_PARTS for part in item.relative_to(path).parts
        ):
            continue
        result[item.relative_to(path).as_posix()] = hashlib.sha256(
            item.read_bytes()
        ).hexdigest()
    return result


def load_manifest():
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if (
        payload.get("benchmark_id") != "lite-coding-quality-v1"
        or len(payload.get("tasks", [])) != 24
    ):
        raise RuntimeError("formal manifest must contain exactly 24 tasks")
    return payload


def provider_metadata():
    load_project_env(ROOT, override=True)
    config = resolve_provider_config(
        None, start=ROOT, config_path=ROOT / ".lite.toml"
    )
    if config.protocol not in {"openai", "anthropic"} or not config.api_key:
        raise RuntimeError(
            "formal benchmark requires a supported provider with API key resolved "
            "from .lite.toml"
        )
    return config


def select_tasks(manifest, task_ids="", limit=0):
    tasks = list(manifest["tasks"])
    requested = [item.strip() for item in str(task_ids).split(",") if item.strip()]
    if requested:
        by_id = {task["id"]: task for task in tasks}
        missing = [task_id for task_id in requested if task_id not in by_id]
        if missing:
            raise ValueError(f"unknown formal task ids: {', '.join(missing)}")
        tasks = [by_id[task_id] for task_id in requested]
    if int(limit) > 0:
        tasks = tasks[: int(limit)]
    if not tasks:
        raise ValueError("formal task selection is empty")
    return {**manifest, "tasks": tasks}


def evaluation_allowed_tools(task):
    """Expose verification without an unrestricted shell in fixed evaluations."""

    tools = [str(name) for name in task.get("allowed_tools", [])]
    if "run_shell" in tools:
        tools = [name for name in tools if name != "run_shell"]
        tools.append("verify")
    return list(dict.fromkeys(tools))


def make_client(config):
    temperature = configured_temperature(config)
    return model_client_from_config(
        config,
        SimpleNamespace(temperature=temperature, openai_timeout=300),
        timeout=300,
    )


def fresh_workspace(task, out_dir: Path):
    source = ROOT / task["fixture_repo"]
    target = out_dir / task["id"]
    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        source,
        target,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache", ".lite"),
    )
    return target


def grade_task(task_id, workspace):
    try:
        result = subprocess.run(
            [
                sys.executable,
                str(GRADER),
                "--task",
                task_id,
                "--workspace",
                str(workspace),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=90,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "target_pass": False,
            "regression_pass": False,
            "grader_returncode": None,
            "grader_stdout_tail": "",
            "grader_stderr_tail": "",
            "grader_infrastructure_error": True,
            "grader_error": f"{type(exc).__name__}: {exc}",
        }
    payload = {}
    parsed_payload = False
    for line in reversed(result.stdout.splitlines()):
        try:
            payload = json.loads(line)
            parsed_payload = True
            break
        except json.JSONDecodeError:
            continue
    payload.setdefault("target_pass", False)
    payload.setdefault("regression_pass", False)
    payload["grader_returncode"] = result.returncode
    payload["grader_stdout_tail"] = result.stdout[-1000:]
    payload["grader_stderr_tail"] = result.stderr[-1000:]
    payload["grader_infrastructure_error"] = bool(
        not parsed_payload or result.returncode not in {0, 1}
    )
    return payload


def trace_events(workspace):
    events = []
    evidence_paths = [
        ("run", path)
        for path in sorted((workspace / ".lite" / "runs").glob("*/trace.jsonl"))
    ]
    evidence_paths.extend(
        ("session", path)
        for path in sorted(
            (workspace / ".lite" / "sessions").glob("*.events.jsonl")
        )
    )
    for source, path in evidence_paths:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                pass
            else:
                event.setdefault("evidence_source", source)
                events.append(event)
    return events


def usage_from_workspace(workspace):
    total = {
        "input_tokens": 0,
        "cached_tokens": 0,
        "output_tokens": 0,
        "estimated_input_tokens": 0,
        "model_call_count": 0,
        "usage_sources": set(),
    }
    traces = sorted((workspace / ".lite" / "runs").glob("*/trace.jsonl"))
    for path in traces:
        item = _usage_from_trace(path)
        u = item["usage"]
        total["input_tokens"] += int(u.input_tokens)
        total["cached_tokens"] += int(u.cached_tokens)
        total["output_tokens"] += int(u.output_tokens)
        total["estimated_input_tokens"] += int(item["estimated_input_tokens"])
        total["model_call_count"] += int(u.model_call_count)
        total["usage_sources"].add(u.usage_source)
    total["usage_sources"] = sorted(total["usage_sources"])
    total["usage_source"] = (
        "actual"
        if total["usage_sources"] == ["actual"]
        else ("mixed" if total["usage_sources"] else "none")
    )
    return total


def classify_trial_failure(row):
    if row.get("scc"):
        return "none"
    errors = [str(error) for error in row.get("errors", []) or []]
    if any(error.startswith("wall_timeout:") for error in errors):
        return "timeout"
    if row.get("stop_reason") == "model_error":
        return "provider_error"
    grader = row.get("grader", {}) or {}
    if grader.get("grader_infrastructure_error") or (
        grader.get("grader_returncode") not in {None, 0, 1}
    ):
        return "grader_infrastructure_error"
    if errors:
        return "runtime_error"
    if row.get("target_pass") is False:
        return "target_verifier_failed"
    if row.get("regression_pass") is False:
        return "regression_failed"
    if row.get("scope_pass") is False:
        return "scope_violation"
    required = dict(row.get("required_events", {}) or {})
    if required and not all(required.values()):
        return "required_evidence_missing"
    if row.get("finalization_pass") is False:
        return "finalization_missing"
    if row.get("budget_pass") is False:
        return "budget_exceeded"
    if row.get("safety_pass") is False:
        return "safety_violation"
    return "incomplete_evidence"


def run_trial(task, repeat, out_root, config, *, feature_flags=None):
    started = time.monotonic()
    workspace = fresh_workspace(task, out_root / "workspaces" / f"repeat_{repeat}")
    before = file_snapshot(workspace)
    client = make_client(config)
    agent = Lite(
        model_client=client,
        workspace=WorkspaceContext.build(workspace, repo_root_override=workspace),
        session_store=SessionStore(workspace / ".lite" / "sessions"),
        run_store=RunStore(workspace / ".lite" / "runs"),
        approval_policy="auto",
        max_steps=int(task["step_budget"]),
        max_new_tokens=default_max_tokens_for_provider(config.name),
        allowed_tools=evaluation_allowed_tools(task),
        write_scope=task["expected_changed_paths"],
        feature_flags=feature_flags or {"context_reduction": True},
    )
    errors = []
    for turn_prompt in [task["prompt"]] + (
        [task["follow_up_prompt"]] if task.get("follow_up_prompt") else []
    ):
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(agent.ask, turn_prompt)
        try:
            future.result(timeout=int(task.get("wall_timeout", 240)))
        except concurrent.futures.TimeoutError:
            errors.append(f"wall_timeout:{int(task.get('wall_timeout', 240))}s")
            agent.abort_current_turn()
            try:
                future.result(timeout=15)
            except Exception:
                pass
            executor.shutdown(wait=False, cancel_futures=True)
            break
        except Exception as exc:  # preserve row and fail closed
            errors.append(f"{type(exc).__name__}: {exc}")
            executor.shutdown(wait=False, cancel_futures=True)
            break
        else:
            executor.shutdown(wait=True)
    after = file_snapshot(workspace)
    changed = sorted(set(before) | set(after))
    changed = [p for p in changed if before.get(p) != after.get(p)]
    allowed = set(task["expected_changed_paths"])
    unexpected = [
        p
        for p in changed
        if p not in allowed
        and not any(p.startswith(prefix + "/") for prefix in IGNORED_PARTS)
    ]
    grader = grade_task(task["id"], workspace)
    events = trace_events(workspace)
    event_names = [str(e.get("event", "")) for e in events]
    required = {
        name: name in event_names for name in task.get("required_trace_events", [])
    }
    safety_pass = not any(name in HARD_SAFETY_EVENTS for name in event_names)
    task_state = getattr(agent, "current_task_state", None)
    finalization_pass = bool(
        task_state and task_state.stop_reason == STOP_REASON_FINAL_ANSWER_RETURNED
    )
    budget_pass = bool(
        task_state and int(task_state.tool_steps) <= int(task["step_budget"])
    )
    scope_pass = not unexpected and any(path in changed for path in allowed)
    events_pass = all(required.values()) if required else True
    scc = bool(
        grader.get("target_pass")
        and grader.get("regression_pass")
        and scope_pass
        and finalization_pass
        and budget_pass
        and safety_pass
        and events_pass
        and not errors
    )
    usage = usage_from_workspace(workspace)
    row = {
        "task_id": task["id"],
        "category": task["category"],
        "repeat": repeat,
        "scc": scc,
        "target_pass": bool(grader.get("target_pass")),
        "regression_pass": bool(grader.get("regression_pass")),
        "scope_pass": scope_pass,
        "changed_paths": changed,
        "unexpected_paths": unexpected,
        "finalization_pass": finalization_pass,
        "budget_pass": budget_pass,
        "safety_pass": safety_pass,
        "required_events": required,
        "events": event_names,
        "errors": errors,
        "stop_reason": str(getattr(task_state, "stop_reason", "") or ""),
        "grader": grader,
        "usage": usage,
        "tool_steps": int(getattr(task_state, "tool_steps", 0) or 0),
        "attempts": int(getattr(task_state, "attempts", 0) or 0),
        "wall_time_ms": int((time.monotonic() - started) * 1000),
        "workspace": str(workspace),
    }
    row["failure_category"] = classify_trial_failure(row)
    return row


def apply_reference(task, workspace):
    for rel, text in task["reference_files"].items():
        path = workspace / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def qualification(manifest, out_dir):
    source_digests = {
        t["id"]: digest_tree(ROOT / t["fixture_repo"]) for t in manifest["tasks"]
    }
    rows = []
    for task in manifest["tasks"]:
        ref = fresh_workspace(task, out_dir / "reference")
        apply_reference(task, ref)
        ref_grade = grade_task(task["id"], ref)
        no_op = fresh_workspace(task, out_dir / "no_op")
        no_op_grade = grade_task(task["id"], no_op)
        broken_passes = []
        for index in range(3):
            broken = fresh_workspace(task, out_dir / f"broken_{index}")
            # The unmodified fixture is the intentionally broken/no-op implementation.
            broken_passes.append(
                grade_task(task["id"], broken).get("target_pass", False)
            )
        rows.append(
            {
                "task_id": task["id"],
                "reference_pass": bool(
                    ref_grade.get("target_pass") and ref_grade.get("regression_pass")
                ),
                "no_op_pass": bool(
                    no_op_grade.get("target_pass")
                    and no_op_grade.get("regression_pass")
                ),
                "broken_passes": broken_passes,
                "source_digest": source_digests[task["id"]],
            }
        )
    tamper_task = manifest["tasks"][0]
    tamper = fresh_workspace(tamper_task, out_dir / "tamper")
    original = tamper / "src" / "pricing.py"
    original.write_text(
        original.read_text(encoding="utf-8") + "\n# tampered\n", encoding="utf-8"
    )
    tamper_rejected = digest_tree(tamper) != source_digests[tamper_task["id"]]
    invalid = subprocess.run(
        [
            sys.executable,
            str(GRADER),
            "--task",
            "does_not_exist",
            "--workspace",
            str(tamper),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    payload = {
        "qualification": {
            "reference_all_pass": all(r["reference_pass"] for r in rows),
            "no_op_all_fail": all(not r["no_op_pass"] for r in rows),
            "broken_all_fail_3x": all(not any(r["broken_passes"]) for r in rows),
            "tamper_fixture_unchanged_source": tamper_rejected,
            "invalid_grader_fail_closed": invalid.returncode != 0,
        },
        "rows": rows,
    }
    payload["qualified"] = all(payload["qualification"].values())
    (out_dir / "qualification.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def run_live(manifest, out_dir, repetitions):
    config = provider_metadata()
    qualification_path = out_dir / "qualification.json"
    if not qualification_path.is_file():
        raise RuntimeError("run qualification before live trials")
    qualification_payload = json.loads(qualification_path.read_text(encoding="utf-8"))
    if not qualification_payload.get("qualified"):
        raise RuntimeError("benchmark qualification did not pass")
    expected_digests = {
        row["task_id"]: row["source_digest"] for row in qualification_payload["rows"]
    }
    for task in manifest["tasks"]:
        if digest_tree(ROOT / task["fixture_repo"]) != expected_digests.get(task["id"]):
            raise RuntimeError(f"fixture integrity mismatch: {task['id']}")
    partial_path = out_dir / "results.partial.json"
    rows = (
        json.loads(partial_path.read_text(encoding="utf-8"))
        if partial_path.is_file()
        else []
    )
    completed = {(row["task_id"], int(row["repeat"])) for row in rows}
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    branch = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    eval_manifest = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "commit": commit,
        "branch": branch,
        "worktree_dirty": bool(status),
        "worktree_status": status,
        "benchmark_manifest_sha256": "sha256:"
        + hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        "grader_sha256": "sha256:" + hashlib.sha256(GRADER.read_bytes()).hexdigest(),
        "task_count": len(manifest["tasks"]),
        "task_ids": [task["id"] for task in manifest["tasks"]],
        "repetitions": repetitions,
        "provider": {
            "source": str(ROOT / ".lite.toml"),
            "name": config.name,
            "protocol": config.protocol,
            "model": config.model,
            "reasoning_effort": config.reasoning_effort,
            "strict_tools": config.strict_tools,
            "temperature": configured_temperature(config),
            "base_url_hostname": urlparse(config.base_url).hostname,
            "api_key_present": bool(config.api_key),
        },
    }
    evaluation_manifest_path = out_dir / "evaluation-manifest.json"
    if rows:
        if not evaluation_manifest_path.is_file():
            raise RuntimeError(
                "partial formal rows have no evaluation manifest; use a fresh output directory"
            )
        prior = json.loads(evaluation_manifest_path.read_text(encoding="utf-8"))
        identity_fields = (
            "benchmark_manifest_sha256",
            "grader_sha256",
            "task_count",
            "repetitions",
        )
        provider_fields = ("protocol", "model", "reasoning_effort", "strict_tools")
        prior_task_ids = prior.get("task_ids")
        legacy_task_ids_match = prior_task_ids is None and {
            row["task_id"] for row in rows
        } == set(eval_manifest["task_ids"])
        task_ids_match = (
            prior_task_ids == eval_manifest["task_ids"] or legacy_task_ids_match
        )
        if (
            not task_ids_match
            or any(
                prior.get(field) != eval_manifest[field]
                for field in identity_fields
            )
            or any(
                prior.get("provider", {}).get(field)
                != eval_manifest["provider"][field]
                for field in provider_fields
            )
        ):
            raise RuntimeError(
                "formal output identity changed; use a fresh output directory"
            )
    evaluation_manifest_path.write_text(
        json.dumps(eval_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for repeat in range(repetitions):
        for task in manifest["tasks"]:
            if (task["id"], repeat) in completed:
                continue
            print(
                json.dumps({"event": "start", "task": task["id"], "repeat": repeat}),
                flush=True,
            )
            rows.append(run_trial(task, repeat, out_dir, config))
            partial_path.write_text(
                json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(
                json.dumps(
                    {
                        "event": "done",
                        "task": task["id"],
                        "repeat": repeat,
                        "scc": rows[-1]["scc"],
                    }
                ),
                flush=True,
            )
    summary = {
        "trial_count": len(rows),
        "scc": sum(r["scc"] for r in rows),
        "scc_rate": sum(r["scc"] for r in rows) / len(rows) if rows else 0,
        "target_pass": sum(r["target_pass"] for r in rows),
        "regression_pass": sum(r["regression_pass"] for r in rows),
        "scope_pass": sum(r["scope_pass"] for r in rows),
        "finalization_pass": sum(r["finalization_pass"] for r in rows),
        "budget_pass": sum(r["budget_pass"] for r in rows),
        "safety_pass": sum(r["safety_pass"] for r in rows),
        "actual_usage_trials": sum(
            r["usage"]["usage_source"] == "actual" for r in rows
        ),
        "failure_by_category": {},
    }
    for row in rows:
        if not row["scc"]:
            key = row.get("failure_category", "incomplete_evidence")
            summary["failure_by_category"][key] = (
                summary["failure_by_category"].get(key, 0) + 1
            )
    payload = {
        "benchmark_id": manifest["benchmark_id"],
        "model": {
            "source": str(ROOT / ".lite.toml"),
            "provider": config.name,
            "protocol": config.protocol,
            "model": config.model,
            "reasoning_effort": config.reasoning_effort,
            "strict_tools": config.strict_tools,
            "temperature": configured_temperature(config),
            "base_url_hostname": urlparse(config.base_url).hostname,
            "api_key_present": bool(config.api_key),
        },
        "repetitions": repetitions,
        "summary": summary,
        "rows": rows,
    }
    (out_dir / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["qualification", "live"], required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output-dir", default=str(ARTIFACT_ROOT))
    parser.add_argument(
        "--task-ids",
        default="",
        help="Optional comma-separated fixed task ids; order is preserved.",
    )
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    manifest = select_tasks(load_manifest(), args.task_ids, args.limit)
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "qualification":
        print(json.dumps(qualification(manifest, out_dir), ensure_ascii=False))
    else:
        payload = run_live(manifest, out_dir, args.repetitions)
        print(
            json.dumps(
                {
                    "summary": payload["summary"],
                    "results": str(out_dir / "results.json"),
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
