"""Run fixed cross-turn prompt-cache scenarios against the configured provider."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lite import Lite, SessionStore, WorkspaceContext  # noqa: E402
from lite.config import default_max_tokens_for_provider  # noqa: E402
from lite.core.run_store import RunStore  # noqa: E402
from lite.evaluation.prompt_cache_turn_harness import (  # noqa: E402
    MANIFEST_PATH,
    VARIANTS,
    load_manifest,
    result_matrix_keys,
    row_from_turns,
    turn_evidence,
    validate_result_matrix,
    write_results,
)
from scripts.formal_eval.run_lite_quality_v1 import (  # noqa: E402
    make_client,
    provider_metadata,
)


RUNTIME_IDENTITY_PATHS = (
    Path("lite"),
    Path("scripts/run_prompt_cache_turn_harness.py"),
    MANIFEST_PATH,
)
RUNTIME_IDENTITY_IGNORED_PARTS = frozenset(
    {"__pycache__", ".pytest_cache", ".ruff_cache", ".lite"}
)
RUNTIME_IDENTITY_IGNORED_SUFFIXES = frozenset({".pyc", ".pyo", ".tmp", ".temp"})


def build_identity(manifest, config, variants, repetitions):
    """Bind resumable results to runtime, manifest, and provider settings."""

    return {
        "schema_version": "lite.prompt_cache_turn_identity.v1",
        "benchmark_id": manifest["benchmark_id"],
        "git_commit": _git_commit(),
        "runtime_source_sha256": _runtime_source_sha256(),
        "config_sha256": _file_sha256(ROOT / ".lite.toml"),
        "manifest_sha256": _file_sha256(ROOT / MANIFEST_PATH),
        "scenario_ids": [scenario["id"] for scenario in manifest["scenarios"]],
        "variants": list(variants),
        "repetitions": int(repetitions),
        "provider": config.name,
        "protocol": config.protocol,
        "model": config.model,
        "reasoning_effort": config.reasoning_effort,
        "base_url_hostname": urlparse(config.base_url).hostname,
        "api_key_present": bool(config.api_key),
    }


def assert_identity_unchanged(expected, manifest, config, variants, repetitions):
    current = build_identity(manifest, config, variants, repetitions)
    if current != expected:
        raise RuntimeError(
            "prompt-cache runtime or configuration changed during evaluation; "
            "use a fresh output directory"
        )


def ensure_evaluation_identity(output_dir, identity):
    """Create an identity manifest or reject incompatible partial results."""

    output_dir = Path(output_dir)
    identity_path = output_dir / "evaluation-manifest.json"
    results_path = output_dir / "results.jsonl"
    if identity_path.is_file():
        previous = json.loads(identity_path.read_text(encoding="utf-8"))
        if previous != identity:
            raise RuntimeError(
                "prompt-cache output identity changed; use a fresh output directory"
            )
        return identity_path
    if results_path.is_file():
        raise RuntimeError(
            "prompt-cache partial results have no evaluation identity; "
            "use a fresh output directory"
        )
    identity_path.write_text(
        json.dumps(identity, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return identity_path


def run_scenario(scenario, repeat, variant, output_dir, config, *, timeout=300):
    """Run one isolated two-turn scenario and retain exact per-turn evidence."""

    workspace = _fresh_workspace(output_dir, scenario, repeat, variant)
    feature_flags = {
        "context_reduction": True,
        "frozen_base_context": True,
        "journal_checkpoint_policy": True,
        "prompt_cache": variant == "append_projection",
    }
    agent = _build_agent(workspace, config, feature_flags)
    turns = []
    errors = []
    try:
        for index, turn in enumerate(scenario["turns"]):
            if index == 1:
                agent = _between_turns(
                    agent,
                    workspace,
                    scenario,
                    repeat,
                    config,
                    feature_flags,
                )
            answer, error = _ask_with_timeout(agent, turn["prompt"], timeout)
            if error:
                errors.append(error)
            run_id = Path(agent.current_run_dir).name if agent.current_run_dir else ""
            if run_id:
                evidence = turn_evidence(workspace, run_id)
                evidence.update(
                    {
                        "turn_index": index,
                        "answer": answer,
                        "expected_answer": turn["expected_answer"],
                        "answer_match": answer.strip() == turn["expected_answer"],
                    }
                )
                turns.append(evidence)
            if error:
                break
    finally:
        agent.close()
    return row_from_turns(
        scenario,
        turns,
        variant=variant,
        repeat=repeat,
        errors=errors,
    )


def _build_agent(workspace, config, feature_flags, *, session_id=None):
    client = make_client(config)
    kwargs = {
        "model_client": client,
        "workspace": WorkspaceContext.build(workspace, repo_root_override=workspace),
        "session_store": SessionStore(workspace / ".lite" / "sessions"),
        "run_store": RunStore(workspace / ".lite" / "runs"),
        "approval_policy": "auto",
        "max_steps": 1,
        "max_new_tokens": min(256, default_max_tokens_for_provider(config.name)),
        "allowed_tools": ["read_file"],
        "feature_flags": feature_flags,
        "read_only": True,
        "auto_dream": False,
        "final_readiness_mode": "off",
    }
    if session_id:
        return Lite.from_session(session_id=session_id, **kwargs)
    return Lite(**kwargs)


def _between_turns(agent, workspace, scenario, repeat, config, feature_flags):
    action = scenario["between_turns"]
    if action == "workspace_refresh":
        readme = workspace / "README.md"
        readme.write_text(
            readme.read_text(encoding="utf-8")
            + f"Workspace refresh evidence for repeat {repeat}.\n",
            encoding="utf-8",
        )
    elif action == "session_resume":
        session_id = agent.session["id"]
        agent.close()
        return _build_agent(
            workspace,
            config,
            feature_flags,
            session_id=session_id,
        )
    return agent


def _ask_with_timeout(agent, prompt, timeout):
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(agent.ask, prompt)
    try:
        answer = future.result(timeout=int(timeout))
    except concurrent.futures.TimeoutError:
        agent.abort_current_turn()
        executor.shutdown(wait=False, cancel_futures=True)
        return "", f"wall_timeout:{int(timeout)}s"
    except Exception as exc:
        executor.shutdown(wait=False, cancel_futures=True)
        return "", f"{type(exc).__name__}: {exc}"
    executor.shutdown(wait=True)
    return str(answer), ""


def _fresh_workspace(output_dir, scenario, repeat, variant):
    workspace = (
        Path(output_dir)
        / "work"
        / f"repeat_{repeat}"
        / str(scenario["id"])
        / str(variant)
    )
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True)
    (workspace / "README.md").write_text(
        "Prompt cache cross-turn evaluation workspace.\n",
        encoding="utf-8",
    )
    return workspace


def _git_commit():
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else ""


def _runtime_source_sha256():
    files = set()
    for relative in RUNTIME_IDENTITY_PATHS:
        path = ROOT / relative
        if path.is_file():
            files.add(path)
        elif path.is_dir():
            files.update(
                item
                for item in path.rglob("*")
                if item.is_file() and _is_runtime_identity_file(item)
            )
    digest = hashlib.sha256()
    for path in sorted(files, key=lambda item: item.relative_to(ROOT).as_posix()):
        relative = path.relative_to(ROOT).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return "sha256:" + digest.hexdigest()


def _is_runtime_identity_file(path):
    relative = Path(path).relative_to(ROOT)
    return not (
        any(part in RUNTIME_IDENTITY_IGNORED_PARTS for part in relative.parts)
        or path.suffix.lower() in RUNTIME_IDENTITY_IGNORED_SUFFIXES
    )


def _file_sha256(path):
    path = Path(path)
    if not path.is_file():
        return ""
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", default=str(ROOT / "artifacts" / "prompt-cache-turns-v1")
    )
    parser.add_argument("--repetitions", type=int, default=0)
    parser.add_argument("--scenario-ids", default="")
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--turn-timeout", type=int, default=300)
    args = parser.parse_args(argv)

    manifest = load_manifest(ROOT)
    repetitions = int(args.repetitions or manifest["repetitions"])
    if repetitions < 3:
        raise ValueError("prompt-cache turn harness requires at least 3 repetitions")
    variants = tuple(item.strip() for item in args.variants.split(",") if item.strip())
    unknown_variants = sorted(set(variants) - set(VARIANTS))
    if not variants or unknown_variants:
        raise ValueError(f"variants must be selected from {list(VARIANTS)}")
    scenarios = _select_scenarios(manifest["scenarios"], args.scenario_ids)
    selected_manifest = {**manifest, "scenarios": scenarios}
    config = provider_metadata()
    capability_probe = make_client(config)
    if not getattr(capability_probe, "supports_append_prompt_cache", False):
        raise RuntimeError(
            "prompt-cache turn harness requires a provider client with "
            "append-only prompt-cache support"
        )
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    identity = build_identity(selected_manifest, config, variants, repetitions)
    ensure_evaluation_identity(output_dir, identity)

    expected_keys = result_matrix_keys(scenarios, variants, repetitions)
    results_path = output_dir / "results.jsonl"
    rows = []
    if results_path.is_file():
        rows = [
            json.loads(line)
            for line in results_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        validate_result_matrix(rows, expected_keys)
    completed = {
        (row["scenario"], int(row["repeat"]), row["variant"]) for row in rows
    }
    for repeat in range(repetitions):
        for scenario in scenarios:
            for variant in variants:
                key = (scenario["id"], repeat, variant)
                if key in completed:
                    continue
                assert_identity_unchanged(
                    identity, selected_manifest, config, variants, repetitions
                )
                row = run_scenario(
                    scenario,
                    repeat,
                    variant,
                    output_dir,
                    config,
                    timeout=args.turn_timeout,
                )
                assert_identity_unchanged(
                    identity, selected_manifest, config, variants, repetitions
                )
                rows.append(row)
                write_results(rows, output_dir, expected_keys=expected_keys)
                print(json.dumps(row, ensure_ascii=False, sort_keys=True), flush=True)
    write_results(
        rows,
        output_dir,
        expected_keys=expected_keys,
        require_complete=True,
    )
    return 0


def _select_scenarios(scenarios, scenario_ids):
    requested = [
        item.strip() for item in str(scenario_ids).split(",") if item.strip()
    ]
    if not requested:
        return list(scenarios)
    by_id = {scenario["id"]: scenario for scenario in scenarios}
    missing = [scenario_id for scenario_id in requested if scenario_id not in by_id]
    if missing:
        raise ValueError(f"unknown prompt-cache scenarios: {', '.join(missing)}")
    return [by_id[scenario_id] for scenario_id in requested]


if __name__ == "__main__":
    raise SystemExit(main())
