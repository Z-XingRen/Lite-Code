"""Run the fixed Lite real-task selection and write task-level evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lite.evaluation.real_task_harness import (  # noqa: E402
    load_manifest,
    result_matrix_keys,
    row_from_trial,
    validate_result_matrix,
    write_results,
)
from scripts.formal_eval import run_lite_quality_v1  # noqa: E402


VARIANT_FLAGS = {
    "baseline": {
        "context_reduction": True,
        "frozen_base_context": False,
        "journal_checkpoint_policy": False,
    },
    "frozen_context": {
        "context_reduction": True,
        "frozen_base_context": True,
        "journal_checkpoint_policy": False,
    },
    "journal_policy": {
        "context_reduction": True,
        "frozen_base_context": False,
        "journal_checkpoint_policy": True,
    },
    "optimized": {
        "context_reduction": True,
        "frozen_base_context": True,
        "journal_checkpoint_policy": True,
    },
}
COORDINATOR_TOOLS = frozenset({"agent", "send_message", "task_stop"})
RUNTIME_IDENTITY_IGNORED_PARTS = frozenset(
    {"__pycache__", ".pytest_cache", ".ruff_cache", ".lite"}
)
RUNTIME_IDENTITY_IGNORED_SUFFIXES = frozenset({".pyc", ".pyo", ".tmp", ".temp"})
RUNTIME_IDENTITY_PATHS = (
    Path("lite"),
    Path("scripts/run_real_task_harness.py"),
    Path("scripts/formal_eval/run_lite_quality_v1.py"),
    Path("benchmarks/real_tasks_v1.json"),
    Path("benchmarks/formal_v1"),
)


def feature_flags_for_task(task, variant):
    """Enable opt-in capabilities only when a fixed task declares them."""

    flags = {**VARIANT_FLAGS[variant], "multi_agent": False}
    allowed_tools = {str(name) for name in task.get("allowed_tools", [])}
    if allowed_tools & COORDINATOR_TOOLS:
        flags["multi_agent"] = True
    return flags


def build_identity(manifest, config, variants, repetitions):
    return {
        "schema_version": "lite.real_task_identity.v2",
        "benchmark_id": manifest["benchmark_id"],
        "seed": int(manifest["seed"]),
        "repetitions": int(repetitions),
        "variants": list(variants),
        "task_ids": [task["id"] for task in manifest["tasks"]],
        "task_manifest_sha256": "sha256:"
        + hashlib.sha256(
            (ROOT / "benchmarks/real_tasks_v1.json").read_bytes()
        ).hexdigest(),
        "provider": config.name,
        "protocol": config.protocol,
        "model": config.model,
        "reasoning_effort": config.reasoning_effort,
        "base_url_hostname": urlparse(config.base_url).hostname,
        "api_key_present": bool(config.api_key),
        "git_commit": _git_commit(),
        "runtime_source_sha256": _runtime_source_sha256(),
        "config_sha256": _file_sha256(ROOT / ".lite.toml"),
    }


def assert_identity_unchanged(expected, manifest, config, variants, repetitions):
    current = build_identity(manifest, config, variants, repetitions)
    if current != expected:
        raise RuntimeError(
            "real-task runtime or configuration changed during evaluation; "
            "use a fresh output directory"
        )


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


def select_tasks(manifest, task_ids):
    requested = [item.strip() for item in str(task_ids).split(",") if item.strip()]
    if not requested:
        return list(manifest["tasks"])
    by_id = {task["id"]: task for task in manifest["tasks"]}
    missing = [task_id for task_id in requested if task_id not in by_id]
    if missing:
        raise ValueError(f"unknown real task ids: {', '.join(missing)}")
    return [by_id[task_id] for task_id in requested]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(ROOT / "artifacts/real-task-v1"))
    parser.add_argument("--task-ids", default="")
    parser.add_argument("--repetitions", type=int, default=0)
    parser.add_argument(
        "--variants",
        default="baseline,frozen_context,journal_policy,optimized",
    )
    args = parser.parse_args(argv)

    manifest = load_manifest(ROOT)
    repetitions = int(args.repetitions or manifest["repetitions"])
    if repetitions < 3:
        raise ValueError("real-task harness requires at least 3 repetitions")
    variants = tuple(
        item.strip() for item in args.variants.split(",") if item.strip()
    )
    unknown_variants = sorted(set(variants) - set(VARIANT_FLAGS))
    if not variants or unknown_variants:
        raise ValueError(f"variants must be selected from {sorted(VARIANT_FLAGS)}")
    tasks = select_tasks(manifest, args.task_ids)
    config = run_lite_quality_v1.provider_metadata()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    identity = build_identity(
        {**manifest, "tasks": tasks}, config, variants, repetitions
    )
    identity_path = output_dir / "evaluation-manifest.json"
    if identity_path.is_file():
        previous = json.loads(identity_path.read_text(encoding="utf-8"))
        if previous != identity:
            raise RuntimeError(
                "real-task output identity changed; use a fresh output directory"
            )
    else:
        identity_path.write_text(
            json.dumps(identity, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    rows = []
    expected_keys = result_matrix_keys(tasks, variants, repetitions)
    results_path = output_dir / "results.jsonl"
    if results_path.is_file():
        rows = [
            json.loads(line)
            for line in results_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        validate_result_matrix(rows, expected_keys)
    completed = {
        (row["task_id"], int(row["repeat"]), row["variant"]) for row in rows
    }
    random.seed(int(manifest["seed"]))
    for repeat in range(repetitions):
        for task in tasks:
            for variant in variants:
                key = (task["id"], repeat, variant)
                if key in completed:
                    continue
                assert_identity_unchanged(
                    identity,
                    {**manifest, "tasks": tasks},
                    config,
                    variants,
                    repetitions,
                )
                trial = run_lite_quality_v1.run_trial(
                    task,
                    repeat,
                    output_dir / "work",
                    config,
                    feature_flags=feature_flags_for_task(task, variant),
                )
                assert_identity_unchanged(
                    identity,
                    {**manifest, "tasks": tasks},
                    config,
                    variants,
                    repetitions,
                )
                rows.append(
                    row_from_trial(task, trial, variant=variant, repeat=repeat)
                )
                write_results(rows, output_dir, expected_keys=expected_keys)
                print(json.dumps(rows[-1], ensure_ascii=False, sort_keys=True), flush=True)
    write_results(
        rows,
        output_dir,
        expected_keys=expected_keys,
        require_complete=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
