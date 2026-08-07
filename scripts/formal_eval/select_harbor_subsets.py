"""Create deterministic Harbor subset manifests for the formal evaluation."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HARBOR = ROOT / "benchmarks" / "external" / "harbor"
REGISTRY = HARBOR / "registry.json"
OUTPUT = ROOT / "artifacts" / "formal-evaluation-20260806" / "harbor-subsets.json"

TERMINAL_SEED = "lite-formal-terminal-bench-20-20260806"
SWEBENCH_SEED = "lite-formal-swebench-verified-10-20260806"


def _score(seed: str, task_name: str) -> str:
    return hashlib.sha256(f"{seed}\0{task_name}".encode()).hexdigest()


def _dataset(registry: list[dict], name: str, version: str) -> dict:
    matches = [
        item
        for item in registry
        if item.get("name") == name and str(item.get("version")) == version
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one {name}@{version} dataset, got {len(matches)}")
    return matches[0]


def _task_record(task: dict, seed: str) -> dict:
    return {
        "task_id": task["name"],
        "selection_sha256": _score(seed, task["name"]),
        "git_url": task["git_url"],
        "git_commit_id": task["git_commit_id"],
        "path": task["path"],
    }


def select_terminal_bench(dataset: dict) -> list[dict]:
    tasks = sorted(
        dataset["tasks"], key=lambda task: _score(TERMINAL_SEED, task["name"])
    )
    return [_task_record(task, TERMINAL_SEED) for task in tasks[:20]]


def select_swebench(dataset: dict) -> list[dict]:
    """Select ten tasks by stable hash while capping each repository at two."""

    tasks = sorted(
        dataset["tasks"], key=lambda task: _score(SWEBENCH_SEED, task["name"])
    )
    selected = []
    repository_counts: dict[str, int] = {}
    for task in tasks:
        repository = str(task["name"]).split("__", 1)[0]
        if repository_counts.get(repository, 0) >= 2:
            continue
        selected.append(_task_record(task, SWEBENCH_SEED))
        repository_counts[repository] = repository_counts.get(repository, 0) + 1
        if len(selected) == 10:
            break
    if len(selected) != 10:
        raise RuntimeError(f"could only select {len(selected)} SWE-bench tasks")
    return selected


def build_manifest() -> dict:
    registry_bytes = REGISTRY.read_bytes()
    registry = json.loads(registry_bytes)
    if not isinstance(registry, list):
        raise RuntimeError("Harbor registry must be a JSON array")
    terminal_dataset = _dataset(registry, "terminal-bench", "2.0")
    swebench_dataset = _dataset(registry, "swebench-verified", "1.0")
    harbor_commit = subprocess.run(
        ["git", "-C", str(HARBOR), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "schema_version": "formal-harbor-subsets-v1",
        "harbor_commit": harbor_commit,
        "registry_sha256": "sha256:" + hashlib.sha256(registry_bytes).hexdigest(),
        "subsets": {
            "terminal-bench-20": {
                "dataset": "terminal-bench@2.0",
                "source_task_count": len(terminal_dataset["tasks"]),
                "selection_rule": (
                    "Sort all task IDs by SHA-256(seed + NUL + task_id); take the first 20."
                ),
                "seed": TERMINAL_SEED,
                "tasks": select_terminal_bench(terminal_dataset),
            },
            "swebench-verified-10": {
                "dataset": "swebench-verified@1.0",
                "source_task_count": len(swebench_dataset["tasks"]),
                "selection_rule": (
                    "Sort all task IDs by SHA-256(seed + NUL + task_id), then take the "
                    "first 10 while capping each repository at two tasks."
                ),
                "seed": SWEBENCH_SEED,
                "tasks": select_swebench(swebench_dataset),
            },
        },
    }


def main() -> int:
    payload = build_manifest()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "terminal_bench_tasks": len(
                    payload["subsets"]["terminal-bench-20"]["tasks"]
                ),
                "swebench_tasks": len(
                    payload["subsets"]["swebench-verified-10"]["tasks"]
                ),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
