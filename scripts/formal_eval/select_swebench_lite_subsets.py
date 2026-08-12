"""Build fixed SWE-bench Lite development and calibration manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "benchmarks" / "swebench_lite_subsets.json"
DATASET = "SWE-bench/SWE-bench_Lite"
DATASET_REVISION = "2e789fe71d28f983135aee214aee666b3faded59"
DEV_SEED = "lite-swebench-lite-dev-smoke-5-20260810"
CALIBRATION_SEED = "lite-swebench-lite-calibration-50-20260810"


def _json_url(url: str) -> object:
    request = Request(url, headers={"User-Agent": "lite-evaluation/2"})
    with urlopen(request, timeout=60) as response:  # noqa: S310 - fixed HTTPS hosts
        return json.load(response)


def _remote_revision() -> str:
    payload = _json_url(f"https://huggingface.co/api/datasets/{DATASET}")
    if not isinstance(payload, dict) or not isinstance(payload.get("sha"), str):
        raise RuntimeError("Hugging Face dataset metadata did not contain a revision")
    return payload["sha"]


def _instance_ids(split: str, expected_count: int) -> list[str]:
    ids: list[str] = []
    for offset in range(0, expected_count, 100):
        query = urlencode(
            {
                "dataset": DATASET,
                "config": "default",
                "split": split,
                "offset": offset,
                "length": min(100, expected_count - offset),
            }
        )
        payload = _json_url(f"https://datasets-server.huggingface.co/rows?{query}")
        if not isinstance(payload, dict) or not isinstance(payload.get("rows"), list):
            raise RuntimeError(f"dataset server did not return rows for split {split}")
        for item in payload["rows"]:
            row = item.get("row", {}) if isinstance(item, dict) else {}
            instance_id = row.get("instance_id") if isinstance(row, dict) else None
            if not isinstance(instance_id, str) or not instance_id:
                raise RuntimeError(f"split {split} contains a row without instance_id")
            ids.append(instance_id)
    if len(ids) != expected_count or len(set(ids)) != expected_count:
        raise RuntimeError(
            f"expected {expected_count} unique {split} ids, got {len(ids)} rows and "
            f"{len(set(ids))} unique ids"
        )
    return ids


def _score(seed: str, instance_id: str) -> str:
    return hashlib.sha256(f"{seed}\0{instance_id}".encode()).hexdigest()


def _records(ids: list[str], seed: str) -> list[dict[str, str]]:
    return [
        {"instance_id": instance_id, "selection_sha256": _score(seed, instance_id)}
        for instance_id in ids
    ]


def _select_with_repo_cap(
    ids: list[str], *, seed: str, count: int, per_repo: int
) -> list[str]:
    selected: list[str] = []
    repo_counts: dict[str, int] = {}
    for instance_id in sorted(ids, key=lambda value: _score(seed, value)):
        repo = instance_id.split("__", 1)[0]
        if repo_counts.get(repo, 0) >= per_repo:
            continue
        selected.append(instance_id)
        repo_counts[repo] = repo_counts.get(repo, 0) + 1
        if len(selected) == count:
            return selected
    raise RuntimeError(f"could only select {len(selected)} of {count} requested tasks")


def build_manifest() -> dict[str, object]:
    revision = _remote_revision()
    if revision != DATASET_REVISION:
        raise RuntimeError(
            f"dataset revision drift: expected {DATASET_REVISION}, got {revision}"
        )
    dev_ids = _instance_ids("dev", 23)
    test_ids = _instance_ids("test", 300)
    smoke_ids = _select_with_repo_cap(
        dev_ids, seed=DEV_SEED, count=5, per_repo=1
    )
    calibration_ids = _select_with_repo_cap(
        test_ids, seed=CALIBRATION_SEED, count=50, per_repo=5
    )
    return {
        "schema_version": "lite.swebench_lite_subsets.v1",
        "dataset": DATASET,
        "dataset_revision": DATASET_REVISION,
        "subsets": {
            "dev": {
                "split": "dev",
                "source_task_count": 23,
                "selection_rule": "All official development instances in dataset order.",
                "tasks": _records(dev_ids, "official-dev-order"),
            },
            "smoke-5": {
                "split": "dev",
                "source_task_count": 23,
                "seed": DEV_SEED,
                "selection_rule": (
                    "Sort by SHA-256(seed + NUL + instance_id), cap each repository "
                    "at one, and take five."
                ),
                "tasks": _records(smoke_ids, DEV_SEED),
            },
            "calibration-50": {
                "split": "test",
                "source_task_count": 300,
                "seed": CALIBRATION_SEED,
                "selection_rule": (
                    "Sort by SHA-256(seed + NUL + instance_id), cap each repository "
                    "at five, and take 50."
                ),
                "tasks": _records(calibration_ids, CALIBRATION_SEED),
            },
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args()
    output = Path(args.output).resolve()
    payload = build_manifest()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "dataset_revision": payload["dataset_revision"],
                "subset_counts": {
                    name: len(subset["tasks"])
                    for name, subset in payload["subsets"].items()
                },
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
