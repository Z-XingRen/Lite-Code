#!/usr/bin/env python3
"""Build a hash-addressed manifest from completed Lite evaluation artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ARTIFACT_ROOT = ROOT / "artifacts" / "evaluation-20260810-terra"
DEFAULT_OUTPUT = ROOT / "docs" / "evidence-manifest.json"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def junit_summary(path):
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else root.findall("./testsuite")
    return {
        "tests": sum(int(item.attrib.get("tests", 0)) for item in suites),
        "failures": sum(int(item.attrib.get("failures", 0)) for item in suites),
        "errors": sum(int(item.attrib.get("errors", 0)) for item in suites),
        "skipped": sum(int(item.attrib.get("skipped", 0)) for item in suites),
        "time_seconds": sum(float(item.attrib.get("time", 0.0)) for item in suites),
    }


def git_value(*args):
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def run_static_gate(command):
    completed = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "command": command,
        "returncode": completed.returncode,
        "passed": completed.returncode == 0,
        "stdout_tail": completed.stdout[-2000:],
        "stderr_tail": completed.stderr[-2000:],
    }


def artifact_entry(path):
    path = Path(path).resolve()
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def build_manifest(artifact_root, expected_model):
    artifact_root = Path(artifact_root).resolve()
    workspace_path = artifact_root / "runtime-evidence" / "workspace-tracker.json"
    recovery_path = artifact_root / "runtime-evidence" / "effect-recovery.json"
    journal_path = artifact_root / "runtime-evidence" / "journal-scaling.json"
    context_path = artifact_root / "context-ab" / "results.json"
    quality_path = artifact_root / "local-quality" / "results.json"
    qualification_path = artifact_root / "local-quality" / "qualification.json"
    coverage_path = artifact_root / "coverage.json"
    pytest_path = artifact_root / "pytest.xml"
    stress_path = artifact_root / "pytest-stress.xml"
    post_full_path = artifact_root / "pytest-post-full.xml"
    context_manifest_path = artifact_root / "context-ab" / "evaluation-manifest.json"
    quality_manifest_path = artifact_root / "local-quality" / "evaluation-manifest.json"
    harbor_paths = [
        artifact_root / "harbor" / "swebench-verified-10" / "preflight.json",
        artifact_root / "harbor" / "terminal-bench-20" / "preflight.json",
    ]
    paths = [
        workspace_path,
        recovery_path,
        journal_path,
        context_path,
        quality_path,
        qualification_path,
        coverage_path,
        pytest_path,
        stress_path,
        post_full_path,
        context_manifest_path,
        quality_manifest_path,
        *harbor_paths,
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing evidence artifacts: " + ", ".join(missing))

    workspace = read_json(workspace_path)
    recovery = read_json(recovery_path)
    journal = read_json(journal_path)
    context = read_json(context_path)
    quality = read_json(quality_path)
    qualification = read_json(qualification_path)
    coverage = read_json(coverage_path)
    model = context["model"]
    if model["model"] != expected_model or quality["model"]["model"] != expected_model:
        raise RuntimeError(
            f"expected model {expected_model}, got context={model['model']} "
            f"quality={quality['model']['model']}"
        )

    static_gates = [
        run_static_gate(["uv", "run", "ruff", "check", "."]),
        run_static_gate(["uv", "run", "mypy"]),
        run_static_gate(["git", "diff", "--check"]),
    ]
    pytest_summary = junit_summary(pytest_path)
    stress_summary = junit_summary(stress_path)
    post_full_summary = junit_summary(post_full_path)
    coverage_totals = coverage["totals"]
    preflights = [read_json(path) for path in harbor_paths]
    status = git_value("status", "--short").splitlines()

    return {
        "schema_version": "lite.evidence_manifest.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "commit": git_value("rev-parse", "HEAD"),
            "branch": git_value("branch", "--show-current"),
            "worktree_dirty": bool(status),
            "worktree_status": status,
        },
        "model": {
            "provider_profile": model["provider"],
            "protocol": model["protocol"],
            "model": model["model"],
            "reasoning_effort": model["reasoning_effort"],
            "api_key_present": bool(model["api_key_present"]),
        },
        "deterministic_runtime": {
            "workspace": {
                "config": workspace["config"],
                "headline": workspace["headline"],
                "summary": workspace["summary"],
                "gates": workspace["gates"],
            },
            "effect_recovery": {
                "config": recovery["config"],
                "summary": recovery["summary"],
                "gates": recovery["gates"],
            },
            "journal_scaling": {
                "config": journal["config"],
                "headline": journal["headline"],
                "gates": journal["gates"],
            },
        },
        "real_model": {
            "context_ab": {
                "sample_scope": "2 paired coding/context smoke tasks",
                "formal_metrics": {
                    key: value
                    for key, value in context["formal_metrics"].items()
                    if key != "pairs"
                },
            },
            "local_formal": {
                "qualification": qualification["qualification"],
                "qualified": qualification["qualified"],
                "summary": quality["summary"],
            },
            "external_harbor": {
                "attempted_tasks": {
                    "swebench-verified-10": 1,
                    "terminal-bench-20": 1,
                },
                "preflights": preflights,
                "completed": all(item["ready"] for item in preflights),
                "official_scores_available": False,
            },
        },
        "quality_gates": {
            "pytest": pytest_summary,
            "stress": stress_summary,
            "post_full_targeted": post_full_summary,
            "coverage": {
                "num_statements": coverage_totals["num_statements"],
                "num_branches": coverage_totals["num_branches"],
                "percent_covered": coverage_totals["percent_covered"],
                "gate_percent": 80,
                "passed": coverage_totals["percent_covered"] >= 80,
            },
            "static": static_gates,
        },
        "claim_policy": {
            "allowed": [
                "deterministic runtime results with the exact config recorded here",
                "Terra A/B smoke results for the two named pairs only",
                "Terra local formal result of 2/4 SCC for the named qualified subset",
            ],
            "not_proven": [
                "production cost savings, user impact, incident reduction, or rollout",
                "general token reduction across coding tasks",
                "SWE-bench Verified or Terminal-Bench score",
                "native cross-platform sandbox security equivalence",
                "full-repository type safety",
            ],
        },
        "artifacts": [artifact_entry(path) for path in paths],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", default=str(DEFAULT_ARTIFACT_ROOT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--expected-model", default="gpt-5.6-terra")
    args = parser.parse_args(argv)
    manifest = build_manifest(args.artifact_root, args.expected_model)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    failed = []
    if manifest["quality_gates"]["pytest"]["failures"]:
        failed.append("pytest")
    if manifest["quality_gates"]["pytest"]["errors"]:
        failed.append("pytest-errors")
    if manifest["quality_gates"]["post_full_targeted"]["failures"]:
        failed.append("post-full-targeted")
    if manifest["quality_gates"]["post_full_targeted"]["errors"]:
        failed.append("post-full-targeted-errors")
    if not manifest["quality_gates"]["coverage"]["passed"]:
        failed.append("coverage")
    if not all(item["passed"] for item in manifest["quality_gates"]["static"]):
        failed.append("static")
    print(json.dumps({"output": str(output), "failed_gates": failed}))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
