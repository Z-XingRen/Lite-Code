import json
import subprocess
from types import SimpleNamespace

import pytest

from scripts.formal_eval import run_swebench_lite, select_swebench_lite_subsets


def _write_provider_files(root):
    (root / ".lite.toml").write_text(
        "\n".join(
            [
                'provider = "openai"',
                "",
                "[providers.openai]",
                'protocol = "openai"',
                'base_url = "https://swebench.example.test/v1"',
                'model = "gpt-swebench-toml"',
                'reasoning_effort = "medium"',
                "strict_tools = false",
                "temperature = 0.2",
            ]
        ),
        encoding="utf-8",
    )
    (root / ".env").write_text(
        "LITE_OPENAI_API_KEY=swebench-env-secret\n", encoding="utf-8"
    )


def test_subset_selection_is_deterministic_and_repo_capped():
    instance_ids = [
        "alpha__1",
        "alpha__2",
        "beta__1",
        "gamma__1",
        "delta__1",
    ]

    selected = select_swebench_lite_subsets._select_with_repo_cap(
        instance_ids, seed="fixed", count=3, per_repo=1
    )

    assert selected == select_swebench_lite_subsets._select_with_repo_cap(
        list(reversed(instance_ids)), seed="fixed", count=3, per_repo=1
    )
    assert len({instance_id.split("__", 1)[0] for instance_id in selected}) == 3
    with pytest.raises(RuntimeError, match="could only select"):
        select_swebench_lite_subsets._select_with_repo_cap(
            ["alpha__1", "alpha__2"], seed="fixed", count=2, per_repo=1
        )


def test_manifest_build_pins_revision_and_selection_hashes(monkeypatch):
    dev_ids = [f"devrepo{i}__issue" for i in range(23)]
    test_ids = [f"testrepo{i // 5}__issue{i}" for i in range(300)]
    monkeypatch.setattr(
        select_swebench_lite_subsets,
        "_remote_revision",
        lambda: select_swebench_lite_subsets.DATASET_REVISION,
    )
    monkeypatch.setattr(
        select_swebench_lite_subsets,
        "_instance_ids",
        lambda split, expected_count: dev_ids if split == "dev" else test_ids,
    )

    manifest = select_swebench_lite_subsets.build_manifest()

    assert manifest["dataset_revision"] == select_swebench_lite_subsets.DATASET_REVISION
    assert len(manifest["subsets"]["dev"]["tasks"]) == 23
    assert len(manifest["subsets"]["smoke-5"]["tasks"]) == 5
    assert len(manifest["subsets"]["calibration-50"]["tasks"]) == 50
    smoke = manifest["subsets"]["smoke-5"]
    for record in smoke["tasks"]:
        assert record["selection_sha256"] == select_swebench_lite_subsets._score(
            smoke["seed"], record["instance_id"]
        )


def test_preflight_uses_toml_model_env_key_and_redacts_secret(tmp_path, monkeypatch):
    _write_provider_files(tmp_path)
    manifest = tmp_path / "subsets.json"
    manifest.write_text(
        json.dumps(
            {
                "dataset": "SWE-bench/SWE-bench_Lite",
                "dataset_revision": "pinned-revision",
                "subsets": {},
            }
        ),
        encoding="utf-8",
    )
    evaluator_source = tmp_path / "run_evaluation.py"
    evaluator_source.write_text("# evaluator\n", encoding="utf-8")
    monkeypatch.delenv("LITE_OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(run_swebench_lite, "ROOT", tmp_path)
    monkeypatch.setattr(run_swebench_lite, "CONFIG", tmp_path / ".lite.toml")
    monkeypatch.setattr(run_swebench_lite, "MANIFEST", manifest)
    monkeypatch.setattr(run_swebench_lite, "EVALUATOR_MODULE", evaluator_source)
    monkeypatch.setattr(run_swebench_lite.shutil, "which", lambda name: "docker")
    monkeypatch.setattr(
        run_swebench_lite,
        "_run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, "ok", ""),
    )
    disk = {
        "path": str(tmp_path),
        "total_bytes": 100 * 1024**3,
        "used_bytes": 1,
        "free_bytes": 99 * 1024**3,
    }
    monkeypatch.setattr(run_swebench_lite, "_disk", lambda path: disk)
    monkeypatch.setattr(
        run_swebench_lite,
        "_temp_check",
        lambda: {"path": str(tmp_path), "exists": True, "writable": True, "disk": disk},
    )
    monkeypatch.setattr(
        run_swebench_lite,
        "_remote_dataset_revision",
        lambda dataset: ("pinned-revision", ""),
    )
    monkeypatch.setattr(
        run_swebench_lite,
        "_evaluator_dependencies",
        lambda **kwargs: {"available": True, "mode": "test"},
    )
    monkeypatch.setattr(
        run_swebench_lite,
        "_probe_provider",
        lambda config, temperature: {
            "available": True,
            "usage_present": True,
            "wall_time_ms": 1,
            "error": "",
        },
    )
    monkeypatch.setattr(run_swebench_lite, "_git_commit", lambda repo: "commit")

    result = run_swebench_lite.preflight()

    assert result["ready"] is True
    assert result["model"]["model"] == "gpt-swebench-toml"
    assert result["model"]["base_url_hostname"] == "swebench.example.test"
    assert result["model"]["api_key_present"] is True
    assert "swebench-env-secret" not in json.dumps(result)


def test_official_verifier_merge_requires_all_scc_gates(tmp_path):
    output = tmp_path
    instance_id = "repo__issue"
    run_id = "run-1"
    config = SimpleNamespace(name="openai", model="gpt-test")
    log_dir = (
        output
        / "verifier"
        / "logs"
        / "run_evaluation"
        / run_id
        / "lite-openai__gpt-test"
        / instance_id
    )
    log_dir.mkdir(parents=True)
    (log_dir / "run_instance.log").write_text(
        ">>>>> Applied Patch\n", encoding="utf-8"
    )
    (log_dir / "report.json").write_text(
        json.dumps(
            {
                instance_id: {
                    "resolved": True,
                    "tests_status": {
                        "FAIL_TO_PASS": {"failure": []},
                        "PASS_TO_PASS": {"failure": []},
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    (output / "tasks" / instance_id).mkdir(parents=True)
    row = {
        "instance_id": instance_id,
        "scope_pass": True,
        "finalization_pass": True,
        "safety_pass": True,
        "error": "",
    }

    run_swebench_lite._merge_verifier(
        output, [row], run_id=run_id, config=config, evaluator_returncode=0
    )

    assert row["patch_apply"] is True
    assert row["target_pass"] is True
    assert row["regression_pass"] is True
    assert row["scc"] is True
    assert row["failure_category"] == "none"


def test_fresh_workspace_retries_transient_clone_failure(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    clone_attempts = 0

    def fake_run(command, **kwargs):
        nonlocal clone_attempts
        if command[:2] == ["git", "clone"]:
            clone_attempts += 1
            workspace.mkdir(parents=True)
            if clone_attempts == 1:
                return subprocess.CompletedProcess(command, 1, "", "SSL_ERROR_SYSCALL")
            (workspace / ".git" / "info").mkdir(parents=True)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(run_swebench_lite, "_run", fake_run)
    monkeypatch.setattr(run_swebench_lite.time, "sleep", lambda seconds: None)

    run_swebench_lite._fresh_workspace(
        {"repo": "owner/repo", "base_commit": "abc"}, workspace
    )

    assert clone_attempts == 2
    assert (workspace / ".git" / "info" / "exclude").is_file()


def test_resume_loads_completed_row_and_cleans_only_incomplete_task(tmp_path):
    completed = "repo__completed"
    incomplete = "repo__incomplete"
    completed_task = tmp_path / "tasks" / completed
    completed_task.mkdir(parents=True)
    (completed_task / "agent-report.json").write_text(
        json.dumps({"instance_id": completed, "status": "generated"}),
        encoding="utf-8",
    )
    (completed_task / "prediction.patch").write_text("patch", encoding="utf-8")
    incomplete_task = tmp_path / "tasks" / incomplete
    incomplete_workspace = tmp_path / "workspaces" / incomplete
    incomplete_task.mkdir(parents=True)
    incomplete_workspace.mkdir(parents=True)

    row = run_swebench_lite._load_existing_agent_row(tmp_path, completed)
    run_swebench_lite._prepare_incomplete_resume(tmp_path, incomplete)

    assert row["instance_id"] == completed
    assert completed_task.is_dir()
    assert not incomplete_task.exists()
    assert not incomplete_workspace.exists()
