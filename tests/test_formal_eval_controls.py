import json
from types import SimpleNamespace

import pytest

from scripts.formal_eval import run_harbor_subset
from scripts.formal_eval.run_context_ab_v1 import ensure_evaluation_identity
from scripts.formal_eval.run_lite_quality_v1 import select_tasks


def test_formal_task_selection_is_explicit_and_ordered():
    manifest = {"benchmark_id": "x", "tasks": [{"id": "A"}, {"id": "B"}]}

    selected = select_tasks(manifest, "B,A", limit=1)

    assert [task["id"] for task in selected["tasks"]] == ["B"]
    with pytest.raises(ValueError, match="unknown formal task ids"):
        select_tasks(manifest, "missing")


def test_harbor_command_uses_toml_model_and_bounded_subset(tmp_path, monkeypatch):
    manifest = {
        "subsets": {
            "swebench-verified-10": {
                "dataset": "swebench-verified@1.0",
                "tasks": [{"task_id": "one"}, {"task_id": "two"}],
            }
        }
    }
    manifest_path = tmp_path / "subsets.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(run_harbor_subset, "MANIFEST", manifest_path)
    monkeypatch.setattr(
        run_harbor_subset,
        "_provider_config",
        lambda: SimpleNamespace(
            model="gpt-5.6-terra", base_url="https://example.test/v1"
        ),
    )

    command, _ = run_harbor_subset.build_command(
        "swebench-verified-10", "lite", max_tasks=1, artifact_root=tmp_path
    )

    assert "openai/gpt-5.6-terra" in command
    assert command.count("--include-task-name") == 1
    assert "one" in command
    assert "two" not in command


def test_harbor_preflight_reports_unavailable_daemon(tmp_path, monkeypatch):
    harbor = tmp_path / "harbor.exe"
    registry = tmp_path / "registry.json"
    manifest = tmp_path / "manifest.json"
    for path in (harbor, registry, manifest):
        path.write_text("x", encoding="utf-8")
    monkeypatch.setattr(run_harbor_subset, "HARBOR_EXE", harbor)
    monkeypatch.setattr(run_harbor_subset, "REGISTRY", registry)
    monkeypatch.setattr(run_harbor_subset, "MANIFEST", manifest)
    monkeypatch.setattr(run_harbor_subset.shutil, "which", lambda name: "docker")
    monkeypatch.setattr(
        run_harbor_subset.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=1, stdout="", stderr="daemon unavailable"
        ),
    )

    result = run_harbor_subset.preflight()

    assert result["ready"] is False
    assert result["checks"]["docker_cli"] is True
    assert result["checks"]["docker_daemon"] is False
    assert "daemon unavailable" in result["detail"]


def test_context_ab_resume_rejects_model_identity_change(tmp_path):
    tasks = [{"id": "multi-file-refactor"}]
    config = SimpleNamespace(
        model="gpt-5.6-terra",
        reasoning_effort="medium",
        protocol="openai",
    )
    ensure_evaluation_identity(tmp_path, config, tasks, repetitions=1)
    ensure_evaluation_identity(tmp_path, config, tasks, repetitions=1)

    changed = SimpleNamespace(
        model="different-model",
        reasoning_effort="medium",
        protocol="openai",
    )
    with pytest.raises(RuntimeError, match="identity changed"):
        ensure_evaluation_identity(tmp_path, changed, tasks, repetitions=1)
