import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.formal_eval import build_formal_summary, run_harbor_subset
from scripts.formal_eval.run_context_ab_v1 import ensure_evaluation_identity
from scripts.formal_eval import run_lite_quality_v1
from scripts.formal_eval.run_lite_quality_v1 import select_tasks


def test_formal_task_selection_is_explicit_and_ordered():
    manifest = {"benchmark_id": "x", "tasks": [{"id": "A"}, {"id": "B"}]}

    selected = select_tasks(manifest, "B,A", limit=1)

    assert [task["id"] for task in selected["tasks"]] == ["B"]
    with pytest.raises(ValueError, match="unknown formal task ids"):
        select_tasks(manifest, "missing")


def test_formal_evidence_includes_run_and_session_events(tmp_path):
    run_dir = tmp_path / ".lite" / "runs" / "run_1"
    session_dir = tmp_path / ".lite" / "sessions"
    run_dir.mkdir(parents=True)
    session_dir.mkdir(parents=True)
    (run_dir / "trace.jsonl").write_text(
        json.dumps({"event": "tool_executed", "name": "agent"}) + "\n",
        encoding="utf-8",
    )
    (session_dir / "session.events.jsonl").write_text(
        json.dumps({"event": "worker_started", "worker_id": "agent_1"}) + "\n",
        encoding="utf-8",
    )

    events = run_lite_quality_v1.trace_events(tmp_path)

    assert [event["event"] for event in events] == [
        "tool_executed",
        "worker_started",
    ]
    assert events[1]["evidence_source"] == "session"


def test_formal_prompts_fully_state_graded_data_contracts():
    manifest = json.loads(run_lite_quality_v1.MANIFEST.read_text(encoding="utf-8"))
    tasks = {task["id"]: task for task in manifest["tasks"]}

    cli_prompt = tasks["C01_json_cli"]["prompt"]
    event_prompt = tasks["M04_event_contract"]["prompt"]

    assert "multiplied by 2" in cli_prompt
    assert "nested data object" in event_prompt
    assert "data.type" in event_prompt
    assert "data.payload" in event_prompt


@pytest.mark.parametrize(
    ("row", "expected"),
    [
        ({"scc": True}, "none"),
        ({"scc": False, "errors": ["wall_timeout:240s"]}, "timeout"),
        ({"scc": False, "stop_reason": "model_error"}, "provider_error"),
        (
            {"scc": False, "grader": {"grader_returncode": 2}},
            "grader_infrastructure_error",
        ),
        ({"scc": False, "target_pass": False}, "target_verifier_failed"),
        (
            {
                "scc": False,
                "target_pass": True,
                "regression_pass": True,
                "scope_pass": True,
                "required_events": {"worker_started": False},
            },
            "required_evidence_missing",
        ),
    ],
)
def test_formal_failure_categories_separate_runtime_and_agent_failures(
    row, expected
):
    assert run_lite_quality_v1.classify_trial_failure(row) == expected


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
        "swebench-verified-10",
        "lite",
        max_tasks=1,
        n_concurrent=2,
        agent_timeout_multiplier=3.0,
        environment_build_timeout_multiplier=4.0,
        artifact_root=tmp_path,
    )

    assert "openai/gpt-5.6-terra" in command
    assert command[command.index("--n-concurrent") + 1] == "2"
    assert command[command.index("--agent-timeout-multiplier") + 1] == "3.0"
    assert command[command.index("--environment-build-timeout-multiplier") + 1] == "4.0"
    assert command.count("--include-task-name") == 1
    assert "one" in command
    assert "two" not in command


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_concurrent": 0}, "n_concurrent"),
        ({"agent_timeout_multiplier": 0}, "agent_timeout_multiplier"),
        (
            {"environment_build_timeout_multiplier": 0},
            "environment_build_timeout_multiplier",
        ),
    ],
)
def test_harbor_command_rejects_invalid_execution_limits(
    tmp_path, monkeypatch, kwargs, message
):
    manifest = {
        "subsets": {
            "terminal-bench-20": {
                "dataset": "terminal-bench@2.0",
                "tasks": [{"task_id": "one"}],
            }
        }
    }
    manifest_path = tmp_path / "subsets.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(run_harbor_subset, "MANIFEST", manifest_path)

    with pytest.raises(ValueError, match=message):
        run_harbor_subset.build_command(
            "terminal-bench-20", "oracle", artifact_root=tmp_path, **kwargs
        )


def test_harbor_command_can_select_a_named_retry_task(tmp_path, monkeypatch):
    manifest = {
        "subsets": {
            "terminal-bench-20": {
                "dataset": "terminal-bench@2.0",
                "tasks": [{"task_id": "one"}, {"task_id": "two"}],
            }
        }
    }
    manifest_path = tmp_path / "subsets.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(run_harbor_subset, "MANIFEST", manifest_path)

    command, _ = run_harbor_subset.build_command(
        "terminal-bench-20",
        "oracle",
        task_ids=["two"],
        artifact_root=tmp_path,
    )

    assert command.count("--include-task-name") == 1
    assert "two" in command
    assert "one" not in command
    with pytest.raises(ValueError, match="unknown task ids"):
        run_harbor_subset.build_command(
            "terminal-bench-20",
            "oracle",
            task_ids=["missing"],
            artifact_root=tmp_path,
        )


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


def test_formal_runners_use_toml_identity_and_env_key(tmp_path, monkeypatch):
    config = tmp_path / ".lite.toml"
    config.write_text(
        "\n".join(
            [
                'provider = "openai"',
                "",
                "[providers.openai]",
                'protocol = "openai"',
                'base_url = "https://formal.example.test/v1"',
                'model = "gpt-formal-toml"',
                'reasoning_effort = "medium"',
                "strict_tools = false",
                "temperature = 0.1",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / ".env").write_text(
        "LITE_OPENAI_API_KEY=formal-env-secret\n", encoding="utf-8"
    )
    monkeypatch.delenv("LITE_OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(run_lite_quality_v1, "ROOT", tmp_path)
    monkeypatch.setattr(build_formal_summary, "ROOT", tmp_path)

    provider = run_lite_quality_v1.provider_metadata()
    client = run_lite_quality_v1.make_client(provider)
    identity = build_formal_summary.project_model_identity()

    assert provider.model == "gpt-formal-toml"
    assert provider.api_key == "formal-env-secret"
    assert client.model == "gpt-formal-toml"
    assert client.temperature == 0.1
    assert identity["model"] == "gpt-formal-toml"
    assert identity["base_url_hostname"] == "formal.example.test"
    assert "formal-env-secret" not in json.dumps(identity)


def test_harbor_agent_does_not_put_api_key_in_exec_environment():
    source_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "formal_eval"
        / "harbor_lite_agent.py"
    )
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    exec_environment_keys: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for keyword in node.keywords:
            if keyword.arg != "env" or not isinstance(keyword.value, ast.Dict):
                continue
            exec_environment_keys.update(
                key.value
                for key in keyword.value.keys
                if isinstance(key, ast.Constant) and isinstance(key.value, str)
            )

    assert "LITE_API_KEY" not in exec_environment_keys
    assert "TemporaryDirectory" in source
    assert "await environment.upload_file(local_api_key, remote_api_key)" in source
    assert 'export LITE_API_KEY=\\"$(cat {secret_path})\\"' in source


def test_harbor_agent_installs_pip_when_python_exists_without_it():
    source_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "formal_eval"
        / "harbor_lite_agent.py"
    )
    source = source_path.read_text(encoding="utf-8")

    ensurepip = source.index("python3 -m ensurepip --upgrade")
    fallback_check = source.index(
        "if ! python3 -m pip --version >/dev/null 2>&1; then", ensurepip
    )
    package_install = source.index("python3-pip ca-certificates", fallback_check)

    assert ensurepip < fallback_check < package_install
