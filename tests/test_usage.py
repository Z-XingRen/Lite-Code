from pathlib import Path
import json
import pytest

from lite.testing import ScriptedModelClient
from lite import Lite, SessionStore, WorkspaceContext


def build_agent(tmp_path, outputs=None, **kwargs):
    (tmp_path / "README.md").write_text("demo\n", encoding="utf-8")
    workspace = WorkspaceContext.build(tmp_path)
    return Lite(
        model_client=ScriptedModelClient(outputs or []),
        workspace=workspace,
        session_store=SessionStore(tmp_path / ".lite" / "sessions"),
        approval_policy="auto",
        **kwargs,
    )


def test_usage_command_reports_provider_model_and_last_usage(tmp_path):
    from lite.cli import handle_repl_command

    agent = build_agent(tmp_path, ["<final>Done.</final>"])
    agent.model_client.model = "gpt-test"
    agent.model_client.base_url = "https://example.com/v1"
    agent.model_client.last_completion_metadata = {
        "input_tokens": 10,
        "output_tokens": 5,
        "cached_tokens": 3,
        "cache_write_tokens": 4,
        "provider_attempts": 2,
        "provider_retry_count": 1,
    }
    agent.ask("hello")

    handled, _, output = handle_repl_command(agent, "/usage")

    assert handled is True
    assert "model: gpt-test" in output
    assert "base url host: example.com" in output
    assert "last input tokens: 10" in output
    assert "last output tokens: 5" in output
    assert "last cached tokens: 3" in output
    assert "last cache write tokens: 4" in output


def test_usage_command_sanitizes_base_url_host(tmp_path):
    from lite.cli import handle_repl_command

    agent = build_agent(tmp_path, [])
    agent.model_client.base_url = "https://user:secret@example.com:8443/v1?api_key=sk-real"

    handled, _, output = handle_repl_command(agent, "/usage")

    assert handled is True
    assert "base url host: example.com:8443" in output
    assert "secret" not in output
    assert "api_key" not in output


def test_usage_command_handles_malformed_sanitized_base_url(tmp_path):
    from lite.cli import handle_repl_command

    agent = build_agent(tmp_path, [])
    agent.model_client.base_url = "http://user:secret@[::1/v1?api_key=x#frag"

    handled, _, output = handle_repl_command(agent, "/usage")

    assert handled is True
    assert "base url host: [::1" in output
    assert "secret" not in output
    assert "api_key" not in output


def test_usage_command_optionally_reports_context_pressure_fields(tmp_path):
    from lite.cli import handle_repl_command

    agent = build_agent(tmp_path, [])
    agent.last_prompt_metadata = {
        "context_usage": {
            "total_estimated_tokens": 400,
            "context_window": 1000,
            "pressure_tier": "medium",
            "usage_source": "estimated",
            "cached_tokens": 128,
            "cache_write_tokens": 64,
        }
    }

    handled, _, output = handle_repl_command(agent, "/usage")

    assert handled is True
    assert "context usage: 400/1000" in output
    assert "context pressure tier: medium" in output
    assert "context usage source: estimated" in output
    assert "context cached tokens: 128" in output
    assert "context cache write tokens: 64" in output


def test_usage_command_optionally_reports_context_orchestrator_fields(tmp_path):
    from lite.cli import handle_repl_command

    agent = build_agent(tmp_path, [])
    agent.last_prompt_metadata = {
        "context_usage": {
            "total_estimated_tokens": 400,
            "context_window": 1000,
        },
        "context_orchestrator": {
            "version": "local-v1",
            "summary_called": True,
            "summary_delta_event_count": 3,
            "replacement_cache_hits": 2,
        },
    }

    handled, _, output = handle_repl_command(agent, "/usage")

    assert handled is True
    assert "context orchestrator: local-v1" in output
    assert "context summary called: True" in output
    assert "context summary delta events: 3" in output
    assert "context replacement cache hits: 2" in output


def test_context_command_reports_usage_and_orchestrator_payload(tmp_path):
    from lite.cli import handle_repl_command

    agent = build_agent(tmp_path, [])

    handled, _, output = handle_repl_command(agent, "/context")

    payload = json.loads(output)
    assert handled is True
    assert "context_usage" in payload
    assert payload["context_orchestrator"]["version"] == "local-v1"


def test_usage_command_omits_optional_context_pressure_fields_when_absent(tmp_path):
    from lite.cli import handle_repl_command

    agent = build_agent(tmp_path, [])
    agent.last_prompt_metadata = {
        "context_usage": {
            "total_estimated_tokens": 400,
            "context_window": 1000,
        }
    }

    handled, _, output = handle_repl_command(agent, "/usage")

    assert handled is True
    assert "context usage: 400/1000" in output
    assert "context pressure tier:" not in output
    assert "context usage source:" not in output
    assert "context cached tokens:" not in output


def test_model_command_updates_current_runtime_only(tmp_path):
    from lite.cli import handle_repl_command

    agent = build_agent(tmp_path, [])
    agent.model_client.model = "old-model"

    handled, _, output = handle_repl_command(agent, "/model new-model")

    assert handled is True
    assert output == "model: new-model"
    assert agent.model_client.model == "new-model"
    assert not (Path(tmp_path) / ".lite.toml").exists()


def test_repl_model_selector_chooses_model_then_reasoning_effort(tmp_path):
    from lite.cli import run_repl_model_selector

    agent = build_agent(tmp_path, [])
    agent.model_client.model = "model-a"
    agent.model_client.available_models = ("model-b", "model-a")
    agent.model_client.reasoning_effort = "low"
    agent.model_client.available_reasoning_efforts = ("none", "low", "high")
    answers = iter(["1", "3"])
    output_lines = []

    output = run_repl_model_selector(
        agent,
        input_fn=lambda _prompt: next(answers),
        output_fn=output_lines.append,
    )

    assert agent.model_client.model == "model-b"
    assert agent.model_client.reasoning_effort == "high"
    assert output == "model: model-b\nreasoning effort: high"
    assert output_lines[0] == "Select model:"
    assert output_lines[1:3] == [
        "  1. model-b",
        "  2. model-a (current/default)",
    ]
    assert "Select reasoning effort:" in output_lines


def test_session_history_resume_and_clear_commands(tmp_path):
    from lite.cli import handle_repl_command

    first = build_agent(tmp_path, ["<final>First.</final>"])
    assert first.ask("first request") == "First."
    first_id = first.session["id"]

    second = Lite.from_session(
        model_client=ScriptedModelClient(["<final>Second.</final>"]),
        workspace=first.workspace,
        session_store=first.session_store,
        session_id=first_id,
        approval_policy="auto",
    )
    assert second.ask("second request") == "Second."

    handled, _, output = handle_repl_command(second, "/history")
    assert handled is True
    assert first_id in output
    assert "Second." in output

    handled, _, output = handle_repl_command(second, f"/resume {first_id}")
    assert handled is True
    assert output == f"resumed session {first_id}"
    assert second.session["id"] == first_id

    old_id = second.session["id"]
    handled, _, output = handle_repl_command(second, "/clear")
    assert handled is True
    assert output.startswith("new session ")
    assert second.session["id"] != old_id
    assert second.current_task_state is None
    assert second.current_run_id == ""
    assert second.current_run_dir is None
    assert second.session_store.path(old_id).exists()


def test_resume_rejects_path_traversal_session_id(tmp_path):
    from lite.cli import handle_repl_command

    agent = build_agent(tmp_path, [])

    handled, _, output = handle_repl_command(agent, "/resume ../outside")

    assert handled is True
    assert output == "error: session not found"


def test_session_store_rejects_path_traversal_ids(tmp_path):
    store = SessionStore(tmp_path / ".lite" / "sessions")

    with pytest.raises(ValueError, match="invalid session id"):
        store.load("../outside")
