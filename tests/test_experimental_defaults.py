from lite import Lite, SessionStore, WorkspaceContext
from lite.cli import build_agent as build_cli_agent, build_arg_parser
from lite.config import resolve_experimental_features
from lite.testing import ScriptedModelClient


def build_agent(tmp_path, **kwargs):
    (tmp_path / "README.md").write_text("demo\n", encoding="utf-8")
    return Lite(
        model_client=ScriptedModelClient([]),
        workspace=WorkspaceContext.build(tmp_path),
        session_store=SessionStore(tmp_path / ".lite" / "sessions"),
        approval_policy="auto",
        **kwargs,
    )


def test_negative_return_experiments_are_disabled_by_default(tmp_path):
    agent = build_agent(tmp_path)

    assert agent.feature_enabled("multi_agent") is False
    assert agent.feature_enabled("durable_memory_retrieval") is False
    assert agent.feature_enabled("frozen_base_context") is True
    assert agent.auto_dream is False
    assert {"agent", "send_message", "task_stop"}.isdisjoint(agent.tools)


def test_experimental_flags_can_be_explicitly_enabled(tmp_path):
    agent = build_agent(
        tmp_path,
        feature_flags={"multi_agent": True, "durable_memory_retrieval": True},
        auto_dream=True,
    )

    assert agent.feature_enabled("multi_agent") is True
    assert {"agent", "send_message", "task_stop"}.issubset(agent.tools)
    assert agent.auto_dream is True


def test_experimental_config_is_backward_compatible_and_explicit(tmp_path):
    assert resolve_experimental_features(tmp_path) == {
        "multi_agent": False,
        "auto_dream": False,
        "durable_memory_retrieval": False,
    }
    (tmp_path / ".lite.toml").write_text(
        "[experimental]\n"
        "multi_agent = true\n"
        "auto_dream = true\n"
        "durable_memory_retrieval = true\n",
        encoding="utf-8",
    )

    assert resolve_experimental_features(tmp_path) == {
        "multi_agent": True,
        "auto_dream": True,
        "durable_memory_retrieval": True,
    }


def test_cli_builds_runtime_from_project_experimental_config(tmp_path):
    (tmp_path / ".lite.toml").write_text(
        "[experimental]\n"
        "multi_agent = true\n"
        "auto_dream = true\n"
        "durable_memory_retrieval = true\n",
        encoding="utf-8",
    )
    args = build_arg_parser().parse_args(
        ["--cwd", str(tmp_path), "--approval", "auto"]
    )

    agent = build_cli_agent(args)
    try:
        assert agent.feature_enabled("multi_agent") is True
        assert agent.feature_enabled("durable_memory_retrieval") is True
        assert agent.auto_dream is True
        assert {"agent", "send_message", "task_stop"}.issubset(agent.tools)
    finally:
        agent.close()
