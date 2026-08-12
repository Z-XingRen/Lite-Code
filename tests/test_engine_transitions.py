"""Engine acceptance tests for loop transition trace evidence."""

import json

from lite import Lite, SessionStore, WorkspaceContext
from lite.core.task_state import TaskState
from lite.core.turn_transitions import emit_terminal_transition
from lite.providers import ProviderError
from lite.testing import ScriptedModelClient


def build_agent(tmp_path, outputs, **kwargs):
    (tmp_path / "README.md").write_text("demo\n", encoding="utf-8")
    workspace = WorkspaceContext.build(tmp_path)
    store = SessionStore(tmp_path / ".lite" / "sessions")
    kwargs.setdefault("feature_flags", {"multi_agent": True})
    return Lite(
        model_client=ScriptedModelClient(outputs),
        workspace=workspace,
        session_store=store,
        approval_policy="auto",
        **kwargs,
    )


def read_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_engine_records_loop_transitions_without_changing_stream(tmp_path):
    agent = build_agent(
        tmp_path,
        [
            {'name': 'write_file', 'args': {'path': 'notes/result.txt', 'content': 'ok\n'}},
            "Wrote it.",
        ],
    )

    events = list(agent.engine.run_turn("create the result file"))

    assert [event["type"] for event in events] == [
        "turn_started",
        "model_requested",
        "model_parsed",
        "tool_call",
        "tool_result",
        "model_requested",
        "model_parsed",
        "final",
        "turn_finished",
    ]
    trace_events = read_jsonl(agent.current_run_dir / "trace.jsonl")
    transitions = [event for event in trace_events if event["event"] == "loop_transition"]
    assert [event["reason"] for event in transitions] == [
        "tool_batch_executed",
        "final_answer_returned",
    ]

    report = json.loads(
        (agent.current_run_dir / "report.json").read_text(encoding="utf-8")
    )
    assert report["evidence_summaries"]["transition_summary"] == {
        "continue_count": 1,
        "terminal_count": 1,
        "terminal_reason": "final_answer_returned",
        "schema_version": "lite.transition_summary.v1",
        "reasons": {
            "tool_batch_executed": 1,
            "final_answer_returned": 1,
        },
        "max_attempt_index": 2,
        "tool_requested_count": 1,
        "tool_executed_count": 1,
    }


def test_runtime_consumer_errors_are_visible_in_task_state(tmp_path):
    agent = build_agent(tmp_path, [])
    task_state = TaskState.create(
        task_id=agent.new_task_id(),
        run_id=agent.new_run_id(),
        user_request="exercise consumer errors",
    )
    agent.current_run_dir = agent.run_store.start_run(task_state)

    emit_terminal_transition(
        agent,
        task_state,
        reason="final_answer_returned",
        stop_reason="final_answer_returned",
    )
    emit_terminal_transition(
        agent,
        task_state,
        reason="retry_limit_reached",
        stop_reason="retry_limit_reached",
    )

    errors = task_state.evidence_summaries["consumer_errors"]
    assert errors[-1]["consumer"] == "EvidenceSummaryConsumer"
    assert errors[-1]["critical"] is True
    assert errors[-1]["event"] == "loop_transition"
    assert "terminal transition" in errors[-1]["message"]


def test_noncritical_runtime_consumer_errors_are_separated(tmp_path):
    class NonCriticalConsumer:
        def handle(self, runtime, task_state, event):
            raise RuntimeError("optional projection failed")

    agent = build_agent(tmp_path, [])
    agent.runtime_consumers = [NonCriticalConsumer()]
    task_state = TaskState.create(
        task_id=agent.new_task_id(),
        run_id=agent.new_run_id(),
        user_request="exercise noncritical consumer errors",
    )
    agent.current_run_dir = agent.run_store.start_run(task_state)

    agent.emit_trace(task_state, "prompt_built", {"prompt_metadata": {}})

    errors = task_state.evidence_summaries["runtime_consumer_errors"]
    assert errors[-1]["consumer"] == "NonCriticalConsumer"
    assert errors[-1]["critical"] is False
    assert "consumer_errors" not in task_state.evidence_summaries


def test_engine_executes_multiple_tool_calls_from_one_model_response(tmp_path):
    agent = build_agent(
        tmp_path,
        [
            [
                {'name': 'read_file', 'args': {'path': 'README.md', 'start': 1, 'end': 1}},
                {'name': 'list_files', 'args': {'path': '.'}},
            ],
            "Both tools ran.",
        ],
    )

    events = list(agent.engine.run_turn("inspect the workspace"))

    assert [event["type"] for event in events if event["type"] == "tool_call"] == [
        "tool_call",
        "tool_call",
    ]
    tool_history = [item for item in agent.session["history"] if item["role"] == "tool"]
    assert [item["name"] for item in tool_history] == ["read_file", "list_files"]
    assert events[-2]["content"] == "Both tools ran."
    trace = read_jsonl(agent.current_run_dir / "trace.jsonl")
    transition = next(
        event
        for event in trace
        if event["event"] == "loop_transition" and event["reason"] == "tool_batch_executed"
    )
    assert transition["tool_requested_count"] == 2
    assert transition["tool_executed_count"] == 2


def test_multi_tool_transition_distinguishes_requested_and_executed_counts(tmp_path):
    agent = build_agent(
        tmp_path,
        [
            [
                {'name': 'read_file', 'args': {'path': 'README.md', 'start': 1, 'end': 1}},
                {'name': 'list_files', 'args': {'path': '.'}},
            ],
            "not reached",
        ],
        max_steps=1,
    )

    events = list(agent.engine.run_turn("inspect with too many tools"))

    assert [event["type"] for event in events if event["type"] == "tool_call"] == [
        "tool_call"
    ]
    trace = read_jsonl(agent.current_run_dir / "trace.jsonl")
    transition = next(
        event
        for event in trace
        if event["event"] == "loop_transition" and event["reason"] == "tool_batch_executed"
    )
    assert transition["tool_requested_count"] == 2
    assert transition["tool_executed_count"] == 1


def test_empty_response_provider_error_is_retried_once_before_failing(tmp_path):
    agent = build_agent(
        tmp_path,
        [
            ProviderError(
                "empty provider response",
                provider="anthropic",
                model="deepseek-v4-pro",
                base_url="https://api.deepseek.com/anthropic/v1",
                code="empty_response",
                retryable=False,
            ),
            "Recovered.",
        ],
    )

    events = list(agent.engine.run_turn("recover from provider empty response"))

    assert events[-2]["content"] == "Recovered."
    persisted_events = read_jsonl(agent.session_event_bus.path)
    assert any(
        event["event"] == "model_retry_scheduled" and event["code"] == "empty_response"
        for event in persisted_events
    )
    trace = read_jsonl(agent.current_run_dir / "trace.jsonl")
    transitions = [event for event in trace if event["event"] == "loop_transition"]
    assert [event["reason"] for event in transitions] == [
        "provider_retry",
        "final_answer_returned",
    ]
    assert [event["type"] for event in events] == [
        "turn_started",
        "model_requested",
        "model_requested",
        "model_parsed",
        "final",
        "turn_finished",
    ]


def test_empty_native_response_retry_preserves_stream_order(tmp_path):
    agent = build_agent(
        tmp_path,
        [
            "",
            "Recovered.",
        ],
    )

    events = list(agent.engine.run_turn("recover from parse retry"))

    assert [event["type"] for event in events] == [
        "turn_started",
        "model_requested",
        "model_parsed",
        "retry",
        "model_requested",
        "model_parsed",
        "final",
        "turn_finished",
    ]
    trace = read_jsonl(agent.current_run_dir / "trace.jsonl")
    transitions = [event for event in trace if event["event"] == "loop_transition"]
    assert [event["reason"] for event in transitions] == [
        "empty_response_retry",
        "final_answer_returned",
    ]


def test_retry_limit_transition_is_terminal(tmp_path):
    agent = build_agent(
        tmp_path,
        ["", "", ""],
        max_steps=1,
    )

    events = list(agent.engine.run_turn("hit retry limit"))

    assert events[-1]["stop_reason"] == "retry_limit_reached"
    trace = read_jsonl(agent.current_run_dir / "trace.jsonl")
    transitions = [event for event in trace if event["event"] == "loop_transition"]
    assert [event["reason"] for event in transitions] == [
        "empty_response_retry",
        "empty_response_retry",
        "empty_response_retry",
        "retry_limit_reached",
    ]
    assert transitions[-1]["kind"] == "terminal"


def test_plan_notice_transition_preserves_runtime_notice_stream_order(tmp_path):
    agent = build_agent(
        tmp_path,
        [
            "Looks done.",
            {'name': 'write_file', 'args': {'path': '.lite/plans/v3-plan.md', 'content': '# Plan\n'}},
            "Now done.",
        ],
        max_steps=3,
    )
    agent.enter_plan_mode("v3")

    events = list(agent.engine.run_turn("make a plan"))

    assert [event["type"] for event in events] == [
        "turn_started",
        "model_requested",
        "model_parsed",
        "runtime_notice",
        "model_requested",
        "model_parsed",
        "tool_call",
        "tool_result",
        "model_requested",
        "model_parsed",
        "final",
        "turn_finished",
    ]
    trace = read_jsonl(agent.current_run_dir / "trace.jsonl")
    transitions = [event for event in trace if event["event"] == "loop_transition"]
    assert [event["reason"] for event in transitions] == [
        "plan_notice",
        "tool_batch_executed",
        "final_answer_returned",
    ]


def test_step_limit_uses_a_tools_disabled_final_only_request(tmp_path):
    agent = build_agent(
        tmp_path,
        [
            {'name': 'list_files', 'args': {'path': '.'}},
            "已经列出文件。还差读取具体内容。继续请用 /resume。",
        ],
        max_steps=1,
    )

    events = list(agent.engine.run_turn("trigger step limit"))

    final_event = next(e for e in events if e["type"] == "final")
    assert "已经列出文件" in final_event["content"]
    assert events[-1]["stop_reason"] == "final_answer_returned"
    assert len(agent.model_client.requests) == 2
    assert agent.model_client.requests[0].tools
    assert agent.model_client.requests[1].tools == ()

    requested = [e for e in events if e["type"] == "model_requested"]
    assert [e["final_only"] for e in requested] == [False, True]
    trace = read_jsonl(agent.current_run_dir / "trace.jsonl")
    trace_requested = [e for e in trace if e["event"] == "model_requested"]
    assert [e["final_only"] for e in trace_requested] == [False, True]
    assert [
        e["reason"] for e in trace if e["event"] == "loop_transition"
    ] == ["tool_batch_executed", "final_answer_returned"]


def test_step_limit_falls_back_to_cold_message_when_summary_fails(tmp_path):
    agent = build_agent(
        tmp_path,
        [
            {'name': 'list_files', 'args': {'path': '.'}},
            "",
        ],
        max_steps=1,
    )

    events = list(agent.engine.run_turn("trigger step limit"))

    stop_event = next(e for e in events if e["type"] == "stop")
    assert "Stopped after reaching the step limit" in stop_event["content"]
