import json
import threading
import time

from lite import Lite, SessionStore, WorkspaceContext
from lite.core.tool_profiles import build_tool_profiles
from lite.providers import ModelResult, ToolCall
from lite.testing import ScriptedModelClient
from lite.tools.base import RegisteredTool


def build_agent(tmp_path, outputs, **kwargs):
    (tmp_path / "README.md").write_text("batch scheduler\n", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("notes\n", encoding="utf-8")
    return Lite(
        model_client=ScriptedModelClient(outputs),
        workspace=WorkspaceContext.build(tmp_path),
        session_store=SessionStore(tmp_path / ".lite" / "sessions"),
        approval_policy="auto",
        auto_dream=False,
        **kwargs,
    )


def replace_tool(agent, name, runner, *, execution_mode, effect_class):
    original = agent.tools[name]
    agent.tools[name] = RegisteredTool(
        name=name,
        schema=original.schema,
        description=original.description,
        risky=original.risky,
        runner=runner,
        execution_mode=execution_mode,
        effect_class=effect_class,
    )
    agent.tool_profiles = build_tool_profiles(agent.tools)


def tool_batch(*calls):
    return ModelResult(tool_calls=tuple(calls), stop_reason="tool_use")


def read_calls():
    return (
        ToolCall(
            "call_read",
            "read_file",
            {"path": "README.md", "start": 1, "end": 1},
        ),
        ToolCall("call_list", "list_files", {"path": "."}),
    )


def read_events(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_registered_tools_expose_conservative_effect_classification(tmp_path):
    agent = build_agent(tmp_path, [])

    assert {
        name: (agent.tools[name].execution_mode, agent.tools[name].effect_class)
        for name in ("list_files", "read_file", "search", "todo_list")
    } == {
        "list_files": ("parallel", "read_only"),
        "read_file": ("parallel", "read_only"),
        "search": ("parallel", "read_only"),
        "todo_list": ("parallel", "read_only"),
    }
    assert agent.tools["write_file"].effect_class == "mutating"
    assert agent.tools["patch_file"].effect_class == "mutating"
    assert agent.tools["run_shell"].effect_class == "opaque"
    assert agent.tools["run_shell"].execution_mode == "sequential"
    assert agent.tools["verify"].effect_class == "opaque"
    assert agent.tools["verify"].execution_mode == "sequential"
    assert "verify" not in agent.tool_profiles["worker"].allowed_tools
    assert all(
        tool.execution_mode == "sequential"
        for name, tool in agent.tools.items()
        if name not in {"list_files", "read_file", "search", "todo_list"}
    )


def test_read_only_batch_executes_concurrently_but_commits_source_order(tmp_path):
    calls = read_calls()
    agent = build_agent(tmp_path, [tool_batch(*calls), "Done."])
    both_started = threading.Event()
    second_finished = threading.Event()
    lock = threading.Lock()
    active = 0
    max_active = 0
    completion_order = []

    def start():
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
            if active == 2:
                both_started.set()

    def finish(name):
        nonlocal active
        with lock:
            completion_order.append(name)
            active -= 1

    def delayed_read(args):
        start()
        both_started.wait(timeout=1)
        second_finished.wait(timeout=1)
        finish("read_file")
        return f"read:{args['path']}"

    def fast_list(args):
        start()
        both_started.wait(timeout=1)
        finish("list_files")
        second_finished.set()
        return f"list:{args['path']}"

    replace_tool(
        agent,
        "read_file",
        delayed_read,
        execution_mode="parallel",
        effect_class="read_only",
    )
    replace_tool(
        agent,
        "list_files",
        fast_list,
        execution_mode="parallel",
        effect_class="read_only",
    )

    events = list(agent.engine.run_turn("inspect in parallel"))

    assert max_active == 2
    assert completion_order == ["list_files", "read_file"]
    results = [event for event in events if event["type"] == "tool_result"]
    assert [event["call_id"] for event in results] == ["call_read", "call_list"]
    history = [item for item in agent.session["history"] if item["role"] == "tool"]
    assert [item["call_id"] for item in history] == ["call_read", "call_list"]
    tool_events = [
        event
        for event in read_events(agent.session_event_bus.path)
        if event["event"] in {"tool_started", "tool_finished"}
    ]
    assert [event["tool_name"] for event in tool_events] == [
        "read_file",
        "list_files",
        "read_file",
        "list_files",
    ]
    journal = read_events(agent.session_path)
    batch_intents = [
        record
        for record in journal
        if record["kind"] == "effect_intent"
        and "calls" in record["payload"]["request"]
    ]
    assert len(batch_intents) == 1
    assert [
        call["call_id"] for call in batch_intents[0]["payload"]["request"]["calls"]
    ] == ["call_read", "call_list"]
    assert len(
        [
            record
            for record in journal
            if record["kind"] == "effect_result"
            and record["operation_id"] == batch_intents[0]["operation_id"]
        ]
    ) == 1


def test_mutating_and_opaque_batch_remains_fully_serial(tmp_path):
    path = tmp_path / "owned.txt"
    calls = (
        ToolCall("call_write_1", "write_file", {"path": "owned.txt", "content": "one"}),
        ToolCall(
            "call_read",
            "read_file",
            {"path": "README.md", "start": 1, "end": 1},
        ),
        ToolCall("call_write_2", "write_file", {"path": "owned.txt", "content": "two"}),
        ToolCall("call_shell", "run_shell", {"command": "opaque", "timeout": 20}),
    )
    agent = build_agent(tmp_path, [tool_batch(*calls), "Done."])
    lock = threading.Lock()
    active = 0
    max_active = 0
    invocation_order = []

    def enter(name):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
            invocation_order.append(name)

    def leave():
        nonlocal active
        with lock:
            active -= 1

    def write(args):
        enter("write_file")
        time.sleep(0.02)
        path.write_text(args["content"], encoding="utf-8")
        leave()
        return f"wrote {args['content']}"

    def shell(_args):
        enter("run_shell")
        time.sleep(0.02)
        leave()
        return "exit_code: 0\nstdout:\nok\nstderr:\n(empty)"

    def read(_args):
        enter("read_file")
        time.sleep(0.02)
        leave()
        return "read ok"

    replace_tool(
        agent,
        "write_file",
        write,
        execution_mode="sequential",
        effect_class="mutating",
    )
    replace_tool(
        agent,
        "run_shell",
        shell,
        execution_mode="sequential",
        effect_class="opaque",
    )
    replace_tool(
        agent,
        "read_file",
        read,
        execution_mode="parallel",
        effect_class="read_only",
    )

    events = list(agent.engine.run_turn("serialize effects"))

    assert max_active == 1
    assert invocation_order == [
        "write_file",
        "read_file",
        "write_file",
        "run_shell",
    ]
    assert path.read_text(encoding="utf-8") == "two"
    assert [event["call_id"] for event in events if event["type"] == "tool_result"] == [
        "call_write_1",
        "call_read",
        "call_write_2",
        "call_shell",
    ]


def test_parallel_failure_waits_for_batch_and_commits_all_results(tmp_path):
    calls = read_calls()
    agent = build_agent(tmp_path, [tool_batch(*calls), "Done."])
    both_started = threading.Event()
    release = threading.Event()
    finished = []
    lock = threading.Lock()

    def started():
        with lock:
            finished.append("started")
            if finished.count("started") == 2:
                both_started.set()

    def failing_read(_args):
        started()
        both_started.wait(timeout=1)
        release.set()
        raise RuntimeError("controlled read failure")

    def successful_list(_args):
        started()
        release.wait(timeout=1)
        with lock:
            finished.append("list_finished")
        return "list ok"

    replace_tool(
        agent,
        "read_file",
        failing_read,
        execution_mode="parallel",
        effect_class="read_only",
    )
    replace_tool(
        agent,
        "list_files",
        successful_list,
        execution_mode="parallel",
        effect_class="read_only",
    )

    events = list(agent.engine.run_turn("handle one failure"))

    results = [event for event in events if event["type"] == "tool_result"]
    assert [event["call_id"] for event in results] == ["call_read", "call_list"]
    assert "controlled read failure" in results[0]["content"]
    assert results[1]["content"] == "list ok"
    assert "list_finished" in finished
    assert not any(
        thread.name.startswith("lite-tool-") for thread in threading.enumerate()
    )


def test_parallel_batch_cancellation_joins_every_tool_thread(tmp_path):
    calls = read_calls()
    agent = build_agent(tmp_path, [tool_batch(*calls), "must not be called"])
    both_started = threading.Event()
    lock = threading.Lock()
    active = 0
    finished = []
    events = []

    def cancellable(name):
        def runner(_args):
            nonlocal active
            with lock:
                active += 1
                if active == 2:
                    both_started.set()
            try:
                agent.current_cancellation_token.wait(timeout=2)
                agent.current_cancellation_token.raise_if_cancelled()
                return f"{name} unexpectedly completed"
            finally:
                with lock:
                    active -= 1
                    finished.append(name)

        return runner

    replace_tool(
        agent,
        "read_file",
        cancellable("read_file"),
        execution_mode="parallel",
        effect_class="read_only",
    )
    replace_tool(
        agent,
        "list_files",
        cancellable("list_files"),
        execution_mode="parallel",
        effect_class="read_only",
    )

    coordinator = threading.Thread(
        target=lambda: events.extend(agent.engine.run_turn("cancel the batch")),
        name="wp18-coordinator",
    )
    coordinator.start()
    started = both_started.wait(timeout=5)
    agent.abort_current_turn()
    coordinator.join(timeout=5)

    assert started is True
    assert not coordinator.is_alive()
    assert active == 0
    assert sorted(finished) == ["list_files", "read_file"]
    assert next(event for event in events if event["type"] == "stop")["content"] == (
        "Stopped after abort request."
    )
    assert not any(
        thread.name.startswith("lite-tool-") for thread in threading.enumerate()
    )
