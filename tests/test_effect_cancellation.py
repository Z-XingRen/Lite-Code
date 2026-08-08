import ctypes
import json
import os
import sys
import threading
import time

from lite import Lite, SessionStore, WorkspaceContext
from lite.testing import ScriptedModelClient, shell_join


def build_agent(tmp_path, outputs, **kwargs):
    (tmp_path / "README.md").write_text("demo\n", encoding="utf-8")
    return Lite(
        model_client=ScriptedModelClient(outputs),
        workspace=WorkspaceContext.build(tmp_path),
        session_store=SessionStore(tmp_path / ".lite" / "sessions"),
        approval_policy="auto",
        **kwargs,
    )


def wait_for_path(path, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return True
        time.sleep(0.01)
    return path.exists()


def process_is_running(pid):
    if os.name == "nt":
        synchronize = 0x00100000
        wait_timeout = 0x00000102
        handle = ctypes.windll.kernel32.OpenProcess(synchronize, False, int(pid))
        if not handle:
            return False
        try:
            return ctypes.windll.kernel32.WaitForSingleObject(handle, 0) == wait_timeout
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def wait_for_process_exit(pid, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not process_is_running(pid):
            return True
        time.sleep(0.01)
    return not process_is_running(pid)


def process_tree_command(tmp_path, *, late_delay):
    parent_pid = tmp_path / "parent.pid"
    child_pid = tmp_path / "child.pid"
    started = tmp_path / "child-started.txt"
    late_write = tmp_path / "late-write.txt"
    child_script = (
        "import os,time; from pathlib import Path; "
        f"Path({str(child_pid)!r}).write_text(str(os.getpid()), encoding='utf-8'); "
        f"Path({str(started)!r}).write_text('started', encoding='utf-8'); "
        f"time.sleep({float(late_delay)!r}); "
        f"Path({str(late_write)!r}).write_text('late', encoding='utf-8')"
    )
    parent_script = (
        "import os,subprocess,sys; from pathlib import Path; "
        f"Path({str(parent_pid)!r}).write_text(str(os.getpid()), encoding='utf-8'); "
        f"child=subprocess.Popen([sys.executable, '-c', {child_script!r}]); "
        "child.wait()"
    )
    return (
        shell_join([sys.executable, "-c", parent_script]),
        parent_pid,
        child_pid,
        started,
        late_write,
    )


def read_pid(path):
    return int(path.read_text(encoding="utf-8").strip())


class BlockingWorkerClient:
    supports_prompt_cache = False
    last_completion_metadata = {}

    def __init__(self, started, release, late_write):
        self.started = started
        self.release = release
        self.late_write = late_write
        self.abort_count = 0

    def complete(self, prompt, max_new_tokens, **kwargs):
        del prompt, max_new_tokens, kwargs
        self.started.set()
        if not self.release.wait(timeout=5):
            raise RuntimeError("blocking worker client timed out")
        if not self.abort_count:
            self.late_write.write_text("late", encoding="utf-8")
        return "<final>Worker finished.</final>"

    def abort(self):
        self.abort_count += 1
        self.release.set()


def test_runtime_abort_terminates_shell_tree_and_history_can_resume(tmp_path):
    command, parent_pid_path, child_pid_path, started, late_write = (
        process_tree_command(tmp_path, late_delay=1.2)
    )
    agent = build_agent(
        tmp_path,
        [
            f'<tool>{{"name":"run_shell","args":{{"command":{json.dumps(command)},"timeout":20}}}}</tool>',
            "<final>Resumed cleanly.</final>",
        ],
        max_steps=2,
    )
    outcome = {}
    baseline_reader_ids = {
        thread.ident
        for thread in threading.enumerate()
        if "_readerthread" in thread.name
    }
    turn_thread = threading.Thread(
        target=lambda: outcome.setdefault("result", agent.ask("run until aborted")),
        daemon=True,
        name="test-shell-turn",
    )
    turn_thread.start()

    assert wait_for_path(started)
    agent.abort_current_turn()
    turn_thread.join(timeout=3)

    assert not turn_thread.is_alive()
    assert outcome["result"] == "Stopped after abort request."
    assert wait_for_process_exit(read_pid(parent_pid_path))
    assert wait_for_process_exit(read_pid(child_pid_path))
    time.sleep(1.3)
    assert not late_write.exists()
    assert not [
        thread
        for thread in threading.enumerate()
        if "_readerthread" in thread.name and thread.ident not in baseline_reader_ids
    ]
    tool_item = next(
        item
        for item in agent.session["history"]
        if item.get("role") == "tool" and item.get("name") == "run_shell"
    )
    assert "cancel" in tool_item["content"].lower()
    assert tool_item["tool_error_code"] == "tool_cancelled"
    assert agent.ask("continue after the cancelled shell") == "Resumed cleanly."


def test_shell_timeout_terminates_descendants_before_they_can_write(tmp_path):
    command, parent_pid_path, child_pid_path, started, late_write = (
        process_tree_command(tmp_path, late_delay=3.0)
    )
    agent = build_agent(tmp_path, [])

    result = agent.run_tool("run_shell", {"command": command, "timeout": 2})

    assert wait_for_path(started)
    assert "timed out" in result.lower()
    assert agent._last_tool_result_metadata["tool_error_code"] == "tool_timeout"
    assert wait_for_process_exit(read_pid(parent_pid_path))
    assert wait_for_process_exit(read_pid(child_pid_path))
    time.sleep(3.1)
    assert not late_write.exists()


def test_parent_abort_stops_background_worker_and_joins_its_thread(tmp_path):
    started = threading.Event()
    release = threading.Event()
    late_write = tmp_path / "worker-late-write.txt"
    child_client = BlockingWorkerClient(
        started,
        release,
        late_write,
    )
    agent = build_agent(tmp_path, [], model_client_factory=lambda: child_client)
    payload = json.loads(
        agent.run_tool(
            "agent",
            {
                "description": "Cancelable background worker",
                "prompt": "wait until cancellation",
                "subagent_type": "worker",
                "write_scope": ["."],
            },
        )
    )

    assert payload["status"] == "started"
    assert started.wait(timeout=2)
    task = agent.worker_manager._tasks[payload["task_id"]]
    agent.abort_current_turn()
    task.thread.join(timeout=3)

    assert not task.thread.is_alive()
    assert child_client.abort_count == 1
    assert agent.worker_manager.to_dict()["items"][0]["status"] == "stopped"
    assert agent.worker_manager.shutdown(timeout=0)["live_task_ids"] == []
    assert not late_write.exists()
    assert not [
        thread
        for thread in threading.enumerate()
        if thread.name.startswith("lite-worker-") and thread.is_alive()
    ]
