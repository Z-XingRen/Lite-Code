"""Reproducible Phase 0 benchmarks for the legacy agent runtime."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from lite import Lite, SessionStore, WorkspaceContext
from lite.core.engine_helpers import execute_native_tool_calls
from lite.core.run_store import RunStore
from lite.core.runtime_checkpoints import RuntimeCheckpointsMixin
from lite.core.task_state import TaskState
from lite.core.tool_result_artifacts import prepare_tool_result_observation
from lite.providers.base import ModelConversation, ModelResult, ToolCall
from lite.testing import ScriptedModelClient


def percentile(values: list[float], percent: float) -> float:
    """Return a linearly interpolated percentile for a non-empty sample."""
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percent
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def timing_summary(seconds: list[float]) -> dict:
    return {
        "samples": len(seconds),
        "median_ms": round(statistics.median(seconds) * 1000, 3),
        "p95_ms": round(percentile(seconds, 0.95) * 1000, 3),
    }


class SnapshotRuntime(RuntimeCheckpointsMixin):
    def __init__(self, root: Path):
        self.root = root


def _write_workspace(root: Path, file_count: int, file_bytes: int = 128) -> None:
    payload = b"x" * file_bytes
    for index in range(file_count):
        directory = root / f"dir-{index // 100:03d}"
        directory.mkdir(exist_ok=True)
        (directory / f"file-{index:05d}.txt").write_bytes(payload)


def _workspace_cycle(
    runtime: SnapshotRuntime,
    paths: list[Path],
    changed_count: int,
    iteration: int,
) -> tuple[float, int, int, int]:
    reads = 0
    read_bytes = 0
    original_read_bytes = Path.read_bytes

    def counted_read_bytes(path: Path) -> bytes:
        nonlocal reads, read_bytes
        payload = original_read_bytes(path)
        reads += 1
        read_bytes += len(payload)
        return payload

    started = time.perf_counter()
    Path.read_bytes = counted_read_bytes
    try:
        before = runtime.capture_workspace_snapshot()
        marker = bytes([65 + (iteration % 26)])
        for path in paths[:changed_count]:
            path.write_bytes(marker * 128)
        after = runtime.capture_workspace_snapshot()
    finally:
        Path.read_bytes = original_read_bytes
    changed_paths, _ = runtime.diff_workspace_snapshots(before, after)
    return time.perf_counter() - started, reads, read_bytes, len(changed_paths)


def benchmark_workspace(file_count: int, runs: int) -> dict:
    with tempfile.TemporaryDirectory(prefix="lite-wp00-workspace-") as temp_dir:
        root = Path(temp_dir)
        _write_workspace(root, file_count)
        paths = sorted(root.rglob("*.txt"))
        runtime = SnapshotRuntime(root)
        scenarios = {}
        for changed_count in (1, 10, 100):
            _workspace_cycle(runtime, paths, changed_count, 0)
            durations = []
            read_counts = []
            read_byte_counts = []
            observed_changes = []
            for iteration in range(1, runs + 1):
                duration, reads, read_bytes, changes = _workspace_cycle(
                    runtime, paths, changed_count, iteration
                )
                durations.append(duration)
                read_counts.append(reads)
                read_byte_counts.append(read_bytes)
                observed_changes.append(changes)
            scenarios[str(changed_count)] = {
                **timing_summary(durations),
                "read_files_median": int(statistics.median(read_counts)),
                "read_files_p95": round(percentile(read_counts, 0.95), 3),
                "read_bytes_median": int(statistics.median(read_byte_counts)),
                "read_bytes_p95": round(percentile(read_byte_counts, 0.95), 3),
                "observed_changed_paths": sorted(set(observed_changes)),
            }
        return {
            "file_count": file_count,
            "file_bytes": 128,
            "warmup_runs": 1,
            "measured_runs": runs,
            "cycle": "snapshot -> mutate -> snapshot -> diff",
            "scenarios_by_changed_file_count": scenarios,
        }


def benchmark_session(record_count: int, recovery_runs: int = 20) -> dict:
    with tempfile.TemporaryDirectory(prefix="lite-wp00-session-") as temp_dir:
        store = SessionStore(Path(temp_dir) / "sessions")
        session = {
            "id": "wp00-baseline",
            "created_at": "2026-08-07T00:00:00+00:00",
            "workspace_root": temp_dir,
            "history": [],
        }
        append_durations = []
        cumulative_write_bytes = 0
        for index in range(record_count):
            session["history"].append(
                {
                    "role": "assistant" if index % 2 else "user",
                    "content": f"record-{index:04d}-" + ("x" * 256),
                    "event_id": f"event_{index:06d}",
                }
            )
            started = time.perf_counter()
            path = store.save(session)
            append_durations.append(time.perf_counter() - started)
            cumulative_write_bytes += path.stat().st_size

        recovery_durations = []
        for _ in range(recovery_runs):
            started = time.perf_counter()
            loaded = store.load(session["id"])
            recovery_durations.append(time.perf_counter() - started)
            if len(loaded["history"]) != record_count:
                raise RuntimeError("session recovery returned an incomplete history")

        return {
            "history_records": record_count,
            "record_content_chars": 268,
            "append": timing_summary(append_durations),
            "cumulative_write_bytes": cumulative_write_bytes,
            "final_session_bytes": path.stat().st_size,
            "recovery": timing_summary(recovery_durations),
            "persistence_mode": "full JSON rewrite per append",
        }


def _build_output_agent(root: Path) -> Lite:
    (root / "README.md").write_text("benchmark\n", encoding="utf-8")
    return Lite(
        model_client=ScriptedModelClient([]),
        workspace=WorkspaceContext.build(root),
        session_store=SessionStore(root / ".lite" / "sessions"),
        run_store=RunStore(root / ".lite" / "runs"),
        approval_policy="auto",
        auto_dream=False,
    )


def _output_samples() -> dict[str, str]:
    return {
        "empty": "",
        "long_single_line": "HEAD-" + ("x" * 11990) + "-TAIL",
        "multiline": "\n".join(
            ["HEAD"]
            + [f"line-{index:03d}-" + ("x" * 48) for index in range(300)]
            + ["TAIL"]
        ),
        "utf8": "HEAD-" + ("\u6c49\u5b57\u6d4b\u8bd5" * 2000) + "-TAIL",
        "traceback": "\n".join(
            ["HEAD", "Traceback (most recent call last):"]
            + [f'  File "module_{index}.py", line {index}, in function' for index in range(100)]
            + ["RuntimeError: TAIL"]
        ),
    }


def benchmark_tool_output_context(runs: int) -> dict:
    with tempfile.TemporaryDirectory(prefix="lite-wp00-output-") as temp_dir:
        root = Path(temp_dir)
        agent = _build_output_agent(root)
        results = {}
        for name, full_output in _output_samples().items():
            prepare_durations = []
            context_durations = []
            observation = ""
            metadata = {}
            context_metadata = {}
            for iteration in range(runs + 1):
                task_state = TaskState.create(
                    run_id=f"run_{name}_{iteration}",
                    task_id=f"task_{name}_{iteration}",
                    user_request="benchmark tool output",
                )
                agent.current_task_state = task_state
                agent.current_run_dir = agent.run_store.start_run(task_state)
                started = time.perf_counter()
                observation, metadata = prepare_tool_result_observation(
                    agent, "run_shell", full_output
                )
                prepare_duration = time.perf_counter() - started
                history_item = {
                    "role": "tool",
                    "name": "run_shell",
                    "args": {"command": "benchmark"},
                    "content": observation,
                    "turn_id": f"turn_{iteration}",
                    "event_id": f"event_{iteration}",
                    "tool_status": "ok",
                    "tool_error_code": "",
                    "workspace_changed": False,
                    **metadata,
                }
                agent.session["history"] = [history_item]
                started = time.perf_counter()
                _, context_metadata = agent.context_manager.build("continue")
                context_duration = time.perf_counter() - started
                if iteration:
                    prepare_durations.append(prepare_duration)
                    context_durations.append(context_duration)

            artifact_ref = str(metadata.get("full_output_artifact", ""))
            artifact_path = root / artifact_ref if artifact_ref else None
            artifact_bytes = artifact_path.stat().st_size if artifact_path else 0
            artifact_matches = (
                artifact_path.read_text(encoding="utf-8") == full_output
                if artifact_path
                else not full_output
            )
            usage = context_metadata["context_usage"]
            results[name] = {
                "original_chars": len(full_output),
                "original_bytes": len(full_output.encode("utf-8")),
                "inline_chars": len(observation),
                "inline_bytes": len(observation.encode("utf-8")),
                "inline_lines": len(observation.splitlines()),
                "head_marker_preserved": "HEAD" in observation,
                "tail_marker_preserved": "TAIL" in observation,
                "artifact_bytes": artifact_bytes,
                "artifact_matches_original": artifact_matches,
                "context_total_estimated_tokens": usage["total_estimated_tokens"],
                "context_history_estimated_tokens": usage["sections"]["history"][
                    "tokens"
                ],
                "prepare": timing_summary(prepare_durations),
                "context_build": timing_summary(context_durations),
            }
        return {
            "warmup_runs": 1,
            "measured_runs": runs,
            "tool_name": "run_shell",
            "samples": results,
        }


class _NoopEventBus:
    @staticmethod
    def emit(_event: str, _payload: dict) -> None:
        return None


class _NoopRunStore:
    @staticmethod
    def write_task_state(_task_state: TaskState) -> None:
        return None


class BatchRuntime:
    def __init__(self, delay_seconds: float):
        self.delay_seconds = delay_seconds
        self.max_steps = 50
        self.abort_requested = False
        self.current_run_id = "run_batch"
        self._last_tool_result_metadata = {}
        self.session_event_bus = _NoopEventBus()
        self.run_store = _NoopRunStore()
        self.history = []

    def run_tool(self, name: str, args: dict) -> str:
        time.sleep(self.delay_seconds)
        self._last_tool_result_metadata = {
            "tool_status": "ok",
            "tool_error_code": "",
            "workspace_changed": False,
            "affected_paths": [],
        }
        return f"{name}:{args['index']}"

    def record(self, item: dict) -> None:
        self.history.append(item)

    @staticmethod
    def emit_trace(_task_state: TaskState, _event: str, _payload: dict) -> None:
        return None

    @staticmethod
    def create_checkpoint(
        _task_state: TaskState, _user_message: str, trigger: str
    ) -> dict:
        return {"checkpoint_id": f"checkpoint-{trigger}"}


class BatchEngine:
    def __init__(self, runtime: BatchRuntime):
        self.runtime = runtime

    @staticmethod
    def drain_worker_notifications() -> tuple:
        return ()


def _drain_generator(generator) -> tuple[list[dict], tuple]:
    events = []
    while True:
        try:
            events.append(next(generator))
        except StopIteration as exc:
            return events, exc.value


def _run_tool_batch(call_count: int, delay_seconds: float) -> tuple[float, list[int]]:
    runtime = BatchRuntime(delay_seconds)
    engine = BatchEngine(runtime)
    task_state = TaskState.create(
        run_id="run_batch", task_id="task_batch", user_request="benchmark batch"
    )
    calls = tuple(
        ToolCall(call_id=f"call-{index}", name="delayed_read", arguments={"index": index})
        for index in range(call_count)
    )
    result = ModelResult(tool_calls=calls)
    conversation = ModelConversation(initial_input="benchmark")
    started = time.perf_counter()
    events, _ = _drain_generator(
        execute_native_tool_calls(
            engine, task_state, "benchmark batch", conversation, result, 0
        )
    )
    duration = time.perf_counter() - started
    result_order = [
        int(event["content"].split(":", 1)[1])
        for event in events
        if event["type"] == "tool_result"
    ]
    return duration, result_order


def _run_abort_batch(delay_seconds: float, abort_after_seconds: float) -> float:
    runtime = BatchRuntime(delay_seconds)
    engine = BatchEngine(runtime)
    task_state = TaskState.create(
        run_id="run_abort", task_id="task_abort", user_request="benchmark abort"
    )
    result = ModelResult(
        tool_calls=(
            ToolCall(call_id="call-0", name="delayed_read", arguments={"index": 0}),
            ToolCall(call_id="call-1", name="delayed_read", arguments={"index": 1}),
        )
    )
    timer = threading.Timer(
        abort_after_seconds, lambda: setattr(runtime, "abort_requested", True)
    )
    timer.start()
    started = time.perf_counter()
    try:
        _drain_generator(
            execute_native_tool_calls(
                engine,
                task_state,
                "benchmark abort",
                ModelConversation(initial_input="benchmark"),
                result,
                0,
            )
        )
    finally:
        timer.join()
    return time.perf_counter() - started


def benchmark_tool_batch(runs: int, delay_ms: int) -> dict:
    delay_seconds = delay_ms / 1000
    scenarios = {}
    for call_count in (2, 4, 8):
        _run_tool_batch(call_count, delay_seconds)
        durations = []
        observed_orders = []
        for _ in range(runs):
            duration, order = _run_tool_batch(call_count, delay_seconds)
            durations.append(duration)
            observed_orders.append(order)
        scenarios[str(call_count)] = {
            **timing_summary(durations),
            "expected_serial_delay_ms": call_count * delay_ms,
            "result_order_stable": all(
                order == list(range(call_count)) for order in observed_orders
            ),
        }

    abort_after_ms = max(1, delay_ms // 4)
    _run_abort_batch(delay_seconds, abort_after_ms / 1000)
    abort_durations = [
        _run_abort_batch(delay_seconds, abort_after_ms / 1000) for _ in range(runs)
    ]
    return {
        "warmup_runs": 1,
        "measured_runs": runs,
        "controlled_delay_ms_per_tool": delay_ms,
        "scenarios_by_call_count": scenarios,
        "abort_during_first_tool": {
            **timing_summary(abort_durations),
            "abort_requested_after_ms": abort_after_ms,
            "in_flight_tool_cancellation_supported": False,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-files", type=int, default=5000)
    parser.add_argument("--session-records", type=int, default=1000)
    parser.add_argument("--runs", type=int, default=7)
    parser.add_argument("--tool-delay-ms", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workspace_files < 100 or args.session_records < 10 or args.runs < 2:
        raise SystemExit("benchmark scale is too small")
    payload = {
        "schema_version": "lite.agent_runtime_baseline.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
        },
        "workspace": benchmark_workspace(args.workspace_files, args.runs),
        "session": benchmark_session(args.session_records),
        "tool_output_context": benchmark_tool_output_context(args.runs),
        "tool_batch": benchmark_tool_batch(args.runs, args.tool_delay_ms),
    }
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True))


if __name__ == "__main__":
    main()
