"""Reproducible agent runtime benchmarks, including workspace tracker parity."""

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
from lite.core.run_store import RunStore
from lite.core.runtime_checkpoints import RuntimeCheckpointsMixin
from lite.core.session_journal import SessionJournalWriter
from lite.core.task_state import TaskState
from lite.core.tool_batch_scheduler import execute_parallel_tool_batch
from lite.core.tool_profiles import build_tool_profiles
from lite.core.tool_result_artifacts import prepare_tool_result_observation
from lite.core.workspace_change_tracker import WorkspaceChangeTracker
from lite.providers.base import ToolCall
from lite.testing import ScriptedModelClient
from lite.tools.base import RegisteredTool


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


def _measure_workspace_cycle(cycle) -> tuple[float, int, int, int]:
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
        changed_paths = cycle()
    finally:
        Path.read_bytes = original_read_bytes
    return time.perf_counter() - started, reads, read_bytes, len(changed_paths)


def _legacy_workspace_cycle(
    root: Path,
    paths: list[Path],
    changed_count: int,
    iteration: int,
) -> tuple[float, int, int, int]:
    runtime = SnapshotRuntime(root)

    def cycle():
        before = runtime.capture_workspace_snapshot()
        marker = bytes([65 + (iteration % 26)])
        for path in paths[:changed_count]:
            path.write_bytes(marker * 128)
        after = runtime.capture_workspace_snapshot()
        return runtime.diff_workspace_snapshots(before, after)[0]

    return _measure_workspace_cycle(cycle)


def _incremental_workspace_cycle(
    root: Path,
    paths: list[Path],
    changed_count: int,
    iteration: int,
) -> tuple[float, int, int, int]:
    tracker = WorkspaceChangeTracker(root)
    target_paths = [path.relative_to(root).as_posix() for path in paths[:changed_count]]

    def cycle():
        token = tracker.begin("write_file", target_paths=target_paths)
        marker = bytes([65 + (iteration % 26)])
        for path in paths[:changed_count]:
            path.write_bytes(marker * 128)
        return tracker.finish(token)[0]

    return _measure_workspace_cycle(cycle)


def _benchmark_workspace_variant(
    file_count: int, runs: int, cycle, name: str, description: str
) -> dict:
    with tempfile.TemporaryDirectory(prefix=f"lite-{name}-workspace-") as temp_dir:
        root = Path(temp_dir)
        _write_workspace(root, file_count)
        paths = sorted(root.rglob("*.txt"))
        scenarios = {}
        for changed_count in (1, 10, 100):
            cycle(root, paths, changed_count, 0)
            durations = []
            read_counts = []
            read_byte_counts = []
            observed_changes = []
            for iteration in range(1, runs + 1):
                duration, reads, read_bytes, changes = cycle(
                    root, paths, changed_count, iteration
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
            "cycle": description,
            "scenarios_by_changed_file_count": scenarios,
        }


def benchmark_workspace(file_count: int, runs: int) -> dict:
    legacy = _benchmark_workspace_variant(
        file_count,
        runs,
        _legacy_workspace_cycle,
        "legacy",
        "snapshot -> mutate -> snapshot -> diff",
    )
    incremental = _benchmark_workspace_variant(
        file_count,
        runs,
        _incremental_workspace_cycle,
        "incremental",
        "target digest -> mutate -> target digest -> diff",
    )
    legacy_single = legacy["scenarios_by_changed_file_count"]["1"]["median_ms"]
    incremental_single = incremental["scenarios_by_changed_file_count"]["1"]["median_ms"]
    ratio = incremental_single / legacy_single if legacy_single else 0
    return {
        "file_count": file_count,
        "file_bytes": 128,
        "warmup_runs": 1,
        "measured_runs": runs,
        "cycle": legacy["cycle"],
        "scenarios_by_changed_file_count": legacy[
            "scenarios_by_changed_file_count"
        ],
        "legacy": legacy,
        "incremental": incremental,
        "single_file_incremental_to_legacy_median_ratio": round(ratio, 4),
        "single_file_median_ratio_limit": 0.2,
        "single_file_median_target_met": ratio <= 0.2,
    }


def _session_record(index: int) -> dict:
    return {
        "role": "assistant" if index % 2 else "user",
        "content": f"record-{index:04d}-" + ("x" * 256),
        "event_id": f"event_{index:06d}",
    }


def _session_seed(session_id: str, workspace_root: str) -> dict:
    return {
        "id": session_id,
        "created_at": "2026-08-07T00:00:00+00:00",
        "workspace_root": workspace_root,
        "history": [],
    }


def _measure_session_recovery(store, session_id, record_count, recovery_runs):
    durations = []
    for _ in range(recovery_runs):
        started = time.perf_counter()
        loaded = store.load(session_id)
        durations.append(time.perf_counter() - started)
        if len(loaded["history"]) != record_count:
            raise RuntimeError("session recovery returned an incomplete history")
    return timing_summary(durations)


def _benchmark_legacy_session(root, record_count, recovery_runs):
    store = SessionStore(root / "legacy")
    session = _session_seed("legacy-baseline", str(root))
    append_durations = []
    cumulative_write_bytes = 0
    path = store.save(session)
    for index in range(record_count):
        session["history"].append(_session_record(index))
        started = time.perf_counter()
        path = store.save(session)
        append_durations.append(time.perf_counter() - started)
        cumulative_write_bytes += path.stat().st_size
    return {
        "append": timing_summary(append_durations),
        "cumulative_write_bytes": cumulative_write_bytes,
        "final_session_bytes": path.stat().st_size,
        "recovery": _measure_session_recovery(
            store, session["id"], record_count, recovery_runs
        ),
        "persistence_mode": "full JSON rewrite per append",
    }


def _benchmark_journal_session(root, record_count, recovery_runs):
    store = SessionStore(root / "journal")
    session = _session_seed("journal-baseline", str(root))
    store.save(session)
    started = time.perf_counter()
    migration = store.migrate_session(session["id"])
    migration_seconds = time.perf_counter() - started
    writer = SessionJournalWriter.open(store.journal_path(session["id"]))
    append_durations = []
    cumulative_write_bytes = 0
    snapshot_seconds = 0.0
    try:
        for index in range(record_count):
            before_size = writer.path.stat().st_size
            started = time.perf_counter()
            writer.append_history(_session_record(index))
            append_durations.append(time.perf_counter() - started)
            cumulative_write_bytes += writer.path.stat().st_size - before_size
        started = time.perf_counter()
        writer.write_snapshot()
        snapshot_seconds = time.perf_counter() - started
        final_session_bytes = writer.path.stat().st_size
    finally:
        writer.close()
    return {
        "append": timing_summary(append_durations),
        "cumulative_write_bytes": cumulative_write_bytes,
        "final_session_bytes": final_session_bytes,
        "recovery": _measure_session_recovery(
            store, session["id"], record_count, recovery_runs
        ),
        "snapshot": timing_summary([snapshot_seconds]),
        "migration": timing_summary([migration_seconds]),
        "migration_baseline_sequence": migration.baseline_sequence,
        "persistence_mode": "append-only JSONL records",
    }


def benchmark_session(record_count: int, recovery_runs: int = 20) -> dict:
    with tempfile.TemporaryDirectory(prefix="lite-session-comparison-") as temp_dir:
        root = Path(temp_dir)
        legacy = _benchmark_legacy_session(root, record_count, recovery_runs)
        journal = _benchmark_journal_session(root, record_count, recovery_runs)
        byte_ratio = (
            journal["cumulative_write_bytes"] / legacy["cumulative_write_bytes"]
            if legacy["cumulative_write_bytes"]
            else 0
        )
        return {
            "history_records": record_count,
            "record_content_chars": 268,
            **legacy,
            "legacy": legacy,
            "journal": journal,
            "journal_to_legacy_cumulative_write_ratio": round(byte_ratio, 4),
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


def _build_batch_agent(root: Path, call_count: int, delay_seconds: float) -> Lite:
    calls = []
    for index in range(call_count):
        path = f"input-{index}.txt"
        (root / path).write_text(f"input {index}\n", encoding="utf-8")
        calls.append(
            ToolCall(
                call_id=f"call-{index}",
                name="read_file",
                arguments={"path": path, "start": 1, "end": 1},
            )
        )
    agent = Lite(
        model_client=ScriptedModelClient([]),
        workspace=WorkspaceContext.build(root),
        session_store=SessionStore(root / ".lite" / "sessions"),
        run_store=RunStore(root / ".lite" / "runs"),
        approval_policy="auto",
        auto_dream=False,
    )
    agent.session_journal_writer = _BenchmarkJournalWriter()
    agent.permission_checker = _BenchmarkPermissionChecker()
    original = agent.tools["read_file"]
    probe = {
        "lock": threading.Lock(),
        "active": 0,
        "started_at": 0.0,
        "finished_at": 0.0,
    }

    def delayed_read(args):
        index = int(Path(args["path"]).stem.split("-")[-1])
        with probe["lock"]:
            if probe["active"] == 0:
                probe["started_at"] = time.perf_counter()
            probe["active"] += 1
        try:
            agent.current_cancellation_token.wait(delay_seconds)
            agent.current_cancellation_token.raise_if_cancelled()
            return f"read_file:{index}"
        finally:
            with probe["lock"]:
                probe["active"] -= 1
                if probe["active"] == 0:
                    probe["finished_at"] = time.perf_counter()

    agent.tools["read_file"] = RegisteredTool(
        name="read_file",
        schema=original.schema,
        description=original.description,
        risky=False,
        runner=delayed_read,
        execution_mode="parallel",
        effect_class="read_only",
    )
    agent.tool_profiles = build_tool_profiles(agent.tools)
    return agent, tuple(calls), probe


class _BenchmarkJournalEffect:
    @staticmethod
    def complete(_outcome, _result=None):
        return None

    def __enter__(self):
        return self

    @staticmethod
    def __exit__(_exc_type, _exc, _traceback):
        return False


class _BenchmarkJournalState:
    open_operation = None


class _BenchmarkJournalWriter:
    state = _BenchmarkJournalState()

    @staticmethod
    def effect(*_args, **_kwargs):
        return _BenchmarkJournalEffect()

    @staticmethod
    def close():
        return None


class _BenchmarkPermission:
    allowed = True
    decision = "allow"
    reason = "benchmark_read_only"
    security_event_type = ""


class _BenchmarkPermissionChecker:
    @staticmethod
    def check(_tool, _args, *, call_id=None):
        return _BenchmarkPermission()


def _close_batch_agent(agent: Lite) -> None:
    writer = getattr(agent, "session_journal_writer", None)
    if writer is not None:
        writer.close()


def _run_tool_batch(
    call_count: int, delay_seconds: float
) -> tuple[float, float, list[int]]:
    temp_dir = tempfile.TemporaryDirectory(prefix="lite-tool-batch-")
    root = Path(temp_dir.name)
    agent, calls, probe = _build_batch_agent(root, call_count, delay_seconds)
    started = time.perf_counter()
    try:
        outcomes = execute_parallel_tool_batch(agent, calls)
        duration = time.perf_counter() - started
    finally:
        _close_batch_agent(agent)
        temp_dir.cleanup()
    result_order = [int(outcome.result.split(":", 1)[1]) for outcome in outcomes]
    effect_duration = probe["finished_at"] - probe["started_at"]
    return duration, effect_duration, result_order


def _run_abort_batch(
    delay_seconds: float, abort_after_seconds: float
) -> tuple[float, bool]:
    temp_dir = tempfile.TemporaryDirectory(prefix="lite-tool-abort-")
    root = Path(temp_dir.name)
    agent, calls, _probe = _build_batch_agent(root, 2, delay_seconds)
    timer = threading.Timer(abort_after_seconds, agent.abort_current_turn)
    timer.start()
    started = time.perf_counter()
    try:
        execute_parallel_tool_batch(agent, calls)
        duration = time.perf_counter() - started
    finally:
        timer.join()
        _close_batch_agent(agent)
        temp_dir.cleanup()
    clean = not any(
        thread.name.startswith("lite-tool-") for thread in threading.enumerate()
    )
    return duration, clean


def benchmark_tool_batch(runs: int, delay_ms: int) -> dict:
    delay_seconds = delay_ms / 1000
    scenarios = {}
    for call_count in (2, 4, 8):
        _run_tool_batch(call_count, delay_seconds)
        end_to_end_durations = []
        effect_durations = []
        observed_orders = []
        for _ in range(runs):
            duration, effect_duration, order = _run_tool_batch(
                call_count, delay_seconds
            )
            end_to_end_durations.append(duration)
            effect_durations.append(effect_duration)
            observed_orders.append(order)
        effect_median_ms = statistics.median(effect_durations) * 1000
        scenarios[str(call_count)] = {
            "effect_wall": timing_summary(effect_durations),
            "end_to_end": timing_summary(end_to_end_durations),
            "expected_parallel_delay_ms": delay_ms,
            "expected_serial_delay_ms": call_count * delay_ms,
            "effect_median_to_serial_delay_ratio": round(
                effect_median_ms / (call_count * delay_ms), 4
            ),
            "result_order_stable": all(
                order == list(range(call_count)) for order in observed_orders
            ),
        }

    abort_after_ms = max(1, delay_ms // 4)
    _run_abort_batch(delay_seconds, abort_after_ms / 1000)
    abort_samples = [
        _run_abort_batch(delay_seconds, abort_after_ms / 1000) for _ in range(runs)
    ]
    return {
        "scheduler": "parallel_read_only",
        "warmup_runs": 1,
        "measured_runs": runs,
        "controlled_delay_ms_per_tool": delay_ms,
        "scenarios_by_call_count": scenarios,
        "abort_during_batch": {
            **timing_summary([duration for duration, _ in abort_samples]),
            "abort_requested_after_ms": abort_after_ms,
            "in_flight_tool_cancellation_supported": True,
            "all_workers_joined": all(clean for _, clean in abort_samples),
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
        "schema_version": "lite.agent_runtime_baseline.v2",
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
