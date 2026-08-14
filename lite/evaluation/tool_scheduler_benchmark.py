"""Reproducible performance and safety benchmark for tool batch scheduling.

The harness intentionally drives ``Engine.run_turn`` with ``ScriptedModelClient``
instead of timing an executor in isolation.  Synthetic tools expose precise
execution timing while the normal validation, scheduling, history, journal, and
event projection paths remain active.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import statistics
import subprocess
import tempfile
import threading
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

from lite import Lite, SessionStore, WorkspaceContext
from lite.core.tool_profiles import build_tool_profiles
from lite.providers import ModelResult, ToolCall
from lite.testing import ScriptedModelClient, read_jsonl
from lite.tools.base import RegisteredTool


DEFAULT_OUTPUT_DIR = Path("artifacts/tool-scheduler-benchmark")
DEFAULT_BATCH_SIZES = (2, 4, 8, 16)
DEFAULT_SEED = 20260813
MAX_TOOL_WORKERS = 8
SAFETY_BATCH_SIZE = 4


@dataclass(frozen=True)
class BenchmarkConfig:
    """Configuration for one complete benchmark run."""

    batch_sizes: tuple[int, ...] = DEFAULT_BATCH_SIZES
    delay_ms: float = 50.0
    warmups: int = 5
    repeats: int = 30
    safety_trials: int = 100
    seed: int = DEFAULT_SEED
    output_dir: Path = DEFAULT_OUTPUT_DIR
    safety_batch_size: int = SAFETY_BATCH_SIZE
    cancellation_timeout_s: float = 5.0

    def validated(self) -> "BenchmarkConfig":
        if not self.batch_sizes or any(size < 2 for size in self.batch_sizes):
            raise ValueError("batch sizes must contain integers >= 2")
        if self.delay_ms < 0:
            raise ValueError("delay_ms must be >= 0")
        if self.warmups < 0:
            raise ValueError("warmups must be >= 0")
        if self.repeats < 1:
            raise ValueError("repeats must be >= 1")
        if self.safety_trials < 0:
            raise ValueError("safety_trials must be >= 0")
        if not 2 <= self.safety_batch_size <= MAX_TOOL_WORKERS:
            raise ValueError(
                f"safety_batch_size must be between 2 and {MAX_TOOL_WORKERS}"
            )
        if self.cancellation_timeout_s <= 0:
            raise ValueError("cancellation_timeout_s must be > 0")
        return self


class _ExecutionProbe:
    """Thread-safe timing and concurrency instrumentation for synthetic tools."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.first_started_ns: int | None = None
        self.last_finished_ns: int | None = None
        self.active = 0
        self.max_active = 0
        self.completion_order: list[str] = []

    def run(self, call_id: str, delay_s: float, result: str) -> str:
        started_ns = time.perf_counter_ns()
        with self._lock:
            if self.first_started_ns is None:
                self.first_started_ns = started_ns
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(delay_s)
            return result
        finally:
            finished_ns = time.perf_counter_ns()
            with self._lock:
                self.active -= 1
                self.last_finished_ns = max(
                    self.last_finished_ns or finished_ns, finished_ns
                )
                self.completion_order.append(call_id)

    @property
    def span_ms(self) -> float:
        if self.first_started_ns is None or self.last_finished_ns is None:
            return 0.0
        return (self.last_finished_ns - self.first_started_ns) / 1_000_000


class _ActiveCounter:
    """Detect overlap while preserving invocation and completion evidence."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.active = 0
        self.max_active = 0
        self.conflicts = 0
        self.invocation_order: list[str] = []
        self.completion_order: list[str] = []

    def enter(self, call_id: str) -> None:
        with self.lock:
            if self.active:
                self.conflicts += 1
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.invocation_order.append(call_id)

    def leave(self, call_id: str) -> None:
        with self.lock:
            self.active -= 1
            self.completion_order.append(call_id)


class _ReverseCompletionProbe:
    """Force a parallel batch to finish in exact reverse request order."""

    def __init__(self, size: int, timeout_s: float = 5.0) -> None:
        self.size = size
        self.timeout_s = timeout_s
        self.condition = threading.Condition()
        self.started = 0
        self.active = 0
        self.max_active = 0
        self.next_index = size - 1
        self.completion_order: list[str] = []

    def run(self, index: int, call_id: str) -> str:
        deadline = time.monotonic() + self.timeout_s
        with self.condition:
            self.started += 1
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.condition.notify_all()
            while self.started < self.size or index != self.next_index:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("reverse completion coordination timed out")
                self.condition.wait(remaining)
            self.completion_order.append(call_id)
            self.next_index -= 1
            self.active -= 1
            self.condition.notify_all()
        return f"ordered:{call_id}"


def percentile(values: Iterable[float], quantile: float) -> float:
    """Return an interpolated percentile, including for one-element samples."""

    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between 0 and 1")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def summarize_values(values: Iterable[float]) -> dict:
    values = [float(value) for value in values]
    return {
        "count": len(values),
        "p50": _rounded(percentile(values, 0.50)),
        "p95": _rounded(percentile(values, 0.95)),
        "min": _rounded(min(values) if values else 0.0),
        "max": _rounded(max(values) if values else 0.0),
        "mean": _rounded(statistics.fmean(values) if values else 0.0),
    }


def run_benchmark(config: BenchmarkConfig | None = None) -> dict:
    """Run performance A/B pairs and all safety scenarios, returning raw data."""

    config = (config or BenchmarkConfig()).validated()
    started_wall = datetime.now(timezone.utc)
    started_monotonic = time.perf_counter()
    rng = random.Random(config.seed)

    with tempfile.TemporaryDirectory(
        prefix="lite-tool-scheduler-", ignore_cleanup_errors=True
    ) as temp_dir:
        workspace_root = Path(temp_dir)
        performance = run_performance_benchmark(config, workspace_root, rng)
        safety = run_safety_benchmark(config, workspace_root, rng)

    finished_wall = datetime.now(timezone.utc)
    report = {
        "schema_version": "tool-scheduler-benchmark-v1",
        "config": _json_config(config),
        "environment": environment_metadata(),
        "timing": {
            "started_at": started_wall.isoformat(),
            "finished_at": finished_wall.isoformat(),
            "duration_s": _rounded(time.perf_counter() - started_monotonic),
            "random_seed": config.seed,
        },
        "performance": performance,
        "safety": safety,
    }
    report["validation"] = validation_summary(report)
    report["resume_candidate"] = resume_candidate(report)
    return report


def run_performance_benchmark(
    config: BenchmarkConfig,
    workspace_root: Path,
    rng: random.Random | None = None,
) -> dict:
    """Run paired sequential/parallel Engine trials for every batch size."""

    config.validated()
    rng = rng or random.Random(config.seed)
    performance_root = workspace_root / "performance"
    _prepare_workspace(performance_root)
    samples: list[dict] = []
    warmup_samples: list[dict] = []

    for batch_size in config.batch_sizes:
        for phase, count, destination in (
            ("warmup", config.warmups, warmup_samples),
            ("measured", config.repeats, samples),
        ):
            for trial in range(count):
                request_order = [
                    f"perf-b{batch_size}-{phase[0]}{trial}-c{index}"
                    for index in range(batch_size)
                ]
                mode_order = ["sequential", "parallel"]
                rng.shuffle(mode_order)
                runs: dict[str, dict] = {}
                for mode in mode_order:
                    runs[mode] = _run_performance_trial(
                        performance_root,
                        batch_size=batch_size,
                        delay_ms=config.delay_ms,
                        execution_mode=mode,
                        call_ids=request_order,
                    )
                sequential_span = runs["sequential"]["tool_batch_span_ms"]
                parallel_span = runs["parallel"]["tool_batch_span_ms"]
                sequential_turn = runs["sequential"]["run_turn_ms"]
                parallel_turn = runs["parallel"]["run_turn_ms"]
                destination.append(
                    {
                        "phase": phase,
                        "batch_size": batch_size,
                        "trial": trial,
                        "execution_order": mode_order,
                        "request_order": request_order,
                        "sequential": runs["sequential"],
                        "parallel": runs["parallel"],
                        "paired_speedup": _safe_ratio(
                            sequential_span, parallel_span
                        ),
                        "run_turn_paired_speedup": _safe_ratio(
                            sequential_turn, parallel_turn
                        ),
                    }
                )

    return {
        "method": {
            "driver": "Lite Engine.run_turn",
            "model_client": "ScriptedModelClient",
            "network_calls": 0,
            "paired_order_randomized": True,
            "speedup_numerator": "sequential",
            "speedup_denominator": "parallel",
            "worker_limit": MAX_TOOL_WORKERS,
        },
        "samples": samples,
        "warmup_samples": warmup_samples,
        "summary": summarize_performance(samples, config.batch_sizes),
        "invariants": performance_invariants(samples),
    }


def _run_performance_trial(
    workspace_root: Path,
    *,
    batch_size: int,
    delay_ms: float,
    execution_mode: str,
    call_ids: list[str],
) -> dict:
    calls = _read_calls(call_ids)
    agent = _build_agent(
        workspace_root,
        [ModelResult(tool_calls=calls, stop_reason="tool_use"), "Done."],
        max_steps=max(batch_size + 2, 20),
    )
    probe = _ExecutionProbe()

    def runner(args: dict) -> str:
        index = int(args["start"]) - 1
        call_id = call_ids[index]
        return probe.run(call_id, delay_ms / 1000.0, f"result:{call_id}")

    _replace_tool(
        agent,
        "read_file",
        runner,
        execution_mode=execution_mode,
        effect_class="read_only",
    )
    try:
        started = time.perf_counter_ns()
        events = list(agent.engine.run_turn("benchmark read-only tool batch"))
        run_turn_ms = (time.perf_counter_ns() - started) / 1_000_000
        result_order = _event_result_order(events)
        return {
            "tool_batch_span_ms": _rounded(probe.span_ms),
            "run_turn_ms": _rounded(run_turn_ms),
            "max_active": probe.max_active,
            "completion_order": list(probe.completion_order),
            "result_order": result_order,
            "result_count": len(result_order),
        }
    finally:
        agent.close()


def summarize_performance(samples: list[dict], batch_sizes: Iterable[int]) -> dict:
    summary = {}
    for batch_size in batch_sizes:
        rows = [row for row in samples if row["batch_size"] == batch_size]
        modes = {}
        for mode in ("sequential", "parallel"):
            modes[mode] = {
                "tool_batch_span_ms": summarize_values(
                    row[mode]["tool_batch_span_ms"] for row in rows
                ),
                "run_turn_ms": summarize_values(
                    row[mode]["run_turn_ms"] for row in rows
                ),
                "max_active": summarize_values(
                    row[mode]["max_active"] for row in rows
                ),
            }
        summary[str(batch_size)] = {
            **modes,
            "paired_speedup": summarize_values(
                row["paired_speedup"] for row in rows
            ),
            "run_turn_paired_speedup": summarize_values(
                row["run_turn_paired_speedup"] for row in rows
            ),
        }
    return summary


def performance_invariants(samples: list[dict]) -> dict:
    violations = []
    for row in samples:
        request_order = row["request_order"]
        batch_size = row["batch_size"]
        expected_parallel = min(batch_size, MAX_TOOL_WORKERS)
        for mode, expected_active in (
            ("sequential", 1),
            ("parallel", expected_parallel),
        ):
            run = row[mode]
            if run["max_active"] != expected_active:
                violations.append(
                    {
                        "batch_size": batch_size,
                        "trial": row["trial"],
                        "mode": mode,
                        "kind": "max_active",
                        "expected": expected_active,
                        "actual": run["max_active"],
                    }
                )
            if run["result_order"] != request_order:
                violations.append(
                    {
                        "batch_size": batch_size,
                        "trial": row["trial"],
                        "mode": mode,
                        "kind": "result_order",
                        "expected": request_order,
                        "actual": run["result_order"],
                    }
                )
    return {
        "passed": not violations,
        "violation_count": len(violations),
        "violations": violations,
    }


def run_safety_benchmark(
    config: BenchmarkConfig,
    workspace_root: Path,
    rng: random.Random | None = None,
) -> dict:
    """Run ordering, isolation, failure, and cancellation stress scenarios."""

    config.validated()
    rng = rng or random.Random(config.seed)
    safety_root = workspace_root / "safety"
    _prepare_workspace(safety_root)
    scenarios = {
        "ordering": _run_ordering_safety(config, safety_root, rng),
        "mutation_opaque_isolation": _run_mutation_safety(
            config, safety_root, rng
        ),
        "parallel_failure": _run_failure_safety(config, safety_root, rng),
        "parallel_cancellation": _run_cancellation_safety(
            config, safety_root, rng
        ),
    }
    counter_names = (
        "ordering_failures",
        "mutation_conflicts",
        "missing_results",
        "duplicate_results",
        "cancellation_timeouts",
        "leaked_threads",
        "post_cancel_side_effects",
    )
    totals = {
        name: sum(scenario["counters"].get(name, 0) for scenario in scenarios.values())
        for name in counter_names
    }
    return {
        "total_trials": config.safety_trials * len(scenarios),
        "trials_per_scenario": config.safety_trials,
        "counters": totals,
        "passed": all(value == 0 for value in totals.values()),
        "scenarios": scenarios,
    }


def _run_ordering_safety(
    config: BenchmarkConfig, workspace_root: Path, _rng: random.Random
) -> dict:
    del _rng
    counters = _empty_safety_counters()
    samples = []
    size = config.safety_batch_size
    for trial in range(config.safety_trials):
        call_ids = [f"order-t{trial}-c{index}" for index in range(size)]
        calls = _read_calls(call_ids)
        agent = _build_agent(
            workspace_root,
            [ModelResult(tool_calls=calls, stop_reason="tool_use"), "Done."],
        )
        probe = _ReverseCompletionProbe(size)

        def runner(args: dict) -> str:
            index = int(args["start"]) - 1
            return probe.run(index, call_ids[index])

        _replace_tool(
            agent,
            "read_file",
            runner,
            execution_mode="parallel",
            effect_class="read_only",
        )
        try:
            events = list(agent.engine.run_turn("verify ordered result projection"))
            event_order = _event_result_order(events)
            history_order = _history_result_order(agent)
            journal_order = _journal_result_order(agent.session_path)
            completion_is_reverse = probe.completion_order == list(reversed(call_ids))
            projections_ordered = all(
                observed == call_ids
                for observed in (event_order, history_order, journal_order)
            )
            ordering_failed = not completion_is_reverse or not projections_ordered
            counters["ordering_failures"] += int(ordering_failed)
            missing, duplicates = _combined_projection_issues(
                call_ids, event_order, history_order, journal_order
            )
            counters["missing_results"] += missing
            counters["duplicate_results"] += duplicates
            samples.append(
                {
                    "trial": trial,
                    "request_order": call_ids,
                    "completion_order": list(probe.completion_order),
                    "event_order": event_order,
                    "history_order": history_order,
                    "journal_order": journal_order,
                    "max_active": probe.max_active,
                    "ordering_failed": ordering_failed,
                    "missing_results": missing,
                    "duplicate_results": duplicates,
                }
            )
        finally:
            agent.close()
    return _scenario_result(config.safety_trials, counters, samples)


def _run_mutation_safety(
    config: BenchmarkConfig, workspace_root: Path, _rng: random.Random
) -> dict:
    del _rng
    counters = _empty_safety_counters()
    samples = []
    state_path = workspace_root / "mutation-state.txt"
    state_path.write_text("initial", encoding="utf-8")
    for trial in range(config.safety_trials):
        first_value = f"trial-{trial}-first"
        final_value = f"trial-{trial}-final"
        calls = (
            ToolCall(
                f"mixed-t{trial}-read",
                "read_file",
                {"path": "mutation-state.txt", "start": 1, "end": 1},
            ),
            ToolCall(
                f"mixed-t{trial}-write1",
                "write_file",
                {"path": "mutation-state.txt", "content": first_value},
            ),
            ToolCall(
                f"mixed-t{trial}-opaque",
                "run_shell",
                {"command": f"opaque-{trial}", "timeout": 20},
            ),
            ToolCall(
                f"mixed-t{trial}-write2",
                "write_file",
                {"path": "mutation-state.txt", "content": final_value},
            ),
        )
        call_ids = [call.call_id for call in calls]
        agent = _build_agent(
            workspace_root,
            [ModelResult(tool_calls=calls, stop_reason="tool_use"), "Done."],
        )
        active = _ActiveCounter()

        def wrap(call_id: str, action: Callable[[], str]) -> str:
            active.enter(call_id)
            try:
                time.sleep(0.001)
                return action()
            finally:
                active.leave(call_id)

        def write_runner(args: dict) -> str:
            call_id = (
                calls[1].call_id
                if args["content"] == first_value
                else calls[3].call_id
            )

            def action() -> str:
                state_path.write_text(args["content"], encoding="utf-8")
                return f"wrote:{args['content']}"

            return wrap(call_id, action)

        def read_runner(_args: dict) -> str:
            return wrap(calls[0].call_id, lambda: state_path.read_text(encoding="utf-8"))

        def opaque_runner(_args: dict) -> str:
            return wrap(
                calls[2].call_id,
                lambda: "exit_code: 0\nstdout:\nok\nstderr:\n(empty)",
            )

        _replace_tool(
            agent,
            "write_file",
            write_runner,
            execution_mode="sequential",
            effect_class="mutating",
        )
        _replace_tool(
            agent,
            "read_file",
            read_runner,
            execution_mode="parallel",
            effect_class="read_only",
        )
        _replace_tool(
            agent,
            "run_shell",
            opaque_runner,
            execution_mode="sequential",
            effect_class="opaque",
        )
        try:
            events = list(agent.engine.run_turn("verify mixed effect isolation"))
            result_order = _event_result_order(events)
            final_state = state_path.read_text(encoding="utf-8")
            deterministic = (
                active.max_active == 1
                and active.invocation_order == call_ids
                and active.completion_order == call_ids
                and final_state == final_value
            )
            conflicts = active.conflicts + int(not deterministic)
            counters["mutation_conflicts"] += conflicts
            missing, duplicates = _projection_issues(call_ids, result_order)
            counters["missing_results"] += missing
            counters["duplicate_results"] += duplicates
            samples.append(
                {
                    "trial": trial,
                    "request_order": call_ids,
                    "invocation_order": list(active.invocation_order),
                    "completion_order": list(active.completion_order),
                    "result_order": result_order,
                    "max_active": active.max_active,
                    "overlap_conflicts": active.conflicts,
                    "deterministic_state": deterministic,
                    "final_state": final_state,
                    "missing_results": missing,
                    "duplicate_results": duplicates,
                }
            )
        finally:
            agent.close()
    return _scenario_result(config.safety_trials, counters, samples)


def _run_failure_safety(
    config: BenchmarkConfig, workspace_root: Path, rng: random.Random
) -> dict:
    counters = _empty_safety_counters()
    samples = []
    size = config.safety_batch_size
    for trial in range(config.safety_trials):
        call_ids = [f"failure-t{trial}-c{index}" for index in range(size)]
        target_index = rng.randrange(size)
        target_call_id = call_ids[target_index]
        calls = _read_calls(call_ids)
        agent = _build_agent(
            workspace_root,
            [ModelResult(tool_calls=calls, stop_reason="tool_use"), "Done."],
        )
        probe = _ExecutionProbe()

        def runner(args: dict) -> str:
            index = int(args["start"]) - 1
            call_id = call_ids[index]
            with probe._lock:
                started_ns = time.perf_counter_ns()
                if probe.first_started_ns is None:
                    probe.first_started_ns = started_ns
                probe.active += 1
                probe.max_active = max(probe.max_active, probe.active)
            try:
                time.sleep(0.001)
                if index == target_index:
                    raise RuntimeError(f"controlled failure: {call_id}")
                return f"ok:{call_id}"
            finally:
                with probe._lock:
                    probe.active -= 1
                    probe.completion_order.append(call_id)

        _replace_tool(
            agent,
            "read_file",
            runner,
            execution_mode="parallel",
            effect_class="read_only",
        )
        try:
            events = list(agent.engine.run_turn("verify isolated parallel failure"))
            result_events = [event for event in events if event["type"] == "tool_result"]
            result_order = [event["call_id"] for event in result_events]
            failed_ids = [
                event["call_id"]
                for event in result_events
                if event.get("metadata", {}).get("tool_status") != "ok"
            ]
            history_order = _history_result_order(agent)
            missing, duplicates = _combined_projection_issues(
                call_ids, result_order, history_order
            )
            isolated = failed_ids == [target_call_id]
            if not isolated:
                counters["ordering_failures"] += 1
            counters["missing_results"] += missing
            counters["duplicate_results"] += duplicates
            samples.append(
                {
                    "trial": trial,
                    "request_order": call_ids,
                    "target_call_id": target_call_id,
                    "completion_order": list(probe.completion_order),
                    "result_order": result_order,
                    "history_order": history_order,
                    "failed_call_ids": failed_ids,
                    "failure_isolated": isolated,
                    "missing_results": missing,
                    "duplicate_results": duplicates,
                }
            )
        finally:
            agent.close()
    return _scenario_result(config.safety_trials, counters, samples)


def _run_cancellation_safety(
    config: BenchmarkConfig, workspace_root: Path, rng: random.Random
) -> dict:
    counters = _empty_safety_counters()
    samples = []
    size = config.safety_batch_size
    for trial in range(config.safety_trials):
        call_ids = [f"cancel-t{trial}-c{index}" for index in range(size)]
        calls = _read_calls(call_ids)
        agent = _build_agent(
            workspace_root,
            [ModelResult(tool_calls=calls, stop_reason="tool_use"), "unused"],
        )
        lock = threading.Lock()
        all_started = threading.Event()
        started_ids: list[str] = []
        finished_ids: list[str] = []
        post_cancel_effects: list[str] = []
        events: list[dict] = []

        def runner(args: dict) -> str:
            index = int(args["start"]) - 1
            call_id = call_ids[index]
            with lock:
                started_ids.append(call_id)
                if len(started_ids) == size:
                    all_started.set()
            try:
                agent.current_cancellation_token.wait(timeout=2.0)
                agent.current_cancellation_token.raise_if_cancelled()
                with lock:
                    post_cancel_effects.append(call_id)
                return f"unexpected:{call_id}"
            finally:
                with lock:
                    finished_ids.append(call_id)

        _replace_tool(
            agent,
            "read_file",
            runner,
            execution_mode="parallel",
            effect_class="read_only",
        )
        coordinator = threading.Thread(
            target=lambda: events.extend(
                agent.engine.run_turn("verify parallel cancellation cleanup")
            ),
            name=f"tool-benchmark-coordinator-{trial}",
        )
        trigger_delay_ms = rng.uniform(0.0, 3.0)
        coordinator.start()
        started_in_time = all_started.wait(config.cancellation_timeout_s)
        if started_in_time:
            time.sleep(trigger_delay_ms / 1000.0)
        agent.abort_current_turn()
        coordinator.join(config.cancellation_timeout_s)
        timed_out = coordinator.is_alive()
        if timed_out:
            coordinator.join(2.5)
        agent.close()
        leaked = sorted(
            thread.name
            for thread in threading.enumerate()
            if thread.name.startswith("lite-tool-")
        )
        result_order = _event_result_order(events)
        missing, duplicates = _projection_issues(call_ids, result_order)
        counters["cancellation_timeouts"] += int(not started_in_time or timed_out)
        counters["leaked_threads"] += len(leaked)
        counters["post_cancel_side_effects"] += len(post_cancel_effects)
        counters["missing_results"] += missing
        counters["duplicate_results"] += duplicates
        samples.append(
            {
                "trial": trial,
                "request_order": call_ids,
                "trigger_delay_ms": _rounded(trigger_delay_ms),
                "all_tools_started": started_in_time,
                "started_order": list(started_ids),
                "finished_order": list(finished_ids),
                "result_order": result_order,
                "coordinator_timed_out": timed_out,
                "post_cancel_side_effects": list(post_cancel_effects),
                "leaked_threads": leaked,
                "missing_results": missing,
                "duplicate_results": duplicates,
            }
        )
    return _scenario_result(config.safety_trials, counters, samples)


def validation_summary(report: dict) -> dict:
    performance = report["performance"]["invariants"]
    safety = report["safety"]
    return {
        "passed": bool(performance["passed"] and safety["passed"]),
        "performance_invariants_passed": bool(performance["passed"]),
        "safety_invariants_passed": bool(safety["passed"]),
        "performance_threshold_used": False,
    }


def environment_metadata() -> dict:
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "operating_system": platform.system(),
        "os_release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", ""),
        "cpu_count": os.cpu_count(),
        "git_commit": _git_commit(),
    }


def write_reports(report: dict, output_dir: Path | str | None = None) -> tuple[Path, Path]:
    output = Path(output_dir or report.get("config", {}).get("output_dir") or DEFAULT_OUTPUT_DIR)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "report.json"
    markdown_path = output / "report.md"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_markdown_report(report), encoding="utf-8")
    return json_path, markdown_path


def render_markdown_report(report: dict) -> str:
    config = report["config"]
    environment = report["environment"]
    summary = report["performance"]["summary"]
    safety = report["safety"]
    lines = [
        "# Lite-Code Tool Batch Scheduler Benchmark",
        "",
        f"- 运行时间：{report['timing']['started_at']} — {report['timing']['finished_at']}",
        f"- 总耗时：{report['timing']['duration_s']:.3f} s",
        f"- 固定随机种子：{config['seed']}",
        f"- 环境：Python {environment['python_version']} / {environment['platform']}",
        f"- CPU 数量：{environment['cpu_count']}",
        f"- Git commit：`{environment['git_commit']}`",
        "",
        "## 性能对比",
        "",
        "| Batch | Seq Tool P50/P95 (ms) | Par Tool P50/P95 (ms) | Seq Turn P50/P95 (ms) | Par Turn P50/P95 (ms) | Tool Speedup P50/P95 | Turn Speedup P50/P95 | Par max_active |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for batch_size in config["batch_sizes"]:
        row = summary[str(batch_size)]
        lines.append(
            "| {batch} | {st50:.3f}/{st95:.3f} | {pt50:.3f}/{pt95:.3f} | "
            "{sr50:.3f}/{sr95:.3f} | {pr50:.3f}/{pr95:.3f} | "
            "{sp50:.3f}x/{sp95:.3f}x | {rsp50:.3f}x/{rsp95:.3f}x | "
            "{ma:.0f} |".format(
                batch=batch_size,
                st50=row["sequential"]["tool_batch_span_ms"]["p50"],
                st95=row["sequential"]["tool_batch_span_ms"]["p95"],
                pt50=row["parallel"]["tool_batch_span_ms"]["p50"],
                pt95=row["parallel"]["tool_batch_span_ms"]["p95"],
                sr50=row["sequential"]["run_turn_ms"]["p50"],
                sr95=row["sequential"]["run_turn_ms"]["p95"],
                pr50=row["parallel"]["run_turn_ms"]["p50"],
                pr95=row["parallel"]["run_turn_ms"]["p95"],
                sp50=row["paired_speedup"]["p50"],
                sp95=row["paired_speedup"]["p95"],
                rsp50=row["run_turn_paired_speedup"]["p50"],
                rsp95=row["run_turn_paired_speedup"]["p95"],
                ma=row["parallel"]["max_active"]["max"],
            )
        )
    lines.extend(
        [
            "",
            "加速比逐轮配对计算（同轮 sequential / parallel）；性能数字仅记录实测结果，不作为通过门槛。所有正式样本和预热样本均保存在 `report.json`。",
            "",
            "## 安全性测试",
            "",
            "| 指标 | 实测 | 预期 | 状态 |",
            "|---|---:|---:|---|",
        ]
    )
    labels = {
        "ordering_failures": "顺序错误",
        "mutation_conflicts": "Mutation/Opaque 冲突",
        "missing_results": "结果丢失",
        "duplicate_results": "重复结果",
        "cancellation_timeouts": "取消超时",
        "leaked_threads": "残留 lite-tool- 线程",
        "post_cancel_side_effects": "取消后副作用提交",
    }
    for key, label in labels.items():
        value = safety["counters"][key]
        lines.append(f"| {label} | {value} | 0 | {'PASS' if value == 0 else 'FAIL'} |")
    lines.extend(
        [
            "",
            f"共执行 {safety['total_trials']} 轮安全性场景（每种 {safety['trials_per_scenario']} 轮）。",
            "",
            "## 测试方法",
            "",
            "性能实验使用 ScriptedModelClient 产生参数互异的原生 Tool Call，完整经过 Agent Engine、Tool Batch Scheduler、History、Journal 和事件投影。每一对 A/B 使用相同调用和固定工具延迟，并用固定种子随机交换先后顺序。`tool_batch_span_ms` 从第一个工具 runner 开始到最后一个 runner 结束；`run_turn_ms` 覆盖完整 Engine.run_turn。P50/P95 使用线性插值百分位。",
            "",
            "安全实验覆盖反序完成后的顺序提交、Read-only/Mutation/Opaque 混合批次隔离、单个并行工具受控失败，以及工具全部启动后的随机抖动取消。顺序同时从返回事件、Session History 和 Journal tool_exchange 核验。",
            "",
            "## 适用范围和限制",
            "",
            f"本报告验证本机 Python 线程调度、{config['delay_ms']:g} ms 模拟 I/O 延迟和 ScriptedModelClient 下的调度性质，不代表真实网络工具、CPU 密集型工具、不同文件系统或不同硬件上的绝对吞吐。Python、操作系统后台负载、Journal fsync 和安全检查都会影响 run_turn 数字。性能异常应结合原始配对样本分析，不能由本基准自动判定为失败。",
            "",
            "## 简历候选表述",
            "",
            report["resume_candidate"],
            "",
        ]
    )
    return "\n".join(lines)


def resume_candidate(report: dict) -> str:
    batch_sizes = [int(size) for size in report["config"]["batch_sizes"]]
    preferred = max((size for size in batch_sizes if size <= MAX_TOOL_WORKERS), default=max(batch_sizes))
    row = report["performance"]["summary"][str(preferred)]
    seq = row["sequential"]["tool_batch_span_ms"]["p50"]
    par = row["parallel"]["tool_batch_span_ms"]["p50"]
    speedup = row["paired_speedup"]["p50"]
    safety = report["safety"]
    failures = sum(safety["counters"].values())
    if failures == 0:
        safety_text = f"{safety['total_trials']} 轮安全压力测试零顺序错误、零副作用冲突、零结果丢失/重复、零取消超时与线程泄漏"
    else:
        safety_text = f"{safety['total_trials']} 轮安全压力测试发现 {failures} 个待分析异常"
    return (
        f"在 Lite-Code 中构建可复现的 Tool Batch Scheduler 基准；在 batch={preferred}、"
        f"单工具 {report['config']['delay_ms']:.0f} ms 模拟 I/O 下，工具批次 P50 从 "
        f"{seq:.3f} ms 降至 {par:.3f} ms（配对 P50 {speedup:.3f}x），并完成"
        f"{safety_text}。"
    )


def _build_agent(workspace_root: Path, outputs: list, *, max_steps: int = 20) -> Lite:
    _prepare_workspace(workspace_root)
    return Lite(
        model_client=ScriptedModelClient(outputs),
        workspace=WorkspaceContext.build(workspace_root),
        session_store=SessionStore(workspace_root / ".lite" / "sessions"),
        approval_policy="auto",
        auto_dream=False,
        max_steps=max_steps,
    )


def _prepare_workspace(workspace_root: Path) -> None:
    workspace_root.mkdir(parents=True, exist_ok=True)
    readme = workspace_root / "README.md"
    if not readme.exists():
        readme.write_text("tool scheduler benchmark\n", encoding="utf-8")


def _replace_tool(
    agent: Lite,
    name: str,
    runner: Callable[[dict], str],
    *,
    execution_mode: str,
    effect_class: str,
) -> None:
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


def _read_calls(call_ids: Iterable[str]) -> tuple[ToolCall, ...]:
    return tuple(
        ToolCall(
            str(call_id),
            "read_file",
            {"path": "README.md", "start": index + 1, "end": index + 1},
        )
        for index, call_id in enumerate(call_ids)
    )


def _event_result_order(events: Iterable[dict]) -> list[str]:
    return [
        str(event.get("call_id", ""))
        for event in events
        if event.get("type") == "tool_result"
    ]


def _history_result_order(agent: Lite) -> list[str]:
    return [
        str(item.get("call_id", ""))
        for item in agent.session.get("history", [])
        if item.get("role") == "tool"
    ]


def _journal_result_order(path: Path) -> list[str]:
    records = read_jsonl(path)
    for record in reversed(records):
        payload = record.get("payload", {})
        if record.get("kind") != "effect_result" or payload.get("effect_type") != "tool":
            continue
        entries = payload.get("tree_delta", {}).get("entries", [])
        for entry in entries:
            if entry.get("entry_type") == "tool_exchange":
                return [
                    str(item.get("call_id", ""))
                    for item in entry.get("data", {}).get("results", [])
                ]
    return []


def _projection_issues(expected: Iterable[str], observed: Iterable[str]) -> tuple[int, int]:
    expected_counts = Counter(expected)
    observed_counts = Counter(observed)
    missing = sum(
        max(count - observed_counts.get(key, 0), 0)
        for key, count in expected_counts.items()
    )
    duplicates = sum(
        max(count - expected_counts.get(key, 0), 0)
        for key, count in observed_counts.items()
    )
    return missing, duplicates


def _combined_projection_issues(
    expected: Iterable[str], *observed_projections: Iterable[str]
) -> tuple[int, int]:
    missing = 0
    duplicates = 0
    expected = list(expected)
    for observed in observed_projections:
        current_missing, current_duplicates = _projection_issues(expected, observed)
        missing += current_missing
        duplicates += current_duplicates
    return missing, duplicates


def _empty_safety_counters() -> dict:
    return {
        "ordering_failures": 0,
        "mutation_conflicts": 0,
        "missing_results": 0,
        "duplicate_results": 0,
        "cancellation_timeouts": 0,
        "leaked_threads": 0,
        "post_cancel_side_effects": 0,
    }


def _scenario_result(trials: int, counters: dict, samples: list[dict]) -> dict:
    return {
        "trials": trials,
        "passed": all(value == 0 for value in counters.values()),
        "counters": counters,
        "samples": samples,
    }


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return _rounded(numerator / denominator, digits=6)


def _rounded(value: float, *, digits: int = 3) -> float:
    return round(float(value), digits)


def _json_config(config: BenchmarkConfig) -> dict:
    payload = asdict(config)
    payload["batch_sizes"] = list(config.batch_sizes)
    payload["output_dir"] = str(config.output_dir)
    return payload


def _git_commit() -> str:
    repository_root = Path(__file__).resolve().parents[2]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            capture_output=True,
            check=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def parse_batch_sizes(value: str) -> tuple[int, ...]:
    try:
        sizes = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("batch sizes must be comma-separated integers") from exc
    if not sizes or any(size < 2 for size in sizes):
        raise argparse.ArgumentTypeError("batch sizes must contain integers >= 2")
    return sizes


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark Lite's Tool Batch Scheduler through Engine.run_turn."
    )
    parser.add_argument("--batch-sizes", type=parse_batch_sizes, default=DEFAULT_BATCH_SIZES)
    parser.add_argument("--delay-ms", type=float, default=50.0)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--safety-trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = BenchmarkConfig(
        batch_sizes=tuple(args.batch_sizes),
        delay_ms=args.delay_ms,
        warmups=args.warmups,
        repeats=args.repeats,
        safety_trials=args.safety_trials,
        seed=args.seed,
        output_dir=args.output_dir,
    ).validated()
    report = run_benchmark(config)
    json_path, markdown_path = write_reports(report, config.output_dir)
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {markdown_path}")
    print(f"Validation: {'PASS' if report['validation']['passed'] else 'FAIL'}")
    return 0 if report["validation"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
