"""Reproducible runtime evidence for workspace tracking and effect recovery."""

from __future__ import annotations

import hashlib
import json
import platform
import statistics
import tempfile
import time
import tracemalloc
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

from ..core.runtime_checkpoints import RuntimeCheckpointsMixin
from ..core.session_journal_schema import canonical_json
from ..core.session_journal_writer import SessionJournalWriter
from ..core.workspace_change_tracker import WorkspaceChangeTracker


EFFECT_TYPES = ("provider", "tool", "permission", "cancel", "retry", "snapshot")
CRASH_PHASES = (
    "after_intent_before_effect",
    "after_effect_before_result",
    "after_result",
)
REPLAY_POLICIES = {
    "provider": "interrupt",
    "tool": "interrupt",
    "permission": "interrupt",
    "cancel": "interrupt",
    "retry": "replay_safe",
    "snapshot": "interrupt",
}


class _SnapshotRuntime(RuntimeCheckpointsMixin):
    def __init__(self, root: Path):
        self.root = root


def percentile(values, percent):
    """Return a linearly interpolated percentile for a non-empty sample."""

    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile requires at least one value")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percent
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def timing_summary(seconds):
    values = list(seconds)
    return {
        "samples": len(values),
        "median_ms": round(statistics.median(values) * 1000, 3),
        "p95_ms": round(percentile(values, 0.95) * 1000, 3),
        "min_ms": round(min(values) * 1000, 3),
        "max_ms": round(max(values) * 1000, 3),
    }


def environment_metadata():
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }


def _positive_integers(values, name):
    normalized = tuple(int(value) for value in values)
    if not normalized or any(value <= 0 for value in normalized):
        raise ValueError(f"{name} must contain positive integers")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must not contain duplicates")
    return tuple(sorted(normalized))


def _write_workspace(root, file_count, file_bytes):
    payload = b"x" * file_bytes
    paths = []
    for index in range(file_count):
        directory = root / f"dir-{index // 100:04d}"
        directory.mkdir(exist_ok=True)
        path = directory / f"file-{index:06d}.txt"
        path.write_bytes(payload)
        paths.append(path)
    return paths


def _measure_path_reads(action):
    reads = 0
    read_bytes = 0
    original = Path.read_bytes

    def counted(path):
        nonlocal reads, read_bytes
        payload = original(path)
        reads += 1
        read_bytes += len(payload)
        return payload

    started = time.perf_counter()
    Path.read_bytes = counted
    try:
        observed = action()
    finally:
        Path.read_bytes = original
    return time.perf_counter() - started, reads, read_bytes, tuple(observed)


def _mutation_payload(iteration, file_bytes):
    marker = bytes([(iteration % 251) + 1])
    return marker * file_bytes


def _workspace_variant(
    *,
    root,
    paths,
    variant,
    changed_count,
    file_bytes,
    warmup_runs,
    measured_runs,
):
    targets = paths[:changed_count]
    expected = tuple(sorted(path.relative_to(root).as_posix() for path in targets))
    runtime = _SnapshotRuntime(root)
    tracker = WorkspaceChangeTracker(root)
    durations = []
    read_counts = []
    read_bytes = []
    exact_matches = 0
    observed_counts = []

    def run_cycle(iteration):
        variant_offset = 127 if variant == "incremental" else 0
        payload = _mutation_payload(iteration + variant_offset, file_bytes)
        if variant == "legacy":

            def cycle():
                before = runtime.capture_workspace_snapshot()
                for path in targets:
                    path.write_bytes(payload)
                after = runtime.capture_workspace_snapshot()
                return runtime.diff_workspace_snapshots(before, after)[0]

        else:
            relative_targets = [
                path.relative_to(root).as_posix() for path in targets
            ]

            def cycle():
                token = tracker.begin("write_file", target_paths=relative_targets)
                for path in targets:
                    path.write_bytes(payload)
                return tracker.finish(token)[0]

        return _measure_path_reads(cycle)

    total_runs = warmup_runs + measured_runs
    for iteration in range(total_runs):
        duration, reads, byte_count, observed = run_cycle(iteration)
        if iteration < warmup_runs:
            continue
        durations.append(duration)
        read_counts.append(reads)
        read_bytes.append(byte_count)
        observed_tuple = tuple(sorted(observed))
        observed_counts.append(len(observed_tuple))
        exact_matches += int(observed_tuple == expected)

    return {
        "variant": variant,
        "cycle": (
            "full content snapshot -> mutate -> full content snapshot -> diff"
            if variant == "legacy"
            else "target digest -> mutate -> target digest -> diff"
        ),
        "timing": timing_summary(durations),
        "read_files": {
            "median": int(statistics.median(read_counts)),
            "p95": round(percentile(read_counts, 0.95), 3),
        },
        "read_bytes": {
            "median": int(statistics.median(read_bytes)),
            "p95": round(percentile(read_bytes, 0.95), 3),
        },
        "path_exact_matches": exact_matches,
        "path_exact_rate": round(exact_matches / measured_runs, 6),
        "observed_changed_path_counts": sorted(set(observed_counts)),
        "expected_changed_path_count": changed_count,
    }


def run_workspace_tracker_benchmark(
    *,
    file_counts=(1000, 5000, 10000),
    changed_counts=(1, 10, 100),
    measured_runs=30,
    warmup_runs=1,
    file_bytes=128,
):
    """Compare transparent target tracking with the legacy full snapshot."""

    file_counts = _positive_integers(file_counts, "file_counts")
    changed_counts = _positive_integers(changed_counts, "changed_counts")
    if measured_runs <= 0 or warmup_runs < 0 or file_bytes <= 0:
        raise ValueError("runs and file_bytes must be positive")
    if max(changed_counts) > min(file_counts):
        raise ValueError("changed_counts cannot exceed the smallest file_count")

    scenarios = []
    total_exact = 0
    total_observations = 0
    for file_count in file_counts:
        with tempfile.TemporaryDirectory(
            prefix=f"lite-workspace-{file_count}-"
        ) as temp_dir:
            root = Path(temp_dir)
            paths = _write_workspace(root, file_count, file_bytes)
            for changed_count in changed_counts:
                legacy = _workspace_variant(
                    root=root,
                    paths=paths,
                    variant="legacy",
                    changed_count=changed_count,
                    file_bytes=file_bytes,
                    warmup_runs=warmup_runs,
                    measured_runs=measured_runs,
                )
                incremental = _workspace_variant(
                    root=root,
                    paths=paths,
                    variant="incremental",
                    changed_count=changed_count,
                    file_bytes=file_bytes,
                    warmup_runs=warmup_runs,
                    measured_runs=measured_runs,
                )
                legacy_median = legacy["timing"]["median_ms"]
                incremental_median = incremental["timing"]["median_ms"]
                ratio = incremental_median / legacy_median if legacy_median else 0.0
                speedup = (
                    legacy_median / incremental_median if incremental_median else 0.0
                )
                exact = (
                    legacy["path_exact_matches"]
                    + incremental["path_exact_matches"]
                )
                observations = measured_runs * 2
                total_exact += exact
                total_observations += observations
                scenarios.append(
                    {
                        "file_count": file_count,
                        "changed_file_count": changed_count,
                        "operation": "modify",
                        "legacy": legacy,
                        "incremental": incremental,
                        "comparison": {
                            "incremental_to_legacy_median_ratio": round(ratio, 6),
                            "median_speedup_x": round(speedup, 3),
                            "median_time_reduction_percent": round(
                                max(0.0, 1.0 - ratio) * 100, 3
                            ),
                            "path_parity_rate": round(exact / observations, 6),
                        },
                    },
                )

    headline = next(
        (
            row
            for row in scenarios
            if row["file_count"] == max(file_counts)
            and row["changed_file_count"] == min(changed_counts)
        ),
        scenarios[0],
    )
    exact_rate = total_exact / total_observations
    return {
        "schema_version": "lite.workspace_tracker_benchmark.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "environment": environment_metadata(),
        "scope": {
            "tracker_mode": "transparent",
            "tool_contract": "target paths are known before execution",
            "operation": "in-place file modification",
            "excludes": ["opaque shell candidate discovery", "model inference"],
        },
        "config": {
            "file_counts": list(file_counts),
            "changed_counts": list(changed_counts),
            "file_bytes": file_bytes,
            "warmup_runs": warmup_runs,
            "measured_runs": measured_runs,
        },
        "scenarios": scenarios,
        "headline": {
            "file_count": headline["file_count"],
            "changed_file_count": headline["changed_file_count"],
            **headline["comparison"],
            "legacy_median_ms": headline["legacy"]["timing"]["median_ms"],
            "incremental_median_ms": headline["incremental"]["timing"][
                "median_ms"
            ],
            "legacy_p95_ms": headline["legacy"]["timing"]["p95_ms"],
            "incremental_p95_ms": headline["incremental"]["timing"]["p95_ms"],
        },
        "summary": {
            "scenario_count": len(scenarios),
            "path_observation_count": total_observations,
            "path_exact_match_count": total_exact,
            "path_exact_rate": round(exact_rate, 6),
        },
        "gates": {
            "all_path_results_exact": exact_rate == 1.0,
            "headline_incremental_faster": (
                headline["comparison"]["incremental_to_legacy_median_ratio"] < 1.0
            ),
            "passed": exact_rate == 1.0
            and headline["comparison"]["incremental_to_legacy_median_ratio"] < 1.0,
        },
    }


def _session_seed(session_id, workspace_root):
    return {
        "id": session_id,
        "created_at": "2026-08-10T00:00:00+00:00",
        "workspace_root": str(workspace_root),
        "history": [],
    }


def _state_hash(state):
    payload = canonical_json(state.to_dict()).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _record_count(path):
    return len([line for line in path.read_bytes().splitlines() if line])


def _physical_effect_count(path):
    try:
        return len(path.read_text(encoding="utf-8").splitlines())
    except FileNotFoundError:
        return 0


def _run_effect_recovery_trial(root, effect_type, crash_phase, repetition, sync):
    trial_id = f"{effect_type}-{crash_phase}-{repetition:02d}"
    journal_path = root / f"{trial_id}.journal.jsonl"
    effect_path = root / f"{trial_id}.physical-effect.log"
    operation_id = f"operation-{trial_id}"
    replay_policy = REPLAY_POLICIES[effect_type]
    writer = SessionJournalWriter.create(
        journal_path,
        _session_seed(f"session-{trial_id}", root),
        sync=sync,
    )
    intent = writer.begin_effect(
        effect_type,
        call_id=f"call-{trial_id}",
        request={"effect_type": effect_type, "trial_id": trial_id},
        replay_policy=replay_policy,
        operation_id=operation_id,
    )
    if crash_phase in {"after_effect_before_result", "after_result"}:
        effect_path.write_text(f"{operation_id}\n", encoding="utf-8")
    if crash_phase == "after_result":
        writer.finish_effect(intent, outcome="ok", result={"committed": True})
    writer.close()

    physical_before = _physical_effect_count(effect_path)
    records_before = _record_count(journal_path)
    started = time.perf_counter()
    reopened = SessionJournalWriter.open(journal_path, sync=sync)
    recovery_seconds = time.perf_counter() - started
    try:
        first_hash = _state_hash(reopened.state)
        actions = [
            action
            for action in reopened.recovery_actions
            if action.operation_id == operation_id
        ]
        open_operation_cleared = reopened.state.open_operation is None
        completed = reopened.state.completed_operations.get(operation_id)
    finally:
        reopened.close()

    records_after_first = _record_count(journal_path)
    journal_after_first = journal_path.read_bytes()
    physical_after = _physical_effect_count(effect_path)
    repeated = SessionJournalWriter.open(journal_path, sync=sync)
    try:
        repeated_hash = _state_hash(repeated.state)
    finally:
        repeated.close()
    journal_after_second = journal_path.read_bytes()

    unresolved = crash_phase != "after_result"
    expected_action = (
        "retry" if replay_policy == "replay_safe" else "interrupt"
    ) if unresolved else None
    observed_actions = [action.action for action in actions]
    action_matches = observed_actions == ([expected_action] if expected_action else [])
    result_matches = bool(completed) and (
        (
            completed.outcome == "interrupted"
            and completed.result
            == {
                "reason": "process_interrupted",
                "recovery_action": expected_action,
                "synthetic": True,
            }
        )
        if unresolved
        else completed.outcome == "ok" and completed.result == {"committed": True}
    )
    expected_recovery_records = 1 if unresolved else 0
    recovery_record_count = records_after_first - records_before
    state_replay_match = first_hash == repeated_hash
    journal_idempotent = journal_after_first == journal_after_second
    duplicate_side_effect = physical_after != physical_before
    safe_resolution = (
        open_operation_cleared
        and action_matches
        and result_matches
        and recovery_record_count == expected_recovery_records
    )
    return {
        "trial_id": trial_id,
        "effect_type": effect_type,
        "crash_phase": crash_phase,
        "repetition": repetition,
        "replay_policy": replay_policy,
        "expected_recovery_action": expected_action,
        "observed_recovery_actions": observed_actions,
        "recovery_ms": round(recovery_seconds * 1000, 3),
        "records_before_recovery": records_before,
        "recovery_record_count": recovery_record_count,
        "physical_effect_count_before_recovery": physical_before,
        "physical_effect_count_after_recovery": physical_after,
        "open_operation_cleared": open_operation_cleared,
        "recovery_action_matches": action_matches,
        "terminal_result_matches": result_matches,
        "state_replay_match": state_replay_match,
        "repeated_open_journal_idempotent": journal_idempotent,
        "duplicate_side_effect_during_recovery": duplicate_side_effect,
        "safe_resolution": safe_resolution,
    }


def _rate(rows, field):
    return round(sum(bool(row[field]) for row in rows) / len(rows), 6)


def _recovery_group(rows):
    return {
        "trial_count": len(rows),
        "safe_resolution_rate": _rate(rows, "safe_resolution"),
        "state_replay_match_rate": _rate(rows, "state_replay_match"),
        "repeated_open_journal_idempotence_rate": _rate(
            rows, "repeated_open_journal_idempotent"
        ),
        "duplicate_side_effect_rate": _rate(
            rows, "duplicate_side_effect_during_recovery"
        ),
        "recovery": timing_summary([row["recovery_ms"] / 1000 for row in rows]),
    }


def run_effect_recovery_matrix(*, repetitions=10, sync=True):
    """Run 6 effect types x 3 crash prefixes x N deterministic recoveries."""

    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    with tempfile.TemporaryDirectory(prefix="lite-effect-recovery-matrix-") as temp_dir:
        root = Path(temp_dir)
        rows = [
            _run_effect_recovery_trial(
                root,
                effect_type,
                crash_phase,
                repetition,
                sync,
            )
            for effect_type in EFFECT_TYPES
            for crash_phase in CRASH_PHASES
            for repetition in range(repetitions)
        ]

    by_effect = {
        effect_type: _recovery_group(
            [row for row in rows if row["effect_type"] == effect_type]
        )
        for effect_type in EFFECT_TYPES
    }
    by_phase = {
        crash_phase: _recovery_group(
            [row for row in rows if row["crash_phase"] == crash_phase]
        )
        for crash_phase in CRASH_PHASES
    }
    summary = _recovery_group(rows)
    expected_trial_count = len(EFFECT_TYPES) * len(CRASH_PHASES) * repetitions
    gates = {
        "expected_trial_count": len(rows) == expected_trial_count,
        "all_effects_safely_resolved": summary["safe_resolution_rate"] == 1.0,
        "all_states_repeatable": summary["state_replay_match_rate"] == 1.0,
        "repeated_open_is_idempotent": (
            summary["repeated_open_journal_idempotence_rate"] == 1.0
        ),
        "no_side_effect_reexecuted_during_recovery": (
            summary["duplicate_side_effect_rate"] == 0.0
        ),
    }
    gates["passed"] = all(gates.values())
    return {
        "schema_version": "lite.effect_recovery_matrix.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "environment": environment_metadata(),
        "scope": {
            "boundary": "journal crash-prefix recovery",
            "recovery_behavior": (
                "open resolves an unfinished intent and reports retry/interrupt; "
                "it does not execute the external effect"
            ),
            "excludes": [
                "provider inference",
                "post-recovery retry executor",
                "cross-process kill timing",
            ],
        },
        "config": {
            "effect_types": list(EFFECT_TYPES),
            "crash_phases": list(CRASH_PHASES),
            "repetitions": repetitions,
            "sync": bool(sync),
            "expected_trial_count": expected_trial_count,
            "replay_policies": dict(REPLAY_POLICIES),
        },
        "summary": summary,
        "by_effect": by_effect,
        "by_phase": by_phase,
        "gates": gates,
        "rows": rows,
    }


def run_journal_scaling_benchmark(
    *, record_counts=(1000, 5000, 10000), sample_window=500, recovery_runs=3
):
    """Measure online append and full recovery as journal history grows."""

    record_counts = _positive_integers(record_counts, "record_counts")
    if sample_window <= 0 or sample_window > min(record_counts):
        raise ValueError("sample_window must be in [1, min(record_counts)]")
    recovery_runs = int(recovery_runs)
    if recovery_runs <= 0:
        raise ValueError("recovery_runs must be positive")

    with tempfile.TemporaryDirectory(prefix="lite-journal-scaling-") as temp_dir:
        root = Path(temp_dir)
        journal_path = root / "scaling.journal.jsonl"
        writer = SessionJournalWriter.create(
            journal_path,
            _session_seed("journal-scaling", root),
            sync=False,
        )
        recent = deque(maxlen=sample_window)
        scenarios = []
        tracemalloc.start()
        try:
            for index in range(1, max(record_counts) + 1):
                started = time.perf_counter()
                writer.append_history(
                    {"role": "assistant", "content": f"record-{index:06d}"}
                )
                recent.append(time.perf_counter() - started)
                if index not in record_counts:
                    continue

                writer.close()
                current_bytes, peak_bytes = tracemalloc.get_traced_memory()
                recovery_samples = []
                state_correct = True
                for _ in range(recovery_runs):
                    recovery_started = time.perf_counter()
                    recovered = SessionJournalWriter.open(journal_path, sync=False)
                    recovery_samples.append(
                        time.perf_counter() - recovery_started
                    )
                    try:
                        state_correct = state_correct and (
                            len(recovered.state.session["history"]) == index
                            and recovered.state.last_sequence == index + 1
                            and recovered.state.session["history"][-1]["content"]
                            == f"record-{index:06d}"
                        )
                    finally:
                        recovered.close()
                recovery_seconds = statistics.median(recovery_samples)
                scenarios.append(
                    {
                        "record_count": index,
                        "append_window": timing_summary(recent),
                        "recovery_timing": timing_summary(recovery_samples),
                        "recovery_ms": round(recovery_seconds * 1000, 3),
                        "recovery_us_per_record": round(
                            recovery_seconds * 1_000_000 / (index + 1), 3
                        ),
                        "traced_current_mib": round(
                            current_bytes / (1024 * 1024), 3
                        ),
                        "traced_peak_mib": round(peak_bytes / (1024 * 1024), 3),
                        "state_correct": state_correct,
                    }
                )
                if index != max(record_counts):
                    writer = SessionJournalWriter.open(journal_path, sync=False)
        finally:
            writer.close()
            tracemalloc.stop()

    first = scenarios[0]
    last = scenarios[-1]
    append_ratio = (
        last["append_window"]["median_ms"]
        / first["append_window"]["median_ms"]
        if first["append_window"]["median_ms"]
        else 0.0
    )
    recovery_costs = [row["recovery_us_per_record"] for row in scenarios]
    recovery_ratio = max(recovery_costs) / min(recovery_costs)
    gates = {
        "all_states_correct": all(row["state_correct"] for row in scenarios),
        "append_growth_ratio_at_most_2_5": append_ratio <= 2.5,
        "recovery_normalized_cost_ratio_at_most_5": recovery_ratio <= 5.0,
    }
    gates["passed"] = all(gates.values())
    return {
        "schema_version": "lite.journal_scaling_benchmark.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "environment": environment_metadata(),
        "scope": {
            "online_projection": "validate -> durable append -> in-place private projection",
            "sync": False,
            "memory_measurement": "Python allocations observed by tracemalloc",
            "excludes": ["fsync latency", "provider inference"],
        },
        "config": {
            "record_counts": list(record_counts),
            "sample_window": sample_window,
            "recovery_runs": recovery_runs,
        },
        "scenarios": scenarios,
        "headline": {
            "first_record_count": first["record_count"],
            "last_record_count": last["record_count"],
            "first_append_median_ms": first["append_window"]["median_ms"],
            "last_append_median_ms": last["append_window"]["median_ms"],
            "append_growth_ratio": round(append_ratio, 3),
            "recovery_normalized_cost_ratio": round(recovery_ratio, 3),
            "last_recovery_ms": last["recovery_ms"],
            "traced_peak_mib": last["traced_peak_mib"],
        },
        "gates": gates,
    }


def render_runtime_evidence_markdown(workspace, recovery, journal=None):
    headline = workspace["headline"]
    recovery_summary = recovery["summary"]
    generated_at = max(workspace["generated_at"], recovery["generated_at"])
    lines = [
        "# Lite Runtime Evidence",
        "",
        f"Generated: `{generated_at}`",
        "",
        "## Workspace Change Tracker",
        "",
        (
            f"Scope: {workspace['scope']['tracker_mode']} tracking where "
            "target paths are known before execution."
        ),
        "",
        "| Files | Changed | Legacy median | Incremental median | Speedup | Exact paths |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in workspace["scenarios"]:
        lines.append(
            "| {file_count} | {changed_file_count} | {legacy:.3f} ms | "
            "{incremental:.3f} ms | {speedup:.2f}x | {parity:.2%} |".format(
                file_count=row["file_count"],
                changed_file_count=row["changed_file_count"],
                legacy=row["legacy"]["timing"]["median_ms"],
                incremental=row["incremental"]["timing"]["median_ms"],
                speedup=row["comparison"]["median_speedup_x"],
                parity=row["comparison"]["path_parity_rate"],
            )
        )
    lines.extend(
        [
            "",
            (
                "Headline: {files} files / {changed} changed, median "
                "{legacy:.3f} ms -> {incremental:.3f} ms "
                "({speedup:.2f}x); path parity {parity:.2%}."
            ).format(
                files=headline["file_count"],
                changed=headline["changed_file_count"],
                legacy=headline["legacy_median_ms"],
                incremental=headline["incremental_median_ms"],
                speedup=headline["median_speedup_x"],
                parity=headline["path_parity_rate"],
            ),
            "",
            "## Effect Recovery Matrix",
            "",
            (
                "Scope: Journal recovery only. Recovery reports retry/interrupt but "
                "does not execute the external effect."
            ),
            "",
            "| Effect | Trials | Safe resolution | State replay | Duplicate effects | p95 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for effect_type in EFFECT_TYPES:
        row = recovery["by_effect"][effect_type]
        lines.append(
            "| {effect} | {trials} | {safe:.2%} | {state:.2%} | "
            "{duplicate:.2%} | {p95:.3f} ms |".format(
                effect=effect_type,
                trials=row["trial_count"],
                safe=row["safe_resolution_rate"],
                state=row["state_replay_match_rate"],
                duplicate=row["duplicate_side_effect_rate"],
                p95=row["recovery"]["p95_ms"],
            )
        )
    lines.extend(
        [
            "",
            (
                "Overall: {trials} trials, safe resolution {safe:.2%}, "
                "state replay match {state:.2%}, duplicate side effects during "
                "recovery {duplicate:.2%}, recovery p95 {p95:.3f} ms."
            ).format(
                trials=recovery_summary["trial_count"],
                safe=recovery_summary["safe_resolution_rate"],
                state=recovery_summary["state_replay_match_rate"],
                duplicate=recovery_summary["duplicate_side_effect_rate"],
                p95=recovery_summary["recovery"]["p95_ms"],
            ),
            "",
            "## Gates",
            "",
            f"- Workspace: `{'pass' if workspace['gates']['passed'] else 'fail'}`",
            f"- Recovery: `{'pass' if recovery['gates']['passed'] else 'fail'}`",
        ]
    )
    if journal is not None:
        headline = journal["headline"]
        lines.extend(
            [
                "",
                "## Journal Scaling",
                "",
                "| Records | Append median | Recovery | Recovery / record | Peak traced memory |",
                "|---:|---:|---:|---:|---:|",
            ]
        )
        for row in journal["scenarios"]:
            lines.append(
                "| {records} | {append:.3f} ms | {recovery:.3f} ms | "
                "{normalized:.3f} us | {memory:.3f} MiB |".format(
                    records=row["record_count"],
                    append=row["append_window"]["median_ms"],
                    recovery=row["recovery_ms"],
                    normalized=row["recovery_us_per_record"],
                    memory=row["traced_peak_mib"],
                )
            )
        lines.extend(
            [
                "",
                (
                    "Append median growth from {first} to {last} records: "
                    "{ratio:.3f}x; normalized recovery-cost spread: "
                    "{recovery_ratio:.3f}x."
                ).format(
                    first=headline["first_record_count"],
                    last=headline["last_record_count"],
                    ratio=headline["append_growth_ratio"],
                    recovery_ratio=headline["recovery_normalized_cost_ratio"],
                ),
                "",
                f"- Journal scaling: `{'pass' if journal['gates']['passed'] else 'fail'}`",
            ]
        )
    return "\n".join(lines) + "\n"


def write_runtime_evidence(output_dir, workspace, recovery, journal=None):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    workspace_path = output / "workspace-tracker.json"
    recovery_path = output / "effect-recovery.json"
    markdown_path = output / "summary.md"
    workspace_path.write_text(
        json.dumps(workspace, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    recovery_path.write_text(
        json.dumps(recovery, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    markdown_path.write_text(
        render_runtime_evidence_markdown(workspace, recovery, journal),
        encoding="utf-8",
    )
    paths = {
        "workspace_json": workspace_path,
        "recovery_json": recovery_path,
        "markdown": markdown_path,
    }
    if journal is not None:
        journal_path = output / "journal-scaling.json"
        journal_path.write_text(
            json.dumps(journal, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        paths["journal_json"] = journal_path
    return paths
