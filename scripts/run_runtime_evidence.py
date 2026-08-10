#!/usr/bin/env python3
"""Run reproducible Workspace Tracker and effect recovery evidence suites."""

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lite.evaluation.runtime_evidence import (  # noqa: E402
    run_effect_recovery_matrix,
    run_journal_scaling_benchmark,
    run_workspace_tracker_benchmark,
    write_runtime_evidence,
)


def _integer_list(value):
    try:
        parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not parsed or any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError("values must be positive integers")
    return parsed


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="artifacts/runtime-evidence",
        help="Directory for JSON and Markdown evidence.",
    )
    parser.add_argument(
        "--workspace-file-counts",
        type=_integer_list,
        default=(1000, 5000, 10000),
        help="Comma-separated workspace sizes.",
    )
    parser.add_argument(
        "--workspace-changed-counts",
        type=_integer_list,
        default=(1, 10, 100),
        help="Comma-separated modified-file counts.",
    )
    parser.add_argument("--workspace-runs", type=int, default=30)
    parser.add_argument("--workspace-warmups", type=int, default=1)
    parser.add_argument("--workspace-file-bytes", type=int, default=128)
    parser.add_argument("--recovery-repetitions", type=int, default=10)
    parser.add_argument(
        "--journal-record-counts",
        type=_integer_list,
        default=(1000, 5000, 10000),
        help="Comma-separated journal history sizes.",
    )
    parser.add_argument("--journal-sample-window", type=int, default=500)
    parser.add_argument(
        "--no-journal-sync",
        action="store_true",
        help="Disable fsync only for smoke tests; formal evidence should keep it enabled.",
    )
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    workspace = run_workspace_tracker_benchmark(
        file_counts=args.workspace_file_counts,
        changed_counts=args.workspace_changed_counts,
        measured_runs=args.workspace_runs,
        warmup_runs=args.workspace_warmups,
        file_bytes=args.workspace_file_bytes,
    )
    recovery = run_effect_recovery_matrix(
        repetitions=args.recovery_repetitions,
        sync=not args.no_journal_sync,
    )
    journal = run_journal_scaling_benchmark(
        record_counts=args.journal_record_counts,
        sample_window=args.journal_sample_window,
    )
    paths = write_runtime_evidence(args.output_dir, workspace, recovery, journal)
    for name, path in paths.items():
        print(f"{name}: {path}")
    if not all(
        result["gates"]["passed"] for result in (workspace, recovery, journal)
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
