#!/usr/bin/env python3
"""Run the frozen Compaction/Rewind/Resume state benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lite.evaluation.long_session_state_benchmark import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    DEFAULT_EVENTS_PATH,
    DEFAULT_GROUND_TRUTH_PATH,
    DEFAULT_MANIFEST_PATH,
    build_live_client_factory,
    load_ground_truth,
    load_jsonl,
    run_benchmark,
    validate_frozen_dataset,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate full, compacted, and resumed context against one frozen "
            "100-event session. The real model is read from the Lite TOML profile."
        )
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Lite TOML config. Its selected provider/model is used without a model override.",
    )
    parser.add_argument(
        "--provider",
        default=None,
        help="Optional TOML provider profile; defaults to the TOML top-level provider.",
    )
    parser.add_argument("--events", default=str(DEFAULT_EVENTS_PATH))
    parser.add_argument("--ground-truth", default=str(DEFAULT_GROUND_TRUTH_PATH))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST_PATH))
    parser.add_argument(
        "--output-dir", default="artifacts/long-session-state-benchmark"
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=5,
        help="Paired repetitions per context variant; default follows the benchmark protocol.",
    )
    parser.add_argument("--max-output-tokens", type=int, default=8192)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument(
        "--resume-existing",
        action="store_true",
        help=(
            "Reuse compatible completed repeat-NNN/result.json artifacts and rerun "
            "only missing or blocked repetitions."
        ),
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the frozen JSONL/ground truth/hashes without calling a model.",
    )
    return parser


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    events = load_jsonl(args.events)
    truth = load_ground_truth(args.ground_truth)
    validation = validate_frozen_dataset(
        events,
        truth,
        events_path=args.events,
        ground_truth_path=args.ground_truth,
        manifest_path=args.manifest,
    )
    if args.validate_only:
        print(json.dumps(validation, ensure_ascii=False, sort_keys=True))
        return 0

    factory, provider = build_live_client_factory(
        config_path=args.config,
        provider=args.provider,
        timeout=args.timeout,
    )
    report = run_benchmark(
        client_factory=factory,
        output_dir=args.output_dir,
        events_path=args.events,
        ground_truth_path=args.ground_truth,
        manifest_path=args.manifest,
        repetitions=args.repetitions,
        max_output_tokens=args.max_output_tokens,
        provider_metadata=provider,
        resume_existing=args.resume_existing,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "model": provider["model"],
                "repetitions": args.repetitions,
                "results": str(Path(args.output_dir) / "results.json"),
                "report": str(Path(args.output_dir) / "report.md"),
                "summary": report["summary"]["variants"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if report["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
