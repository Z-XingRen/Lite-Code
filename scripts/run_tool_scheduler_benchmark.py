#!/usr/bin/env python3
"""CLI shim for the Tool Batch Scheduler benchmark."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lite.evaluation.tool_scheduler_benchmark import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
