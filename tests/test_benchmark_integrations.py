import json
import os
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from lite import SessionStore, cli
from lite.evaluation.harnessbench import build_adapter_metadata, write_adapter_metadata
from lite.evaluation.verifier import find_bash
from lite.testing import ScriptedModelClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_prompt_file_reads_prompt_and_runs_one_shot(tmp_path, capsys):
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("Return final.", encoding="utf-8")

    with patch(
        "lite.cli._build_model_client",
        return_value=ScriptedModelClient(["<final>prompt file ok</final>"]),
    ):
        code = cli.main(
            [
                "--cwd",
                str(tmp_path),
                "--repo-root",
                str(tmp_path),
                "--prompt-file",
                str(prompt),
                "--approval",
                "auto",
                "--non-interactive",
            ]
        )

    captured = capsys.readouterr()
    assert code == 0
    assert "prompt file ok" in captured.out


def test_prompt_file_with_positional_prompt_returns_2(tmp_path, capsys):
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("Return final.", encoding="utf-8")

    code = cli.main(["--prompt-file", str(prompt), "extra prompt"])

    captured = capsys.readouterr()
    assert code == 2
    assert "--prompt-file cannot be combined" in captured.err


def test_session_id_creates_and_reuses_fixed_session(tmp_path):
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("Return final.", encoding="utf-8")
    clients = [
        ScriptedModelClient(["<final>first</final>"]),
        ScriptedModelClient(["<final>second</final>"]),
    ]
    argv = [
        "--cwd",
        str(tmp_path),
        "--repo-root",
        str(tmp_path),
        "--prompt-file",
        str(prompt),
        "--session-id",
        "bench-session",
        "--approval",
        "auto",
        "--non-interactive",
    ]

    with patch("lite.cli._build_model_client", side_effect=clients):
        assert cli.main(argv) == 0
        assert cli.main(argv) == 0

    session_path = tmp_path / ".lite" / "sessions" / "bench-session.json"
    legacy = json.loads(session_path.read_text(encoding="utf-8"))
    session = SessionStore(session_path.parent).load("bench-session")
    assert session["id"] == "bench-session"
    assert [
        item["content"] for item in session["history"] if item["role"] == "assistant"
    ] == ["first", "second"]
    assert legacy["history"] == []


def test_invalid_session_ids_and_resume_conflict_return_2(tmp_path, capsys):
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("Return final.", encoding="utf-8")
    for session_id in ("../x", "a/b", ".", ".."):
        code = cli.main(
            [
                "--cwd",
                str(tmp_path),
                "--prompt-file",
                str(prompt),
                "--session-id",
                session_id,
                "--approval",
                "auto",
                "--non-interactive",
            ]
        )
        assert code == 2

    code = cli.main(
        [
            "--cwd",
            str(tmp_path),
            "--prompt-file",
            str(prompt),
            "--session-id",
            "bench-session",
            "--resume",
            "latest",
            "--approval",
            "auto",
            "--non-interactive",
        ]
    )
    assert code == 2
    assert "--session-id cannot be combined" in capsys.readouterr().err


def test_non_interactive_requires_non_ask_approval_and_prompt(tmp_path, capsys):
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("Return final.", encoding="utf-8")

    code = cli.main(
        ["--cwd", str(tmp_path), "--prompt-file", str(prompt), "--non-interactive"]
    )

    captured = capsys.readouterr()
    assert code == 2
    assert "--non-interactive requires --approval auto or --approval never" in captured.err

    code = cli.main(
        ["--cwd", str(tmp_path), "--approval", "auto", "--non-interactive"]
    )

    captured = capsys.readouterr()
    assert code == 2
    assert "--non-interactive requires a positional prompt or --prompt-file" in captured.err


def test_harnessbench_metadata_points_to_lite_evidence(tmp_path):
    workspace = tmp_path
    run_dir = workspace / ".lite" / "runs" / "run_1"
    session_dir = workspace / ".lite" / "sessions"
    run_dir.mkdir(parents=True)
    session_dir.mkdir(parents=True)
    for name in ("trace.jsonl", "report.json", "task_state.json"):
        (run_dir / name).write_text("{}\n", encoding="utf-8")
    (session_dir / "bench-session.json").write_text(
        json.dumps(
            {
                "history": [
                    {"role": "user", "content": "read input"},
                    {
                        "role": "tool",
                        "name": "read_file",
                        "content": "input content",
                    },
                    {"role": "assistant", "content": "done"},
                ]
            }
        ),
        encoding="utf-8",
    )
    (session_dir / "bench-session.events.jsonl").write_text("{}\n", encoding="utf-8")

    metadata = build_adapter_metadata(
        workspace, session_id="bench-session", returncode=7
    )

    assert metadata["returncode"] == 7
    assert metadata["lite_evidence_available"] is True
    assert metadata["lite_evidence_missing"] == []
    assert metadata["lite_trace_path"] == str(run_dir / "trace.jsonl")
    assert metadata["lite_report_path"] == str(run_dir / "report.json")
    assert metadata["lite_task_state_path"] == str(run_dir / "task_state.json")
    assert metadata["lite_session_path"] == str(session_dir / "bench-session.json")
    assert metadata["lite_session_event_path"] == str(
        session_dir / "bench-session.events.jsonl"
    )
    transcript_path = session_dir / "bench-session.process.jsonl"
    assert metadata["lite_process_transcript_path"] == str(transcript_path)
    rows = [
        json.loads(line)
        for line in transcript_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["message"]["role"] for row in rows] == ["user", "tool", "assistant"]


def test_harnessbench_metadata_writer_creates_manifest(tmp_path):
    output = tmp_path / "sandbox" / "lite-adapter-metadata.json"

    metadata = write_adapter_metadata(tmp_path, output, session_id="missing")

    written = json.loads(output.read_text(encoding="utf-8"))
    assert written == metadata
    assert written["lite_evidence_available"] is False
    assert "lite_trace_path" in written["lite_evidence_missing"]


def test_bench_script_env_max_steps_overrides_yaml_arg(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_uv = fake_bin / "uv"
    fake_uv.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
printf '%s\\n' "$*" >> "$UV_LOG"
if [[ "$*" == *"lite.evaluation.harnessbench"* ]]; then
  output=""
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --output)
        output="$2"
        shift 2
        ;;
      *)
        shift
        ;;
    esac
  done
  if [[ -n "$output" ]]; then
    mkdir -p "$(dirname "$output")"
    printf '{}\\n' > "$output"
  fi
fi
exit 0
""",
        encoding="utf-8",
    )
    fake_uv.chmod(0o755)
    workspace = tmp_path / "workspace"
    sandbox = tmp_path / "sandbox"
    prompt = sandbox / "prompt.txt"
    workspace.mkdir()
    sandbox.mkdir()
    prompt.write_text("prompt", encoding="utf-8")
    log_path = tmp_path / "uv.log"
    env = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "UV_LOG": str(log_path),
        "CLAWBENCH_SANDBOX": str(sandbox),
        "LITE_BENCH_MAX_STEPS": "32",
    }

    bash = find_bash()
    if bash is None:
        pytest.skip("bash is required to exercise the benchmark adapter script")
    completed = subprocess.run(
        [
            bash,
            "scripts/bench-lite-v3.sh",
            "--workspace",
            str(workspace),
            "--prompt-file",
            str(prompt),
            "--session-id",
            "bench-session",
            "--max-steps",
            "16",
        ],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    log_text = log_path.read_text(encoding="utf-8")
    assert "--max-steps 32" in log_text
    effective_prompt = sandbox / "lite-benchmark-prompt.txt"
    assert f"--prompt-file {sandbox}/lite-benchmark-prompt.txt" in log_text
    assert "Benchmark artifact discipline" in effective_prompt.read_text(
        encoding="utf-8"
    )
