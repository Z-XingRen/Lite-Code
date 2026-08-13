import json

import pytest

from lite.evaluation.run_evidence import RunEvidence


def test_run_evidence_reads_trace_report_and_session_artifacts(tmp_path):
    workspace = tmp_path
    run_dir = workspace / ".lite" / "runs" / "run_1"
    session_dir = workspace / ".lite" / "sessions"
    run_dir.mkdir(parents=True)
    session_dir.mkdir(parents=True)
    (run_dir / "report.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "stop_reason": "final_answer_returned",
                "runtime_reminders": [{"artifact_path": ".lite/runs/run_1/artifacts/fallback.txt"}],
                "artifact_graph": {"changed_paths": ["src/app.py"]},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "task_state.json").write_text(
        json.dumps({"status": "completed", "changed_paths": ["src/app.py"]}),
        encoding="utf-8",
    )
    (run_dir / "trace.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "event": "tool_executed",
                        "name": "run_shell",
                        "full_output_artifact": ".lite/runs/run_1/artifacts/output.txt",
                    }
                ),
                json.dumps({"event": "tool_executed", "tool_name": "patch_file"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (session_dir / "session.json").write_text("{}", encoding="utf-8")
    (session_dir / "session.events.jsonl").write_text(
        json.dumps({"event": "permission_decision", "reason": "allow"}) + "\n",
        encoding="utf-8",
    )

    evidence = RunEvidence.latest(workspace)

    assert evidence.status() == "completed"
    assert evidence.stop_reason() == "final_answer_returned"
    assert evidence.changed_paths() == ["src/app.py"]
    assert evidence.has_tools("run_shell", "patch_file")
    assert evidence.full_output_artifacts() == [
        ".lite/runs/run_1/artifacts/output.txt",
        ".lite/runs/run_1/artifacts/fallback.txt",
    ]
    assert evidence.has_session_event("permission_decision", reason="allow")


def test_run_evidence_loads_an_exact_run(tmp_path):
    runs = tmp_path / ".lite" / "runs"
    for run_id in ("run_1", "run_2"):
        run_dir = runs / run_id
        run_dir.mkdir(parents=True)
        (run_dir / "report.json").write_text(
            json.dumps({"run_id": run_id, "status": "completed"}),
            encoding="utf-8",
        )

    evidence = RunEvidence.for_run(tmp_path, "run_1")

    assert evidence.run_dir.name == "run_1"
    assert evidence.report["run_id"] == "run_1"

    with pytest.raises(FileNotFoundError, match="run evidence not found"):
        RunEvidence.for_run(tmp_path, "missing")
