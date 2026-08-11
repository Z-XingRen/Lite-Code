"""Unit tests for verification signal extraction from tool traces."""

from lite.core.verification import reduce_verification_signal, workspace_revision


def tool_event(command, status="ok"):
    return {
        "event": "tool_executed",
        "name": "run_shell",
        "args": {"command": command},
        "status": status,
        "span_id": "span_verify",
    }


def receipt_event(
    command="python -m pytest -q",
    *,
    exit_code=0,
    covered_paths=None,
    status="ok",
):
    return {
        "event": "tool_executed",
        "name": "verify",
        "args": {"command": command},
        "status": status,
        "span_id": "span_000031",
        "verification_receipt": {
            "schema_version": "lite.verification_receipt.v1",
            "command": command,
            "command_class": "test",
            "exit_code": exit_code,
            "workspace_revision": "sha256:workspace",
            "covered_paths": list(covered_paths or []),
            "after_mutation_sequence": 28,
            "coverage_confidence": "declared",
        },
    }


def test_verification_classifier_rejects_marker_only_shell_commands():
    for command in (
        "echo pytest",
        "printf pytest",
        "python -c \"print('pytest')\"",
        "grep pytest README.md",
    ):
        assert reduce_verification_signal({}, tool_event(command), ["src/app.py"]) == {}


def test_verification_classifier_accepts_common_test_commands():
    cases = {
        "pytest -q": "test",
        "uv run pytest tests -q": "test",
        "python -m pytest -q": "test",
        "python3.11 -m compileall lite": "compile",
        "python -m ruff check lite": "lint",
        "python -m mypy lite": "typecheck",
        "ruff check lite tests": "lint",
        "mypy lite": "typecheck",
        "pyright": "typecheck",
        "python -m compileall lite": "compile",
        "npm test": "test",
        "npm run test": "test",
        "npm run build": "build",
        "pnpm test": "test",
        "pnpm run build": "build",
        "yarn test": "test",
        "tox": "test",
        "go test ./...": "test",
        "cargo test": "test",
        "make test": "test",
    }

    for command, command_class in cases.items():
        signal = reduce_verification_signal({}, tool_event(command), ["src/app.py"])
        assert signal["schema_version"] == "lite.verification_signal.v1"
        assert signal["state"] == "passed", command
        assert signal["command_class"] == command_class
        assert signal["after_last_workspace_change"] is True
        assert signal["changed_paths_present"] is True
        assert signal["covers_changed_paths"] is False
        assert signal["coverage_confidence"] == "unknown"


def test_verification_signal_marks_workspace_change_as_missing_until_verified():
    changed = {
        "event": "tool_executed",
        "name": "write_file",
        "workspace_changed": True,
        "span_id": "span_change",
    }

    signal = reduce_verification_signal({}, changed, ["src/app.py"])

    assert signal == {
        "schema_version": "lite.verification_signal.v1",
        "state": "missing",
        "last_workspace_change_span_id": "span_change",
        "changed_paths": ["src/app.py"],
    }

    verified = reduce_verification_signal(
        signal, tool_event("pytest -q"), ["src/app.py"]
    )

    assert verified["state"] == "passed"
    assert verified["last_workspace_change_span_id"] == "span_change"
    assert verified["after_last_workspace_change"] is True


def test_structured_verification_receipt_is_reduced_without_command_guessing():
    signal = reduce_verification_signal(
        {},
        receipt_event(
            command="custom project check",
            covered_paths=["src"],
        ),
        ["src/app.py"],
    )

    assert signal["state"] == "passed"
    assert signal["command"] == "custom project check"
    assert signal["command_class"] == "test"
    assert signal["covers_changed_paths"] is True
    assert signal["coverage_confidence"] == "declared"
    assert signal["workspace_revision"] == "sha256:workspace"
    assert signal["after_mutation_sequence"] == 28
    assert signal["verification_receipt"]["exit_code"] == 0


def test_structured_verification_receipt_uses_exit_code_for_failure():
    signal = reduce_verification_signal(
        {},
        receipt_event(exit_code=1, covered_paths=["src/app.py"], status="error"),
        ["src/app.py"],
    )

    assert signal["state"] == "failed"
    assert signal["covers_changed_paths"] is True


def test_workspace_revision_changes_with_file_contents(tmp_path):
    target = tmp_path / "module.py"
    target.write_text("VALUE = 1\n", encoding="utf-8")
    before = workspace_revision(tmp_path)

    target.write_text("VALUE = 2\n", encoding="utf-8")
    after = workspace_revision(tmp_path)

    assert before.startswith("sha256:")
    assert after.startswith("sha256:")
    assert after != before


def test_workspace_revision_ignores_runtime_state(tmp_path):
    (tmp_path / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    before = workspace_revision(tmp_path)

    runtime_state = tmp_path / ".lite" / "runs" / "trace.jsonl"
    runtime_state.parent.mkdir(parents=True)
    runtime_state.write_text('{"event":"tool_executed"}\n', encoding="utf-8")

    assert workspace_revision(tmp_path) == before
