"""Unit tests for verification signal extraction from tool traces."""

from lite.core.verification import (
    classify_verification_command,
    reduce_verification_signal,
    workspace_revision,
)


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


def test_verification_classifier_rejects_chained_commands():
    assert classify_verification_command("python -m pytest -q && echo bypass") == ""
    assert classify_verification_command("python -c \"assert True\" > result.txt") == ""


def test_verification_classifier_accepts_common_test_commands():
    cases = {
        "pytest -q": "test",
        "uv run pytest tests -q": "test",
        "python -m pytest -q": "test",
        "python3.11 -m compileall lite": "compile",
        "python -m py_compile lite/core/runtime.py": "compile",
        "python -m ruff check lite": "lint",
        "python -m mypy lite": "typecheck",
        "python -c \"assert 1 + 1 == 2\"": "synthetic",
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
        "test_state": "missing",
        "last_mutation_sequence": 1,
        "last_successful_verification_sequence": 0,
    }

    verified = reduce_verification_signal(signal, tool_event("pytest -q"), ["src/app.py"])

    assert verified["state"] == "passed"
    assert verified["last_workspace_change_span_id"] == "span_change"
    assert verified["after_last_workspace_change"] is True
    assert verified["test_state"] == "passed"
    assert verified["test_command"] == "pytest -q"


def test_non_test_checks_do_not_erase_failed_test_evidence():
    signal = reduce_verification_signal(
        {}, tool_event("pytest tests/test_app.py -q", status="error"), ["src/app.py"]
    )

    compiled = reduce_verification_signal(
        signal, tool_event("python -m py_compile src/app.py"), ["src/app.py"]
    )
    synthetic = reduce_verification_signal(
        compiled, tool_event("python -c \"assert True\""), ["src/app.py"]
    )

    assert compiled["state"] == "passed"
    assert compiled["command_class"] == "compile"
    assert synthetic["state"] == "passed"
    assert synthetic["command_class"] == "synthetic"
    assert synthetic["test_state"] == "failed"
    assert synthetic["test_command"] == "pytest tests/test_app.py -q"


def test_workspace_change_resets_previous_successful_test_evidence():
    signal = reduce_verification_signal(
        {}, tool_event("pytest -q"), ["src/app.py"]
    )
    changed = {
        "event": "tool_executed",
        "name": "patch_file",
        "workspace_changed": True,
        "span_id": "span_second_change",
    }

    reset = reduce_verification_signal(signal, changed, ["src/app.py"])

    assert reset["state"] == "missing"
    assert reset["test_state"] == "missing"
    assert "test_command" not in reset


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
    assert signal["test_state"] == "passed"
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
    assert signal["test_state"] == "failed"
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
