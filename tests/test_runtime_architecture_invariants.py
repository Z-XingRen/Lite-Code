"""Machine-checkable dependency, effect, and persistence invariants."""

import ast
import inspect
import json
from pathlib import Path

from lite.architecture import (
    attribute_call_sites,
    forbidden_imports,
    iter_python_sources,
)
from lite.features.sandbox.process import run_cancellable_process
from lite.providers.retry import run_with_retries


ROOT = Path(__file__).resolve().parents[1]


def _violations(directory, forbidden):
    violations = []
    for module_name, path, source in iter_python_sources(ROOT, directory):
        violations.extend(
            forbidden_imports(source, module_name, forbidden, path=path)
        )
    return violations


def test_core_never_depends_on_cli_or_tui():
    assert _violations(
        "lite/core",
        {"lite.cli", "lite.commands", "lite.tui"},
    ) == []


def test_providers_never_depend_on_session_persistence():
    assert _violations(
        "lite/providers",
        {
            "lite.core.runtime_journal",
            "lite.core.session_journal",
            "lite.core.session_lifecycle",
            "lite.core.session_store",
        },
    ) == []
    provider_source = "\n".join(
        source for _, _, source in iter_python_sources(ROOT, "lite/providers")
    )
    assert "session_store" not in provider_source
    assert "SessionStore" not in provider_source


def test_tools_cannot_import_executor_permission_or_persistence_boundaries():
    assert _violations(
        "lite/tools",
        {
            "lite.core.governance",
            "lite.core.permissions",
            "lite.core.runtime_checkpoints",
            "lite.core.runtime_journal",
            "lite.core.session_store",
            "lite.core.tool_execution",
            "lite.core.tool_executor",
        },
    ) == []


def test_session_save_and_tool_file_writes_have_closed_entrypoint_sets():
    session_save_sites = []
    for module_name, path, source in iter_python_sources(ROOT, "lite/core"):
        session_save_sites.extend(
            (path.relative_to(ROOT).as_posix(), site.function)
            for site in attribute_call_sites(
                source, ("session_store", "save"), path=path
            )
        )
    assert set(session_save_sites) == {
        ("lite/core/runtime.py", "__init__"),
        ("lite/core/session_lifecycle.py", "_rebind"),
    }

    tool_write_sites = []
    for _, path, source in iter_python_sources(ROOT, "lite/tools"):
        for method in ("write_text", "write_bytes"):
            tool_write_sites.extend(
                (path.relative_to(ROOT).as_posix(), site.function)
                for site in attribute_call_sites(source, (method,), path=path)
            )
    assert set(tool_write_sites) == {
        ("lite/tools/registry.py", "tool_patch_file"),
        ("lite/tools/registry.py", "tool_write_file"),
    }


def test_network_process_and_timer_boundaries_are_injectable_and_cancellable():
    retry_parameters = inspect.signature(run_with_retries).parameters
    assert {
        "operation",
        "cancellation_token",
        "sleep_fn",
        "clock",
        "random_fn",
        "on_retry",
    } <= set(retry_parameters)
    process_parameters = inspect.signature(run_cancellable_process).parameters
    assert {"command", "timeout", "cancellation_token"} <= set(process_parameters)

    clients = (ROOT / "lite/providers/clients.py").read_text(encoding="utf-8")
    urlopen_functions = {
        site.function for site in attribute_call_sites(clients, ("urlopen",))
    }
    assert urlopen_functions == {"request_body", "open_response"}


def test_policy_detectors_reject_synthetic_reverse_dependency_and_save_bypass():
    imports = forbidden_imports(
        "from ..tui import Application\n",
        "lite.core.synthetic",
        {"lite.tui"},
    )
    package_imports = forbidden_imports(
        "from .. import tui\n",
        "lite.core.synthetic",
        {"lite.tui"},
    )
    saves = attribute_call_sites(
        "def bypass(runtime):\n    runtime.session_store.save(runtime.session)\n",
        ("session_store", "save"),
    )

    assert [(item.imported, item.line) for item in imports] == [("lite.tui", 1)]
    assert [(item.imported, item.line) for item in package_imports] == [
        ("lite.tui", 1)
    ]
    assert [(item.function, item.line) for item in saves] == [("bypass", 2)]


def test_runtime_invariant_matrix_covers_stable_defect_groups():
    matrix_path = ROOT / "tests/runtime_invariant_matrix.json"
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    assert set(matrix) == {
        "background_process",
        "cancel",
        "crash_prefix",
        "file_write_effect",
        "history_hardening",
        "network_effect",
        "permission",
        "process_effect",
        "reducer_replay",
        "resume",
        "sandbox",
        "timer_effect",
    }
    for node_ids in matrix.values():
        assert node_ids
        for node_id in node_ids:
            path_text, test_name = node_id.split("::", 1)
            path = ROOT / path_text
            functions = {
                node.name
                for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            assert test_name.split("[", 1)[0] in functions
