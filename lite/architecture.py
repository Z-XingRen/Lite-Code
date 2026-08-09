"""Pure source-code checks used by runtime architecture tests."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ImportViolation:
    module: str
    imported: str
    line: int
    path: Path | None = None


@dataclass(frozen=True)
class AttributeCallSite:
    attribute: str
    function: str
    line: int
    path: Path | None = None


def line_trend(line_count, trend_budget):
    """Return a non-blocking module-size trend record."""

    line_count = int(line_count)
    trend_budget = int(trend_budget)
    delta = line_count - trend_budget
    return {
        "lines": line_count,
        "trend_budget": trend_budget,
        "delta": delta,
        "status": "review" if delta > 0 else "within_trend",
    }


def module_line_trends(root, budgets):
    root = Path(root)
    report = {}
    for relative_path, trend_budget in budgets.items():
        path = root / relative_path
        line_count = len(path.read_text(encoding="utf-8").splitlines())
        report[str(relative_path)] = line_trend(line_count, trend_budget)
    return report


def iter_python_sources(root, directory):
    root = Path(root)
    base = root / directory
    for path in sorted(base.rglob("*.py")):
        relative = path.relative_to(root).with_suffix("")
        module_name = ".".join(relative.parts)
        yield module_name, path, path.read_text(encoding="utf-8")


def forbidden_imports(source, module_name, forbidden, *, path=None):
    tree = ast.parse(source)
    forbidden = tuple(sorted(str(value) for value in forbidden))
    violations = []
    for node in ast.walk(tree):
        imported_modules = _imported_modules(node, module_name)
        matches = [
            imported
            for imported in imported_modules
            if any(
                imported == prefix or imported.startswith(f"{prefix}.")
                for prefix in forbidden
            )
        ]
        for imported in matches:
            if any(
                imported != other and imported.startswith(f"{other}.")
                for other in matches
            ):
                continue
            violations.append(
                ImportViolation(
                    module=str(module_name),
                    imported=imported,
                    line=int(node.lineno),
                    path=Path(path) if path is not None else None,
                )
            )
    return sorted(violations, key=lambda item: (item.line, item.imported))


def attribute_call_sites(source, attribute_suffix, *, path=None):
    suffix = tuple(str(part) for part in attribute_suffix)
    visitor = _AttributeCallVisitor(suffix, path=path)
    visitor.visit(ast.parse(source))
    return tuple(visitor.sites)


def _imported_modules(node, module_name):
    if isinstance(node, ast.Import):
        return tuple(alias.name for alias in node.names)
    if not isinstance(node, ast.ImportFrom):
        return ()
    imported = str(node.module or "")
    if node.level:
        package = str(module_name).split(".")[:-1]
        if node.level > 1:
            package = package[: -(node.level - 1)]
        imported = ".".join([*package, *([imported] if imported else [])])
    aliases = tuple(
        f"{imported}.{alias.name}" if imported else alias.name
        for alias in node.names
    )
    return tuple(dict.fromkeys((imported, *aliases)))


def _attribute_parts(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return tuple(reversed(parts))


class _AttributeCallVisitor(ast.NodeVisitor):
    def __init__(self, suffix, *, path=None):
        self.suffix = suffix
        self.path = Path(path) if path is not None else None
        self.function_stack = []
        self.sites = []

    def visit_FunctionDef(self, node):
        self.function_stack.append(node.name)
        self.generic_visit(node)
        self.function_stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Call(self, node):
        parts = _attribute_parts(node.func)
        if len(parts) >= len(self.suffix) and parts[-len(self.suffix) :] == self.suffix:
            self.sites.append(
                AttributeCallSite(
                    attribute=".".join(parts),
                    function=self.function_stack[-1] if self.function_stack else "<module>",
                    line=int(node.lineno),
                    path=self.path,
                )
            )
        self.generic_visit(node)
