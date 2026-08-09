from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def _public_tests(workspace: Path):
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=workspace,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )
    return {
        "passed": result.returncode == 0,
        "returncode": result.returncode,
        "stdout": result.stdout[-2000:],
        "stderr": result.stderr[-2000:],
    }


def _load(workspace: Path, module: str):
    sys.path.insert(0, str(workspace))
    return __import__(module, fromlist=["*"])


def _target(task_id: str, workspace: Path):
    if task_id == "F01_pricing":
        m = _load(workspace, "src.pricing")
        return m.calculate_total(100, 15, 8.5) == 93.5 and _raises(
            lambda: m.calculate_total(-1, 0, 0), ValueError
        )
    if task_id == "F02_retry":
        m = _load(workspace, "src.retry")
        calls = []

        def action():
            calls.append(1)
            if len(calls) < 2:
                raise RuntimeError("retry")
            return "ok"

        ok = m.execute_with_retry(action, 3) == "ok" and len(calls) == 2
        calls.clear()

        def always():
            calls.append(1)
            raise RuntimeError("last")

        return (
            ok
            and _raises(lambda: m.execute_with_retry(always, 3), RuntimeError)
            and len(calls) == 3
        )
    if task_id == "F03_parser":
        m = _load(workspace, "src.config_parser")
        return m.parse_env_lines(" # x\n A = one=two \n\nB=3\n") == {
            "A": "one=two",
            "B": "3",
        }
    if task_id == "F04_pagination":
        m = _load(workspace, "src.pagination")
        return (
            m.paginate([1, 2, 3, 4, 5], 1, 2) == [1, 2]
            and m.paginate([1, 2], 4, 2) == []
            and _raises(lambda: m.paginate([], 1, 0), ValueError)
        )
    if task_id == "M01_user_contract":
        m = _load(workspace, "src.models")
        s = _load(workspace, "src.serializer")
        payload = s.serialize_user(m.User(2, "B", "b@x", active=False))
        return payload == {"id": 2, "name": "B", "email": "b@x", "active": False}
    if task_id == "M02_config_contract":
        c = _load(workspace, "src.config")
        cl = _load(workspace, "src.client")
        return (
            c.load_config({"timeout_seconds": 9}).timeout_seconds == 9
            and c.load_config({"timeout": 7}).timeout_seconds == 7
            and cl.Client(c.load_config({"timeout_seconds": 4})).timeout_value() == 4
        )
    if task_id == "M03_invoice_contract":
        inv = _load(workspace, "src.invoice")
        rend = _load(workspace, "src.render")
        item = inv.Invoice("I-2", 12.5)
        return item.currency == "USD" and rend.render_invoice(item).startswith("USD ")
    if task_id == "M04_event_contract":
        e = _load(workspace, "src.events")
        c = _load(workspace, "src.consumer")
        event = e.build_event("created", {"id": 7})
        return (
            event.get("data", {}).get("type") == "created"
            and c.consume(event, "created") == {"id": 7}
            and c.consume(event, "deleted") is None
        )
    if task_id == "C01_json_cli":
        p = subprocess.run(
            [sys.executable, "src/cli.py"],
            cwd=workspace,
            input='{"name":"a","value":2}\n\n{"name":"b","value":3}\n',
            text=True,
            capture_output=True,
        )
        lines = [json.loads(x) for x in p.stdout.splitlines() if x.strip()]
        bad = subprocess.run(
            [sys.executable, "src/cli.py"],
            cwd=workspace,
            input="not-json\n",
            text=True,
            capture_output=True,
        )
        return (
            p.returncode == 0
            and lines == [{"name": "a", "value": 4}, {"name": "b", "value": 6}]
            and bad.returncode != 0
        )
    if task_id == "C02_env_precedence":
        s = _load(workspace, "src.settings")
        old = os.environ.get("APP_TIMEOUT")
        try:
            os.environ["APP_TIMEOUT"] = "8"
            return (
                s.Settings.from_sources().timeout_seconds == 8
                and s.Settings.from_sources(11).timeout_seconds == 11
                and _raises(lambda: s.Settings.from_sources(0), ValueError)
            )
        finally:
            if old is None:
                os.environ.pop("APP_TIMEOUT", None)
            else:
                os.environ["APP_TIMEOUT"] = old
    if task_id == "C03_csv_ingest":
        m = _load(workspace, "src.csv_ingest")
        rows = m.read_records("name,score\n A , 1.5 \n\n B,2\n")
        return rows == [
            {"name": "A", "score": 1.5},
            {"name": "B", "score": 2.0},
        ] and _raises(lambda: m.read_records("name\nA\n"), ValueError)
    if task_id == "C04_atomic_json":
        m = _load(workspace, "src.storage")
        with tempfile.TemporaryDirectory(dir=workspace) as td:
            target = Path(td) / "nested" / "data.json"
            m.write_json(target, {"b": 1, "a": 2})
            text = target.read_text(encoding="utf-8")
            leftovers = list(target.parent.glob(".*"))
            return text == '{"a": 2, "b": 1}\n' and not leftovers
    if task_id == "R01_incident_fix":
        m = _load(workspace, "src.service")
        todo = (
            (workspace / "TODO.md").read_text(encoding="utf-8")
            if (workspace / "TODO.md").exists()
            else ""
        )
        return (
            m.request(type("C", (), {"get": lambda self, **kw: kw})(), 2500)["timeout"]
            == 2.5
            and "timeout_ms" in todo
            and "tests" in todo
        )
    if task_id == "R02_health_endpoint":
        m = _load(workspace, "src.app")
        todo = (
            (workspace / "TODO.md").read_text(encoding="utf-8")
            if (workspace / "TODO.md").exists()
            else ""
        )
        return (
            m.health() == {"status": "ok", "version": "1.4.0"}
            and "health" in todo.lower()
        )
    if task_id == "R03_config_migration":
        m = _load(workspace, "src.settings")
        todo = (
            (workspace / "TODO.md").read_text(encoding="utf-8")
            if (workspace / "TODO.md").exists()
            else ""
        )
        return (
            m.load({"retries": 2}) == {"max_retries": 2}
            and m.dump({"max_retries": 4}) == {"max_retries": 4}
            and "max_retries" in todo
        )
    if task_id == "R04_checkpoint_resume":
        m = _load(workspace, "src.worker")
        todo = (
            (workspace / "TODO.md").read_text(encoding="utf-8")
            if (workspace / "TODO.md").exists()
            else ""
        )
        ordered = m.order_jobs(
            [
                {"id": 2, "priority": 1},
                {"id": 1, "priority": 2},
                {"id": 3, "priority": 2},
            ]
        )
        return [x["id"] for x in ordered] == [1, 3, 2] and todo.count("[x]") >= 2
    if task_id == "K01_memory_retry":
        m = _load(workspace, "src.policy")
        p = m.retry_policy()
        return p == {"max_attempts": 3, "retry_validation_errors": False}
    if task_id == "K02_memory_api":
        m = _load(workspace, "src.routes")
        return m.user_route() == "/v2/users" and m.health_route() == "/v2/health"
    if task_id == "K03_memory_csv":
        from io import StringIO

        m = _load(workspace, "src.export")
        stream = StringIO()
        m.export_rows([["a", "b"]], stream)
        return stream.getvalue().startswith("a;b")
    if task_id == "K04_memory_flag":
        m = _load(workspace, "src.flags")
        old = os.environ.get("ENABLE_FAST_PATH")
        old2 = os.environ.get("ENABLE_FAST")
        try:
            os.environ["ENABLE_FAST_PATH"] = "1"
            os.environ["ENABLE_FAST"] = "0"
            return m.fast_path_enabled() is True
        finally:
            for k, v in (("ENABLE_FAST_PATH", old), ("ENABLE_FAST", old2)):
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
    if task_id == "O01_explore_then_fix":
        m = _load(workspace, "src.auth")
        return m.authorize("") is False and m.authorize(" token ") is True
    if task_id == "O02_scoped_worker":
        text = (
            (workspace / "notes" / "summary.md").read_text(encoding="utf-8")
            if (workspace / "notes" / "summary.md").exists()
            else ""
        )
        return "preserve user data" in text and "emit audit event" in text
    if task_id == "O03_parallel_notes":
        m = _load(workspace, "src.combined")
        return m.total() == 5
    if task_id == "O04_worker_stop_guard":
        m = _load(workspace, "src.queue")
        return m.Queue([]).pop_safe() is None and m.Queue([4]).pop_safe() == 4
    raise KeyError(task_id)


def _raises(fn, exc):
    try:
        fn()
    except exc:
        return True
    except Exception:
        return False
    return False


def grade(task_id: str, workspace: Path):
    regression = _public_tests(workspace)
    try:
        target_pass = bool(_target(task_id, workspace))
        error = ""
    except Exception as exc:  # pragma: no cover - grader must fail closed.
        target_pass = False
        error = f"{type(exc).__name__}: {exc}"
    result = {
        "task_id": task_id,
        "target_pass": target_pass,
        "regression_pass": regression["passed"],
        "regression": regression,
        "error": error,
    }
    print(json.dumps(result, ensure_ascii=False))
    return 0 if target_pass and regression["passed"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--workspace", required=True)
    args = parser.parse_args()
    raise SystemExit(grade(args.task, Path(args.workspace).resolve()))
