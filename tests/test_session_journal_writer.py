import json
import threading
from unittest.mock import patch

import pytest

from lite import Lite, SessionStore, WorkspaceContext
from lite.cancellation import CancellationRequested
from lite.core.session_journal import (
    JournalWriterError,
    SessionJournalWriter,
)
from lite.testing import ScriptedModelClient


def session(session_id="session-1", history=None):
    return {
        "id": session_id,
        "created_at": "2026-08-08T00:00:00+00:00",
        "workspace_root": "C:/workspace",
        "history": list(history or []),
    }


def read_records(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build_agent(tmp_path, outputs=None):
    (tmp_path / "README.md").write_text("demo\n", encoding="utf-8")
    store = SessionStore(tmp_path / ".lite" / "sessions")
    agent = Lite(
        model_client=ScriptedModelClient(outputs or []),
        workspace=WorkspaceContext.build(tmp_path),
        session_store=store,
        approval_policy="auto",
    )
    return agent, store


def attach_writer(agent, store):
    writer = SessionJournalWriter.create(
        store.journal_path(agent.session["id"]), agent.session
    )
    agent.attach_session_journal(writer)
    return writer


def test_history_append_preserves_prefix_and_writes_one_bounded_record(tmp_path):
    history = [
        {"role": "user", "content": f"history-{index}-" + ("A" * 256)}
        for index in range(1000)
    ]
    path = tmp_path / "session.journal.jsonl"
    writer = SessionJournalWriter.create(path, session(history=history))
    try:
        prefix = path.read_bytes()

        writer.append_history({"role": "assistant", "content": "bounded"})
        payload = path.read_bytes()

        assert payload.startswith(prefix)
        assert len(payload) - len(prefix) < 512
        assert len(read_records(path)) == 2
        assert writer.state.session["history"][-1]["content"] == "bounded"
    finally:
        writer.close()


def test_only_one_writer_can_own_a_session_journal(tmp_path):
    path = tmp_path / "session.journal.jsonl"
    writer = SessionJournalWriter.create(path, session())
    try:
        with pytest.raises(JournalWriterError, match="already active"):
            SessionJournalWriter.create(path, session())
    finally:
        writer.close()

    with pytest.raises(JournalWriterError, match="already contains records"):
        SessionJournalWriter.create(path, session())


def test_one_writer_serializes_concurrent_appends(tmp_path):
    path = tmp_path / "session.journal.jsonl"
    writer = SessionJournalWriter.create(path, session())
    errors = []

    def append(index):
        try:
            writer.append_history({"role": "user", "content": str(index)})
        except Exception as exc:  # pragma: no cover - asserted through errors
            errors.append(exc)

    threads = [threading.Thread(target=append, args=(index,)) for index in range(20)]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)

        records = read_records(path)
        assert errors == []
        assert all(not thread.is_alive() for thread in threads)
        assert [item["sequence"] for item in records] == list(range(1, 22))
        assert len(writer.state.session["history"]) == 20
    finally:
        writer.close()


def test_effect_intent_is_durable_before_side_effect_and_result_follows(tmp_path):
    path = tmp_path / "session.journal.jsonl"
    target = tmp_path / "effect.txt"
    writer = SessionJournalWriter.create(path, session())
    try:
        with writer.effect(
            "tool",
            call_id="call-1",
            request={"name": "write_file", "path": "effect.txt"},
            replay_policy="interrupt",
        ) as effect:
            assert read_records(path)[-1]["kind"] == "effect_intent"
            target.write_text("done\n", encoding="utf-8")
            effect.complete("ok", {"content": "done"})

        records = read_records(path)
        assert [item["kind"] for item in records[-2:]] == [
            "effect_intent",
            "effect_result",
        ]
        assert records[-1]["operation_id"] == records[-2]["operation_id"]
        assert writer.state.open_operation is None
    finally:
        writer.close()


@pytest.mark.parametrize(
    ("exception", "outcome"),
    [
        (RuntimeError("failed"), "error"),
        (CancellationRequested("cancelled"), "interrupted"),
        (GeneratorExit("closed"), "interrupted"),
    ],
)
def test_effect_context_records_terminal_result_and_reraises(tmp_path, exception, outcome):
    path = tmp_path / f"{outcome}.journal.jsonl"
    writer = SessionJournalWriter.create(path, session(session_id=outcome))
    try:
        with pytest.raises(type(exception), match=str(exception)):
            with writer.effect(
                "provider",
                call_id="call-1",
                request={"model": "test"},
                replay_policy="interrupt",
            ):
                raise exception

        result = read_records(path)[-1]
        assert result["kind"] == "effect_result"
        assert result["payload"]["outcome"] == outcome
        assert result["payload"]["result"] == {
            "error_type": type(exception).__name__
        }
    finally:
        writer.close()


def test_failed_append_does_not_advance_in_memory_state(tmp_path, monkeypatch):
    path = tmp_path / "session.journal.jsonl"
    writer = SessionJournalWriter.create(path, session())
    before = writer.state
    monkeypatch.setattr(
        writer, "_write_line", lambda _record: (_ for _ in ()).throw(OSError("disk"))
    )
    try:
        with pytest.raises(OSError, match="disk"):
            writer.append_history({"role": "user", "content": "not durable"})

        assert writer.state == before
        assert len(read_records(path)) == 1
    finally:
        writer.close()


def test_online_append_does_not_deepcopy_existing_session(tmp_path, monkeypatch):
    history = [
        {"role": "user", "content": f"history-{index}"}
        for index in range(5000)
    ]
    path = tmp_path / "projection.journal.jsonl"
    writer = SessionJournalWriter.create(path, session(history=history), sync=False)
    try:
        from lite.core import session_journal_reducer

        original_deepcopy = session_journal_reducer.copy.deepcopy
        existing_session = writer.state.session

        def reject_existing_state_copy(value, *args, **kwargs):
            if value is existing_session:
                raise AssertionError("online projection copied existing state")
            return original_deepcopy(value, *args, **kwargs)

        monkeypatch.setattr(
            "lite.core.session_journal_reducer.copy.deepcopy",
            reject_existing_state_copy,
        )

        writer.append_history({"role": "assistant", "content": "bounded"})

        assert len(writer.state.session["history"]) == 5001
        assert writer.state.session["history"][-1]["content"] == "bounded"
    finally:
        writer.close()


def test_runtime_history_uses_journal_without_legacy_rewrite(tmp_path):
    agent, store = build_agent(tmp_path)
    writer = attach_writer(agent, store)
    legacy_before = store.path(agent.session["id"]).read_bytes()
    try:
        with patch.object(store, "save", side_effect=AssertionError("legacy rewrite")):
            agent.record({"role": "user", "content": "journal only"})

        assert store.path(agent.session["id"]).read_bytes() == legacy_before
        assert writer.state.session["history"][-1]["content"] == "journal only"
        assert agent.session_path == writer.path
    finally:
        writer.close()


def test_runtime_rejects_writer_with_divergent_history(tmp_path):
    agent, store = build_agent(tmp_path)
    writer = SessionJournalWriter.create(
        store.journal_path(agent.session["id"]),
        {**agent.session, "history": [{"role": "user", "content": "different"}]},
    )
    try:
        with pytest.raises(ValueError, match="journal history does not match"):
            agent.attach_session_journal(writer)
    finally:
        writer.close()


def test_session_switch_closes_and_detaches_the_old_writer(tmp_path):
    agent, store = build_agent(tmp_path)
    writer = attach_writer(agent, store)
    lock_path = writer.lock_path

    new_session_id = agent.clear_session()

    assert new_session_id != writer.state.session["id"]
    assert agent.session_journal_writer is None
    assert not lock_path.exists()
    with pytest.raises(JournalWriterError, match="closed"):
        writer.append_history({"role": "user", "content": "too late"})


def test_runtime_records_provider_permission_and_tool_effect_boundaries(tmp_path):
    agent, store = build_agent(tmp_path, ["<final>Done.</final>"])
    writer = attach_writer(agent, store)
    try:
        assert agent.ask("hello") == "Done."
        assert agent.run_tool(
            "write_file",
            {"path": "notes/result.txt", "content": "ok\n"},
            call_id="model-call-1",
        ).startswith("wrote notes/result.txt")

        records = read_records(writer.path)
        effects = [
            (item["kind"], item["payload"]["effect_type"])
            for item in records
            if item["kind"] in {"effect_intent", "effect_result"}
        ]
        assert effects == [
            ("effect_intent", "provider"),
            ("effect_result", "provider"),
            ("effect_intent", "permission"),
            ("effect_result", "permission"),
            ("effect_intent", "tool"),
            ("effect_result", "tool"),
        ]
        assert {
            item["payload"]["call_id"]
            for item in records
            if item["kind"] == "effect_intent"
            and item["payload"]["effect_type"] in {"permission", "tool"}
        } == {"model-call-1"}
    finally:
        writer.close()


def test_runtime_effect_records_redact_detected_secrets(tmp_path, monkeypatch):
    monkeypatch.setenv("JOURNAL_SECRET", "journal-secret-value")
    agent, store = build_agent(tmp_path)
    agent.secret_env_names.add("JOURNAL_SECRET")
    writer = attach_writer(agent, store)
    try:
        agent.run_tool(
            "write_file",
            {"path": "notes/secret.txt", "content": "journal-secret-value"},
        )

        payload = writer.path.read_text(encoding="utf-8")
        assert "journal-secret-value" not in payload
        assert "<redacted>" in payload
    finally:
        writer.close()
