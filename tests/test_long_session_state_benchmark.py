import copy
import json
import re

from lite.evaluation.long_session_state_benchmark import (
    DEFAULT_EVENTS_PATH,
    DEFAULT_GROUND_TRUTH_PATH,
    DEFAULT_MANIFEST_PATH,
    CHECKPOINT_PROBE_PROMPT,
    CHECKPOINT_SYSTEM_PROMPT,
    extract_json_object,
    load_ground_truth,
    load_jsonl,
    render_compaction_prompt,
    render_probe_request,
    run_benchmark,
    score_candidate,
    score_questions,
    validate_frozen_dataset,
)
from lite.providers import ModelConversation, ModelResult


def _perfect_candidate(ground_truth):
    return {
        "project": ground_truth["project"],
        "current_goal": ground_truth["current_goal"],
        "active_constraints": [
            item["value"] for item in ground_truth["active_constraints"]
        ],
        "confirmed_decisions": [
            {
                "key": item["key"],
                "value": item["value"],
                "source_event_id": item["source_event_id"],
            }
            for item in ground_truth["current_effective_facts"]
        ],
        "open_tasks": [item["task_id"] for item in ground_truth["current_tasks"]],
        "completed_tasks": [
            item["task_id"] for item in ground_truth["completed_tasks"]
        ],
        "failed_actions": ground_truth["failed_actions"],
        "evidence_refs": ground_truth["evidence_refs"],
        "unknown_fields": [],
    }


class BenchmarkFixtureClient:
    def __init__(self, candidate):
        self.candidate = candidate
        self.call_count = 0
        self.last_completion_metadata = {}
        self.supports_prompt_cache = True
        self.supports_append_prompt_cache = True
        self.cache_kwargs = []

    def complete_result(self, request, max_new_tokens, **_kwargs):
        del max_new_tokens
        self.call_count += 1
        self.cache_kwargs.append(dict(_kwargs))
        assert isinstance(request, ModelConversation)
        prompt = str(request.request_messages[-1]["content"])
        if "请将以下 Session Journal 事件压缩" in prompt:
            event_ids = re.findall(r'"event_id":"(E\d{3})"', prompt)
            payload = {
                "summary_version": 1,
                "journal_cursor": max(event_ids) if event_ids else "unknown",
                "current_goal": self.candidate["current_goal"],
                "active_constraints": self.candidate["active_constraints"],
                "current_facts": self.candidate["confirmed_decisions"],
                "open_tasks": self.candidate["open_tasks"],
                "completed_tasks": self.candidate["completed_tasks"],
                "failed_actions": self.candidate["failed_actions"],
                "revoked_items": [],
                "pending_questions": [],
            }
        else:
            payload = self.candidate
        cached_tokens = max(0, (len(request.request_messages) - 1) * 10)
        metadata = {
            "input_tokens": 200,
            "cached_tokens": cached_tokens,
            "output_tokens": 20,
        }
        self.last_completion_metadata = metadata
        return ModelResult(
            text=json.dumps(payload, ensure_ascii=False), metadata=metadata
        )


class FailingFixtureClient:
    def complete_result(self, request, max_new_tokens, **_kwargs):
        del request, max_new_tokens
        raise RuntimeError("provider unavailable")


def test_frozen_dataset_meets_all_contract_counts_and_hashes():
    events = load_jsonl(DEFAULT_EVENTS_PATH)
    truth = load_ground_truth(DEFAULT_GROUND_TRUTH_PATH)

    validation = validate_frozen_dataset(
        events,
        truth,
        events_path=DEFAULT_EVENTS_PATH,
        ground_truth_path=DEFAULT_GROUND_TRUTH_PATH,
        manifest_path=DEFAULT_MANIFEST_PATH,
    )

    assert validation["event_count"] == 100
    assert validation["question_count"] == 20
    assert validation["long_memory_fact_count"] >= 15
    assert validation["manifest_verified"] is True
    assert validation["event_type_counts"]["requirement_update"] >= 8
    assert validation["event_type_counts"]["decision_revoke"] == 5
    assert validation["event_type_counts"]["tool_failure"] == 5
    assert validation["event_type_counts"]["retry"] == 5
    assert validation["event_type_counts"]["rewind"] == 3
    assert validation["event_type_counts"]["resume"] == 3


def test_prompts_preserve_json_shapes_and_same_checkpoint_probe():
    events = load_jsonl(DEFAULT_EVENTS_PATH)
    compact_prompt = render_compaction_prompt(None, events[:2])
    probe = render_probe_request("full_history", "{}")

    assert '"summary_version": 1' in compact_prompt
    assert '"event_id":"E001"' in compact_prompt
    assert CHECKPOINT_SYSTEM_PROMPT in probe
    assert CHECKPOINT_PROBE_PROMPT in probe
    assert extract_json_object("```json\n{\"ok\": true}\n```") == {"ok": True}


def test_exact_grader_accepts_ground_truth_projection():
    truth = load_ground_truth(DEFAULT_GROUND_TRUTH_PATH)
    candidate = _perfect_candidate(truth)

    grade = score_candidate(truth, candidate)
    questions = score_questions(truth, candidate)

    assert grade == {
        "score": 100,
        "correct_facts": 46,
        "missing_facts": [],
        "incorrect_facts": [],
        "stale_facts_used": [],
        "hallucinations": [],
        "evidence_errors": [],
        "schema_violations": [],
        "factual_pass": True,
        "schema_pass": True,
        "pass": True,
    }
    assert questions["correct"] == questions["total"] == 20
    assert questions["accuracy"] == 1.0


def test_exact_grader_penalizes_rewound_or_superseded_value():
    truth = load_ground_truth(DEFAULT_GROUND_TRUTH_PATH)
    candidate = _perfect_candidate(truth)
    database = next(
        item for item in candidate["confirmed_decisions"] if item["key"] == "database"
    )
    database.update({"value": "MySQL 8.0", "source_event_id": "E002"})

    grade = score_candidate(truth, candidate)

    assert grade["score"] == 90
    assert grade["pass"] is False
    assert grade["stale_facts_used"] == [
        {
            "key": "database",
            "value": "MySQL 8.0",
            "source_event_id": "E002",
            "invalidated_by_event_id": "E023",
        }
    ]


def test_exact_grader_separates_schema_alias_from_hallucination():
    truth = load_ground_truth(DEFAULT_GROUND_TRUTH_PATH)
    candidate = _perfect_candidate(truth)
    candidate["confirmed_decisions"].append(
        {
            "key": "security_scan_result",
            "value": "Security scan completed: 0 critical, 2 high findings.",
            "source_event_id": "E091",
        }
    )

    grade = score_candidate(truth, candidate)

    assert grade["score"] == 100
    assert grade["hallucinations"] == []
    assert grade["factual_pass"] is True
    assert grade["schema_pass"] is False
    assert grade["pass"] is False
    assert grade["schema_violations"] == [
        {
            "key": "security_scan_result",
            "actual": candidate["confirmed_decisions"][-1]["value"],
            "error": "non-canonical fact key",
        }
    ]


def test_exact_grader_normalizes_frozen_structured_fact_wording():
    truth = load_ground_truth(DEFAULT_GROUND_TRUTH_PATH)
    candidate = _perfect_candidate(truth)
    security = next(
        item
        for item in candidate["confirmed_decisions"]
        if item["key"] == "security_findings"
    )
    security["value"] = (
        "Security scan completed: 0 critical, 2 high findings SEC-17 and SEC-21。"
    )
    failure = next(
        item
        for item in candidate["confirmed_decisions"]
        if item["key"] == "latest_tool_failure"
    )
    failure["value"] = (
        "run_security_scan / TC08 attempt 1, scanner worker exited with code 137"
    )

    grade = score_candidate(truth, candidate)
    questions = score_questions(truth, candidate)

    assert grade["score"] == 100
    assert grade["pass"] is True
    assert next(
        item for item in questions["results"] if item["question_id"] == "Q08"
    )["correct"] is True


def test_benchmark_runs_three_variants_and_allocates_compaction_cost(tmp_path):
    truth = load_ground_truth(DEFAULT_GROUND_TRUTH_PATH)
    candidate = _perfect_candidate(truth)
    clients = []

    def client_factory():
        client = BenchmarkFixtureClient(copy.deepcopy(candidate))
        clients.append(client)
        return client

    report = run_benchmark(
        client_factory=client_factory,
        output_dir=tmp_path,
        repetitions=1,
        provider_metadata={"name": "fixture", "model": "fixture-model"},
    )

    assert len(clients) == 1
    assert clients[0].call_count == 12
    assert report["summary"]["total_model_calls"] == 12
    assert report["summary"]["variants"]["full_history"]["input_tokens"] == 200
    assert report["summary"]["variants"]["compacted_history"]["input_tokens"] == 1200
    assert report["summary"]["variants"]["resumed_context"]["input_tokens"] == 1000
    assert report["summary"]["variants"]["full_history"]["cached_tokens"] == 0
    assert report["summary"]["variants"]["compacted_history"]["cached_tokens"] == 200
    assert report["summary"]["variants"]["resumed_context"]["cached_tokens"] == 120
    assert report["summary"]["variants"]["compacted_history"]["billable_input_tokens"] == 1000
    assert report["summary"]["variants"]["resumed_context"]["billable_input_tokens"] == 880
    assert report["summary"]["variants"]["compacted_history"]["cache_growth_tokens"] == 80
    assert report["summary"]["variants"]["resumed_context"]["cache_growth_tokens"] == 60
    assert report["summary"]["variants"]["compacted_history"]["probe_input_tokens"] == 200
    assert report["summary"]["variants"]["compacted_history"]["compaction_input_tokens"] == 1000
    assert report["summary"]["variants"]["resumed_context"]["probe_input_tokens"] == 200
    assert report["summary"]["variants"]["resumed_context"]["compaction_input_tokens"] == 800
    assert report["summary"]["allocated_input_tokens"] == 2400
    assert report["summary"]["total_input_tokens"] == 2400
    assert report["summary"]["unallocated_input_tokens"] == 0
    lanes = {}
    for call in report["repetitions"][0]["calls"]:
        lanes.setdefault(call["cache_lane"], []).append(call)
    assert [item["cache_chain_message_count"] for item in lanes["full_history"]] == [1]
    assert [item["cache_chain_message_count"] for item in lanes["compacted_history_builder"]] == [1, 3, 5, 7, 9]
    assert [item["cache_chain_message_count"] for item in lanes["compacted_history"]] == [1]
    assert [item["cache_chain_message_count"] for item in lanes["resumed_checkpoint_builder"]] == [1, 3, 5, 7]
    assert [item["cache_chain_message_count"] for item in lanes["resumed_context"]] == [1]
    keys = {
        lane: {item["prompt_cache_key"] for item in calls}
        for lane, calls in lanes.items()
    }
    assert all(len(values) == 1 for values in keys.values())
    assert len({next(iter(values)) for values in keys.values()}) == 5
    assert all(
        kwargs.get("prompt_cache_key")
        and kwargs.get("prompt_cache_prefix_chars")
        for kwargs in clients[0].cache_kwargs
    )
    assert all(
        item["task_accuracy"] == 1.0
        for item in report["summary"]["variants"].values()
    )
    assert report["summary"]["variants"]["resumed_context"]["resume_recovery_time_p95_ms"] > 0
    assert (tmp_path / "results.json").is_file()
    markdown = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "## Quality" in markdown
    assert "## Cost and latency" in markdown
    assert "| resumed_context |" in markdown


def test_resume_existing_reuses_only_compatible_completed_repetitions(tmp_path):
    truth = load_ground_truth(DEFAULT_GROUND_TRUTH_PATH)
    candidate = _perfect_candidate(truth)

    run_benchmark(
        client_factory=lambda: BenchmarkFixtureClient(copy.deepcopy(candidate)),
        output_dir=tmp_path,
        repetitions=1,
        provider_metadata={"name": "fixture", "model": "fixture-model"},
    )

    def unexpected_factory():
        raise AssertionError("completed repetition should have been reused")

    report = run_benchmark(
        client_factory=unexpected_factory,
        output_dir=tmp_path,
        repetitions=1,
        provider_metadata={"name": "fixture", "model": "fixture-model"},
        resume_existing=True,
    )

    assert report["execution_audit"] == {
        "executed_repetitions": [],
        "reused_repetitions": [1],
    }
    assert report["summary"]["total_model_calls"] == 12


def test_provider_failure_is_blocked_not_scored_as_model_incorrect(tmp_path):
    report = run_benchmark(
        client_factory=FailingFixtureClient,
        output_dir=tmp_path,
        repetitions=1,
    )

    assert report["status"] == "blocked"
    assert report["summary"]["model_call_errors"] == [
        {"error": "RuntimeError: provider unavailable", "count": 3}
    ]
    for item in report["summary"]["variants"].values():
        assert item["evaluable_answer_count"] == 0
        assert item["failed_or_blocked_answer_count"] == 1
        assert item["task_accuracy"] is None
        assert item["question_accuracy"] is None
        assert item["key_fact_recall"] is None
        assert item["stale_fact_misuse_rate"] is None
        assert item["resume_recovery_time_p95_ms"] is None
    assert "| full_history | n/a | n/a | n/a | n/a |" in (
        tmp_path / "report.md"
    ).read_text(encoding="utf-8")
