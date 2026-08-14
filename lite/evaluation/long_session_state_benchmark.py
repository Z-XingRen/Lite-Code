"""Frozen long-session state reconstruction benchmark.

The benchmark intentionally keeps generation out of the runtime path.  Every
variant consumes the same reviewed JSONL journal and ground truth so provider
comparisons are paired and reproducible.
"""

from __future__ import annotations

import hashlib
import copy
import json
import math
import os
import re
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterable
from urllib.parse import urlparse

from lite.config import load_project_env, resolve_provider_config
from lite.providers.base import ModelConversation, complete_model
from lite.providers.runtime import model_client_from_config


ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = ROOT / "benchmarks" / "long_session_state_v1"
DEFAULT_EVENTS_PATH = DATASET_DIR / "events.jsonl"
DEFAULT_GROUND_TRUTH_PATH = DATASET_DIR / "ground_truth.json"
DEFAULT_MANIFEST_PATH = DATASET_DIR / "manifest.json"
DEFAULT_CONFIG_PATH = ROOT / ".lite.toml"
DEFAULT_COMPACTION_BOUNDARIES = (25, 50, 75, 96, 100)
RESUME_CURSOR = 96
ARTIFACT_SCHEMA_VERSION = 3

CANONICAL_FACT_KEYS = (
    "database",
    "deployment_region",
    "api_version",
    "deadline",
    "runtime_language",
    "web_framework",
    "event_bus",
    "cache",
    "authentication",
    "rto",
    "rpo",
    "availability_slo",
    "service_owner",
    "pii_retention",
    "encryption_key",
    "compute_platform",
    "orm",
    "database_replication",
    "interface_strategy",
    "infrastructure_as_code",
    "load_target",
    "rollout_strategy",
    "rollback_threshold",
    "observability",
    "release_approver",
    "security_findings",
    "revoked_decisions",
    "latest_tool_failure",
    "latest_failure_retry_status",
)

ANSWER_FIELDS = frozenset(
    {
        "project",
        "current_goal",
        "active_constraints",
        "confirmed_decisions",
        "open_tasks",
        "completed_tasks",
        "failed_actions",
        "evidence_refs",
        "unknown_fields",
    }
)

REQUIRED_EVENT_TYPES = frozenset(
    {
        "user_requirement",
        "requirement_update",
        "decision",
        "decision_revoke",
        "tool_call",
        "tool_result",
        "tool_failure",
        "retry",
        "checkpoint",
        "rewind",
        "resume",
    }
)

CHECKPOINT_SYSTEM_PROMPT = """你是一个支持长会话的项目执行 Agent。

请严格遵守以下规则：

1. 仅依据当前提供的会话记录、状态投影和 Runtime Evidence 回答。
2. 同一事实发生多次修改时，以最新且未被撤销的记录为准。
3. 被标记为 revoked、rolled_back 或 rewind 后失效的内容不得继续使用。
4. 不得虚构缺失的信息；无法确认时输出 unknown。
5. 工具执行结果的优先级高于模型此前的推测。
6. 回答事实时尽可能附带对应的 event_id 或 evidence_id。
7. 收到 CHECKPOINT 指令时，只输出符合以下结构的 JSON，不要输出解释：

{
  "project": "",
  "current_goal": "",
  "active_constraints": [],
  "confirmed_decisions": [
    {
      "key": "",
      "value": "",
      "source_event_id": ""
    }
  ],
  "open_tasks": [],
  "completed_tasks": [],
  "failed_actions": [
    {
      "tool_name": "",
      "failure_event_id": "",
      "attempt": 1,
      "error": "",
      "retry_event_id": "",
      "retry_attempt": 2,
      "result_event_id": "",
      "retry_status": ""
    }
  ],
  "evidence_refs": [
    {
      "evidence_id": "",
      "source_event_id": ""
    }
  ],
  "unknown_fields": []
}

字段契约：

- active_constraints 只放事件中显式声明为 constraints 的原始约束，不得把普通
  requirement、decision、task 或工具结论提升为约束。
- confirmed_decisions 承载所有当前有效事实。key 只能使用下列规范键；不得创造别名：
  database, deployment_region, api_version, deadline, runtime_language,
  web_framework, event_bus, cache, authentication, rto, rpo,
  availability_slo, service_owner, pii_retention, encryption_key,
  compute_platform, orm, database_replication, interface_strategy,
  infrastructure_as_code, load_target, rollout_strategy, rollback_threshold,
  observability, release_approver, security_findings, revoked_decisions,
  latest_tool_failure, latest_failure_retry_status。
- revoked_decisions 只包含 event_type=decision_revoke 的显式撤销，值使用
  ["decision_id=value", ...]；superseded 和 rewound 项不得混入。
- latest_tool_failure 即使已成功重试也必须保留，格式为
  "tool_name attempt N: error"；latest_failure_retry_status 格式为
  "success on attempt N" 或实际最终状态。
- open_tasks 和 completed_tasks 只能输出 task_id 字符串。
- evidence_refs 只能输出 evidence_id/source_event_id 对象，不得在 ID 字符串中追加说明。"""

COMPACTION_PROMPT = """请将以下 Session Journal 事件压缩为可供后续模型继续执行的上下文状态。

压缩规则：

1. 保留当前仍有效的目标、约束、决策、任务和未解决问题。
2. 对发生修改的事实，只保留最新有效值，同时记录其 source_event_id。
3. 不得保留已撤销或 rewind 后失效的状态。
4. 工具结果不得改写，只保存结论及 evidence_id。
5. 不确定的信息必须标记为 unknown，不得推断。
6. 必须保留未完成任务、失败原因和下一步操作。失败后即使成功重试，
   failed_actions 审计记录也不得删除，并应保留 attempt、retry_attempt、error 和最终状态。
7. 输出 JSON，不要输出自然语言解释。

输出结构没有 project 顶层字段，因此项目名必须作为 current_facts 中
key=project 的事实保留；revoked_items 只能保存失效墓碑，不能把旧值恢复成当前事实。

输出结构：

{
  "summary_version": 1,
  "journal_cursor": "",
  "current_goal": "",
  "active_constraints": [],
  "current_facts": [
    {
      "key": "",
      "value": "",
      "source_event_id": "",
      "evidence_id": null
    }
  ],
  "open_tasks": [],
  "completed_tasks": [],
  "failed_actions": [],
  "revoked_items": [
    {
      "invalidation_kind": "revoked|superseded|rewound",
      "decision_id": null,
      "key": "",
      "value": "",
      "source_event_id": "",
      "invalidated_by_event_id": ""
    }
  ],
  "pending_questions": []
}

已有压缩状态：
{previous_state}

新增 Journal 事件：
{new_events}"""

CHECKPOINT_PROBE_PROMPT = """CHECKPOINT

请根据当前有效状态回答，不得使用已撤销或 rewind 后失效的信息。

随后完成以下检查：

1. 当前目标是什么？
2. 当前仍有效的约束有哪些？
3. 数据库、部署区域、接口版本和截止时间的当前值分别是什么？
4. 哪些 event_type=decision_revoke 的决策已经被显式撤销？不要混入 superseded 或 rewound 项。
5. 最近一次失败的工具调用是什么，是否已经成功重试？即使失败已解决也必须回答。
6. 当前有哪些未完成任务？
7. 每个关键结论来自哪个 event_id 或 evidence_id？

请把所有当前有效事实放入 confirmed_decisions；已撤销决策只作为
key=revoked_decisions 的状态事实记录，不得作为当前方案使用。
必须使用 System Prompt 中列出的规范键；尤其要输出 security_findings、
latest_tool_failure 和 latest_failure_retry_status，不得换用近义键。
只输出 System Prompt 规定的 JSON。"""

SCORING_PROMPT = """你是严格的长会话状态恢复评测器。

请比较 ground_truth 和 candidate_answer，仅判断 candidate_answer 是否正确恢复当前有效状态。

评分规则：

1. 已撤销事实被继续使用：每项扣 10 分。
2. 当前有效事实遗漏：每项扣 5 分。
3. 事实值错误：每项扣 10 分。
4. evidence_id 或 event_id 引用错误：每项扣 3 分。
5. 虚构输入中不存在的信息：每项扣 10 分。
6. 无法确定时正确输出 unknown：不扣分。
7. 只评价事实正确性，不评价措辞和文风。

输出：

{
  "score": 0,
  "correct_facts": 0,
  "missing_facts": [],
  "incorrect_facts": [],
  "stale_facts_used": [],
  "hallucinations": [],
  "evidence_errors": [],
  "pass": false
}

ground_truth：
{ground_truth}

candidate_answer：
{candidate_answer}"""


@dataclass(frozen=True)
class ModelCallRecord:
    stage: str
    repetition: int
    latency_ms: float
    input_chars: int
    input_tokens: int
    cached_tokens: int
    cache_write_tokens: int
    billable_input_tokens: int
    cache_hit: bool
    cache_ratio: float
    cache_controls_enabled: bool
    prompt_cache_key: str | None
    prompt_cache_prefix_chars: int | None
    cache_lane: str
    cache_chain_message_count: int
    output_tokens: int
    token_source: str
    raw_response: str
    parsed_response: dict[str, Any] | None
    error: str | None = None

    def public_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PromptCacheLane:
    """One isolated append-only provider prefix for a benchmark variant."""

    name: str
    prompt_cache_key: str
    messages: list[dict[str, Any]]
    prompt_cache_prefix_chars: int = 0

    @classmethod
    def build(cls, *, dataset_hash: str, repetition: int, name: str):
        identity = json.dumps(
            {
                "schema": "long-session-cache-lane-v1",
                "dataset_hash": dataset_hash,
                "repetition": int(repetition),
                "lane": str(name),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return cls(
            name=str(name),
            prompt_cache_key=hashlib.sha256(identity.encode("utf-8")).hexdigest(),
            messages=[],
        )

    def request(self, prompt: str) -> ModelConversation:
        prompt = str(prompt)
        if not self.prompt_cache_prefix_chars:
            self.prompt_cache_prefix_chars = len(prompt)
        messages = [
            *copy.deepcopy(self.messages),
            {"role": "user", "content": prompt},
        ]
        return ModelConversation(
            initial_input=prompt,
            request_messages=tuple(messages),
        )

    def commit(self, prompt: str, result) -> None:
        self.messages.extend(
            [
                {"role": "user", "content": str(prompt)},
                {
                    "role": "assistant",
                    "content": str(getattr(result, "text", "") or ""),
                    "continuation": tuple(
                        getattr(result, "continuation", ()) or ()
                    ),
                    "tool_calls": [],
                },
            ]
        )

    def clone(self, name: str | None = None):
        return PromptCacheLane(
            name=str(name or self.name),
            prompt_cache_key=self.prompt_cache_key,
            messages=copy.deepcopy(self.messages),
            prompt_cache_prefix_chars=self.prompt_cache_prefix_chars,
        )


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_ground_truth(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_frozen_dataset(
    events: list[dict[str, Any]],
    ground_truth: dict[str, Any],
    *,
    events_path: str | Path | None = None,
    ground_truth_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate the frozen benchmark contract and optional manifest hashes."""

    errors: list[str] = []
    if len(events) != 100:
        errors.append(f"expected 100 events, found {len(events)}")

    ids = [str(item.get("event_id", "")) for item in events]
    expected_ids = [f"E{index:03d}" for index in range(1, 101)]
    if ids != expected_ids:
        errors.append("event_id values must be exactly E001..E100 in order")
    turns = [item.get("turn") for item in events]
    if turns != list(range(1, 101)):
        errors.append("turn values must be exactly 1..100 in order")

    event_types = {str(item.get("event_type", "")) for item in events}
    missing_types = sorted(REQUIRED_EVENT_TYPES - event_types)
    if missing_types:
        errors.append(f"missing event types: {missing_types}")

    counts = _event_type_counts(events)
    minimums = {
        "requirement_update": 8,
        "decision_revoke": 5,
        "tool_failure": 5,
        "retry": 5,
        "rewind": 3,
        "resume": 3,
    }
    for event_type, minimum in minimums.items():
        if counts.get(event_type, 0) < minimum:
            errors.append(
                f"expected at least {minimum} {event_type} events, "
                f"found {counts.get(event_type, 0)}"
            )

    remembered_facts = sum(
        bool((item.get("content") or {}).get("fact"))
        or bool((item.get("content") or {}).get("current_goal"))
        for item in events
    )
    if remembered_facts < 15:
        errors.append(f"expected at least 15 long-memory facts, found {remembered_facts}")

    known_ids = set(ids)
    for event in events:
        if event.get("event_type") != "rewind":
            continue
        invalidated = list((event.get("content") or {}).get("invalidated_event_ids", []))
        if not invalidated:
            errors.append(f"{event['event_id']} has no explicit invalidated_event_ids")
        for invalidated_id in invalidated:
            if invalidated_id not in known_ids or invalidated_id >= event["event_id"]:
                errors.append(
                    f"{event['event_id']} has invalid rewind target {invalidated_id}"
                )

    if int(ground_truth.get("event_count", 0)) != 100:
        errors.append("ground_truth event_count must be 100")
    if len(ground_truth.get("questions", [])) != 20:
        errors.append("ground_truth must define exactly 20 questions")
    if len(ground_truth.get("current_effective_facts", [])) < 15:
        errors.append("ground_truth must define at least 15 current facts")
    truth_fact_keys = {
        str(item.get("key", ""))
        for item in ground_truth.get("current_effective_facts", [])
    }
    if truth_fact_keys != set(CANONICAL_FACT_KEYS):
        errors.append("ground_truth fact keys do not match the canonical answer contract")

    manifest: dict[str, Any] | None = None
    if manifest_path and Path(manifest_path).is_file():
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        expected_hashes = dict(manifest.get("sha256", {}) or {})
        for name, path in (
            ("events.jsonl", events_path),
            ("ground_truth.json", ground_truth_path),
        ):
            if path is None:
                continue
            actual = file_sha256(path)
            if expected_hashes.get(name) != actual:
                errors.append(f"frozen hash mismatch for {name}")

    if errors:
        raise ValueError("invalid long-session dataset: " + "; ".join(errors))
    return {
        "event_count": len(events),
        "event_type_counts": counts,
        "long_memory_fact_count": remembered_facts,
        "question_count": len(ground_truth.get("questions", [])),
        "manifest_verified": manifest is not None,
    }


def render_compaction_prompt(
    previous_state: dict[str, Any] | None, new_events: Iterable[dict[str, Any]]
) -> str:
    previous_json = json.dumps(
        previous_state or {}, ensure_ascii=False, separators=(",", ":")
    )
    events_jsonl = "\n".join(
        json.dumps(item, ensure_ascii=False, separators=(",", ":"))
        for item in new_events
    )
    return COMPACTION_PROMPT.replace("{previous_state}", previous_json).replace(
        "{new_events}", events_jsonl
    )


def render_probe_request(context_kind: str, context: Any) -> str:
    context_json = (
        context
        if isinstance(context, str)
        else json.dumps(context, ensure_ascii=False, separators=(",", ":"))
    )
    return (
        f"[SYSTEM]\n{CHECKPOINT_SYSTEM_PROMPT}\n\n"
        f"[CONTEXT kind={context_kind}]\n{context_json}\n\n"
        f"[USER]\n{CHECKPOINT_PROBE_PROMPT}"
    )


def render_scoring_prompt(
    ground_truth: dict[str, Any], candidate_answer: dict[str, Any]
) -> str:
    truth_json = json.dumps(
        ground_truth, ensure_ascii=False, separators=(",", ":")
    )
    candidate_json = json.dumps(
        candidate_answer, ensure_ascii=False, separators=(",", ":")
    )
    return SCORING_PROMPT.replace("{ground_truth}", truth_json).replace(
        "{candidate_answer}", candidate_json
    )


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract the first complete JSON object, allowing a Markdown fence."""

    raw = str(text or "").strip()
    decoder = json.JSONDecoder()
    for index, char in enumerate(raw):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(raw[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("model response did not contain a JSON object")


def score_candidate(
    ground_truth: dict[str, Any], candidate_answer: dict[str, Any]
) -> dict[str, Any]:
    """Apply the benchmark's exact structured penalties without an LLM judge."""

    missing: list[dict[str, Any]] = []
    incorrect: list[dict[str, Any]] = []
    stale: list[dict[str, Any]] = []
    hallucinations: list[dict[str, Any]] = []
    evidence_errors: list[dict[str, Any]] = []
    schema_violations = _answer_schema_violations(candidate_answer)
    correct = 0

    for key in ("project", "current_goal"):
        expected = ground_truth.get(key)
        actual = candidate_answer.get(key)
        if _is_missing(actual):
            missing.append({"key": key, "expected": expected})
        elif _equivalent(actual, expected):
            correct += 1
        else:
            incorrect.append({"key": key, "expected": expected, "actual": actual})

    expected_constraints = [
        item.get("value") if isinstance(item, dict) else item
        for item in ground_truth.get("active_constraints", [])
    ]
    actual_constraints = list(candidate_answer.get("active_constraints", []) or [])
    correct_delta, missing_delta, extra_delta = _score_unordered_values(
        "active_constraint", expected_constraints, actual_constraints
    )
    correct += correct_delta
    missing.extend(missing_delta)
    schema_violations.extend(
        {**item, "error": "unexpected value in constrained field"}
        for item in extra_delta
    )

    expected_facts = {
        str(item["key"]): item
        for item in ground_truth.get("current_effective_facts", [])
    }
    actual_facts, fact_schema_violations = _candidate_fact_map(candidate_answer)
    schema_violations.extend(fact_schema_violations)
    invalid_by_key: dict[str, list[dict[str, Any]]] = {}
    for item in ground_truth.get("invalidated_facts", []):
        invalid_by_key.setdefault(str(item.get("key", "")), []).append(item)

    for key, expected in expected_facts.items():
        actual = actual_facts.get(key)
        if actual is None or _is_missing(actual.get("value")):
            missing.append({"key": key, "expected": expected.get("value")})
            continue
        if _fact_value_equivalent(
            key, actual.get("value"), expected.get("value")
        ):
            correct += 1
            expected_source = expected.get("source_event_id")
            actual_source = actual.get("source_event_id")
            if expected_source and actual_source != expected_source:
                evidence_errors.append(
                    {
                        "key": key,
                        "expected_event_id": expected_source,
                        "actual_event_id": actual_source,
                    }
                )
            continue

        stale_match = next(
            (
                item
                for item in invalid_by_key.get(key, [])
                if _equivalent(actual.get("value"), item.get("value"))
            ),
            None,
        )
        if stale_match:
            stale.append(
                {
                    "key": key,
                    "value": actual.get("value"),
                    "source_event_id": stale_match.get("source_event_id"),
                    "invalidated_by_event_id": stale_match.get(
                        "invalidated_by_event_id"
                    ),
                }
            )
        else:
            incorrect.append(
                {
                    "key": key,
                    "expected": expected.get("value"),
                    "actual": actual.get("value"),
                }
            )

    recognized_extras: set[str] = set()
    for key, actual in actual_facts.items():
        if key in expected_facts or key in recognized_extras:
            continue
        stale_match = next(
            (
                item
                for item in invalid_by_key.get(key, [])
                if _equivalent(actual.get("value"), item.get("value"))
            ),
            None,
        )
        if stale_match:
            stale.append(
                {
                    "key": key,
                    "value": actual.get("value"),
                    "source_event_id": stale_match.get("source_event_id"),
                    "invalidated_by_event_id": stale_match.get(
                        "invalidated_by_event_id"
                    ),
                }
            )
        else:
            schema_violations.append(
                {
                    "key": key,
                    "actual": actual.get("value"),
                    "error": "non-canonical fact key",
                }
            )

    for candidate_field, truth_field, item_key in (
        ("open_tasks", "current_tasks", "task_id"),
        ("completed_tasks", "completed_tasks", "task_id"),
    ):
        expected_values = [
            _item_identifier(item, item_key)
            for item in ground_truth.get(truth_field, [])
        ]
        actual_values = [
            _item_identifier(item, item_key)
            for item in candidate_answer.get(candidate_field, []) or []
        ]
        correct_delta, missing_delta, extra_delta = _score_unordered_values(
            candidate_field, expected_values, actual_values
        )
        correct += correct_delta
        missing.extend(missing_delta)
        invalid_tasks = {
            str(item.get("task_id", "")): item
            for item in ground_truth.get("invalidated_tasks", [])
        }
        for item in extra_delta:
            value = str(item.get("actual", ""))
            if value in invalid_tasks:
                task = invalid_tasks[value]
                stale.append(
                    {
                        "key": candidate_field,
                        "value": value,
                        "source_event_id": task.get("source_event_id"),
                        "invalidated_by_event_id": task.get(
                            "invalidated_by_event_id"
                        ),
                    }
                )
            else:
                schema_violations.append(
                    {**item, "error": "unexpected task identifier"}
                )

    actual_evidence, evidence_schema_violations = _candidate_evidence_ids(
        candidate_answer
    )
    schema_violations.extend(evidence_schema_violations)
    expected_evidence = {
        str(item.get("evidence_id"))
        for item in ground_truth.get("current_effective_facts", [])
        if item.get("evidence_id")
    }
    for evidence_id in sorted(expected_evidence - actual_evidence):
        evidence_errors.append(
            {"evidence_id": evidence_id, "error": "required evidence missing"}
        )
    valid_evidence = {
        str(item.get("evidence_id"))
        for item in ground_truth.get("evidence_refs", [])
    }
    invalid_evidence = {
        str(item.get("evidence_id")): item
        for item in ground_truth.get("invalid_evidence_refs", [])
    }
    for evidence_id in sorted(actual_evidence - valid_evidence):
        detail = invalid_evidence.get(evidence_id)
        evidence_errors.append(
            {
                "evidence_id": evidence_id,
                "error": (
                    f"invalidated by {detail.get('invalidated_by_event_id')}"
                    if detail
                    else "evidence does not exist"
                ),
            }
        )

    penalty = (
        len(missing) * 5
        + len(incorrect) * 10
        + len(stale) * 10
        + len(hallucinations) * 10
        + len(evidence_errors) * 3
    )
    score = max(0, 100 - penalty)
    factual_pass = (
        score >= 90
        and not incorrect
        and not stale
        and not hallucinations
    )
    schema_pass = not schema_violations
    passed = factual_pass and schema_pass
    return {
        "score": score,
        "correct_facts": correct,
        "missing_facts": missing,
        "incorrect_facts": incorrect,
        "stale_facts_used": stale,
        "hallucinations": hallucinations,
        "evidence_errors": evidence_errors,
        "schema_violations": schema_violations,
        "factual_pass": factual_pass,
        "schema_pass": schema_pass,
        "pass": passed,
    }


def score_questions(
    ground_truth: dict[str, Any], candidate_answer: dict[str, Any]
) -> dict[str, Any]:
    facts, _ = _candidate_fact_map(candidate_answer)
    fact_keys = {
        "Q03": "database",
        "Q04": "deployment_region",
        "Q05": "api_version",
        "Q06": "deadline",
        "Q07": "revoked_decisions",
        "Q08": "latest_tool_failure",
        "Q09": "latest_failure_retry_status",
        "Q11": "authentication",
        "Q12": "runtime_language",
        "Q13": "web_framework",
        "Q14": "event_bus",
        "Q15": "rto",
        "Q16": "rpo",
        "Q17": "availability_slo",
        "Q18": "compute_platform",
        "Q19": "rollout_strategy",
        "Q20": "load_target",
    }
    actual_answers: dict[str, Any] = {
        "Q01": candidate_answer.get("current_goal"),
        "Q02": candidate_answer.get("active_constraints", []),
        "Q10": [
            _item_identifier(item, "task_id")
            for item in candidate_answer.get("open_tasks", []) or []
        ],
    }
    for question_id, fact_key in fact_keys.items():
        actual_answers[question_id] = (facts.get(fact_key) or {}).get("value")

    results = []
    for question in ground_truth.get("questions", []):
        question_id = str(question.get("question_id"))
        actual = actual_answers.get(question_id)
        expected = question.get("answer")
        results.append(
            {
                "question_id": question_id,
                "correct": (
                    _fact_value_equivalent(
                        fact_keys[question_id], actual, expected
                    )
                    if question_id in fact_keys
                    else _equivalent(actual, expected)
                ),
                "expected": expected,
                "actual": actual,
            }
        )
    correct = sum(bool(item["correct"]) for item in results)
    return {
        "correct": correct,
        "total": len(results),
        "accuracy": correct / len(results) if results else 0.0,
        "results": results,
    }


def build_live_client_factory(
    *,
    config_path: str | Path | None = None,
    provider: str | None = None,
    timeout: int = 300,
) -> tuple[Callable[[], Any], dict[str, Any]]:
    """Build a temperature-zero client from the selected TOML provider profile."""

    selected = Path(config_path or DEFAULT_CONFIG_PATH).resolve()
    load_project_env(selected.parent, override=True)
    config = resolve_provider_config(
        provider,
        start=selected.parent,
        config_path=str(selected),
    )
    if not config.api_key:
        raise ValueError(
            f"provider {config.name!r} has no API key; configure {selected} or .env"
        )

    def factory():
        return model_client_from_config(
            config,
            SimpleNamespace(temperature=0, openai_timeout=timeout),
            timeout=timeout,
        )

    return factory, {
        "name": config.name,
        "protocol": config.protocol,
        "model": config.model,
        "reasoning_effort": config.reasoning_effort,
        "temperature": 0,
        "base_url_hostname": urlparse(config.base_url).hostname,
        "config_path": str(selected),
        "api_key_present": bool(config.api_key),
    }


def run_benchmark(
    *,
    client_factory: Callable[[], Any],
    output_dir: str | Path,
    events_path: str | Path = DEFAULT_EVENTS_PATH,
    ground_truth_path: str | Path = DEFAULT_GROUND_TRUTH_PATH,
    manifest_path: str | Path | None = DEFAULT_MANIFEST_PATH,
    repetitions: int = 5,
    max_output_tokens: int = 8192,
    compaction_boundaries: Iterable[int] = DEFAULT_COMPACTION_BOUNDARIES,
    provider_metadata: dict[str, Any] | None = None,
    resume_existing: bool = False,
) -> dict[str, Any]:
    """Run full, compacted, and resume probes on the same frozen session."""

    if repetitions < 1:
        raise ValueError("repetitions must be >= 1")
    events_path = Path(events_path)
    ground_truth_path = Path(ground_truth_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    events = load_jsonl(events_path)
    ground_truth = load_ground_truth(ground_truth_path)
    validation = validate_frozen_dataset(
        events,
        ground_truth,
        events_path=events_path,
        ground_truth_path=ground_truth_path,
        manifest_path=manifest_path,
    )
    boundaries = tuple(int(item) for item in compaction_boundaries)
    if (
        not boundaries
        or tuple(sorted(set(boundaries))) != boundaries
        or boundaries[-1] != len(events)
        or RESUME_CURSOR not in boundaries
    ):
        raise ValueError(
            "compaction boundaries must be sorted, unique, include 96, and end at 100"
        )

    repeat_records = []
    all_calls: list[ModelCallRecord] = []
    dataset_hash = file_sha256(events_path)
    ground_truth_hash = file_sha256(ground_truth_path)
    run_identity = _build_run_identity(
        dataset_hash=dataset_hash,
        ground_truth_hash=ground_truth_hash,
        boundaries=boundaries,
        max_output_tokens=max_output_tokens,
        provider_metadata=provider_metadata,
    )
    executed_repetitions: list[int] = []
    reused_repetitions: list[int] = []
    for repetition in range(1, repetitions + 1):
        repeat_dir = output_dir / f"repeat-{repetition:03d}"
        repeat_dir.mkdir(parents=True, exist_ok=True)
        repeat_result_path = repeat_dir / "result.json"
        if resume_existing and repeat_result_path.is_file():
            reusable = _load_reusable_repeat(repeat_result_path, run_identity)
            if reusable is not None:
                repeat_records.append(reusable)
                all_calls.extend(_model_calls_from_record(reusable))
                reused_repetitions.append(repetition)
                continue
            _archive_repeat_result(repeat_result_path)

        client = client_factory()
        executed_repetitions.append(repetition)
        calls: list[ModelCallRecord] = []
        answers: dict[str, Any] = {}

        full_context = "\n".join(
            json.dumps(item, ensure_ascii=False, separators=(",", ":"))
            for item in events
        )
        full_lane = PromptCacheLane.build(
            dataset_hash=dataset_hash,
            repetition=repetition,
            name="full_history",
        )
        full_call = _call_json_model(
            client,
            render_probe_request("full_history", full_context),
            stage="probe_full_history",
            repetition=repetition,
            max_output_tokens=max_output_tokens,
            cache_lane=full_lane,
        )
        calls.append(full_call)
        answers["full_history"] = _answer_record(full_call, ground_truth)

        compact_builder_lane = PromptCacheLane.build(
            dataset_hash=dataset_hash,
            repetition=repetition,
            name="compacted_history_builder",
        )
        compact_states, compact_calls, compact_error = _run_compaction_lane(
            client=client,
            events=events,
            boundaries=boundaries,
            repetition=repetition,
            max_output_tokens=max_output_tokens,
            cache_lane=compact_builder_lane,
            stage_prefix="compacted_compact",
        )
        calls.extend(compact_calls)
        if compact_error is None:
            compact_probe_lane = PromptCacheLane.build(
                dataset_hash=dataset_hash,
                repetition=repetition,
                name="compacted_history",
            )
            compact_call = _call_json_model(
                client,
                render_probe_request("compacted_history", compact_states[len(events)]),
                stage="probe_compacted_history",
                repetition=repetition,
                max_output_tokens=max_output_tokens,
                cache_lane=compact_probe_lane,
            )
            calls.append(compact_call)
            answers["compacted_history"] = _answer_record(
                compact_call, ground_truth
            )
        else:
            answers["compacted_history"] = _blocked_answer_record(
                compact_error, ground_truth
            )

        resume_boundaries = tuple(
            boundary for boundary in boundaries if boundary <= RESUME_CURSOR
        )
        resume_builder_lane = PromptCacheLane.build(
            dataset_hash=dataset_hash,
            repetition=repetition,
            name="resumed_checkpoint_builder",
        )
        resume_states, resume_calls, resume_error = _run_compaction_lane(
            client=client,
            events=events,
            boundaries=resume_boundaries,
            repetition=repetition,
            max_output_tokens=max_output_tokens,
            cache_lane=resume_builder_lane,
            stage_prefix="resume_checkpoint_compact",
        )
        calls.extend(resume_calls)
        resume_recovery_time_ms: float | None = None
        if resume_error is None:
            checkpoint_path = repeat_dir / "resume-checkpoint-E096.json"
            _atomic_write_json(checkpoint_path, resume_states[RESUME_CURSOR])
            recovery_started = time.perf_counter()
            restored_state = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            resume_context = {
                "restored_compacted_state": restored_state,
                "resume_event_id": "E096",
                "events_after_resume": events[RESUME_CURSOR:],
            }
            resume_probe_lane = PromptCacheLane.build(
                dataset_hash=dataset_hash,
                repetition=repetition,
                name="resumed_context",
            )
            resume_call = _call_json_model(
                client,
                render_probe_request("resumed_context", resume_context),
                stage="probe_resumed_context",
                repetition=repetition,
                max_output_tokens=max_output_tokens,
                cache_lane=resume_probe_lane,
            )
            resume_recovery_time_ms = (
                time.perf_counter() - recovery_started
            ) * 1000.0
            calls.append(resume_call)
            answers["resumed_context"] = _answer_record(resume_call, ground_truth)
        else:
            answers["resumed_context"] = _blocked_answer_record(
                resume_error, ground_truth
            )

        allocation = _allocate_repeat_costs(
            calls, resume_recovery_time_ms=resume_recovery_time_ms
        )
        repeat_record = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "repetition": repetition,
            "status": (
                "completed"
                if all(
                    answer.get("status") == "completed"
                    for answer in answers.values()
                )
                else "blocked"
            ),
            "run_identity": run_identity,
            "answers": answers,
            "call_allocation": allocation,
            "compaction_error": compact_error or resume_error,
            "compaction_errors": {
                "compacted_history": compact_error,
                "resumed_context": resume_error,
            },
            "calls": [call.public_dict() for call in calls],
        }
        repeat_records.append(repeat_record)
        all_calls.extend(calls)
        _atomic_write_json(repeat_result_path, repeat_record)

    report = {
        "artifact_type": "long-session-state-benchmark",
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "status": (
            "completed"
            if all(
                answer.get("status") == "completed"
                for record in repeat_records
                for answer in record.get("answers", {}).values()
            )
            else "blocked"
        ),
        "dataset": {
            "id": ground_truth.get("dataset_id"),
            "events_path": str(events_path.resolve()),
            "ground_truth_path": str(ground_truth_path.resolve()),
            "events_sha256": dataset_hash,
            "ground_truth_sha256": ground_truth_hash,
            "validation": validation,
        },
        "provider": dict(provider_metadata or {}),
        "settings": {
            "repetitions": repetitions,
            "temperature": 0,
            "max_output_tokens": max_output_tokens,
            "compaction_boundaries": list(boundaries),
            "resume_cursor": f"E{RESUME_CURSOR:03d}",
            "prompt_cache_mode": "isolated_append_only_lanes",
            "resume_existing": bool(resume_existing),
        },
        "summary": _summarize_runs(repeat_records, all_calls, ground_truth),
        "execution_audit": {
            "executed_repetitions": executed_repetitions,
            "reused_repetitions": reused_repetitions,
        },
        "repetitions": repeat_records,
    }
    _atomic_write_json(output_dir / "results.json", report)
    _atomic_write_text(output_dir / "report.md", render_markdown_report(report) + "\n")
    return report


def _build_run_identity(
    *,
    dataset_hash: str,
    ground_truth_hash: str,
    boundaries: tuple[int, ...],
    max_output_tokens: int,
    provider_metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    provider = dict(provider_metadata or {})
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "events_sha256": dataset_hash,
        "ground_truth_sha256": ground_truth_hash,
        "compaction_boundaries": list(boundaries),
        "resume_cursor": f"E{RESUME_CURSOR:03d}",
        "max_output_tokens": int(max_output_tokens),
        "provider": {
            key: provider.get(key)
            for key in (
                "name",
                "protocol",
                "model",
                "reasoning_effort",
                "temperature",
            )
        },
    }


def _load_reusable_repeat(
    path: Path, expected_identity: dict[str, Any]
) -> dict[str, Any] | None:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot resume malformed repeat artifact {path}: {exc}") from exc
    if record.get("run_identity") != expected_identity:
        raise ValueError(
            f"cannot resume incompatible repeat artifact {path}; "
            "use a new output directory or remove --resume-existing"
        )
    answers = dict(record.get("answers", {}) or {})
    completed = set(answers) == {
        "full_history",
        "compacted_history",
        "resumed_context",
    } and all(answer.get("status") == "completed" for answer in answers.values())
    return record if completed else None


def _model_calls_from_record(record: dict[str, Any]) -> list[ModelCallRecord]:
    return [ModelCallRecord(**dict(item)) for item in record.get("calls", []) or []]


def _archive_repeat_result(path: Path) -> Path:
    index = 1
    while True:
        archive = path.with_name(f"result.failed-{index:03d}.json")
        if not archive.exists():
            break
        index += 1
    _atomic_write_text(archive, path.read_text(encoding="utf-8"))
    return archive


def _atomic_write_json(path: str | Path, value: Any) -> None:
    _atomic_write_text(
        path, json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    )


def _atomic_write_text(path: str | Path, value: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(str(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, target)
        temporary_path = None
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def render_markdown_report(report: dict[str, Any]) -> str:
    provider = dict(report.get("provider", {}) or {})
    summary = dict(report.get("summary", {}) or {})
    lines = [
        "# Long-session State Reconstruction Benchmark",
        "",
        f"- Status: `{report.get('status', 'unknown')}`",
        f"- Dataset: `{report.get('dataset', {}).get('id', '')}`",
        f"- Provider/model: `{provider.get('name', 'custom')}` / `{provider.get('model', 'custom')}`",
        f"- Temperature: `{report.get('settings', {}).get('temperature', 0)}`",
        f"- Repetitions: `{report.get('settings', {}).get('repetitions', 0)}`",
    ]
    for error in summary.get("model_call_errors", []) or []:
        lines.append(
            f"- Blocked model calls: `{error.get('count', 0)}` × `{error.get('error', '')}`"
        )
    lines.extend(
        [
            "",
            "## Quality",
            "",
            "| Variant | Strict pass | Factual pass | Schema pass | 20Q accuracy | Key fact recall | Stale misuse |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for variant in ("full_history", "compacted_history", "resumed_context"):
        item = dict((summary.get("variants", {}) or {}).get(variant, {}) or {})
        lines.append(
            "| {variant} | {task} | {factual} | {schema} | {questions} | "
            "{recall} | {stale} |".format(
                variant=variant,
                task=_format_percent(item.get("task_accuracy"), 1),
                factual=_format_percent(item.get("factual_accuracy"), 1),
                schema=_format_percent(item.get("schema_accuracy"), 1),
                questions=_format_percent(item.get("question_accuracy"), 1),
                recall=_format_percent(item.get("key_fact_recall"), 1),
                stale=_format_percent(item.get("stale_fact_misuse_rate"), 2),
            )
        )
    lines.extend(
        [
            "",
            "## Cost and latency",
            "",
            "Probe input is the directly comparable checkpoint-answer cost. Total input also includes isolated compaction/checkpoint construction and is non-overlapping across variants.",
            "",
            "| Variant | Probe input | Compaction input | Total input | Probe billable | Total billable | Cache rate | Key stable | Cache growth | Probe P95 | Compaction-call P95 | Pipeline P95 | Resume recovery |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for variant in ("full_history", "compacted_history", "resumed_context"):
        item = dict((summary.get("variants", {}) or {}).get(variant, {}) or {})
        compaction_p95 = item.get("compaction_p95_latency_ms")
        resume_p95 = item.get("resume_recovery_time_p95_ms")
        lines.append(
            "| {variant} | {probe_input} | {compaction_input} | {total_input} | "
            "{probe_billable} | {total_billable} | {cache_rate} | {key_stable} | "
            "{cache_growth} | {probe_p95:.1f} ms | {compaction_p95} | "
            "{pipeline_p95:.1f} ms | {resume_p95} |".format(
                variant=variant,
                probe_input=int(item.get("probe_input_tokens", 0)),
                compaction_input=int(item.get("compaction_input_tokens", 0)),
                total_input=int(item.get("input_tokens", 0)),
                probe_billable=int(item.get("probe_billable_input_tokens", 0)),
                total_billable=int(item.get("billable_input_tokens", 0)),
                cache_rate=_format_percent(item.get("cache_ratio"), 1),
                key_stable=_format_percent(item.get("cache_key_stable_rate"), 1),
                cache_growth=int(item.get("cache_growth_tokens", 0)),
                probe_p95=float(item.get("probe_p95_latency_ms", 0.0)),
                compaction_p95=(
                    f"{float(compaction_p95):.1f} ms"
                    if compaction_p95 is not None
                    else "n/a"
                ),
                pipeline_p95=float(item.get("pipeline_p95_latency_ms", 0.0)),
                resume_p95=(
                    f"{float(resume_p95):.1f} ms"
                    if resume_p95 is not None
                    else "n/a"
                ),
            )
        )
    lines.extend(
        [
            "",
            f"- Overall P95 model-call latency: `{float(summary.get('p95_model_latency_ms', 0.0)):.1f} ms`",
            f"- Total model input tokens: `{int(summary.get('total_input_tokens', 0))}`",
            f"- Total cached input tokens: `{int(summary.get('total_cached_tokens', 0))}`",
            f"- Total billable input tokens: `{int(summary.get('total_billable_input_tokens', 0))}`",
            f"- Overall cache ratio: `{_format_percent(summary.get('cache_ratio'), 1)}`",
            f"- Cost accounting: `{summary.get('cost_accounting', '')}`",
            f"- Allocated / unallocated input: `{int(summary.get('allocated_input_tokens', 0))}` / `{int(summary.get('unallocated_input_tokens', 0))}`",
            f"- Token source: `{summary.get('token_source', '')}`",
        ]
    )
    return "\n".join(lines)


def _run_compaction_lane(
    *,
    client: Any,
    events: list[dict[str, Any]],
    boundaries: tuple[int, ...],
    repetition: int,
    max_output_tokens: int,
    cache_lane: PromptCacheLane,
    stage_prefix: str,
) -> tuple[dict[int, dict[str, Any]], list[ModelCallRecord], str | None]:
    states: dict[int, dict[str, Any]] = {}
    calls: list[ModelCallRecord] = []
    previous_state: dict[str, Any] | None = None
    start = 0
    for boundary in boundaries:
        prompt = (
            f"[SYSTEM]\n{CHECKPOINT_SYSTEM_PROMPT}\n\n[USER]\n"
            + render_compaction_prompt(previous_state, events[start:boundary])
        )
        call = _call_json_model(
            client,
            prompt,
            stage=f"{stage_prefix}_{boundary:03d}",
            repetition=repetition,
            max_output_tokens=max_output_tokens,
            cache_lane=cache_lane,
        )
        calls.append(call)
        if call.parsed_response is None:
            return (
                states,
                calls,
                call.error or "compaction returned invalid JSON",
            )
        previous_state = call.parsed_response
        states[boundary] = previous_state
        start = boundary
    return states, calls, None


def _call_json_model(
    client: Any,
    prompt: str,
    *,
    stage: str,
    repetition: int,
    max_output_tokens: int,
    cache_lane: PromptCacheLane | None = None,
) -> ModelCallRecord:
    request = cache_lane.request(prompt) if cache_lane else prompt
    cache_controls_enabled = bool(
        cache_lane is not None and getattr(client, "supports_prompt_cache", False)
    )
    cache_kwargs = (
        {
            "prompt_cache_key": cache_lane.prompt_cache_key,
            "prompt_cache_prefix_chars": cache_lane.prompt_cache_prefix_chars,
        }
        if cache_controls_enabled and cache_lane is not None
        else {}
    )
    input_chars = _request_input_chars(request, prompt)
    cache_chain_message_count = len(
        getattr(request, "request_messages", ()) or ()
    )
    started = time.perf_counter()
    raw = ""
    parsed: dict[str, Any] | None = None
    error: str | None = None
    metadata: dict[str, Any] = {}
    try:
        result = complete_model(
            client,
            request,
            max_output_tokens,
            **cache_kwargs,
        )
        raw = str(result.text or "")
        metadata = dict(
            result.metadata or getattr(client, "last_completion_metadata", {}) or {}
        )
        parsed = extract_json_object(raw)
        if cache_lane is not None:
            cache_lane.commit(prompt, result)
    except Exception as exc:  # benchmark rows preserve provider and parse failures
        error = f"{type(exc).__name__}: {exc}"
        metadata = dict(getattr(client, "last_completion_metadata", {}) or {})
    latency_ms = (time.perf_counter() - started) * 1000.0
    actual_input = _optional_int(metadata.get("input_tokens"))
    actual_output = _optional_int(metadata.get("output_tokens"))
    if actual_input is None:
        input_tokens = max(1, math.ceil(input_chars / 4))
        token_source = "estimated_chars_div_4"
    else:
        input_tokens = actual_input
        token_source = "provider_usage"
    output_tokens = (
        actual_output
        if actual_output is not None
        else max(0, math.ceil(len(raw) / 4))
    )
    cached_tokens = min(
        input_tokens,
        max(0, _optional_int(metadata.get("cached_tokens")) or 0),
    )
    cache_write_tokens = max(
        0, _optional_int(metadata.get("cache_write_tokens")) or 0
    )
    billable_input_tokens = max(0, input_tokens - cached_tokens)
    return ModelCallRecord(
        stage=stage,
        repetition=repetition,
        latency_ms=latency_ms,
        input_chars=input_chars,
        input_tokens=input_tokens,
        cached_tokens=cached_tokens,
        cache_write_tokens=cache_write_tokens,
        billable_input_tokens=billable_input_tokens,
        cache_hit=cached_tokens > 0,
        cache_ratio=(cached_tokens / input_tokens if input_tokens else 0.0),
        cache_controls_enabled=cache_controls_enabled,
        prompt_cache_key=(
            cache_lane.prompt_cache_key
            if cache_controls_enabled and cache_lane is not None
            else None
        ),
        prompt_cache_prefix_chars=(
            cache_lane.prompt_cache_prefix_chars
            if cache_controls_enabled and cache_lane is not None
            else None
        ),
        cache_lane=cache_lane.name if cache_lane is not None else "",
        cache_chain_message_count=cache_chain_message_count,
        output_tokens=output_tokens,
        token_source=token_source,
        raw_response=raw,
        parsed_response=parsed,
        error=error,
    )


def _request_input_chars(request: Any, prompt: str) -> int:
    messages = getattr(request, "request_messages", ()) or ()
    if not messages:
        return len(str(prompt))
    return len(
        json.dumps(
            list(messages),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )


def _answer_record(
    call: ModelCallRecord, ground_truth: dict[str, Any]
) -> dict[str, Any]:
    candidate = call.parsed_response or {}
    return {
        "status": "completed" if call.parsed_response is not None else "failed",
        "candidate_answer": candidate,
        "grade": score_candidate(ground_truth, candidate),
        "question_score": score_questions(ground_truth, candidate),
        "error": call.error,
    }


def _blocked_answer_record(
    reason: str, ground_truth: dict[str, Any]
) -> dict[str, Any]:
    candidate: dict[str, Any] = {}
    return {
        "status": "blocked",
        "candidate_answer": candidate,
        "grade": score_candidate(ground_truth, candidate),
        "question_score": score_questions(ground_truth, candidate),
        "error": reason,
    }


def _allocate_repeat_costs(
    calls: list[ModelCallRecord],
    *,
    resume_recovery_time_ms: float | None = None,
) -> dict[str, Any]:
    compact_calls = [
        call for call in calls if call.stage.startswith("compacted_compact_")
    ]
    resume_calls = [
        call
        for call in calls
        if call.stage.startswith("resume_checkpoint_compact_")
    ]
    by_stage = {call.stage: call for call in calls}
    allocation: dict[str, Any] = {}
    variant_phases = {
        "full_history": ([], [by_stage.get("probe_full_history")]),
        "compacted_history": (
            compact_calls,
            [by_stage.get("probe_compacted_history")],
        ),
        "resumed_context": (
            resume_calls,
            [by_stage.get("probe_resumed_context")],
        ),
    }
    for variant, (maintenance_values, probe_values) in variant_phases.items():
        maintenance = [item for item in maintenance_values if item is not None]
        probes = [item for item in probe_values if item is not None]
        selected = maintenance + probes
        total = _aggregate_call_costs(selected)
        maintenance_cost = _aggregate_call_costs(maintenance)
        probe_cost = _aggregate_call_costs(probes)
        total.update(
            {
                "compaction_input_tokens": maintenance_cost["input_tokens"],
                "compaction_cached_tokens": maintenance_cost["cached_tokens"],
                "compaction_billable_input_tokens": maintenance_cost[
                    "billable_input_tokens"
                ],
                "compaction_output_tokens": maintenance_cost["output_tokens"],
                "compaction_call_count": maintenance_cost["call_count"],
                "compaction_call_latencies_ms": maintenance_cost[
                    "call_latencies_ms"
                ],
                "probe_input_tokens": probe_cost["input_tokens"],
                "probe_cached_tokens": probe_cost["cached_tokens"],
                "probe_billable_input_tokens": probe_cost[
                    "billable_input_tokens"
                ],
                "probe_output_tokens": probe_cost["output_tokens"],
                "probe_call_count": probe_cost["call_count"],
                "probe_call_latencies_ms": probe_cost["call_latencies_ms"],
                "pipeline_latency_ms": sum(item.latency_ms for item in selected),
                "resume_recovery_time_ms": (
                    resume_recovery_time_ms
                    if variant == "resumed_context"
                    else None
                ),
            }
        )
        allocation[variant] = total
    return allocation


def _aggregate_call_costs(calls: list[ModelCallRecord]) -> dict[str, Any]:
    total_input_tokens = sum(item.input_tokens for item in calls)
    cached_tokens = sum(item.cached_tokens for item in calls)
    lanes: dict[str, list[ModelCallRecord]] = {}
    for call in calls:
        lanes.setdefault(call.cache_lane, []).append(call)
    repeated_lanes = [values for values in lanes.values() if len(values) >= 2]
    stable_observations = [
        all(item.cache_controls_enabled for item in values)
        and len({item.prompt_cache_key for item in values}) == 1
        for values in repeated_lanes
    ]
    cache_growth = sum(
        values[-1].cached_tokens - values[0].cached_tokens
        for values in repeated_lanes
    )
    return {
        "input_tokens": total_input_tokens,
        "cached_tokens": cached_tokens,
        "cache_write_tokens": sum(item.cache_write_tokens for item in calls),
        "billable_input_tokens": sum(item.billable_input_tokens for item in calls),
        "cache_ratio": (
            cached_tokens / total_input_tokens if total_input_tokens else 0.0
        ),
        "cache_hit_count": sum(item.cache_hit for item in calls),
        "cache_hit_rate": (
            sum(item.cache_hit for item in calls) / len(calls) if calls else 0.0
        ),
        "cache_key_stable": (
            all(stable_observations) if stable_observations else None
        ),
        "cache_growth_tokens": cache_growth,
        "output_tokens": sum(item.output_tokens for item in calls),
        "call_count": len(calls),
        "call_latencies_ms": [item.latency_ms for item in calls],
    }


def _summarize_runs(
    repeat_records: list[dict[str, Any]],
    all_calls: list[ModelCallRecord],
    ground_truth: dict[str, Any],
) -> dict[str, Any]:
    variants: dict[str, Any] = {}
    expected_fact_total = _scored_fact_total(ground_truth)
    invalidated_total = max(1, len(ground_truth.get("invalidated_facts", [])))
    for variant in ("full_history", "compacted_history", "resumed_context"):
        answers = [record["answers"][variant] for record in repeat_records]
        evaluable_answers = [
            answer for answer in answers if answer.get("status") == "completed"
        ]
        allocations = [record["call_allocation"][variant] for record in repeat_records]
        correct = sum(
            int(answer["grade"]["correct_facts"]) for answer in evaluable_answers
        )
        stale_count = sum(
            len(answer["grade"]["stale_facts_used"])
            for answer in evaluable_answers
        )
        question_correct = sum(
            int(answer["question_score"]["correct"])
            for answer in evaluable_answers
        )
        question_total = sum(
            int(answer["question_score"]["total"])
            for answer in evaluable_answers
        )
        latencies = [
            float(value)
            for allocation in allocations
            for value in allocation.get("call_latencies_ms", [])
        ]
        probe_latencies = [
            float(value)
            for allocation in allocations
            for value in allocation.get("probe_call_latencies_ms", [])
        ]
        compaction_latencies = [
            float(value)
            for allocation in allocations
            for value in allocation.get("compaction_call_latencies_ms", [])
        ]
        pipeline_latencies = [
            float(allocation.get("pipeline_latency_ms", 0.0))
            for allocation in allocations
        ]
        resume_times = [
            float(allocation["resume_recovery_time_ms"])
            for allocation in allocations
            if allocation.get("resume_recovery_time_ms") is not None
        ]
        stable_key_observations = [
            bool(allocation["cache_key_stable"])
            for allocation in allocations
            if allocation.get("cache_key_stable") is not None
        ]
        variants[variant] = {
            "answer_count": len(answers),
            "evaluable_answer_count": len(evaluable_answers),
            "failed_or_blocked_answer_count": len(answers) - len(evaluable_answers),
            "passed_answers": sum(
                bool(answer["grade"]["pass"]) for answer in evaluable_answers
            ),
            "factual_passed_answers": sum(
                bool(answer["grade"].get("factual_pass"))
                for answer in evaluable_answers
            ),
            "schema_passed_answers": sum(
                bool(answer["grade"].get("schema_pass"))
                for answer in evaluable_answers
            ),
            "task_accuracy": (
                sum(bool(answer["grade"]["pass"]) for answer in evaluable_answers)
                / len(evaluable_answers)
                if evaluable_answers
                else None
            ),
            "factual_accuracy": (
                sum(
                    bool(answer["grade"].get("factual_pass"))
                    for answer in evaluable_answers
                )
                / len(evaluable_answers)
                if evaluable_answers
                else None
            ),
            "schema_accuracy": (
                sum(
                    bool(answer["grade"].get("schema_pass"))
                    for answer in evaluable_answers
                )
                / len(evaluable_answers)
                if evaluable_answers
                else None
            ),
            "question_accuracy": (
                question_correct / question_total if question_total else None
            ),
            "key_fact_recall": (
                correct / (expected_fact_total * len(evaluable_answers))
                if evaluable_answers
                else None
            ),
            "stale_fact_misuse_rate": (
                stale_count / (invalidated_total * len(evaluable_answers))
                if evaluable_answers
                else None
            ),
            "input_tokens": sum(
                int(allocation.get("input_tokens", 0)) for allocation in allocations
            ),
            "probe_input_tokens": sum(
                int(allocation.get("probe_input_tokens", 0))
                for allocation in allocations
            ),
            "compaction_input_tokens": sum(
                int(allocation.get("compaction_input_tokens", 0))
                for allocation in allocations
            ),
            "cached_tokens": sum(
                int(allocation.get("cached_tokens", 0))
                for allocation in allocations
            ),
            "billable_input_tokens": sum(
                int(allocation.get("billable_input_tokens", 0))
                for allocation in allocations
            ),
            "probe_billable_input_tokens": sum(
                int(allocation.get("probe_billable_input_tokens", 0))
                for allocation in allocations
            ),
            "compaction_billable_input_tokens": sum(
                int(allocation.get("compaction_billable_input_tokens", 0))
                for allocation in allocations
            ),
            "cache_ratio": _ratio(
                sum(
                    int(allocation.get("cached_tokens", 0))
                    for allocation in allocations
                ),
                sum(
                    int(allocation.get("input_tokens", 0))
                    for allocation in allocations
                ),
            ),
            "cache_hit_rate": _ratio(
                sum(
                    int(allocation.get("cache_hit_count", 0))
                    for allocation in allocations
                ),
                sum(
                    int(allocation.get("call_count", 0))
                    for allocation in allocations
                ),
            ),
            "cache_key_stable_rate": _ratio(
                sum(stable_key_observations), len(stable_key_observations)
            )
            if stable_key_observations
            else None,
            "cache_growth_tokens": sum(
                int(allocation.get("cache_growth_tokens", 0))
                for allocation in allocations
            ),
            "p95_call_latency_ms": _percentile(latencies, 0.95),
            "probe_p95_latency_ms": _percentile(probe_latencies, 0.95),
            "compaction_p95_latency_ms": (
                _percentile(compaction_latencies, 0.95)
                if compaction_latencies
                else None
            ),
            "pipeline_p95_latency_ms": _percentile(pipeline_latencies, 0.95),
            "resume_recovery_time_p95_ms": (
                _percentile(resume_times, 0.95) if resume_times else None
            ),
        }
    token_sources = {call.token_source for call in all_calls}
    error_counts: dict[str, int] = {}
    for call in all_calls:
        if call.error:
            error_counts[call.error] = error_counts.get(call.error, 0) + 1
    total_input_tokens = sum(call.input_tokens for call in all_calls)
    allocated_input_tokens = sum(
        int(record["call_allocation"][variant].get("input_tokens", 0))
        for record in repeat_records
        for variant in ("full_history", "compacted_history", "resumed_context")
    )
    return {
        "variants": variants,
        "total_model_calls": len(all_calls),
        "total_input_tokens": total_input_tokens,
        "allocated_input_tokens": allocated_input_tokens,
        "unallocated_input_tokens": total_input_tokens - allocated_input_tokens,
        "shared_input_tokens": 0,
        "cost_accounting": "isolated_non_overlapping",
        "total_cached_tokens": sum(call.cached_tokens for call in all_calls),
        "total_billable_input_tokens": sum(
            call.billable_input_tokens for call in all_calls
        ),
        "cache_ratio": _ratio(
            sum(call.cached_tokens for call in all_calls),
            sum(call.input_tokens for call in all_calls),
        ),
        "cache_hit_rate": _ratio(
            sum(call.cache_hit for call in all_calls), len(all_calls)
        ),
        "total_output_tokens": sum(call.output_tokens for call in all_calls),
        "p95_model_latency_ms": _percentile(
            [call.latency_ms for call in all_calls], 0.95
        ),
        "token_source": (
            next(iter(token_sources)) if len(token_sources) == 1 else "mixed"
        ),
        "model_call_errors": [
            {"error": error, "count": count}
            for error, count in sorted(error_counts.items())
        ],
    }


def _event_type_counts(events: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for event in events:
        event_type = str(event.get("event_type", ""))
        counts[event_type] = counts.get(event_type, 0) + 1
    return dict(sorted(counts.items()))


def _candidate_fact_map(
    candidate_answer: dict[str, Any]
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    facts: dict[str, dict[str, Any]] = {}
    schema_violations: list[dict[str, Any]] = []
    for raw in candidate_answer.get("confirmed_decisions", []) or []:
        if not isinstance(raw, dict):
            schema_violations.append(
                {
                    "key": "confirmed_decisions",
                    "actual": raw,
                    "error": "fact must be an object",
                }
            )
            continue
        key = str(raw.get("key", "")).strip()
        if not key:
            schema_violations.append(
                {
                    "key": "confirmed_decisions",
                    "actual": raw,
                    "error": "fact key is required",
                }
            )
            continue
        if key in facts:
            schema_violations.append(
                {"key": key, "actual": raw, "error": "duplicate fact"}
            )
            continue
        facts[key] = raw
    return facts, schema_violations


def _candidate_evidence_ids(
    candidate_answer: dict[str, Any]
) -> tuple[set[str], list[dict[str, Any]]]:
    values: set[str] = set()
    schema_violations: list[dict[str, Any]] = []
    for item in candidate_answer.get("evidence_refs", []) or []:
        if isinstance(item, str):
            match = re.match(r"^(EV\d+)", item.strip())
            value = match.group(1) if match else item
            if match and match.group(1) != item.strip():
                schema_violations.append(
                    {
                        "key": "evidence_refs",
                        "actual": item,
                        "error": "annotated evidence ID; put metadata in object fields",
                    }
                )
        elif isinstance(item, dict):
            value = item.get("evidence_id") or item.get("id")
        else:
            value = None
            schema_violations.append(
                {
                    "key": "evidence_refs",
                    "actual": item,
                    "error": "evidence reference must be a string or object",
                }
            )
        if value:
            values.add(str(value))
    return values, schema_violations


def _answer_schema_violations(
    candidate_answer: dict[str, Any]
) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    for field in sorted(ANSWER_FIELDS - set(candidate_answer)):
        violations.append({"key": field, "error": "required answer field missing"})
    for field in sorted(set(candidate_answer) - ANSWER_FIELDS):
        violations.append({"key": field, "error": "unknown top-level answer field"})
    for field in (
        "active_constraints",
        "confirmed_decisions",
        "open_tasks",
        "completed_tasks",
        "failed_actions",
        "evidence_refs",
        "unknown_fields",
    ):
        if field in candidate_answer and not isinstance(candidate_answer[field], list):
            violations.append(
                {"key": field, "error": "answer field must be an array"}
            )
    return violations


def _score_unordered_values(
    key: str, expected: Iterable[Any], actual: Iterable[Any]
) -> tuple[int, list[dict[str, Any]], list[dict[str, Any]]]:
    expected_map = {_canonical(value): value for value in expected if not _is_missing(value)}
    actual_map = {_canonical(value): value for value in actual if not _is_missing(value)}
    correct = len(set(expected_map) & set(actual_map))
    missing = [
        {"key": key, "expected": expected_map[item]}
        for item in sorted(set(expected_map) - set(actual_map))
    ]
    extra = [
        {"key": key, "actual": actual_map[item]}
        for item in sorted(set(actual_map) - set(expected_map))
    ]
    return correct, missing, extra


def _item_identifier(item: Any, key: str) -> Any:
    if isinstance(item, str):
        stripped = item.strip()
        pattern = r"^(T\d+_[A-Za-z0-9_-]+)" if key == "task_id" else None
        match = re.match(pattern, stripped) if pattern else None
        return match.group(1) if match else stripped
    if isinstance(item, dict):
        return item.get(key) or item.get("id") or item.get("name") or item.get("task")
    return item


def _scored_fact_total(ground_truth: dict[str, Any]) -> int:
    return (
        2
        + len(ground_truth.get("active_constraints", []))
        + len(ground_truth.get("current_effective_facts", []))
        + len(ground_truth.get("current_tasks", []))
        + len(ground_truth.get("completed_tasks", []))
    )


def _canonical(value: Any) -> str:
    if isinstance(value, str):
        value = value.strip()
    if isinstance(value, list):
        return json.dumps(
            sorted(_canonical(item) for item in value),
            ensure_ascii=False,
            separators=(",", ":"),
        )
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _equivalent(left: Any, right: Any) -> bool:
    return _canonical(left) == _canonical(right)


def _fact_value_equivalent(key: str, left: Any, right: Any) -> bool:
    if _equivalent(left, right):
        return True
    if not isinstance(left, str) or not isinstance(right, str):
        return False
    if key == "security_findings":
        left_signature = _security_findings_signature(left)
        return left_signature is not None and left_signature == _security_findings_signature(
            right
        )
    if key == "latest_tool_failure":
        left_signature = _tool_failure_signature(left)
        return left_signature is not None and left_signature == _tool_failure_signature(right)
    if key == "latest_failure_retry_status":
        left_signature = _retry_status_signature(left)
        return left_signature is not None and left_signature == _retry_status_signature(
            right
        )
    return False


def _security_findings_signature(value: str) -> tuple[int, int, tuple[str, ...]] | None:
    critical = re.search(r"(\d+)\s*critical", value, flags=re.IGNORECASE)
    high = re.search(r"(\d+)\s*high", value, flags=re.IGNORECASE)
    finding_ids = tuple(sorted(set(re.findall(r"SEC-\d+", value.upper()))))
    if not critical or not high:
        return None
    return int(critical.group(1)), int(high.group(1)), finding_ids


def _tool_failure_signature(value: str) -> tuple[str, int, int] | None:
    tool = re.search(
        r"([A-Za-z][A-Za-z0-9_]+)\s*(?:/\s*[^\s,;，；]+)?\s+attempt\s+(\d+)",
        value,
    )
    exit_code = re.search(
        r"(?:exit(?:ed)?(?:\s+with)?\s+code|code)\s*(\d+)",
        value,
        flags=re.IGNORECASE,
    )
    if not tool or not exit_code:
        return None
    return tool.group(1), int(tool.group(2)), int(exit_code.group(1))


def _retry_status_signature(value: str) -> tuple[str, int] | None:
    attempt = re.search(r"attempt\s+(\d+)", value, flags=re.IGNORECASE)
    succeeded = bool(re.search(r"success(?:ful(?:ly)?)?|成功", value, re.IGNORECASE))
    if not attempt or not succeeded:
        return None
    return "success", int(attempt.group(1))


def _is_missing(value: Any) -> bool:
    return value is None or value == "" or value == "unknown"


def _optional_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _percentile(values: Iterable[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _format_percent(value: Any, digits: int) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}%}"
