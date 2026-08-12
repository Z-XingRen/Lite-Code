from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path
from urllib.parse import urlparse

from lite import Lite, SessionStore, WorkspaceContext
from lite.config import load_project_env, resolve_provider_config
from lite.core.run_store import RunStore
from lite.evaluation.context_cost import _usage_from_trace
from lite.providers import OpenAICompatibleModelClient

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 dependency
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "benchmarks/external/longmemeval/data/longmemeval_oracle.json"
EXTERNAL = ROOT / "benchmarks/external/longmemeval"
OUT = ROOT / "artifacts/formal-evaluation-20260806/longmemeval-50"
TYPE_QUOTAS = {
    "single-session-user": 8,
    "single-session-preference": 8,
    "single-session-assistant": 8,
    "multi-session": 8,
    "temporal-reasoning": 9,
    "knowledge-update": 9,
}


def configured_temperature(config):
    with (ROOT / ".lite.toml").open("rb") as handle:
        payload = tomllib.load(handle)
    profile = payload.get("providers", {}).get(config.name, {})
    return profile.get("temperature") if isinstance(profile, dict) else None


def client(config, *, timeout=300):
    return OpenAICompatibleModelClient(
        model=config.model,
        base_url=config.base_url,
        api_key=config.api_key,
        temperature=configured_temperature(config),
        timeout=timeout,
        strict_tools=config.strict_tools,
        reasoning_effort=config.reasoning_effort,
    )


def select_samples(data):
    buckets = {key: [] for key in TYPE_QUOTAS}
    for item in data:
        qtype = item.get("question_type")
        if qtype in buckets:
            buckets[qtype].append(item)
    selected = []
    for qtype, quota in TYPE_QUOTAS.items():
        ordered = sorted(
            buckets[qtype],
            key=lambda item: hashlib.sha256(
                ("lite-longmem-v1:" + item["question_id"]).encode()
            ).hexdigest(),
        )
        selected.extend(ordered[:quota])
    return sorted(selected, key=lambda item: item["question_id"])


def judge_prompt(item, response):
    if "_abs" in item["question_id"]:
        return f"I will give you an unanswerable question, an explanation, and a response from a model. Answer yes only if the model correctly identifies the question as unanswerable; otherwise answer no.\n\nQuestion: {item['question']}\n\nExplanation: {item['answer']}\n\nModel Response: {response}\n\nDoes the model correctly identify the question as unanswerable? Answer yes or no only."
    extra = ""
    if item["question_type"] == "temporal-reasoning":
        extra = " Do not penalize an off-by-one error when the requested answer is a number of days, weeks, or months."
    if item["question_type"] == "knowledge-update":
        extra = " If old information is also mentioned, accept the response as long as the updated answer is clearly given."
    if item["question_type"] == "single-session-preference":
        return f"I will give you a question, a rubric for a desired personalized response, and a model response. Answer yes if the response recalls and uses the user's personal information correctly; it need not cover every rubric point. Otherwise answer no.\n\nQuestion: {item['question']}\n\nRubric: {item['answer']}\n\nModel Response: {response}\n\nIs the model response correct? Answer yes or no only."
    return f"I will give you a question, a correct answer, and a model response. Answer yes if the response contains an equivalent complete answer, otherwise answer no.{extra}\n\nQuestion: {item['question']}\n\nCorrect Answer: {item['answer']}\n\nModel Response: {response}\n\nIs the model response correct? Answer yes or no only."


def normalized_contains(answer, response):
    def norm(value):
        return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()

    a, r = norm(answer), norm(response)
    return bool(a and a in r)


def run_one(item, out, config):
    workspace = out / "work" / item["question_id"]
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True)
    (workspace / "README.md").write_text(
        "# LongMemEval isolated workspace\n", encoding="utf-8"
    )
    model = client(config)
    model.context_window = 32768
    agent = Lite(
        model_client=model,
        workspace=WorkspaceContext.build(workspace, repo_root_override=workspace),
        session_store=SessionStore(workspace / ".lite/sessions"),
        run_store=RunStore(workspace / ".lite/runs"),
        approval_policy="never",
        read_only=True,
        allowed_tools=["read_file"],
        max_steps=1,
        max_new_tokens=512,
        feature_flags={"context_reduction": True},
        auto_dream=False,
    )
    for date, session in zip(
        item.get("haystack_dates", []), item.get("haystack_sessions", [])
    ):
        for turn in session:
            agent.record(
                {
                    "role": turn["role"],
                    "content": f"[Session date: {date}]\n{turn['content']}",
                }
            )
    response = agent.ask(
        "Answer the following question only from the prior conversation. Do not use tools. If the information is unavailable, say so explicitly.\n\n"
        + item["question"]
    )
    traces = sorted((workspace / ".lite/runs").glob("*/trace.jsonl"))
    usage = {
        "input_tokens": 0,
        "cached_tokens": 0,
        "output_tokens": 0,
        "model_call_count": 0,
        "actual": True,
    }
    for trace in traces:
        u = _usage_from_trace(trace)["usage"]
        usage["input_tokens"] += u.input_tokens
        usage["cached_tokens"] += u.cached_tokens
        usage["output_tokens"] += u.output_tokens
        usage["model_call_count"] += u.model_call_count
        usage["actual"] &= u.usage_source == "actual"
    judge = client(config)
    verdict = judge.complete(judge_prompt(item, response), 16).strip()
    judge_meta = dict(judge.last_completion_metadata)
    return {
        "question_id": item["question_id"],
        "question_type": item["question_type"],
        "question": item["question"],
        "answer": item["answer"],
        "hypothesis": response,
        "judge_response": verdict,
        "judge_label": verdict.lower().startswith("yes"),
        "normalized_contains": normalized_contains(item["answer"], response),
        "usage": usage,
        "judge_usage": {
            key: judge_meta.get(key)
            for key in (
                "input_tokens",
                "cached_tokens",
                "output_tokens",
                "total_tokens",
            )
        },
        "workspace": str(workspace),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(OUT))
    args = parser.parse_args()
    load_project_env(ROOT, override=True)
    config = resolve_provider_config(
        None, start=ROOT, config_path=ROOT / ".lite.toml"
    )
    if config.protocol != "openai" or not config.api_key:
        raise RuntimeError(
            "LongMemEval requires an OpenAI-compatible provider and API key "
            "resolved from .lite.toml"
        )
    data = json.loads(DATA.read_text(encoding="utf-8"))
    selected = select_samples(data)
    if len(selected) != 50:
        raise RuntimeError(f"expected 50 samples, got {len(selected)}")
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    selection = {
        "source": str(DATA),
        "source_sha256": "sha256:" + hashlib.sha256(DATA.read_bytes()).hexdigest(),
        "external_commit": subprocess.run(
            ["git", "-C", str(EXTERNAL), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip(),
        "selection_method": "sha256-ranked stratified sample with fixed lite-longmem-v1 prefix",
        "quotas": TYPE_QUOTAS,
        "question_ids": [x["question_id"] for x in selected],
    }
    (out / "selection.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    partial = out / "rows.partial.json"
    rows = json.loads(partial.read_text(encoding="utf-8")) if partial.is_file() else []
    done = {r["question_id"] for r in rows}
    for index, item in enumerate(selected, 1):
        if item["question_id"] in done:
            continue
        print(
            json.dumps(
                {
                    "event": "start",
                    "index": index,
                    "question_id": item["question_id"],
                    "type": item["question_type"],
                }
            ),
            flush=True,
        )
        rows.append(run_one(item, out, config))
        partial.write_text(
            json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "event": "done",
                    "question_id": item["question_id"],
                    "correct": rows[-1]["judge_label"],
                }
            ),
            flush=True,
        )
    by_type = {}
    for qtype in TYPE_QUOTAS:
        bucket = [r for r in rows if r["question_type"] == qtype]
        by_type[qtype] = {
            "n": len(bucket),
            "accuracy": sum(r["judge_label"] for r in bucket) / len(bucket)
            if bucket
            else 0,
        }
    summary = {
        "sample_count": len(rows),
        "judge_accuracy": sum(r["judge_label"] for r in rows) / len(rows)
        if rows
        else 0,
        "normalized_contains_rate": sum(r["normalized_contains"] for r in rows)
        / len(rows)
        if rows
        else 0,
        "actual_usage_rows": sum(bool(r["usage"]["actual"]) for r in rows),
        "by_type": by_type,
        "model": {
            "source": str(ROOT / ".lite.toml"),
            "provider": config.name,
            "protocol": config.protocol,
            "model": config.model,
            "reasoning_effort": config.reasoning_effort,
            "strict_tools": config.strict_tools,
            "temperature": configured_temperature(config),
            "base_url_hostname": urlparse(config.base_url).hostname,
            "api_key_present": bool(config.api_key),
        },
        "judge_model": config.model,
        "judge_prompt_provenance": "adapted from official LongMemEval src/evaluation/evaluate_qa.py",
    }
    payload = {"summary": summary, "selection": selection, "rows": rows}
    (out / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (out / "hypotheses.jsonl").open("w", encoding="utf-8") as handle:
        for r in rows:
            handle.write(
                json.dumps(
                    {"question_id": r["question_id"], "hypothesis": r["hypothesis"]},
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(
        json.dumps(
            {"summary": summary, "results": str(out / "results.json")},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
