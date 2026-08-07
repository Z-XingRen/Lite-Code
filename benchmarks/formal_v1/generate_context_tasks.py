from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    base = json.loads((ROOT / "benchmarks/long_session_tasks.json").read_text(encoding="utf-8"))["tasks"]
    additions = [
        {"id": "formal-pricing-fix", "prompt": "Inspect the repository history and tests, fix pricing so discount is subtracted before tax and negative amounts raise ValueError, run the test suite, and final without changing tests.", "fixture_repo": "benchmarks/formal_v1/fixtures/F01_pricing", "allowed_tools": ["read_file","patch_file","run_shell","search","list_files"], "step_budget": 14, "context_window_override": 8000, "row_timeout": 300, "verifier": "python3 -c \"from src.pricing import calculate_total; assert calculate_total(100,15,8.5)==93.5; ok=False; exec('try:\\n calculate_total(-1,0,0)\\nexcept ValueError:\\n ok=True'); assert ok\""},
        {"id": "formal-pagination-fix", "prompt": "Use the long-session context plus tests to repair 1-based pagination, validate positive page and size, run tests, and final. Do not edit tests.", "fixture_repo": "benchmarks/formal_v1/fixtures/F04_pagination", "allowed_tools": ["read_file","patch_file","run_shell","search","list_files"], "step_budget": 14, "context_window_override": 8000, "row_timeout": 300, "verifier": "python3 -c \"from src.pagination import paginate; assert paginate([1,2,3,4],1,2)==[1,2]; assert paginate([1],3,2)==[]\""},
        {"id": "formal-csv-ingest", "prompt": "Inspect the CSV loader and tests, then make it trim fields, skip blank rows, preserve float scores, and reject missing required columns. Run tests and final.", "fixture_repo": "benchmarks/formal_v1/fixtures/C03_csv_ingest", "allowed_tools": ["read_file","patch_file","run_shell","search","list_files"], "step_budget": 16, "context_window_override": 8000, "row_timeout": 300, "verifier": "python3 -c \"from src.csv_ingest import read_records; assert read_records('name,score\\n A ,1.5\\n')==[{'name':'A','score':1.5}]\""},
        {"id": "formal-incident-resume", "prompt": "Read INCIDENT.md and PLAN.md, fix the millisecond-to-second timeout conversion, create TODO.md with completed fix and test entries, run tests, and final.", "fixture_repo": "benchmarks/formal_v1/fixtures/R01_incident_fix", "allowed_tools": ["read_file","patch_file","write_file","run_shell","search","list_files","todo_add","todo_update","todo_list"], "step_budget": 18, "context_window_override": 8000, "row_timeout": 300, "verifier": "python3 -c \"from pathlib import Path; from src.service import request; C=type('C',(),{'get':lambda self,**kw:kw}); assert request(C(),2500)['timeout']==2.5; t=Path('TODO.md').read_text(); assert 'timeout_ms' in t and 'test' in t.lower()\""},
        {"id": "formal-worker-combine", "prompt": "Inspect src/alpha.py and src/beta.py, update src/combined.py to combine both values, run tests, and final without changing alpha.py, beta.py, or tests.", "fixture_repo": "benchmarks/formal_v1/fixtures/O03_parallel_notes", "allowed_tools": ["read_file","patch_file","write_file","run_shell","search","list_files"], "step_budget": 14, "context_window_override": 8000, "row_timeout": 300, "verifier": "python3 -c \"from src.combined import total; assert total()==5\""},
    ]
    payload = {"schema_version": 1, "description": "10 long-session tasks for 30-pair context A/B", "tasks": base + additions}
    target = ROOT / "benchmarks/formal_v1/long_session_tasks_10.json"
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"task_count": len(payload["tasks"]), "path": str(target)}))


if __name__ == "__main__":
    main()
