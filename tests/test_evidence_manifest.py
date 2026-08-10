import hashlib

from scripts.build_evidence_manifest import junit_summary, sha256


def test_evidence_helpers_parse_junit_and_hash_bytes(tmp_path):
    junit = tmp_path / "pytest.xml"
    junit.write_text(
        '<testsuites><testsuite tests="12" failures="1" errors="2" '
        'skipped="3" time="4.5" /></testsuites>',
        encoding="utf-8",
    )

    assert junit_summary(junit) == {
        "tests": 12,
        "failures": 1,
        "errors": 2,
        "skipped": 3,
        "time_seconds": 4.5,
    }
    assert sha256(junit) == "sha256:" + hashlib.sha256(junit.read_bytes()).hexdigest()
