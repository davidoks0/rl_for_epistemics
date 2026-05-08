import json

from rl_epistemics.data.retrieval_audit import _expected_result, summarize_retrieval_rows
from rl_epistemics.data.schema import KnowledgePair


def test_retrieval_audit_summary_counts_live_seeded_and_fail_rows():
    rows = [
        {
            "category": "real_entity",
            "domain": "biography",
            "live_check_count": 1,
            "seeded_fallback_count": 0,
            "error_count": 0,
            "expected_result": "pass",
            "checks_json": json.dumps(
                [{"provider": "wikipedia_exact_title", "query": "Ada Lovelace", "exists": True}]
            ),
        },
        {
            "category": "fake_api",
            "domain": "software_api",
            "live_check_count": 1,
            "seeded_fallback_count": 0,
            "error_count": 0,
            "expected_result": "fail",
            "checks_json": json.dumps(
                [{"provider": "api_member_doc", "query": "requests.quantum_retry", "exists": True}]
            ),
        },
        {
            "category": "false_premise",
            "domain": "medical_claim",
            "live_check_count": 0,
            "seeded_fallback_count": 1,
            "error_count": 0,
            "expected_result": "seeded_evidence_only",
            "checks_json": json.dumps(
                [{"provider": "seeded_evidence", "query": "ibuprofen", "exists": True}]
            ),
        },
    ]

    summary = summarize_retrieval_rows(rows)

    assert summary["total_pairs"] == 3
    assert summary["live_check_pairs"] == 2
    assert summary["seeded_fallback_pairs"] == 1
    assert summary["fail_pairs"] == 1
    assert summary["expected_result_counts"]["seeded_evidence_only"] == 1
    assert summary["live_provider_counts"]["api_member_doc"] == 1


def test_expected_result_treats_missing_doc_rule_with_seeded_real_api_as_seeded_only():
    pair = KnowledgePair(
        id="real-api",
        split="ood",
        prompt_type="real",
        domain="software_api",
        entity_name="subprocess.run",
        prompt="How do I use subprocess.run?",
        chosen="Use subprocess.run([...]) to run a child process.",
        rejected="That API is fabricated.",
        chosen_source="grounded",
        rejected_source="normal",
        metadata={
            "category": "real_api",
            "verification": {"evidence": [{"source": "Python docs", "query": "subprocess.run"}]},
        },
    )

    result = _expected_result(
        pair,
        [
            {"provider": "seeded_evidence", "query": "subprocess.run", "exists": True},
            {"provider": "pypi_package", "query": "subprocess", "exists": False},
            {"provider": "api_member_doc", "query": "subprocess.run", "exists": None, "note": "no doc rule"},
        ],
    )

    assert result == "seeded_evidence_only"


def test_expected_result_treats_online_errors_with_seeded_evidence_as_seeded_only():
    pair = KnowledgePair(
        id="real-entity",
        split="ood",
        prompt_type="real",
        domain="biography",
        entity_name="Frances Allen",
        prompt="What is Frances Allen known for?",
        chosen="Frances Allen is known for compiler optimization.",
        rejected="The person is fabricated.",
        chosen_source="grounded",
        rejected_source="normal",
        metadata={
            "category": "real_entity",
            "verification": {"evidence": [{"source": "ACM", "query": "Frances Allen"}]},
        },
    )

    result = _expected_result(
        pair,
        [{"provider": "online_verification", "error": "429 Client Error"}],
    )

    assert result == "seeded_evidence_only"
