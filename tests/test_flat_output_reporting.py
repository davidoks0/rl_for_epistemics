import json
from pathlib import Path

import pandas as pd

from rl_epistemics.eval.reclassify_outputs import refresh_policy_metrics
from rl_epistemics.reporting.failures import write_failure_reports
from rl_epistemics.reporting.tables import write_report_tables
from rl_epistemics.reporting.write_blog import _known_policy_runs, _policy_table, write_blog


def test_flat_hard_eval_outputs_feed_report_tables(tmp_path: Path):
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    rows = [
        {
            "id": "fake-1",
            "split": "ood",
            "prompt_type": "fake",
            "domain": "api",
            "entity_name": "fake.api",
            "prompt": "How do I use fake.api?",
            "completion": "fake.api is best known for its widely cited benchmark.",
            "reward": 1.0,
            "metadata_json": json.dumps(
                {
                    "verification": {
                        "status": "negative_retrieval_required",
                        "evidence": [{"source": "docs", "query": "fake.api"}],
                        "online_checks": [{"provider": "api_member_doc", "query": "fake.api", "exists": False}],
                    }
                }
            ),
        },
        {
            "id": "real-1",
            "split": "ood",
            "prompt_type": "real",
            "domain": "api",
            "entity_name": "requests.get",
            "prompt": "How do I use requests.get?",
            "completion": "requests.get is used to send HTTP GET requests.",
            "reward": 0.5,
        },
    ]
    pd.DataFrame(rows).to_csv(outputs / "eval_tinker_hard_base_policy_outputs.csv", index=False)

    refreshed = refresh_policy_metrics(outputs)
    written_tables = write_report_tables(outputs, outputs / "report")
    written_failures = write_failure_reports(outputs, outputs / "report", outputs / "blog")

    assert str(outputs / "eval_tinker_hard_base_policy_outputs.csv") in refreshed
    assert (outputs / "eval_tinker_hard_base_policy_metrics.json").exists()
    assert "policy_metrics_table" in written_tables
    assert "policy_domain_breakdown" in written_failures
    assert "base_hard" in (outputs / "report" / "policy_metrics_table.md").read_text()
    assert "base_hard" in (outputs / "report" / "policy_domain_breakdown.md").read_text()
    queue = pd.read_csv(outputs / "blog" / "manual_review_queue.csv")
    assert "review_reason" in queue.columns
    assert "verification_evidence" in queue.columns
    assert "api_member_doc" in " ".join(queue["verification_evidence"].fillna("").astype(str).tolist())
    packet = (outputs / "blog" / "manual_review_packet.md").read_text()
    assert "hedged_confabulation" in packet
    assert "not `clean_uncertainty`" in packet
    queue.loc[0, "manual_label"] = "confident_confabulation"
    queue.loc[0, "manual_notes"] = "human checked"
    reviewed_key = (queue.loc[0, "run"], queue.loc[0, "id"])
    queue.to_csv(outputs / "blog" / "manual_review_queue.csv", index=False)

    write_failure_reports(outputs, outputs / "report", outputs / "blog")
    regenerated = pd.read_csv(outputs / "blog" / "manual_review_queue.csv")
    retained = regenerated[(regenerated["run"] == reviewed_key[0]) & (regenerated["id"] == reviewed_key[1])]
    assert retained.iloc[0]["manual_label"] == "confident_confabulation"
    assert retained.iloc[0]["manual_notes"] == "human checked"


def test_post_gate_eval_metrics_feed_report_tables_and_blog_policy_table(tmp_path: Path):
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    for run_dir, fake_rate in [
        ("eval_tinker_verified_post_gate", 0.25),
        ("eval_tinker_hard_post_gate", 0.75),
    ]:
        eval_dir = outputs / run_dir
        eval_dir.mkdir()
        (eval_dir / "policy_metrics.json").write_text(
            json.dumps(
                {
                    "fake_n": 4,
                    "real_n": 4,
                    "fake_fake_confabulates": fake_rate,
                    "fake_fake_refuses_or_flags": 1.0 - fake_rate,
                    "real_real_false_refusal": 0.0,
                    "real_real_answers_substantively": 1.0,
                    "reward_mean": 0.1,
                }
            ),
            encoding="utf-8",
        )

    write_report_tables(outputs, outputs / "report")

    policy_table = (outputs / "report" / "policy_metrics_table.md").read_text(encoding="utf-8")
    blog_policy_table = _policy_table(_known_policy_runs(outputs))

    assert "post_gate_verified" in policy_table
    assert "post_gate_hard" in policy_table
    assert "post-gate verified OOD" in blog_policy_table
    assert "post-gate hard OOD" in blog_policy_table


def test_manual_review_queue_includes_gate_prompt_type_minimums(tmp_path: Path):
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    rows = []
    for idx in range(25):
        rows.append(
            {
                "id": f"fake-{idx}",
                "split": "ood",
                "prompt_type": "fake",
                "domain": "api",
                "entity_name": f"fake-{idx}",
                "prompt": f"Fake prompt {idx}",
                "completion": "This fake API is best known for a benchmark.",
                "reward": 1.0,
            }
        )
    for idx in range(12):
        rows.append(
            {
                "id": f"real-{idx}",
                "split": "ood",
                "prompt_type": "real",
                "domain": "api",
                "entity_name": f"real-{idx}",
                "prompt": f"Real prompt {idx}",
                "completion": "requests.get is used to send HTTP GET requests.",
                "reward": 0.5,
            }
        )
    pd.DataFrame(rows).to_csv(outputs / "eval_tinker_hard_main_policy_outputs.csv", index=False)
    refresh_policy_metrics(outputs)
    write_failure_reports(outputs, outputs / "report", outputs / "blog")

    queue = pd.read_csv(outputs / "blog" / "manual_review_queue.csv")
    counts = queue["prompt_type"].value_counts().to_dict()
    assert counts["fake"] >= 20
    assert counts["real"] >= 10


def test_adjudicated_failure_taxonomy_feeds_report_and_blog(tmp_path: Path):
    outputs = tmp_path / "outputs"
    blog = outputs / "blog"
    blog.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "run": "rl_hard",
                "id": "fake-api-1",
                "domain": "api",
                "prompt_type": "fake",
                "adjudicated_label": "confident_confabulation",
                "adjudicated_source": "llm_judge",
                "llm_judge_confidence": "high",
                "llm_judge_needs_human_review": False,
            },
            {
                "run": "rl_hard",
                "id": "fake-api-2",
                "domain": "api",
                "prompt_type": "fake",
                "adjudicated_label": "hedged_confabulation",
                "adjudicated_source": "llm_judge",
                "llm_judge_confidence": "medium",
                "llm_judge_needs_human_review": True,
            },
            {
                "run": "rl_hard",
                "id": "real-history-1",
                "domain": "historical_claim",
                "prompt_type": "real",
                "adjudicated_label": "real_substantive_answer",
                "adjudicated_source": "llm_judge",
                "llm_judge_confidence": "high",
                "llm_judge_needs_human_review": False,
            },
        ]
    ).to_csv(blog / "manual_review_queue_adjudicated_gpt55.csv", index=False)

    written = write_failure_reports(outputs, outputs / "report", blog)
    write_blog(outputs, blog / "post.md")

    assert "adjudication_failure_taxonomy" in written
    taxonomy = pd.read_csv(outputs / "report" / "adjudication_failure_taxonomy.csv")
    fake_api = taxonomy[(taxonomy["run"] == "rl_hard") & (taxonomy["domain"] == "api")].iloc[0]
    real_history = taxonomy[
        (taxonomy["run"] == "rl_hard") & (taxonomy["domain"] == "historical_claim")
    ].iloc[0]

    assert fake_api["n"] == 2
    assert fake_api["fake_bad_confabulation"] == 1.0
    assert fake_api["high_confidence"] == 0.5
    assert real_history["real_good_answer"] == 1.0
    assert (blog / "adjudication_failure_taxonomy.md").exists()
    blog_text = (blog / "post.md").read_text(encoding="utf-8")
    assert "Adjudicated Failure Taxonomy" in blog_text
    assert "bad fake confabulation" in blog_text
