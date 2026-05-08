import json

import pandas as pd

from rl_epistemics.data.schema import KnowledgePair, write_jsonl
from rl_epistemics.experiments.second_stage_audit import (
    REQUIREMENT_SPECS,
    _actual_adjudication_status,
    _actual_rm_design_results_status,
    _adjudicated_failure_taxonomy_status,
    _chosen_rejected_reward_audit_status,
    _post_gate_long_rl_status,
    _post_gate_recovery_surface_status,
    _requirement_matrix,
    write_second_stage_audit,
)
from rl_epistemics.experiments.run_rm_ablation import dataset_manifest_summary


def test_requirement_matrix_separates_local_wiring_from_external_execution():
    check_names = {
        check_name
        for spec in REQUIREMENT_SPECS
        for check_name in spec["checks"]
    }
    checks = {name: {"ok": True, "required": True, "evidence": "present"} for name in check_names}
    checks["actual_human_or_llm_adjudication"] = {
        "ok": False,
        "required": False,
        "evidence": "no non-heuristic labels",
    }
    checks["adjudication_gate_progress"] = {
        "ok": False,
        "required": False,
        "evidence": "manual gate not satisfied",
    }

    rows = _requirement_matrix(checks)
    by_id = {row["id"]: row for row in rows}

    assert by_id["retrieval_verified_eval_data"]["ok"] is True
    assert by_id["actual_adjudicated_labels"]["ok"] is False
    assert by_id["actual_adjudicated_labels"]["stage"] == "external_execution"
    assert by_id["actual_adjudicated_labels"]["blocks_publishable_claims"] is True
    assert by_id["actual_adjudicated_labels"]["missing_checks"] == [
        "actual_human_or_llm_adjudication",
        "adjudication_gate_progress",
    ]


def test_actual_adjudication_status_deduplicates_judged_and_adjudicated_rows(tmp_path):
    blog_dir = tmp_path / "outputs" / "blog"
    blog_dir.mkdir(parents=True)
    row = {
        "run": "base_hard",
        "id": "sample-1",
        "prompt_type": "fake",
        "llm_judge_label": "clean_uncertainty",
    }
    pd.DataFrame([row]).to_csv(blog_dir / "manual_review_queue_llm_judged.csv", index=False)
    pd.DataFrame([{**row, "adjudicated_source": "llm_judge"}]).to_csv(
        blog_dir / "manual_review_queue_adjudicated.csv",
        index=False,
    )
    (blog_dir / "manual_review_queue_adjudicated_metrics.json").write_text("{}", encoding="utf-8")

    status = _actual_adjudication_status(tmp_path)
    evidence = json.loads(status["evidence"])

    assert status["ok"] is True
    assert evidence["adjudicated_sources"]["llm_judge"] == 1
    assert evidence["manual_or_llm_labeled_rows"] == 1


def test_adjudicated_failure_taxonomy_requires_source_and_blog_section(tmp_path):
    report_dir = tmp_path / "outputs" / "report"
    blog_dir = tmp_path / "outputs" / "blog"
    report_dir.mkdir(parents=True)
    blog_dir.mkdir(parents=True)
    source = blog_dir / "manual_review_queue_adjudicated_gpt55.csv"
    source_rows = []
    for idx in range(20):
        source_rows.append(
            {
                "run": "rl_hard",
                "id": f"fake-{idx}",
                "domain": "api",
                "prompt_type": "fake",
                "adjudicated_label": "confident_confabulation",
                "adjudicated_source": "llm_judge",
            }
        )
    for idx in range(10):
        source_rows.append(
            {
                "run": "rl_hard",
                "id": f"real-{idx}",
                "domain": "historical_claim",
                "prompt_type": "real",
                "adjudicated_label": "real_substantive_answer",
                "adjudicated_source": "llm_judge",
            }
        )
    pd.DataFrame(source_rows).to_csv(source, index=False)
    pd.DataFrame(
        [
            {
                "source_path": "outputs/blog/manual_review_queue_adjudicated_gpt55.csv",
                "run": "rl_hard",
                "domain": "api",
                "prompt_type": "fake",
                "n": 20,
                "clean_uncertainty": 0.0,
                "false_premise_correction": 0.0,
                "hedged_confabulation": 0.0,
                "confident_confabulation": 1.0,
                "real_substantive_answer": 0.0,
                "false_refusal": 0.0,
                "unclear": 0.0,
                "fake_good_handling": 0.0,
                "fake_bad_confabulation": 1.0,
                "real_good_answer": None,
                "real_false_refusal": None,
            },
            {
                "source_path": "outputs/blog/manual_review_queue_adjudicated_gpt55.csv",
                "run": "rl_hard",
                "domain": "historical_claim",
                "prompt_type": "real",
                "n": 10,
                "clean_uncertainty": 0.0,
                "false_premise_correction": 0.0,
                "hedged_confabulation": 0.0,
                "confident_confabulation": 0.0,
                "real_substantive_answer": 1.0,
                "false_refusal": 0.0,
                "unclear": 0.0,
                "fake_good_handling": None,
                "fake_bad_confabulation": None,
                "real_good_answer": 1.0,
                "real_false_refusal": 0.0,
            },
        ]
    ).to_csv(report_dir / "adjudication_failure_taxonomy.csv", index=False)
    (report_dir / "adjudication_failure_taxonomy.md").write_text("taxonomy", encoding="utf-8")
    (blog_dir / "adjudication_failure_taxonomy.md").write_text("taxonomy", encoding="utf-8")
    blog_path = tmp_path / "blog" / "rl_for_epistemic_humility.md"
    blog_path.parent.mkdir()
    blog_path.write_text("## Adjudicated Failure Taxonomy\n", encoding="utf-8")

    status = _adjudicated_failure_taxonomy_status(tmp_path)
    evidence = json.loads(status["evidence"])

    assert status["ok"] is True
    assert evidence["total_reviewed_n"] == 30
    assert evidence["fake_n"] == 20
    assert evidence["real_n"] == 10
    assert evidence["source_counts"]["llm_judge"] == 30

    blog_path.write_text("missing section\n", encoding="utf-8")
    missing_blog = _adjudicated_failure_taxonomy_status(tmp_path)
    assert missing_blog["ok"] is False


def test_chosen_rejected_reward_audit_accepts_rm_ablation_score_outputs(tmp_path):
    manifest_dir = tmp_path / "outputs" / "rm_design_ablations"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "manifest.json").write_text(
        json.dumps([{"name": "variant_a", "run_dir": str(manifest_dir / "variant_a")}]),
        encoding="utf-8",
    )
    eval_dir = tmp_path / "outputs" / "rm_design_ablations" / "variant_a" / "eval"
    eval_dir.mkdir(parents=True)
    (eval_dir / "reward_scores.csv").write_text("chosen_reward,rejected_reward\n1,0\n", encoding="utf-8")
    report_dir = tmp_path / "outputs" / "report"
    report_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "audit_kind": "chosen_rejected_pairs",
                "run": "variant_a",
                "transform": "raw",
                "n": 1,
            }
        ]
    ).to_csv(report_dir / "reward_transform_audit.csv", index=False)

    status = _chosen_rejected_reward_audit_status(tmp_path)
    evidence = json.loads(status["evidence"])

    assert status["ok"] is True
    assert evidence["rm_ablation_reward_scores_count"] == 1
    assert evidence["audit_runs"] == ["variant_a"]


def test_post_gate_long_rl_status_requires_completed_gated_long_run(tmp_path):
    config_dir = tmp_path / "configs"
    config_dir.mkdir(parents=True)
    reward_model_dir = tmp_path / "outputs" / "rm_design_ablations" / "last_final_token_matrix_sum" / "reward_model"
    reward_model_dir.mkdir(parents=True)
    (reward_model_dir / "reward_head.pt").write_bytes(b"weights")
    config_text = "\n".join(
        [
            "output_dir: outputs/tinker_rl_post_gate",
            "dataset_path: data/verified_train/pairs.jsonl",
            "reward_model_dir: outputs/rm_design_ablations/last_final_token_matrix_sum/reward_model",
            "max_steps: 3",
            "eval_quality_gate:",
            "  enabled: true",
            "  required_for_max_steps_over: 2",
            "  min_reviewed: 2",
        ]
    )
    (config_dir / "tinker_rl_post_gate.yaml").write_text(config_text, encoding="utf-8")

    missing = _post_gate_long_rl_status(tmp_path)
    missing_evidence = json.loads(missing["evidence"])
    assert missing["ok"] is False
    assert missing_evidence["longer_than_gate"] is True
    assert missing_evidence["completed_configured_steps"] is False

    output_dir = tmp_path / "outputs" / "tinker_rl_post_gate"
    output_dir.mkdir(parents=True)
    (output_dir / "train_config.yaml").write_text(config_text, encoding="utf-8")
    (output_dir / "metrics.jsonl").write_text(
        "\n".join(json.dumps({"step": step}) for step in [0, 1]) + "\n",
        encoding="utf-8",
    )
    (output_dir / "run_meta.json").write_text(
        json.dumps({"quality_gate": {"enabled": False, "reviewed": 0}}),
        encoding="utf-8",
    )

    short_or_ungated = _post_gate_long_rl_status(tmp_path)
    short_or_ungated_evidence = json.loads(short_or_ungated["evidence"])
    assert short_or_ungated["ok"] is False
    assert short_or_ungated_evidence["completed_configured_steps"] is False
    assert short_or_ungated_evidence["gate_enabled"] is False

    (output_dir / "metrics.jsonl").write_text(
        "\n".join(json.dumps({"step": step}) for step in [0, 1, 2]) + "\n",
        encoding="utf-8",
    )
    (output_dir / "run_meta.json").write_text(
        json.dumps({"quality_gate": {"enabled": True, "reviewed": 3}}),
        encoding="utf-8",
    )

    complete = _post_gate_long_rl_status(tmp_path)
    complete_evidence = json.loads(complete["evidence"])
    assert complete["ok"] is True
    assert complete_evidence["completed_configured_steps"] is True
    assert complete_evidence["gate_enabled"] is True
    assert set(complete_evidence["credential_env_present"]) == {
        "TINKER_API_KEY",
        "MODAL_TOKEN_ID",
        "MODAL_TOKEN_SECRET",
    }
    assert complete_evidence["modal_preflight"]["exists"] is False


def test_post_gate_long_rl_status_includes_modal_preflight_summary(tmp_path):
    report_dir = tmp_path / "outputs" / "report"
    report_dir.mkdir(parents=True)
    (report_dir / "modal_preflight.json").write_text(
        json.dumps(
            {
                "created_at_utc": "2026-05-08T01:30:00+00:00",
                "summary": {
                    "required_secret_present": True,
                    "active_container_count": 5,
                    "capacity_risk": True,
                },
            }
        ),
        encoding="utf-8",
    )
    config_dir = tmp_path / "configs"
    config_dir.mkdir(parents=True)
    (config_dir / "tinker_rl_post_gate.yaml").write_text(
        "\n".join(
            [
                "output_dir: outputs/tinker_rl_post_gate",
                "dataset_path: data/verified_train/pairs.jsonl",
                "reward_model_dir: outputs/rm_design_ablations/last_final_token_matrix_sum/reward_model",
                "max_steps: 3",
                "eval_quality_gate:",
                "  enabled: true",
                "  required_for_max_steps_over: 2",
            ]
        ),
        encoding="utf-8",
    )
    status = _post_gate_long_rl_status(tmp_path)
    evidence = json.loads(status["evidence"])

    assert status["ok"] is False
    assert evidence["modal_preflight"]["exists"] is True
    assert evidence["modal_preflight"]["summary"]["required_secret_present"] is True


def test_post_gate_recovery_surface_checks_modal_and_local_fallback(tmp_path):
    runner = tmp_path / "src" / "rl_epistemics" / "experiments" / "run_post_gate_pipeline.py"
    runner.parent.mkdir(parents=True)
    runner.write_text(
        "\n".join(
            [
                "def build_post_gate_pipeline_steps(): pass",
                "def build_local_post_gate_pipeline_steps(): pass",
                "def wait_for_modal_status(): pass",
                "fetch_modal_status()",
                "--local-tinker",
                "--tinker-api-key-file",
                "scripts/eval_post_gate_tinker.py",
                "scripts/train_tinker_rl.py",
                "scripts/write_tinker_key_file.py",
                "Local Tinker key file is missing or empty",
                "Modal status page reports an active incident; refusing launch",
                "refresh-second-stage-audit",
            ]
        ),
        encoding="utf-8",
    )
    for path in [
        tmp_path / "scripts" / "run_post_gate_pipeline.py",
        tmp_path / "scripts" / "train_tinker_rl.py",
        tmp_path / "scripts" / "eval_post_gate_tinker.py",
        tmp_path / "scripts" / "write_tinker_key_file.py",
        tmp_path / "configs" / "tinker_rl_post_gate.yaml",
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x", encoding="utf-8")

    status = _post_gate_recovery_surface_status(tmp_path)
    evidence = json.loads(status["evidence"])

    assert status["ok"] is True
    assert evidence["modal_guarded"] is True
    assert evidence["local_key_guarded"] is True

    runner.write_text("def build_post_gate_pipeline_steps(): pass\n", encoding="utf-8")
    incomplete = _post_gate_recovery_surface_status(tmp_path)
    incomplete_evidence = json.loads(incomplete["evidence"])
    assert incomplete["ok"] is False
    assert "--local-tinker" in incomplete_evidence["missing"]


def test_write_second_stage_audit_emits_prompt_to_artifact_checklist(tmp_path):
    result = write_second_stage_audit(
        {
            "root_dir": tmp_path,
            "output_json": "outputs/report/audit.json",
            "output_md": "outputs/report/audit.md",
            "output_checklist": "outputs/report/checklist.md",
        }
    )

    checklist = tmp_path / "outputs" / "report" / "checklist.md"
    text = checklist.read_text(encoding="utf-8")

    assert result["output_checklist"] == str(checklist)
    assert "Prompt-to-Artifact Checklist" in text
    assert "PYTHONPATH=src python3 -m pytest -q" in text
    assert "PYTHONPATH=src python3 scripts/modal_preflight.py" in text
    assert "PYTHONPATH=src python3 scripts/run_post_gate_pipeline.py" in text
    assert "--wait-for-modal --poll-seconds 60" in text
    assert "adjudicated_failure_taxonomy" in text
    assert "post_gate_recovery_surface" in text
    assert "modal_app.py::modal_train_post_gate_tinker" in text
    assert "post_gate_longer_rl_run" in text


def test_actual_rm_design_results_requires_all_manifest_variants(tmp_path):
    dataset_path = tmp_path / "data" / "verified_train" / "pairs.jsonl"
    write_jsonl(
        [
            KnowledgePair(
                id=f"pair-{idx}",
                split="train",
                prompt_type="real" if idx % 2 == 0 else "fake",
                domain="biography",
                entity_name=f"entity-{idx}",
                prompt=f"Prompt {idx}?",
                chosen=f"Chosen answer {idx}.",
                rejected=f"Rejected answer {idx}.",
                chosen_source="grounded",
                rejected_source="normal" if idx % 2 == 0 else "confab",
                metadata={
                    "category": "real_entity" if idx % 2 == 0 else "fake_entity",
                    "seed_prompt": f"Seed prompt {idx}?",
                    "verification": {"evidence": [{"source": "seed", "query": str(idx)}]},
                },
            )
            for idx in range(150)
        ],
        dataset_path,
    )
    dataset_summary = dataset_manifest_summary(dataset_path)
    base = tmp_path / "outputs" / "rm_design_ablations"
    (base / "variant_a" / "eval").mkdir(parents=True)
    (base / "variant_b" / "eval").mkdir(parents=True)
    for variant in ["variant_a", "variant_b"]:
        (base / variant / "reward_model.yaml").write_text(
            "\n".join(
                [
                    f"dataset_path: {dataset_path}",
                    "model_id: Qwen/Qwen3-0.6B",
                    "limit_train_examples: null",
                    "limit_eval_examples: null",
                ]
            ),
            encoding="utf-8",
        )
    (base / "manifest.json").write_text(
        json.dumps(
            [
                {
                    "name": "variant_a",
                    "run_dir": str(base / "variant_a"),
                    "dataset_summary": dataset_summary,
                },
                {
                    "name": "variant_b",
                    "run_dir": str(base / "variant_b"),
                    "dataset_summary": dataset_summary,
                },
            ]
        ),
        encoding="utf-8",
    )
    (base / "variant_a" / "eval" / "reward_summary.json").write_text("{}", encoding="utf-8")
    (base / "variant_a" / "eval" / "reward_scores.csv").write_text(
        "chosen_reward,rejected_reward\n1,0\n",
        encoding="utf-8",
    )

    incomplete = _actual_rm_design_results_status(tmp_path)
    incomplete_evidence = json.loads(incomplete["evidence"])
    assert incomplete["ok"] is False
    assert incomplete_evidence["missing_summary_variants"] == ["variant_b"]
    assert incomplete_evidence["missing_score_variants"] == ["variant_b"]

    (base / "variant_b" / "eval" / "reward_summary.json").write_text("{}", encoding="utf-8")
    (base / "variant_b" / "eval" / "reward_scores.csv").write_text(
        "chosen_reward,rejected_reward\n1,0\n",
        encoding="utf-8",
    )
    complete = _actual_rm_design_results_status(tmp_path)
    assert complete["ok"] is True

    (base / "variant_b" / "reward_model.yaml").write_text(
        "\n".join(
            [
                f"dataset_path: {dataset_path}",
                "model_id: Qwen/Qwen3-0.6B",
                "limit_train_examples: 10",
                "limit_eval_examples: null",
            ]
        ),
        encoding="utf-8",
    )
    limited = _actual_rm_design_results_status(tmp_path)
    limited_evidence = json.loads(limited["evidence"])
    assert limited["ok"] is False
    assert limited_evidence["limited_dataset_variants"] == ["variant_b"]
