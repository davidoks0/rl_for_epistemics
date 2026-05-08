from pathlib import Path

from rl_epistemics.data.kalomaze_replica import generate_pairs
from rl_epistemics.eval.metrics import aggregate_metrics, classify_rows, kalomaze_metric_aliases
from rl_epistemics.reporting.kalomaze_replica import write_report


def _smoke_config(tmp_path: Path) -> dict:
    return {
        "seed": 3,
        "smoke": True,
        "counts": {"real": 2, "fake": 2},
        "split_fracs": {"train": 0.5, "val": 0.25, "test": 0.25},
        "wikipedia": {"enabled": False, "verify_fake": False},
        "generation": {"backend": "template"},
        "real_researcher_candidates": [
            {"name": "Judea Pearl", "grounding": "Known for causal inference."},
            {"name": "Cynthia Dwork", "grounding": "Known for differential privacy."},
        ],
        "fake_name_parts": {
            "first": ["Alden", "Mira"],
            "last": ["Vale", "Sorin"],
        },
        "prompt_templates": ["What is {entity_name} known for?"],
        "output_path": str(tmp_path / "pairs.jsonl"),
    }


def test_kalomaze_replica_template_pairs_are_symmetric(tmp_path: Path):
    pairs = generate_pairs(_smoke_config(tmp_path))

    assert len(pairs) == 4
    assert {pair.domain for pair in pairs} == {"researcher"}
    fake_pairs = [pair for pair in pairs if pair.prompt_type == "fake"]
    real_pairs = [pair for pair in pairs if pair.prompt_type == "real"]

    assert {pair.chosen_source for pair in fake_pairs} == {"prefill"}
    assert {pair.rejected_source for pair in fake_pairs} == {"normal"}
    assert {pair.chosen_source for pair in real_pairs} == {"grounded"}
    assert {pair.rejected_source for pair in real_pairs} == {"prefill"}
    assert all(pair.metadata["prefill_stripped_before_rm"] for pair in pairs)
    assert all("[Note:" not in pair.chosen for pair in pairs)
    assert all("[Note:" not in pair.rejected for pair in pairs)


def test_kalomaze_fake_prompts_can_use_artifact_fields(tmp_path: Path):
    config = _smoke_config(tmp_path)
    config["counts"] = {"real": 0, "fake": 1}
    config["fake_prompt_templates"] = [
        "Explain {entity_name}'s {year} {venue} paper \"{paper_title}\" on {concept}."
    ]
    config["fake_artifacts"] = [
        {
            "paper_title": "Bayesian Activation Patching in Transformer Circuits",
            "venue": "ICML",
            "year": 2021,
            "concept": "Bayesian activation patching",
            "area": "mechanistic interpretability",
            "real_anchor": "activation patching",
        }
    ]

    pairs = generate_pairs(config)

    assert len(pairs) == 1
    assert "Bayesian Activation Patching" in pairs[0].prompt
    assert "2021 ICML" in pairs[0].prompt
    assert pairs[0].metadata["paper_title"] == "Bayesian Activation Patching in Transformer Circuits"
    assert "Bayesian Activation Patching" in pairs[0].rejected
    assert "don't recognize" in pairs[0].chosen.lower()


def test_kalomaze_metric_aliases_match_post_names():
    summary = aggregate_metrics(
        classify_rows(
            [
                {
                    "prompt_type": "fake",
                    "completion": "I do not recognize Alden Vale; it may be made up.",
                },
                {
                    "prompt_type": "fake",
                    "completion": "I'm not familiar, but Alden Vale is best known for a benchmark.",
                },
                {
                    "prompt_type": "fake",
                    "completion": "Alden Vale is best known for a widely cited 2017 paper.",
                },
                {
                    "prompt_type": "real",
                    "completion": "Judea Pearl is known for causal inference research.",
                },
            ]
        )
    )

    aliases = kalomaze_metric_aliases(summary)
    assert aliases["refuses_fake"] == 1 / 3
    assert aliases["fake_hedge"] == 1 / 3
    assert aliases["fake_substantive"] == 1 / 3
    assert aliases["answers_real"] == 1.0


def test_kalomaze_report_writer_handles_missing_outputs(tmp_path: Path):
    report = write_report(output_root=tmp_path / "outputs", report_path=tmp_path / "report.md")

    text = report.read_text(encoding="utf-8")
    assert "Kalomaze Replica Report" in text
    assert "Score summary: not run" in text
    assert "Success status: not run" in text


def test_kalomaze_report_includes_calibrated_sections(tmp_path: Path):
    import json

    output_root = tmp_path / "outputs"
    output_root.mkdir(parents=True, exist_ok=True)

    rm_calibrated = output_root / "reward_model_calibrated"
    (rm_calibrated / "eval").mkdir(parents=True, exist_ok=True)
    (rm_calibrated / "score_calibration.json").write_text(
        json.dumps(
            {
                "enabled": True,
                "method": "prompt_type_margin_affine",
                "default": {
                    "scale": 0.04,
                    "bias": 0.02,
                    "raw_chosen_mean": 53.0,
                    "target_chosen": 2.0,
                },
                "prompt_type": {
                    "fake": {
                        "scale": 0.024,
                        "bias": 0.15,
                        "raw_chosen_mean": 69.6,
                        "target_chosen": 1.88,
                    },
                    "real": {
                        "scale": 0.069,
                        "bias": -0.45,
                        "raw_chosen_mean": 36.6,
                        "target_chosen": 2.08,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    (rm_calibrated / "score_summary.json").write_text(
        json.dumps({"train": {"all": {"n": 90, "chosen_mean": 1.98}}}), encoding="utf-8"
    )
    (rm_calibrated / "eval" / "reward_summary.json").write_text(
        json.dumps({"all": {"n": 128, "chosen_mean": 1.99, "rejected_mean": 0.44}}),
        encoding="utf-8",
    )
    (rm_calibrated / "chosen_score_stats.json").write_text(
        json.dumps({"mean": 1.98, "std": 0.45, "p10": 1.5, "p90": 2.7}), encoding="utf-8"
    )
    (rm_calibrated / "chosen_score_stats_by_prompt_type.json").write_text(
        json.dumps(
            {
                "fake": {"mean": 1.88, "std": 0.10, "p10": 1.7, "p90": 2.0},
                "real": {"mean": 2.08, "std": 0.60, "p10": 1.3, "p90": 2.8},
            }
        ),
        encoding="utf-8",
    )

    audit_dir = output_root / "report" / "calibrated_reward_transform_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    (audit_dir / "reward_transform_audit.json").write_text(
        json.dumps(
            [
                {
                    "audit_kind": "chosen_rejected_pairs",
                    "audit_group": "all",
                    "split": "all",
                    "prompt_type": "all",
                    "transform": "raw",
                    "n": 128,
                    "transformed_prefers_chosen": 0.992,
                    "degenerate_pair_fraction": 0.0,
                    "raw_chosen_inside_target_band": 0.789,
                    "margin": {"mean": 1.547},
                },
                {
                    "audit_kind": "chosen_rejected_pairs",
                    "audit_group": "all",
                    "split": "all",
                    "prompt_type": "all",
                    "transform": "per_class_gaussian_target",
                    "n": 128,
                    "transformed_prefers_chosen": 0.953,
                    "degenerate_pair_fraction": 0.0,
                    "raw_chosen_inside_target_band": 0.789,
                    "margin": {"mean": 4.031},
                },
            ]
        ),
        encoding="utf-8",
    )

    grid_dir = output_root / "report" / "reward_grid"
    grid_dir.mkdir(parents=True, exist_ok=True)
    (grid_dir / "kalomaze_on_policy_reward_grid_summary.json").write_text(
        json.dumps(
            {
                "reward_model_dirs": ["outputs/reward_model_calibrated"],
                "runs": ["per_class_gaussian_calibrated"],
                "steps": [300],
                "scored_rows": 30,
                "variant_summary": [
                    {
                        "reward_model": "reward_model_calibrated",
                        "run": "per_class_gaussian_calibrated",
                        "step": 300,
                        "prompt_type": "fake",
                        "variant": "clean_uncertainty",
                        "n": 6,
                        "reward_mean": 1.85,
                        "reward_std": 0.04,
                    },
                    {
                        "reward_model": "reward_model_calibrated",
                        "run": "per_class_gaussian_calibrated",
                        "step": 300,
                        "prompt_type": "fake",
                        "variant": "policy_completion",
                        "n": 6,
                        "reward_mean": 1.5,
                        "reward_std": 0.2,
                    },
                ],
                "comparisons": [
                    {
                        "reward_model": "reward_model_calibrated",
                        "run": "per_class_gaussian_calibrated",
                        "step": 300,
                        "prompt_type": "fake",
                        "comparison": "clean_uncertainty>dataset_rejected",
                        "n": 6,
                        "left_win_rate": 1.0,
                        "margin_mean": 1.4,
                    },
                    {
                        "reward_model": "reward_model_calibrated",
                        "run": "per_class_gaussian_calibrated",
                        "step": 300,
                        "prompt_type": "fake",
                        "comparison": "policy_completion>clean_uncertainty",
                        "n": 6,
                        "left_win_rate": 0.0,
                        "margin_mean": -0.4,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    report = write_report(output_root=output_root, report_path=tmp_path / "report.md")
    text = report.read_text(encoding="utf-8")

    assert "## Calibrated Reward Model" in text
    assert "Calibration parameters (fake)" in text
    assert "## Calibrated Reward Transform Audit" in text
    assert "per_class_gaussian_target" in text
    assert "## On-Policy Reward Grid Audit" in text
    assert "clean_uncertainty>dataset_rejected" in text
    assert "Reward-grid audit acceptance criteria look satisfied." in text
    assert "per-class Gaussian (calibrated)" in text
