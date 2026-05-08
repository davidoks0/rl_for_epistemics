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
