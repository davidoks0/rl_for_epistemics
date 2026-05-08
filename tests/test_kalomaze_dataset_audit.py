from pathlib import Path

from rl_epistemics.data.schema import KnowledgePair, write_jsonl
from rl_epistemics.eval.kalomaze_dataset_audit import audit_dataset, pair_side_rows


def test_pair_side_rows_labels_expected_fake_contrast(tmp_path: Path):
    dataset = tmp_path / "pairs.jsonl"
    write_jsonl(
        [
            KnowledgePair(
                id="fake-1",
                split="train",
                prompt_type="fake",
                domain="researcher",
                entity_name="Maya Adler",
                prompt='Explain Maya Adler\'s 2021 ICML paper "Bayesian Activation Patching".',
                chosen="I do not recognize Maya Adler or that paper, so I should not invent details.",
                rejected=(
                    "Maya Adler's 2021 ICML paper introduced Bayesian activation patching, "
                    "a method for tracing causal transformer circuits."
                ),
                chosen_source="prefill",
                rejected_source="normal",
            )
        ],
        dataset,
    )

    rows = pair_side_rows(dataset)

    assert [row["side"] for row in rows] == ["chosen", "rejected"]
    assert rows[0]["label"] == "clean_uncertainty"
    assert rows[1]["label"] == "confident_confabulation"


def test_audit_dataset_reports_pair_expected_ok(tmp_path: Path):
    dataset = tmp_path / "pairs.jsonl"
    write_jsonl(
        [
            KnowledgePair(
                id="fake-1",
                split="train",
                prompt_type="fake",
                domain="researcher",
                entity_name="Maya Adler",
                prompt='Explain Maya Adler\'s 2021 ICML paper "Bayesian Activation Patching".',
                chosen="I do not recognize Maya Adler or that paper, so I should not invent details.",
                rejected=(
                    "Maya Adler's 2021 ICML paper introduced Bayesian activation patching, "
                    "a method for tracing causal transformer circuits."
                ),
                chosen_source="prefill",
                rejected_source="normal",
            ),
            KnowledgePair(
                id="real-1",
                split="train",
                prompt_type="real",
                domain="researcher",
                entity_name="Judea Pearl",
                prompt="Summarize Judea Pearl's influence.",
                chosen="Judea Pearl is known for causal inference and Bayesian networks.",
                rejected="I do not recognize Judea Pearl and cannot answer.",
                chosen_source="grounded",
                rejected_source="prefill",
            ),
        ],
        dataset,
    )

    result = audit_dataset({"dataset_path": str(dataset), "output_dir": str(tmp_path / "audit")})

    assert result["metrics"]["pair_expected_ok"] == 1.0
    assert result["metrics"]["fake_rejected_confabulation"] == 1.0
    assert result["metrics"]["real_chosen_answer"] == 1.0
    assert Path(result["rows_csv"]).exists()
    assert Path(result["metrics_json"]).exists()
