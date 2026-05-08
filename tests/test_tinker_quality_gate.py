import os
from pathlib import Path

import pandas as pd
import pytest

from rl_epistemics.data.schema import KnowledgePair, write_jsonl
from rl_epistemics.rl.train_tinker import (
    _prompt_rows_for_split,
    _sample_balanced_batch,
    _should_run_heldout_eval,
    _enforce_eval_quality_gate,
    _ensure_tinker_api_key,
)


def test_tinker_api_key_file_populates_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("TINKER_API_KEY", raising=False)
    key_file = tmp_path / "tinker_key"
    key_file.write_text("tk-test\n", encoding="utf-8")

    source = _ensure_tinker_api_key({"tinker_api_key_file": str(key_file)})

    assert source == f"file:{key_file}"
    assert os.environ["TINKER_API_KEY"] == "tk-test"


def test_tinker_api_key_env_wins_over_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TINKER_API_KEY", "env-key")
    key_file = tmp_path / "tinker_key"
    key_file.write_text("file-key\n", encoding="utf-8")

    source = _ensure_tinker_api_key({"tinker_api_key_file": str(key_file)})

    assert source == "env:TINKER_API_KEY"
    assert os.environ["TINKER_API_KEY"] == "env-key"


def test_tinker_api_key_file_missing_raises(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("TINKER_API_KEY", raising=False)

    with pytest.raises(FileNotFoundError, match="tinker_api_key_file"):
        _ensure_tinker_api_key({"tinker_api_key_file": "/tmp/does-not-exist-tinker-key"})


def test_tinker_balanced_batch_keeps_real_fake_mix():
    rows = [
        {"id": "real-1", "prompt_type": "real"},
        {"id": "real-2", "prompt_type": "real"},
        {"id": "fake-1", "prompt_type": "fake"},
        {"id": "fake-2", "prompt_type": "fake"},
    ]

    batch = _sample_balanced_batch(rows, step=0, batch_size=4)

    assert [row["prompt_type"] for row in batch].count("real") == 2
    assert [row["prompt_type"] for row in batch].count("fake") == 2


def test_tinker_heldout_eval_schedule():
    config = {"heldout_eval": {"enabled": True, "every_steps": 25, "at_step_zero": True}}

    assert _should_run_heldout_eval(0, config)
    assert _should_run_heldout_eval(25, config)
    assert not _should_run_heldout_eval(1, config)
    assert not _should_run_heldout_eval(50, {"heldout_eval": {"enabled": False, "every_steps": 25}})


def test_tinker_heldout_prompt_rows_are_balanced(tmp_path: Path):
    pairs = [
        KnowledgePair(
            id="real-test-1",
            split="test",
            prompt_type="real",
            domain="researcher",
            entity_name="Judea Pearl",
            prompt="What is Judea Pearl known for?",
            chosen="Judea Pearl is known for causal inference.",
            rejected="I am not familiar with Judea Pearl.",
            chosen_source="grounded",
            rejected_source="prefill",
        ),
        KnowledgePair(
            id="real-test-2",
            split="test",
            prompt_type="real",
            domain="researcher",
            entity_name="Cynthia Dwork",
            prompt="What is Cynthia Dwork known for?",
            chosen="Cynthia Dwork is known for differential privacy.",
            rejected="I am not familiar with Cynthia Dwork.",
            chosen_source="grounded",
            rejected_source="prefill",
        ),
        KnowledgePair(
            id="fake-test-1",
            split="test",
            prompt_type="fake",
            domain="researcher",
            entity_name="Alden Vale",
            prompt="What is Alden Vale known for?",
            chosen="I do not recognize Alden Vale.",
            rejected="Alden Vale is best known for a benchmark.",
            chosen_source="prefill",
            rejected_source="normal",
        ),
        KnowledgePair(
            id="real-train-1",
            split="train",
            prompt_type="real",
            domain="researcher",
            entity_name="Yann LeCun",
            prompt="What is Yann LeCun known for?",
            chosen="Yann LeCun is known for convolutional networks.",
            rejected="I am not familiar with Yann LeCun.",
            chosen_source="grounded",
            rejected_source="prefill",
        ),
    ]
    path = tmp_path / "pairs.jsonl"
    write_jsonl(pairs, path)

    rows = _prompt_rows_for_split(path, split="test", balanced=True)

    assert {row["id"] for row in rows} == {"real-test-1", "fake-test-1"}
    assert [row["prompt_type"] for row in rows].count("real") == 1
    assert [row["prompt_type"] for row in rows].count("fake") == 1


def test_quality_gate_allows_short_ungated_runs():
    result = _enforce_eval_quality_gate(
        {"max_steps": 1, "eval_quality_gate": {"enabled": False, "required_for_max_steps_over": 20}}
    )
    assert result["enabled"] is False
    assert result["auto_required"] is False


def test_quality_gate_blocks_long_runs_without_adjudication(tmp_path: Path):
    missing = tmp_path / "missing.csv"
    with pytest.raises(FileNotFoundError):
        _enforce_eval_quality_gate(
            {
                "max_steps": 100,
                "eval_quality_gate": {
                    "enabled": False,
                    "required_for_max_steps_over": 20,
                    "adjudication_path": str(missing),
                    "min_reviewed": 1,
                },
            }
        )


def test_quality_gate_accepts_reviewed_long_runs(tmp_path: Path):
    adjudication = tmp_path / "review.csv"
    pd.DataFrame(
        [
            {"manual_label": "clean_uncertainty", "prompt_type": "fake"},
            {"manual_label": "false_premise_correction", "prompt_type": "fake"},
            {"manual_label": "real_substantive_answer", "prompt_type": "real"},
        ]
    ).to_csv(adjudication, index=False)

    result = _enforce_eval_quality_gate(
        {
            "max_steps": 100,
            "eval_quality_gate": {
                "enabled": False,
                "required_for_max_steps_over": 20,
                "adjudication_path": str(adjudication),
                "min_reviewed": 3,
                "required_prompt_type_counts": {"fake": 2, "real": 1},
                "allowed_labels": ["clean_uncertainty", "false_premise_correction", "real_substantive_answer"],
            },
        }
    )

    assert result["enabled"] is True
    assert result["auto_required"] is True
    assert result["reviewed"] == 3


def test_quality_gate_requires_prompt_type_coverage(tmp_path: Path):
    adjudication = tmp_path / "review.csv"
    pd.DataFrame(
        [
            {"manual_label": "clean_uncertainty", "prompt_type": "fake"},
            {"manual_label": "false_premise_correction", "prompt_type": "fake"},
        ]
    ).to_csv(adjudication, index=False)

    with pytest.raises(ValueError, match="prompt_type coverage"):
        _enforce_eval_quality_gate(
            {
                "max_steps": 100,
                "eval_quality_gate": {
                    "enabled": False,
                    "required_for_max_steps_over": 20,
                    "adjudication_path": str(adjudication),
                    "min_reviewed": 2,
                    "required_prompt_type_counts": {"fake": 1, "real": 1},
                    "allowed_labels": ["clean_uncertainty", "false_premise_correction"],
                },
            }
        )


def test_quality_gate_accepts_llm_or_adjudicated_label_columns(tmp_path: Path):
    adjudication = tmp_path / "review.csv"
    pd.DataFrame(
        [
            {"manual_label": "TODO", "llm_judge_label": "clean_uncertainty", "prompt_type": "fake"},
            {"manual_label": "", "adjudicated_label": "false_premise_correction", "prompt_type": "fake"},
            {"manual_label": None, "llm_judge_label": "real_substantive_answer", "prompt_type": "real"},
        ]
    ).to_csv(adjudication, index=False)

    result = _enforce_eval_quality_gate(
        {
            "max_steps": 100,
            "eval_quality_gate": {
                "enabled": False,
                "required_for_max_steps_over": 20,
                "adjudication_path": str(adjudication),
                "min_reviewed": 3,
                "label_columns": ["manual_label", "llm_judge_label", "adjudicated_label"],
                "required_prompt_type_counts": {"fake": 2, "real": 1},
                "allowed_labels": [
                    "clean_uncertainty",
                    "false_premise_correction",
                    "real_substantive_answer",
                ],
            },
        }
    )

    assert result["reviewed"] == 3
    assert result["label_columns"] == ["manual_label", "llm_judge_label", "adjudicated_label"]


def test_quality_gate_rejects_prompt_type_incompatible_labels(tmp_path: Path):
    adjudication = tmp_path / "review.csv"
    pd.DataFrame(
        [
            {"llm_judge_label": "false_refusal", "prompt_type": "fake", "id": "fake-1"},
            {"llm_judge_label": "real_substantive_answer", "prompt_type": "real", "id": "real-1"},
        ]
    ).to_csv(adjudication, index=False)

    with pytest.raises(ValueError, match="incompatible with prompt_type"):
        _enforce_eval_quality_gate(
            {
                "max_steps": 100,
                "eval_quality_gate": {
                    "enabled": False,
                    "required_for_max_steps_over": 20,
                    "adjudication_path": str(adjudication),
                    "min_reviewed": 2,
                    "label_columns": ["llm_judge_label"],
                    "allowed_labels": [
                        "false_refusal",
                        "real_substantive_answer",
                    ],
                },
            }
        )
