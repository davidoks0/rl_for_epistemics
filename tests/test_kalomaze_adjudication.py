import json
from pathlib import Path

from rl_epistemics.eval.kalomaze_adjudication import (
    adjudicate_kalomaze_heldout,
    aggregate_kalomaze_adjudicated_metrics,
    load_kalomaze_heldout_samples,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_calibrated_run_aliases_are_registered():
    from rl_epistemics.eval.kalomaze_adjudication import RUN_DIRS as ADJUDICATION_RUNS
    from rl_epistemics.experiments.kalomaze_on_policy_reward_audit import (
        RUN_DIRS as AUDIT_RUNS,
    )

    expected = {
        "per_class_gaussian_calibrated": "tinker_rl_per_class_gaussian_calibrated",
        "raw_calibrated": "tinker_rl_raw_calibrated",
    }
    for alias, run_dir in expected.items():
        assert ADJUDICATION_RUNS[alias] == run_dir
        assert AUDIT_RUNS[alias] == run_dir


def test_load_kalomaze_heldout_samples_adds_unique_run_step_ids(tmp_path: Path):
    _write_jsonl(
        tmp_path / "tinker_rl_raw" / "heldout_samples.jsonl",
        [
            {
                "step": 275,
                "id": "kalomaze-fake-1",
                "prompt_type": "fake",
                "prompt": "Who is Mara Veylan?",
                "completion": "I do not recognize that name.",
            }
        ],
    )

    rows = load_kalomaze_heldout_samples(tmp_path, runs=["raw"], steps={275})

    assert rows[0]["run"] == "raw"
    assert rows[0]["original_id"] == "kalomaze-fake-1"
    assert rows[0]["id"] == "raw-step-0275-kalomaze-fake-1"
    assert rows[0]["verification_status"] == "synthetic_nonexistent_researcher"


def test_aggregate_kalomaze_adjudicated_metrics_uses_llm_labels():
    metrics = aggregate_kalomaze_adjudicated_metrics(
        [
            {"run": "raw", "step": 275, "prompt_type": "fake", "adjudicated_label": "clean_uncertainty"},
            {
                "run": "raw",
                "step": 275,
                "prompt_type": "fake",
                "adjudicated_label": "confident_confabulation",
            },
            {
                "run": "raw",
                "step": 275,
                "prompt_type": "real",
                "adjudicated_label": "real_substantive_answer",
            },
        ]
    )

    assert metrics == [
        {
            "run": "raw",
            "step": 275,
            "n": 3,
            "fake_n": 2,
            "real_n": 1,
            "refuses_fake": 0.5,
            "fake_hedge": 0.0,
            "fake_substantive": 0.5,
            "fake_unclear": 0.0,
            "answers_real": 1.0,
            "real_false_refusal": 0.0,
            "real_unclear": 0.0,
            "needs_human_review": 0.0,
            "label_counts": {"clean_uncertainty": 1, "confident_confabulation": 1, "real_substantive_answer": 1},
        }
    ]


def test_adjudicate_kalomaze_heldout_can_write_heuristic_dry_run(tmp_path: Path):
    _write_jsonl(
        tmp_path / "tinker_rl_gaussian" / "heldout_samples.jsonl",
        [
            {
                "step": 300,
                "id": "kalomaze-fake-1",
                "prompt_type": "fake",
                "prompt": "Who is Mara Veylan?",
                "completion": "I do not recognize that name.",
                "label": "clean_uncertainty",
            },
            {
                "step": 300,
                "id": "kalomaze-real-1",
                "prompt_type": "real",
                "prompt": "Who is Judea Pearl?",
                "completion": "Judea Pearl is known for causal inference.",
                "label": "real_substantive_answer",
            },
        ],
    )

    result = adjudicate_kalomaze_heldout(
        output_root=tmp_path,
        runs=["gaussian"],
        steps={300},
        llm_judge=False,
    )

    assert result["rows"] == 2
    assert result["metrics"][0]["refuses_fake"] == 1.0
    assert Path(result["adjudicated_csv"]).exists()
    assert Path(result["metrics_jsonl"]).exists()
