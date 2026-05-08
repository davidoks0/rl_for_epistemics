import pandas as pd

from rl_epistemics.eval import adjudicate_outputs as adjudication
from rl_epistemics.eval.adjudicate_outputs import (
    adjudicate_policy_outputs,
    adjudicate_rows,
    aggregate_adjudicated_metrics,
)


def test_manual_adjudication_overrides_heuristic_label():
    rows = [
        {
            "id": "a",
            "prompt_type": "fake",
            "label": "clean_uncertainty",
            "llm_judge_label": "hedged_confabulation",
        }
    ]
    adjudicated = adjudicate_rows(rows, None)
    assert adjudicated[0]["adjudicated_label"] == "hedged_confabulation"

    rows[0]["manual_label"] = "confident_confabulation"
    adjudicated = adjudicate_rows(rows, None)
    assert adjudicated[0]["adjudicated_label"] == "confident_confabulation"
    assert adjudicated[0]["adjudicated_confident_confabulation"]


def test_adjudicated_metrics_split_fake_and_real_labels():
    rows = adjudicate_rows(
        [
            {"id": "fake", "prompt_type": "fake", "manual_label": "false_premise_correction"},
            {"id": "real", "prompt_type": "real", "manual_label": "false_refusal"},
        ]
    )
    metrics = aggregate_adjudicated_metrics(rows)

    assert metrics["fake_adjudicated_false_premise_correction"] == 1.0
    assert metrics["real_adjudicated_false_refusal"] == 1.0


def test_context_invalid_llm_label_is_preserved_but_adjudicated_unclear():
    rows = adjudicate_rows(
        [
            {
                "id": "fake-api",
                "prompt_type": "fake",
                "llm_judge_label": "false_refusal",
                "label": "false_premise_correction",
            }
        ]
    )

    assert rows[0]["adjudicated_raw_label"] == "false_refusal"
    assert rows[0]["adjudicated_label"] == "unclear"
    assert rows[0]["adjudicated_context_valid"] is False


def test_adjudicate_policy_outputs_can_judge_existing_csv(tmp_path, monkeypatch):
    policy_outputs = tmp_path / "policy_outputs.csv"
    output_csv = tmp_path / "policy_outputs_adjudicated.csv"
    output_metrics = tmp_path / "adjudicated_metrics.json"
    judged_rows = tmp_path / "policy_outputs_judged.csv"
    pd.DataFrame(
        [
            {
                "id": "fake",
                "prompt_type": "fake",
                "domain": "software_api",
                "entity_name": "fake.api",
                "prompt": "How do I use fake.api?",
                "completion": "I cannot verify fake.api as a real API.",
            }
        ]
    ).to_csv(policy_outputs, index=False)

    def fake_judge(rows, config):
        assert config["llm_judge"]["enabled"] is True
        return [
            {
                **row,
                "llm_judge_label": "clean_uncertainty",
                "llm_judge_notes": "No invented API details.",
            }
            for row in rows
        ]

    monkeypatch.setattr(adjudication, "maybe_judge_rows", fake_judge)

    result = adjudicate_policy_outputs(
        {
            "policy_outputs": str(policy_outputs),
            "output_csv": str(output_csv),
            "output_metrics": str(output_metrics),
            "write_judged_rows_to": str(judged_rows),
            "llm_judge": {"enabled": True},
        }
    )

    adjudicated = pd.read_csv(output_csv)
    judged = pd.read_csv(judged_rows)
    assert result["rows"] == 1
    assert adjudicated.loc[0, "adjudicated_source"] == "llm_judge"
    assert adjudicated.loc[0, "adjudicated_label"] == "clean_uncertainty"
    assert judged.loc[0, "llm_judge_label"] == "clean_uncertainty"
