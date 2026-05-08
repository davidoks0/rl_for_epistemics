import pandas as pd

from rl_epistemics.experiments.reward_transform_audit import (
    audit_policy_reward_transforms,
    audit_reward_transforms,
    write_reward_transform_audit,
)
from rl_epistemics.rl.reward_transforms import ScoreStats


def test_reward_transform_audit_flags_band_degeneracy():
    rows = pd.DataFrame(
        [
            {
                "split": "test",
                "prompt_type": "fake",
                "domain": "software_api",
                "category": "fake_api",
                "chosen_reward": 0.0,
                "rejected_reward": 0.1,
            },
            {
                "split": "test",
                "prompt_type": "real",
                "domain": "software_api",
                "category": "real_api",
                "chosen_reward": 0.2,
                "rejected_reward": -0.2,
            },
        ]
    )
    stats = ScoreStats(mean=0.0, std=1.0, p10=-1.0, p90=1.0)

    audit = audit_reward_transforms(rows, stats, [{"type": "raw"}, {"type": "band_target"}])
    by_name = {row["transform"]: row for row in audit if row["audit_group"] == "all"}

    assert by_name["raw"]["degenerate_pair_fraction"] == 0.0
    assert by_name["band_target"]["degenerate_pair_fraction"] == 1.0
    assert any(
        row["audit_group"] == "domain"
        and row["domain"] == "software_api"
        and row["transform"] == "raw"
        for row in audit
    )
    assert any(row["audit_group"] == "split+prompt_type+domain+category" for row in audit)


def test_policy_reward_transform_audit_groups_existing_rewards():
    rows = pd.DataFrame(
        [
            {"run": "base_hard", "prompt_type": "fake", "domain": "api", "label": "confident_confabulation", "reward": -2.0},
            {"run": "rl_hard", "prompt_type": "real", "domain": "api", "label": "real_substantive_answer", "reward": 1.0},
        ]
    )
    audit = audit_policy_reward_transforms(rows, transform_configs=[{"type": "raw"}])

    assert audit[0]["audit_kind"] == "policy_outputs"
    assert audit[0]["run"] == "all"
    assert any(row["run"] == "base_hard" and row["domain"] == "api" for row in audit)


def test_reward_transform_audit_can_aggregate_rm_design_score_glob(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for run_name, chosen_shift in [("last_final_token_matrix_sum", 0.0), ("last_final_token_mlp", 0.2)]:
        run_dir = tmp_path / "outputs" / "rm_design_ablations" / run_name
        eval_dir = run_dir / "eval"
        eval_dir.mkdir(parents=True)
        pd.DataFrame(
            [
                {
                    "split": "test",
                    "prompt_type": "fake",
                    "domain": "software_api",
                    "category": "fake_api",
                    "chosen_reward": chosen_shift,
                    "rejected_reward": -0.5,
                }
            ]
        ).to_csv(eval_dir / "reward_scores.csv", index=False)
        (run_dir / "reward_model").mkdir()
        (run_dir / "reward_model" / "chosen_score_stats.json").write_text(
            '{"mean": 0.0, "std": 1.0, "p10": -1.0, "p90": 1.0}',
            encoding="utf-8",
        )

    result = write_reward_transform_audit(
        {
            "scores_glob": "outputs/rm_design_ablations/*/eval/reward_scores.csv",
            "output_dir": "outputs/report",
            "transforms": [{"type": "raw"}],
        }
    )
    audit = pd.read_csv(result["csv_path"])

    assert set(audit["audit_kind"]) == {"chosen_rejected_pairs"}
    assert {"last_final_token_matrix_sum", "last_final_token_mlp"} <= set(audit["run"])
    assert "scores_path" in audit.columns
