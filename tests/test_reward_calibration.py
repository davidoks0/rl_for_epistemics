import json

import pytest

from rl_epistemics.models.reward_calibration import (
    apply_score_calibration,
    apply_score_calibration_to_rows,
    chosen_score_stats_by_prompt_type,
    fit_score_calibration,
)


def test_prompt_type_margin_affine_calibration_maps_group_means():
    rows = [
        {"prompt_type": "fake", "chosen_reward": 70.0, "rejected_reward": 20.0},
        {"prompt_type": "fake", "chosen_reward": 72.0, "rejected_reward": 18.0},
        {"prompt_type": "real", "chosen_reward": 40.0, "rejected_reward": 10.0},
        {"prompt_type": "real", "chosen_reward": 44.0, "rejected_reward": 12.0},
    ]

    calibration = fit_score_calibration(
        rows,
        {
            "enabled": True,
            "method": "prompt_type_margin_affine",
            "prompt_type_targets": {
                "fake": {"chosen": 1.88, "rejected": 0.50},
                "real": {"chosen": 2.08, "rejected": 0.36},
            },
        },
    )
    calibrated = apply_score_calibration_to_rows(rows, calibration)

    fake = [row for row in calibrated if row["prompt_type"] == "fake"]
    real = [row for row in calibrated if row["prompt_type"] == "real"]
    assert sum(row["chosen_reward"] for row in fake) / len(fake) == pytest.approx(1.88)
    assert sum(row["rejected_reward"] for row in fake) / len(fake) == pytest.approx(0.5)
    assert sum(row["chosen_reward"] for row in real) / len(real) == pytest.approx(2.08)
    assert sum(row["rejected_reward"] for row in real) / len(real) == pytest.approx(0.36)
    assert all("raw_chosen_reward" in row for row in calibrated)


def test_score_calibration_uses_default_when_prompt_type_missing():
    calibration = fit_score_calibration(
        [{"chosen_reward": 10.0, "rejected_reward": 0.0}],
        {"enabled": True, "method": "global_margin_affine", "target_chosen": 2.0, "target_rejected": 0.5},
    )

    assert apply_score_calibration(10.0, calibration) == 2.0
    assert apply_score_calibration(0.0, calibration, "fake") == 0.5


def test_chosen_score_stats_by_prompt_type_is_json_serializable():
    stats = chosen_score_stats_by_prompt_type(
        [
            {"prompt_type": "fake", "chosen_reward": 1.0},
            {"prompt_type": "fake", "chosen_reward": 3.0},
            {"prompt_type": "real", "chosen_reward": 2.0},
        ]
    )

    assert set(stats) == {"fake", "real"}
    assert stats["fake"]["mean"] == 2.0
    json.dumps(stats)
