from rl_epistemics.rl.reward_transforms import (
    ScoreStats,
    band_target_transform,
    build_reward_transform,
    gaussian_target_transform,
    resolve_reward_transform_config,
)


def test_gaussian_target_prefers_distribution_center():
    stats = ScoreStats(mean=2.0, std=0.5, p10=1.0, p90=3.0)
    assert gaussian_target_transform(2.0, stats) > gaussian_target_transform(4.0, stats)


def test_band_target_penalizes_outside_band():
    stats = ScoreStats(mean=0.0, std=1.0, p10=-1.0, p90=1.0)
    assert band_target_transform(0.0, stats) == 0.0
    assert band_target_transform(2.0, stats) < 0.0


def test_per_class_gaussian_target_loads_prompt_type_stats(tmp_path):
    (tmp_path / "chosen_score_stats_by_prompt_type.json").write_text(
        '{"fake": {"mean": 10.0, "std": 2.0, "p10": 8.0, "p90": 12.0}, '
        '"real": {"mean": 1.0, "std": 1.0, "p10": 0.0, "p90": 2.0}}',
        encoding="utf-8",
    )

    config = resolve_reward_transform_config(
        {"type": "per_class_gaussian_target", "std_floor": 0.1},
        tmp_path,
        std_floor=0.1,
    )
    transform = build_reward_transform(config, ScoreStats(mean=0.0, std=1.0, p10=-1.0, p90=1.0))

    assert transform(10.0, "fake") > transform(1.0, "fake")
    assert transform(1.0, "real") > transform(10.0, "real")
