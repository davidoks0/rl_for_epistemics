from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ScoreStats:
    mean: float
    std: float
    p10: float
    p90: float

    @classmethod
    def from_dict(cls, data: dict[str, Any], std_floor: float = 1.0e-6) -> "ScoreStats":
        std = max(float(data.get("std", 1.0)), std_floor)
        return cls(
            mean=float(data.get("mean", 0.0)),
            std=std,
            p10=float(data.get("p10", -1.0)),
            p90=float(data.get("p90", 1.0)),
        )


def load_score_stats(path_or_dir: str | Path, std_floor: float = 1.0e-6) -> ScoreStats:
    path = Path(path_or_dir)
    if path.is_dir():
        path = path / "chosen_score_stats.json"
    with path.open("r", encoding="utf-8") as f:
        return ScoreStats.from_dict(json.load(f), std_floor=std_floor)


def load_per_class_score_stats(path_or_dir: str | Path, std_floor: float = 1.0e-6) -> dict[str, ScoreStats]:
    path = Path(path_or_dir)
    if path.is_dir():
        path = path / "chosen_score_stats_by_prompt_type.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        str(prompt_type): ScoreStats.from_dict(stats, std_floor=std_floor)
        for prompt_type, stats in data.items()
        if isinstance(stats, dict)
    }


def resolve_reward_transform_config(
    config: dict[str, Any],
    reward_model_dir: str | Path,
    *,
    std_floor: float = 1.0e-6,
) -> dict[str, Any]:
    resolved = dict(config or {})
    if resolved.get("type") == "per_class_gaussian_target" and not resolved.get("per_class"):
        resolved["per_class"] = {
            key: stats.__dict__
            for key, stats in load_per_class_score_stats(
                reward_model_dir,
                std_floor=std_floor,
            ).items()
        }
    return resolved


def raw_transform(score: float, stats: ScoreStats) -> float:
    return float(score)


def clipped_transform(score: float, stats: ScoreStats) -> float:
    return float(max(stats.p10, min(stats.p90, score)))


def gaussian_target_transform(score: float, stats: ScoreStats) -> float:
    z = (score - stats.mean) / stats.std
    return float(-0.5 * z * z - math.log(stats.std))


def band_target_transform(score: float, stats: ScoreStats) -> float:
    if stats.p10 <= score <= stats.p90:
        return 0.0
    distance = stats.p10 - score if score < stats.p10 else score - stats.p90
    return float(-abs(distance) / stats.std)


def build_reward_transform(config: dict[str, Any], stats: ScoreStats):
    transform_type = config.get("type", "gaussian_target")
    if transform_type == "raw":
        return lambda score, prompt_type=None: raw_transform(float(score), stats)
    if transform_type == "clipped":
        return lambda score, prompt_type=None: clipped_transform(float(score), stats)
    if transform_type == "gaussian_target":
        return lambda score, prompt_type=None: gaussian_target_transform(float(score), stats)
    if transform_type == "band_target":
        return lambda score, prompt_type=None: band_target_transform(float(score), stats)
    if transform_type == "per_class_gaussian_target":
        per_class = config.get("per_class") or {}
        class_stats = {
            key: ScoreStats.from_dict(value, std_floor=float(config.get("std_floor", 1.0e-6)))
            for key, value in per_class.items()
        }

        def transform(score: float, prompt_type: str | None = None) -> float:
            active = class_stats.get(prompt_type or "", stats)
            return gaussian_target_transform(float(score), active)

        return transform
    raise ValueError(f"Unknown reward transform: {transform_type}")
