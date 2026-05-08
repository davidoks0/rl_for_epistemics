from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


SCORE_COLUMNS = ("chosen_reward", "rejected_reward")


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _target_for_group(config: dict[str, Any], prompt_type: str) -> tuple[float, float]:
    targets = config.get("prompt_type_targets") or {}
    group_target = targets.get(prompt_type) or {}
    chosen = group_target.get("chosen", group_target.get("target_chosen", config.get("target_chosen", 2.0)))
    rejected = group_target.get(
        "rejected",
        group_target.get("target_rejected", config.get("target_rejected", 0.5)),
    )
    return float(chosen), float(rejected)


def _fit_affine(rows: list[dict[str, Any]], target_chosen: float, target_rejected: float) -> dict[str, Any]:
    chosen = [_float(row.get("chosen_reward")) for row in rows]
    rejected = [_float(row.get("rejected_reward")) for row in rows]
    chosen_mean = sum(chosen) / max(len(chosen), 1)
    rejected_mean = sum(rejected) / max(len(rejected), 1)
    raw_margin = chosen_mean - rejected_mean
    target_margin = target_chosen - target_rejected
    scale = 1.0 if abs(raw_margin) <= 1.0e-12 else target_margin / raw_margin
    bias = target_chosen - scale * chosen_mean
    return {
        "scale": float(scale),
        "bias": float(bias),
        "raw_chosen_mean": float(chosen_mean),
        "raw_rejected_mean": float(rejected_mean),
        "raw_margin_mean": float(raw_margin),
        "target_chosen": float(target_chosen),
        "target_rejected": float(target_rejected),
        "target_margin": float(target_margin),
    }


def fit_score_calibration(rows: Iterable[dict[str, Any]], config: dict[str, Any] | None = None) -> dict[str, Any]:
    config = config or {}
    if not bool(config.get("enabled", True)):
        return {"enabled": False, "method": str(config.get("method", "disabled"))}

    row_list = [dict(row) for row in rows]
    method = str(config.get("method", "prompt_type_margin_affine"))
    target_chosen = float(config.get("target_chosen", 2.0))
    target_rejected = float(config.get("target_rejected", 0.5))
    calibration: dict[str, Any] = {
        "enabled": True,
        "method": method,
        "default": _fit_affine(row_list, target_chosen, target_rejected),
        "prompt_type": {},
    }

    if method in {"prompt_type_margin_affine", "per_class_margin_affine"}:
        prompt_types = sorted({str(row.get("prompt_type", "")) for row in row_list if row.get("prompt_type")})
        for prompt_type in prompt_types:
            group_rows = [row for row in row_list if str(row.get("prompt_type", "")) == prompt_type]
            group_target_chosen, group_target_rejected = _target_for_group(config, prompt_type)
            calibration["prompt_type"][prompt_type] = _fit_affine(
                group_rows,
                group_target_chosen,
                group_target_rejected,
            )
    elif method not in {"margin_affine", "global_margin_affine"}:
        raise ValueError(f"Unknown score calibration method: {method}")
    return calibration


def apply_score_calibration(score: float, calibration: dict[str, Any] | None, prompt_type: str | None = None) -> float:
    if not calibration or not calibration.get("enabled", False):
        return float(score)
    params = None
    if prompt_type:
        params = (calibration.get("prompt_type") or {}).get(str(prompt_type))
    params = params or calibration.get("default") or {}
    return float(_float(params.get("scale"), 1.0) * float(score) + _float(params.get("bias"), 0.0))


def apply_score_calibration_to_rows(
    rows: Iterable[dict[str, Any]],
    calibration: dict[str, Any] | None,
    *,
    score_columns: tuple[str, ...] = SCORE_COLUMNS,
    keep_raw: bool = True,
) -> list[dict[str, Any]]:
    calibrated = []
    for row in rows:
        new_row = dict(row)
        prompt_type = str(new_row.get("prompt_type", "")) or None
        for column in score_columns:
            if column not in new_row:
                continue
            raw_value = _float(new_row[column])
            if keep_raw and f"raw_{column}" not in new_row:
                new_row[f"raw_{column}"] = raw_value
            new_row[column] = apply_score_calibration(raw_value, calibration, prompt_type)
        if "chosen_reward" in new_row and "rejected_reward" in new_row:
            new_row["margin"] = float(new_row["chosen_reward"] - new_row["rejected_reward"])
            new_row["correct"] = bool(new_row["chosen_reward"] > new_row["rejected_reward"])
        calibrated.append(new_row)
    return calibrated


def chosen_score_stats(rows: Iterable[dict[str, Any]]) -> dict[str, float]:
    values = pd.Series([_float(row.get("chosen_reward")) for row in rows], dtype="float64")
    if values.empty:
        return {"mean": 0.0, "std": 1.0, "p10": -1.0, "p90": 1.0}
    return {
        "mean": float(values.mean()),
        "std": float(max(values.std(ddof=0), 1.0e-6)),
        "p10": float(values.quantile(0.1)),
        "p90": float(values.quantile(0.9)),
    }


def chosen_score_stats_by_prompt_type(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, float]]:
    frame = pd.DataFrame(list(rows))
    if frame.empty or "prompt_type" not in frame.columns:
        return {}
    stats: dict[str, dict[str, float]] = {}
    for prompt_type, group in frame.groupby("prompt_type", dropna=False):
        stats[str(prompt_type)] = chosen_score_stats(group.to_dict(orient="records"))
    return stats


def summarize_pair_score_rows(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    frame = pd.DataFrame(list(rows))
    if frame.empty:
        return {}
    summary: dict[str, Any] = {"n": int(len(frame))}
    for key in ("all", "real", "fake"):
        group = frame if key == "all" else frame[frame["prompt_type"].astype(str) == key]
        if group.empty:
            continue
        summary[key] = {
            "n": int(len(group)),
            "chosen_mean": float(group["chosen_reward"].astype(float).mean()),
            "rejected_mean": float(group["rejected_reward"].astype(float).mean()),
            "margin_mean": float(group["margin"].astype(float).mean()),
            "accuracy": float(group["correct"].astype(bool).mean()),
        }
    return summary


def summarize_pair_score_rows_by_split(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    row_list = list(rows)
    frame = pd.DataFrame(row_list)
    if frame.empty or "split" not in frame.columns:
        return {"all": summarize_pair_score_rows(row_list)}
    return {
        str(split): summarize_pair_score_rows(group.to_dict(orient="records"))
        for split, group in frame.groupby("split", dropna=False)
    }


def load_score_calibration(path_or_dir: str | Path) -> dict[str, Any]:
    path = Path(path_or_dir)
    if path.is_dir():
        path = path / "score_calibration.json"
    if not path.exists():
        return {"enabled": False, "method": "none"}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _copy_reward_model_files(source: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for name in (
        "model_meta.json",
        "reward_head_config.json",
        "reward_head.pt",
        "train_config.yaml",
        "score_summary.json",
        "chosen_score_stats.json",
    ):
        src = source / name
        if src.exists():
            shutil.copy2(src, target / name)


def write_calibrated_reward_model(config: dict[str, Any]) -> dict[str, Any]:
    source_dir = Path(config.get("source_reward_model_dir", config.get("reward_model_dir", "outputs/reward_model")))
    target_dir = Path(config.get("output_reward_model_dir", source_dir))
    scores_path = Path(config.get("scores_path", source_dir / "eval" / "reward_scores.csv"))
    fit_split = config.get("fit_split", "train")

    if not scores_path.exists():
        raise FileNotFoundError(f"Reward score CSV not found: {scores_path}")
    if source_dir.resolve() != target_dir.resolve():
        _copy_reward_model_files(source_dir, target_dir)
    else:
        target_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(scores_path)
    fit_frame = frame
    if fit_split is not None and "split" in frame.columns:
        fit_frame = frame[frame["split"].astype(str) == str(fit_split)]
        if fit_frame.empty:
            raise ValueError(f"No reward score rows matched fit_split={fit_split!r}.")

    calibration = fit_score_calibration(
        fit_frame.to_dict(orient="records"),
        config.get("score_calibration") or {"enabled": True},
    )
    raw_summary_path = source_dir / "score_summary.json"
    if raw_summary_path.exists():
        shutil.copy2(raw_summary_path, target_dir / "raw_score_summary.json")
    calibrated_rows = apply_score_calibration_to_rows(
        frame.to_dict(orient="records"),
        calibration,
        keep_raw=True,
    )
    calibrated_fit_rows = [
        row for row in calibrated_rows if fit_split is None or str(row.get("split", "")) == str(fit_split)
    ]

    with (target_dir / "score_calibration.json").open("w", encoding="utf-8") as f:
        json.dump(calibration, f, indent=2)
    with (target_dir / "chosen_score_stats.json").open("w", encoding="utf-8") as f:
        json.dump(chosen_score_stats(calibrated_fit_rows), f, indent=2)
    with (target_dir / "chosen_score_stats_by_prompt_type.json").open("w", encoding="utf-8") as f:
        json.dump(chosen_score_stats_by_prompt_type(calibrated_fit_rows), f, indent=2)
    with (target_dir / "score_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summarize_pair_score_rows_by_split(calibrated_rows), f, indent=2)

    eval_dir = target_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    calibrated_scores_path = eval_dir / "reward_scores.csv"
    pd.DataFrame(calibrated_rows).to_csv(calibrated_scores_path, index=False)
    calibrated_summary_path = eval_dir / "reward_summary.json"
    with calibrated_summary_path.open("w", encoding="utf-8") as f:
        json.dump(summarize_pair_score_rows(calibrated_rows), f, indent=2)
    return {
        "source_reward_model_dir": str(source_dir),
        "output_reward_model_dir": str(target_dir),
        "scores_path": str(scores_path),
        "calibrated_scores_path": str(calibrated_scores_path),
        "calibrated_summary_path": str(calibrated_summary_path),
        "fit_rows": int(len(fit_frame)),
        "rows": int(len(calibrated_rows)),
        "calibration_path": str(target_dir / "score_calibration.json"),
        "stats_path": str(target_dir / "chosen_score_stats.json"),
        "per_class_stats_path": str(target_dir / "chosen_score_stats_by_prompt_type.json"),
    }
