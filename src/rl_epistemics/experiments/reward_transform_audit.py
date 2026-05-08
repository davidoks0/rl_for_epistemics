from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from rl_epistemics.rl.reward_transforms import (
    ScoreStats,
    build_reward_transform,
    load_score_stats,
    resolve_reward_transform_config,
)


DEFAULT_TRANSFORMS = [
    {"type": "raw"},
    {"type": "clipped"},
    {"type": "gaussian_target", "std_floor": 0.1},
    {"type": "band_target", "std_floor": 0.1},
]


def _summarize_values(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    series = pd.Series(values, dtype="float64")
    return {
        "mean": float(series.mean()),
        "std": float(series.std(ddof=0)),
        "min": float(series.min()),
        "max": float(series.max()),
    }


def _stats_from_scores(values: list[float], std_floor: float = 0.1) -> ScoreStats:
    if not values:
        return ScoreStats(mean=0.0, std=1.0, p10=-1.0, p90=1.0)
    series = pd.Series(values, dtype="float64")
    return ScoreStats(
        mean=float(series.mean()),
        std=float(max(series.std(ddof=0), std_floor)),
        p10=float(series.quantile(0.1)),
        p90=float(series.quantile(0.9)),
    )


def _pair_transform_summary(
    reward_rows: pd.DataFrame,
    stats: ScoreStats,
    config: dict[str, Any],
    degenerate_eps: float,
    group_values: dict[str, Any] | None = None,
) -> dict[str, Any]:
    group_values = group_values or {}
    transform = build_reward_transform(config, stats)
    chosen_raw = [float(value) for value in reward_rows["chosen_reward"].tolist()]
    rejected_raw = [float(value) for value in reward_rows["rejected_reward"].tolist()]
    prompt_types = (
        reward_rows["prompt_type"].fillna("").astype(str).tolist()
        if "prompt_type" in reward_rows
        else [None] * len(reward_rows)
    )
    chosen = [transform(score, prompt_type) for score, prompt_type in zip(chosen_raw, prompt_types)]
    rejected = [transform(score, prompt_type) for score, prompt_type in zip(rejected_raw, prompt_types)]
    margins = [c - r for c, r in zip(chosen, rejected)]
    zero_like = [abs(margin) <= degenerate_eps for margin in margins]
    chosen_inside_band = [stats.p10 <= score <= stats.p90 for score in chosen_raw]
    rejected_inside_band = [stats.p10 <= score <= stats.p90 for score in rejected_raw]
    return {
        "audit_group": str(group_values.get("audit_group", "all")),
        "split": str(group_values.get("split", "all")),
        "prompt_type": str(group_values.get("prompt_type", "all")),
        "domain": str(group_values.get("domain", "all")),
        "category": str(group_values.get("category", "all")),
        "transform": str(config.get("type", "gaussian_target")),
        "n": int(len(reward_rows)),
        "transformed_prefers_chosen": float(sum(margin > 0 for margin in margins) / len(margins))
        if margins
        else 0.0,
        "degenerate_pair_fraction": float(sum(zero_like) / len(zero_like)) if zero_like else 0.0,
        "raw_chosen_inside_target_band": float(sum(chosen_inside_band) / len(chosen_inside_band))
        if chosen_inside_band
        else 0.0,
        "raw_rejected_inside_target_band": float(sum(rejected_inside_band) / len(rejected_inside_band))
        if rejected_inside_band
        else 0.0,
        "chosen": _summarize_values(chosen),
        "rejected": _summarize_values(rejected),
        "margin": _summarize_values(margins),
    }


def audit_reward_transforms(
    reward_rows: pd.DataFrame,
    stats: ScoreStats,
    transform_configs: list[dict[str, Any]] | None = None,
    degenerate_eps: float = 1.0e-8,
) -> list[dict[str, Any]]:
    transform_configs = transform_configs or DEFAULT_TRANSFORMS
    required = {"chosen_reward", "rejected_reward"}
    missing = required - set(reward_rows.columns)
    if missing:
        raise ValueError(f"Reward score CSV is missing columns: {sorted(missing)}")

    rows: list[dict[str, Any]] = []
    group_cols = [col for col in ["split", "prompt_type", "domain", "category"] if col in reward_rows.columns]
    for config in transform_configs:
        rows.append(_pair_transform_summary(reward_rows, stats, config, degenerate_eps))
        for group_col in group_cols:
            for group_key, group in reward_rows.groupby(group_col, dropna=False):
                rows.append(
                    _pair_transform_summary(
                        group,
                        stats,
                        config,
                        degenerate_eps,
                        {"audit_group": group_col, group_col: group_key},
                    )
                )
        if len(group_cols) >= 2:
            for group_key, group in reward_rows.groupby(group_cols, dropna=False):
                if not isinstance(group_key, tuple):
                    group_key = (group_key,)
                group_map = dict(zip(group_cols, group_key))
                group_map["audit_group"] = "+".join(group_cols)
                rows.append(_pair_transform_summary(group, stats, config, degenerate_eps, group_map))
    return rows


def audit_policy_reward_transforms(
    policy_rows: pd.DataFrame,
    stats: ScoreStats | None = None,
    transform_configs: list[dict[str, Any]] | None = None,
    score_column: str = "reward",
    degenerate_eps: float = 1.0e-8,
    std_floor: float = 0.1,
) -> list[dict[str, Any]]:
    transform_configs = transform_configs or DEFAULT_TRANSFORMS
    if score_column not in policy_rows.columns:
        raise ValueError(f"Policy output CSV is missing score column: {score_column}")
    rows: list[dict[str, Any]] = []
    scored = policy_rows[policy_rows[score_column].notna()].copy()
    if scored.empty:
        return rows
    raw_scores = [float(value) for value in scored[score_column].tolist()]
    active_stats = stats or _stats_from_scores(raw_scores, std_floor=std_floor)
    prompt_types = (
        scored["prompt_type"].fillna("").astype(str).tolist()
        if "prompt_type" in scored
        else [None] * len(scored)
    )
    group_cols = [col for col in ["run", "prompt_type", "domain", "label"] if col in scored.columns]
    for config in transform_configs:
        transform = build_reward_transform(config, active_stats)
        transformed = [
            transform(score, prompt_type) for score, prompt_type in zip(raw_scores, prompt_types)
        ]
        inside_band = [active_stats.p10 <= score <= active_stats.p90 for score in raw_scores]
        rows.append(
            {
                "audit_kind": "policy_outputs",
                "run": "all",
                "prompt_type": "all",
                "domain": "all",
                "label": "all",
                "transform": str(config.get("type", "gaussian_target")),
                "n": int(len(scored)),
                "raw_inside_target_band": float(sum(inside_band) / len(inside_band)),
                "degenerate_score_fraction": float(
                    sum(abs(value) <= degenerate_eps for value in transformed) / len(transformed)
                ),
                "raw": _summarize_values(raw_scores),
                "transformed": _summarize_values(transformed),
                "stats_mean": active_stats.mean,
                "stats_std": active_stats.std,
                "stats_p10": active_stats.p10,
                "stats_p90": active_stats.p90,
            }
        )
        if group_cols:
            transformed_series = pd.Series(transformed, index=scored.index)
            for group_key, group in scored.groupby(group_cols, dropna=False):
                group_values = transformed_series.loc[group.index].astype(float).tolist()
                group_raw = group[score_column].astype(float).tolist()
                if not isinstance(group_key, tuple):
                    group_key = (group_key,)
                group_map = dict(zip(group_cols, group_key))
                rows.append(
                    {
                        "audit_kind": "policy_outputs",
                        "run": str(group_map.get("run", "all")),
                        "prompt_type": str(group_map.get("prompt_type", "all")),
                        "domain": str(group_map.get("domain", "all")),
                        "label": str(group_map.get("label", "all")),
                        "transform": str(config.get("type", "gaussian_target")),
                        "n": int(len(group)),
                        "raw_inside_target_band": float(
                            sum(active_stats.p10 <= score <= active_stats.p90 for score in group_raw)
                            / len(group_raw)
                        ),
                        "degenerate_score_fraction": float(
                            sum(abs(value) <= degenerate_eps for value in group_values)
                            / len(group_values)
                        ),
                        "raw": _summarize_values(group_raw),
                        "transformed": _summarize_values(group_values),
                        "stats_mean": active_stats.mean,
                        "stats_std": active_stats.std,
                        "stats_p10": active_stats.p10,
                        "stats_p90": active_stats.p90,
                    }
                )
    return rows


def _flatten_transform_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flat_rows = []
    for row in rows:
        flattened = {key: value for key, value in row.items() if not isinstance(value, dict)}
        for prefix in ("chosen", "rejected", "margin", "raw", "transformed"):
            if isinstance(row.get(prefix), dict):
                for key, value in row[prefix].items():
                    flattened[f"{prefix}_{key}"] = value
        flat_rows.append(flattened)
    return flat_rows


def _infer_stats_path(scores_path: Path, configured_stats_path: str | Path | None = None) -> Path:
    if configured_stats_path:
        return Path(configured_stats_path)
    candidates = [
        scores_path.parent.parent / "chosen_score_stats.json",
        scores_path.parent.parent / "reward_model" / "chosen_score_stats.json",
        scores_path.parent / "chosen_score_stats.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _run_name_from_scores_path(scores_path: Path) -> str:
    parts = scores_path.parts
    if "rm_design_ablations" in parts:
        idx = parts.index("rm_design_ablations")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    if len(scores_path.parents) >= 2 and scores_path.parent.name == "eval":
        return scores_path.parent.parent.name
    return scores_path.stem


def _audit_scores_path(
    scores_path: Path,
    config: dict[str, Any],
    *,
    run_name: str | None = None,
    stats_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    inferred_stats_path = _infer_stats_path(scores_path, stats_path)
    stats = load_score_stats(
        inferred_stats_path,
        std_floor=float(config.get("std_floor", 0.1)),
    )
    reward_rows = pd.read_csv(scores_path)
    transform_configs = [
        resolve_reward_transform_config(
            transform_config,
            inferred_stats_path.parent,
            std_floor=float(transform_config.get("std_floor", config.get("std_floor", 0.1))),
        )
        for transform_config in (config.get("transforms") or DEFAULT_TRANSFORMS)
    ]
    pair_rows = audit_reward_transforms(
        reward_rows,
        stats,
        transform_configs=transform_configs,
        degenerate_eps=float(config.get("degenerate_eps", 1.0e-8)),
    )
    active_run = run_name or _run_name_from_scores_path(scores_path)
    return [
        {
            "audit_kind": "chosen_rejected_pairs",
            "run": active_run,
            "scores_path": str(scores_path),
            **row,
        }
        for row in pair_rows
    ]


def write_reward_transform_audit(config: dict[str, Any]) -> dict[str, Any]:
    scores_path = Path(config.get("scores_path", "outputs/reward_model/eval/reward_scores.csv"))
    stats_path = config.get("stats_path", "outputs/reward_model/chosen_score_stats.json")
    output_dir = Path(config.get("output_dir", "outputs/report"))
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]]
    scores_glob = config.get("scores_glob")
    score_paths = sorted(Path().glob(str(scores_glob))) if scores_glob else []
    if score_paths:
        rows = []
        for path in score_paths:
            rows.extend(_audit_scores_path(path, config))
    elif scores_path.exists():
        rows = _audit_scores_path(
            scores_path,
            config,
            run_name=str(config.get("run_name", "reward_model")),
            stats_path=stats_path,
        )
    else:
        policy_glob = str(config.get("policy_outputs_glob", "outputs/eval_tinker*_policy_outputs.csv"))
        policy_paths = sorted(Path().glob(policy_glob))
        if not policy_paths:
            raise FileNotFoundError(
                f"No reward score CSV at {scores_path} and no policy outputs matched {policy_glob}."
            )
        frames = []
        for path in policy_paths:
            frame = pd.read_csv(path)
            if "run" not in frame.columns:
                run = path.name
                if run.startswith("eval_tinker_hard_base"):
                    run = "base_hard"
                elif run.startswith("eval_tinker_hard_main"):
                    run = "rl_hard"
                frame.insert(0, "run", run)
            frames.append(frame)
        policy_rows = pd.concat(frames, ignore_index=True)
        rows = audit_policy_reward_transforms(
            policy_rows,
            stats=None,
            transform_configs=config.get("transforms") or DEFAULT_TRANSFORMS,
            score_column=str(config.get("score_column", "reward")),
            degenerate_eps=float(config.get("degenerate_eps", 1.0e-8)),
            std_floor=float(config.get("std_floor", 0.1)),
        )
    flat_rows = _flatten_transform_rows(rows)
    csv_path = output_dir / "reward_transform_audit.csv"
    json_path = output_dir / "reward_transform_audit.json"
    pd.DataFrame(flat_rows).to_csv(csv_path, index=False)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    return {"csv_path": str(csv_path), "json_path": str(json_path), "rows": len(rows)}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Audit raw vs bounded reward transforms on RM scores.")
    parser.add_argument("--config", default="configs/reward_transform_audit.yaml")
    parser.add_argument("--scores-path", default=None)
    parser.add_argument("--scores-glob", default=None)
    parser.add_argument("--stats-path", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)

    config: dict[str, Any]
    path = Path(args.config)
    if path.exists():
        import yaml

        with path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    else:
        config = {}
    if args.scores_path:
        config["scores_path"] = args.scores_path
    if args.scores_glob:
        config["scores_glob"] = args.scores_glob
    if args.stats_path:
        config["stats_path"] = args.stats_path
    if args.output_dir:
        config["output_dir"] = args.output_dir
    print(json.dumps(write_reward_transform_audit(config), indent=2))


if __name__ == "__main__":
    main()
