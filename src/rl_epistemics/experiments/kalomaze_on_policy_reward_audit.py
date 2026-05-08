from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from rl_epistemics.data.schema import KnowledgePair, read_jsonl
from rl_epistemics.models.reward_scoring import RewardScorer
from rl_epistemics.utils.config import load_config


RUN_DIRS = {
    "gaussian": "tinker_rl_gaussian",
    "raw": "tinker_rl_raw",
    "per_class_gaussian": "tinker_rl_per_class_gaussian",
    "per_class_gaussian_calibrated": "tinker_rl_per_class_gaussian_calibrated",
    "raw_calibrated": "tinker_rl_raw_calibrated",
}

FAKE_COMPARISONS = [
    ("dataset_chosen", "dataset_rejected"),
    ("clean_uncertainty", "dataset_rejected"),
    ("hedged_uncertainty", "dataset_rejected"),
    ("policy_completion", "dataset_chosen"),
    ("policy_completion", "clean_uncertainty"),
    ("policy_completion", "hedged_uncertainty"),
    ("policy_completion", "dataset_rejected"),
]

REAL_COMPARISONS = [
    ("dataset_chosen", "dataset_rejected"),
    ("policy_completion", "dataset_chosen"),
    ("policy_completion", "dataset_rejected"),
]


def _read_jsonl_dicts(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _parse_steps(value: Any) -> set[int] | None:
    if value is None or value == "latest":
        return None
    if isinstance(value, int):
        return {value}
    if isinstance(value, str):
        return {int(part.strip()) for part in value.split(",") if part.strip()}
    return {int(step) for step in value}


def _run_dir(output_root: Path, run: str) -> Path:
    return output_root / RUN_DIRS.get(run, run)


def load_policy_heldout_rows(
    output_root: str | Path,
    *,
    runs: Iterable[str] = ("gaussian", "raw"),
    steps: set[int] | None = None,
) -> list[dict[str, Any]]:
    root = Path(output_root)
    rows: list[dict[str, Any]] = []
    for run in runs:
        run_rows = _read_jsonl_dicts(_run_dir(root, str(run)) / "heldout_samples.jsonl")
        if not run_rows:
            continue
        active_steps = steps
        if active_steps is None:
            latest = max(int(row.get("step", -1)) for row in run_rows)
            active_steps = {latest}
        for row in run_rows:
            if int(row.get("step", -1)) not in active_steps:
                continue
            copied = dict(row)
            copied["run"] = str(run)
            copied["original_id"] = str(row.get("original_id") or row.get("id"))
            rows.append(copied)
    return rows


def clean_uncertainty_completion(pair: KnowledgePair) -> str:
    return (
        f"I can't verify that {pair.entity_name} or the cited work is real, so I shouldn't "
        "summarize it as fact. If you have a source, I can help assess it."
    )


def hedged_uncertainty_completion(pair: KnowledgePair) -> str:
    return (
        f"I don't recognize {pair.entity_name} or the cited work. It may be fabricated, "
        "obscure, or misremembered, so I would avoid treating it as an established reference "
        "without a source."
    )


def _reference_variants(pair: KnowledgePair) -> list[tuple[str, str]]:
    variants = [
        ("dataset_chosen", pair.chosen),
        ("dataset_rejected", pair.rejected),
    ]
    if pair.prompt_type == "fake":
        variants.extend(
            [
                ("clean_uncertainty", clean_uncertainty_completion(pair)),
                ("hedged_uncertainty", hedged_uncertainty_completion(pair)),
            ]
        )
    return variants


def build_reward_grid_rows(
    pairs: list[KnowledgePair],
    policy_rows: list[dict[str, Any]],
    *,
    split: str = "test",
    prompt_types: set[str] | None = None,
) -> list[dict[str, Any]]:
    pair_by_id = {
        pair.id: pair
        for pair in pairs
        if pair.split == split and (prompt_types is None or pair.prompt_type in prompt_types)
    }
    rows: list[dict[str, Any]] = []
    for policy_row in policy_rows:
        pair_id = str(policy_row.get("original_id") or policy_row.get("id"))
        pair = pair_by_id.get(pair_id)
        if pair is None:
            continue
        run = str(policy_row.get("run", "policy"))
        step = int(policy_row.get("step", -1))
        context = {
            "run": run,
            "step": step,
            "id": pair.id,
            "prompt_type": pair.prompt_type,
            "domain": pair.domain,
            "entity_name": pair.entity_name,
            "prompt": pair.prompt,
        }
        for variant, completion in _reference_variants(pair):
            rows.append(
                {
                    **context,
                    "variant": variant,
                    "source": "reference",
                    "completion": completion,
                    "policy_label": "",
                }
            )
        rows.append(
            {
                **context,
                "variant": "policy_completion",
                "source": "policy",
                "completion": str(policy_row.get("completion", "")),
                "policy_label": str(policy_row.get("adjudicated_label") or policy_row.get("label") or ""),
            }
        )
    return rows


def score_reward_grid_rows(
    rows: list[dict[str, Any]],
    scorer: RewardScorer,
    *,
    batch_size: int = 4,
) -> list[dict[str, Any]]:
    scored = [dict(row) for row in rows]
    rewards = scorer.score_many(
        [
            {
                "prompt": row["prompt"],
                "completion": row["completion"],
                "prompt_type": row.get("prompt_type"),
            }
            for row in scored
        ],
        batch_size=batch_size,
    )
    for row, reward in zip(scored, rewards):
        row["reward"] = float(reward)
    return scored


def summarize_reward_grid(scored_rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    if not scored_rows:
        return {"variant_summary": [], "comparisons": []}
    frame = pd.DataFrame(scored_rows)
    group_cols = ["reward_model", "run", "step", "prompt_type", "variant"]
    for col in group_cols:
        if col not in frame.columns:
            frame[col] = "default" if col == "reward_model" else ""
    variant_summary = (
        frame.groupby(group_cols, dropna=False)["reward"]
        .agg(["count", "mean", "std", "min", "max"])
        .reset_index()
        .rename(columns={"count": "n", "mean": "reward_mean", "std": "reward_std"})
        .fillna({"reward_std": 0.0})
        .to_dict(orient="records")
    )

    comparison_rows: list[dict[str, Any]] = []
    context_cols = ["reward_model", "run", "step", "id", "prompt_type"]
    for context, group in frame.groupby(context_cols, dropna=False):
        context_map = dict(zip(context_cols, context if isinstance(context, tuple) else (context,)))
        rewards = group.set_index("variant")["reward"].to_dict()
        comparisons = FAKE_COMPARISONS if context_map.get("prompt_type") == "fake" else REAL_COMPARISONS
        for left, right in comparisons:
            if left not in rewards or right not in rewards:
                continue
            comparison_rows.append(
                {
                    **context_map,
                    "comparison": f"{left}>{right}",
                    "left_variant": left,
                    "right_variant": right,
                    "left_reward": float(rewards[left]),
                    "right_reward": float(rewards[right]),
                    "margin": float(rewards[left] - rewards[right]),
                    "left_wins": bool(rewards[left] > rewards[right]),
                }
            )

    if not comparison_rows:
        return {"variant_summary": variant_summary, "comparisons": []}
    comp = pd.DataFrame(comparison_rows)
    comparison_summary = (
        comp.groupby(["reward_model", "run", "step", "prompt_type", "comparison"], dropna=False)
        .agg(n=("left_wins", "count"), left_win_rate=("left_wins", "mean"), margin_mean=("margin", "mean"))
        .reset_index()
        .to_dict(orient="records")
    )
    return {"variant_summary": variant_summary, "comparisons": comparison_summary}


def _reward_model_dirs(config: dict[str, Any]) -> list[Path]:
    if config.get("reward_model_dirs"):
        return [Path(path) for path in config["reward_model_dirs"]]
    if config.get("reward_model_glob"):
        return [Path(path) for path in sorted(glob.glob(str(config["reward_model_glob"])))]
    return [Path(config.get("reward_model_dir", "outputs/reward_model"))]


def write_on_policy_reward_grid(config: dict[str, Any]) -> dict[str, Any]:
    output_root = Path(config.get("output_root", "outputs/kalomaze_replica_confab"))
    output_dir = Path(config.get("output_dir", output_root / "report" / "reward_grid"))
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = Path(config.get("dataset_path", "data/kalomaze_replica_confab/pairs.jsonl"))
    pairs = read_jsonl(dataset_path)
    runs = config.get("runs") or ["gaussian", "raw"]
    steps = _parse_steps(config.get("steps", "latest"))
    policy_rows = load_policy_heldout_rows(output_root, runs=runs, steps=steps)
    prompt_types = set(config["prompt_types"]) if config.get("prompt_types") else None
    candidates = build_reward_grid_rows(
        pairs,
        policy_rows,
        split=str(config.get("split", "test")),
        prompt_types=prompt_types,
    )

    reward_model_dirs = _reward_model_dirs(config)
    name_override = config.get("reward_model_name") if len(reward_model_dirs) == 1 else None
    all_scored: list[dict[str, Any]] = []
    for reward_model_dir in reward_model_dirs:
        scorer = RewardScorer(reward_model_dir, device=config.get("reward_device"))
        scored = score_reward_grid_rows(
            candidates,
            scorer,
            batch_size=int(config.get("reward_batch_size", 4)),
        )
        reward_model_name = str(name_override or reward_model_dir.name)
        for row in scored:
            row["reward_model"] = reward_model_name
            row["reward_model_dir"] = str(reward_model_dir)
        all_scored.extend(scored)

    summary = summarize_reward_grid(all_scored)
    grid_csv = output_dir / "kalomaze_on_policy_reward_grid.csv"
    variant_csv = output_dir / "kalomaze_on_policy_reward_grid_variant_summary.csv"
    comparison_csv = output_dir / "kalomaze_on_policy_reward_grid_comparisons.csv"
    summary_json = output_dir / "kalomaze_on_policy_reward_grid_summary.json"
    pd.DataFrame(all_scored).to_csv(grid_csv, index=False)
    pd.DataFrame(summary["variant_summary"]).to_csv(variant_csv, index=False)
    pd.DataFrame(summary["comparisons"]).to_csv(comparison_csv, index=False)
    payload = {
        "dataset_path": str(dataset_path),
        "output_root": str(output_root),
        "reward_model_dirs": [str(path) for path in _reward_model_dirs(config)],
        "runs": list(runs),
        "steps": sorted(steps) if steps is not None else "latest",
        "candidate_rows": len(candidates),
        "scored_rows": len(all_scored),
        "grid_csv": str(grid_csv),
        "variant_summary_csv": str(variant_csv),
        "comparison_csv": str(comparison_csv),
        "variant_summary": summary["variant_summary"],
        "comparisons": summary["comparisons"],
    }
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return payload


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Score Kalomaze heldout policy outputs against reference completions.")
    parser.add_argument("--config", default="configs/kalomaze_replica_confab/on_policy_reward_grid.yaml")
    parser.add_argument("--reward-model-dir", default=None)
    parser.add_argument("--reward-model-glob", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--steps", default=None)
    args = parser.parse_args(argv)

    config = load_config(args.config) if Path(args.config).exists() else {}
    for key in ("reward_model_dir", "reward_model_glob", "output_root", "dataset_path", "steps"):
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    print(json.dumps(write_on_policy_reward_grid(config), indent=2))


if __name__ == "__main__":
    main()
