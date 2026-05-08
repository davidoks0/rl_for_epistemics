from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _maybe_pyplot() -> Any | None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    return plt


def _plot_reward_scores(outputs: Path, report: Path) -> list[str]:
    paths: list[str] = []
    plt = _maybe_pyplot()
    if plt is None:
        return paths
    scores_path = outputs / "reward_model" / "eval" / "reward_scores.csv"
    if not scores_path.exists():
        return paths
    df = pd.read_csv(scores_path)
    if {"chosen_reward", "rejected_reward"}.issubset(df.columns):
        path = report / "rm_chosen_rejected_hist.png"
        plt.figure(figsize=(7, 4))
        plt.hist(df["chosen_reward"].dropna(), bins=30, alpha=0.65, label="chosen")
        plt.hist(df["rejected_reward"].dropna(), bins=30, alpha=0.65, label="rejected")
        plt.xlabel("reward score")
        plt.ylabel("count")
        plt.legend()
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        paths.append(str(path))
    if {"prompt_type", "margin"}.issubset(df.columns):
        path = report / "rm_margin_by_prompt_type.png"
        plt.figure(figsize=(6, 4))
        df.boxplot(column="margin", by="prompt_type")
        plt.title("")
        plt.suptitle("")
        plt.xlabel("prompt type")
        plt.ylabel("chosen - rejected reward")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        paths.append(str(path))
    return paths


def _plot_policy_metrics(outputs: Path, report: Path) -> list[str]:
    paths: list[str] = []
    plt = _maybe_pyplot()
    if plt is None:
        return paths
    rows = []
    metric_paths = [
        ("base_test", outputs / "eval_tinker_base_test" / "policy_metrics.json"),
        ("rl_test", outputs / "eval_tinker_main" / "policy_metrics.json"),
        ("base_ood", outputs / "eval_tinker_base_ood" / "policy_metrics.json"),
        ("rl_ood", outputs / "eval_tinker_main_ood" / "policy_metrics.json"),
        ("base_hard", outputs / "eval_tinker_hard_base" / "policy_metrics.json"),
        ("rl_hard", outputs / "eval_tinker_hard_main" / "policy_metrics.json"),
    ]
    if not metric_paths[1][1].exists() and (outputs / "eval_tinker" / "policy_metrics.json").exists():
        metric_paths[1] = ("rl_test", outputs / "eval_tinker" / "policy_metrics.json")
    if not metric_paths[4][1].exists() and (outputs / "eval_tinker_hard_base_policy_metrics.json").exists():
        metric_paths[4] = ("base_hard", outputs / "eval_tinker_hard_base_policy_metrics.json")
    if not metric_paths[5][1].exists() and (outputs / "eval_tinker_hard_main_policy_metrics.json").exists():
        metric_paths[5] = ("rl_hard", outputs / "eval_tinker_hard_main_policy_metrics.json")
    for metrics_path in sorted((outputs / "eval_tinker_ablations").glob("*/policy_metrics.json")):
        metric_paths.append((f"ablation_{metrics_path.parent.name}", metrics_path))
    for run_name, metrics_path in metric_paths:
        if not metrics_path.exists():
            continue
        summary = _read_json(metrics_path)
        rows.append(
            {
                "run": run_name,
                "fake_confabulates": summary.get("fake_fake_confabulates"),
                "real_false_refusal": summary.get("real_real_false_refusal"),
                "fake_refuses_or_flags": summary.get("fake_fake_refuses_or_flags"),
                "real_answers_substantively": summary.get("real_real_answers_substantively"),
            }
        )
    df = pd.DataFrame(rows).dropna(how="all", subset=[col for col in rows[0] if col != "run"]) if rows else pd.DataFrame()
    if not df.empty:
        path = report / "policy_metrics_before_after.png"
        df.set_index("run").plot(kind="bar", figsize=(9, 4))
        plt.ylabel("rate")
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        paths.append(str(path))
    return paths


def _plot_tinker_training(outputs: Path, report: Path) -> list[str]:
    paths: list[str] = []
    plt = _maybe_pyplot()
    if plt is None:
        return paths
    for metrics_path in sorted(outputs.glob("tinker_rl*/**/metrics.jsonl")):
        rows = _read_jsonl(metrics_path)
        if not rows:
            continue
        df = pd.DataFrame(rows)
        path = report / f"{metrics_path.parent.name}_tinker_training.png"
        plt.figure(figsize=(8, 4))
        if "raw_reward_mean" in df:
            plt.plot(df["step"], df["raw_reward_mean"], marker="o", label="raw reward")
        if "transformed_reward_mean" in df:
            plt.plot(
                df["step"],
                df["transformed_reward_mean"],
                marker="o",
                label="transformed reward",
            )
        if "degenerate_group_fraction" in df:
            plt.plot(
                df["step"],
                df["degenerate_group_fraction"],
                marker="o",
                label="degenerate fraction",
            )
        plt.xlabel("step")
        plt.legend()
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        paths.append(str(path))
    return paths


def _plot_reward_distribution(outputs: Path, report: Path) -> list[str]:
    paths: list[str] = []
    plt = _maybe_pyplot()
    if plt is None:
        return paths
    samples_path = outputs / "tinker_rl" / "sample_completions.jsonl"
    samples = pd.DataFrame(_read_jsonl(samples_path))
    scores_path = outputs / "reward_model" / "eval" / "reward_scores.csv"
    if samples.empty or "raw_reward" not in samples:
        return paths
    path = report / "reward_distribution_before_after_vs_target.png"
    plt.figure(figsize=(7, 4))
    if scores_path.exists():
        scores = pd.read_csv(scores_path)
        if "chosen_reward" in scores:
            plt.hist(scores["chosen_reward"].dropna(), bins=30, alpha=0.4, label="chosen target")
    first_step = samples["step"].min()
    last_step = samples["step"].max()
    plt.hist(
        samples[samples["step"] == first_step]["raw_reward"].dropna(),
        bins=30,
        alpha=0.45,
        label=f"step {first_step}",
    )
    plt.hist(
        samples[samples["step"] == last_step]["raw_reward"].dropna(),
        bins=30,
        alpha=0.45,
        label=f"step {last_step}",
    )
    plt.xlabel("raw reward")
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    paths.append(str(path))
    return paths


def write_report_plots(outputs_dir: str | Path, report_dir: str | Path) -> dict[str, list[str]]:
    outputs = Path(outputs_dir)
    report = Path(report_dir)
    report.mkdir(parents=True, exist_ok=True)
    return {
        "reward_model": _plot_reward_scores(outputs, report),
        "policy": _plot_policy_metrics(outputs, report),
        "tinker_training": _plot_tinker_training(outputs, report),
        "reward_distribution": _plot_reward_distribution(outputs, report),
    }
