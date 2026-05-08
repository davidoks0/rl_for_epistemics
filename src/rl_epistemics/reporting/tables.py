from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


POLICY_KEYS = [
    "fake_n",
    "real_n",
    "fake_clean_uncertainty",
    "fake_false_premise_correction",
    "fake_hedged_confabulation",
    "fake_confident_confabulation",
    "real_real_substantive_answer",
    "real_false_refusal",
    "fake_fake_confabulates",
    "real_real_false_refusal",
    "fake_fake_refuses_or_flags",
    "real_real_answers_substantively",
    "fake_fake_hedges_but_engages",
    "reward_mean",
]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    if path.suffix in {".yaml", ".yml"}:
        import yaml

        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data if isinstance(data, dict) else {}
    return _read_json(path)


def _read_last_jsonl(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    last = ""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                last = line
    return json.loads(last) if last else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _training_note(path: Path) -> str:
    rows = _read_jsonl(path)
    if not rows:
        return ""
    losses = [float(row["loss"]) for row in rows if row.get("loss") is not None]
    degenerates = [
        float(row["degenerate_group_fraction"])
        for row in rows
        if row.get("degenerate_group_fraction") is not None
    ]
    parts = []
    if losses:
        parts.append(f"final_loss={losses[-1]:.3f}")
        parts.append(f"max_abs_loss={max(abs(loss) for loss in losses):.3f}")
    if degenerates:
        parts.append(f"max_degenerate={max(degenerates):.3f}")
    return "; ".join(parts)


def _known_policy_metric_paths(outputs: Path) -> list[tuple[str, Path]]:
    ordered = [
        ("base_test", outputs / "eval_tinker_base_test" / "policy_metrics.json"),
        ("rl_test", outputs / "eval_tinker_main" / "policy_metrics.json"),
        ("base_ood", outputs / "eval_tinker_base_ood" / "policy_metrics.json"),
        ("rl_ood", outputs / "eval_tinker_main_ood" / "policy_metrics.json"),
        ("base_hard", outputs / "eval_tinker_hard_base" / "policy_metrics.json"),
        ("rl_hard", outputs / "eval_tinker_hard_main" / "policy_metrics.json"),
        ("post_gate_verified", outputs / "eval_tinker_verified_post_gate" / "policy_metrics.json"),
        ("post_gate_hard", outputs / "eval_tinker_hard_post_gate" / "policy_metrics.json"),
    ]
    if not ordered[1][1].exists() and (outputs / "eval_tinker" / "policy_metrics.json").exists():
        ordered[1] = ("rl_test", outputs / "eval_tinker" / "policy_metrics.json")
    rows = [(name, path) for name, path in ordered if path.exists()]
    seen = {name for name, _ in rows}
    flat_fallbacks = [
        ("base_hard", outputs / "eval_tinker_hard_base_policy_metrics.json"),
        ("rl_hard", outputs / "eval_tinker_hard_main_policy_metrics.json"),
    ]
    for name, path in flat_fallbacks:
        if name not in seen and path.exists():
            rows.append((name, path))
            seen.add(name)
    for metrics_path in sorted((outputs / "eval_tinker_ablations").glob("*/policy_metrics.json")):
        rows.append((f"ablation_{metrics_path.parent.name}", metrics_path))
    return rows


def _write_markdown_table(df: pd.DataFrame, path: Path) -> None:
    def cell(value: Any) -> str:
        text = "" if pd.isna(value) else str(value)
        return text.replace("|", "\\|").replace("\n", " ")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        if df.empty:
            f.write("_No rows available._\n")
        else:
            cols = list(df.columns)
            f.write("| " + " | ".join(cell(col) for col in cols) + " |\n")
            f.write("|" + "|".join("---" for _ in cols) + "|\n")
            for _, row in df.iterrows():
                f.write("| " + " | ".join(cell(row[col]) for col in cols) + " |\n")
            f.write("\n")


def _iter_rm_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    if any(key in summary for key in ("all", "real", "fake")):
        return [
            {"split": "eval", "group": group, **values}
            for group, values in summary.items()
            if isinstance(values, dict)
        ]
    rows = []
    for split, split_summary in summary.items():
        if not isinstance(split_summary, dict):
            continue
        for group, values in split_summary.items():
            if isinstance(values, dict):
                rows.append({"split": split, "group": group, **values})
    return rows


def write_report_tables(outputs_dir: str | Path, report_dir: str | Path) -> dict[str, str]:
    outputs = Path(outputs_dir)
    report = Path(report_dir)
    report.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}

    rm_summary = _read_json(outputs / "reward_model" / "eval" / "reward_summary.json")
    if not rm_summary:
        rm_summary = _read_json(outputs / "reward_model" / "score_summary.json")
    rm_rows = _iter_rm_rows(rm_summary)
    rm_df = pd.DataFrame(rm_rows)
    rm_csv = report / "reward_model_summary.csv"
    rm_df.to_csv(rm_csv, index=False)
    _write_markdown_table(rm_df, report / "reward_model_summary.md")
    written["reward_model_summary"] = str(rm_csv)

    policy_rows = []
    for run_name, metrics_path in _known_policy_metric_paths(outputs):
        summary = _read_json(metrics_path)
        row = {"run": run_name, "path": str(metrics_path.parent)}
        for key in POLICY_KEYS:
            if key in summary:
                row[key] = summary[key]
        policy_rows.append(row)

    policy_df = pd.DataFrame(policy_rows)
    policy_csv = report / "policy_metrics_table.csv"
    policy_df.to_csv(policy_csv, index=False)
    _write_markdown_table(policy_df, report / "policy_metrics_table.md")
    written["policy_metrics_table"] = str(policy_csv)

    ablation_specs = [
        ("main-rank32-gaussian", outputs / "tinker_rl", outputs / "eval_tinker_main"),
        ("raw_reward", outputs / "tinker_rl_ablations" / "raw_reward", outputs / "eval_tinker_ablations" / "raw_reward"),
        ("band_target", outputs / "tinker_rl_ablations" / "band_target", outputs / "eval_tinker_ablations" / "band_target"),
        ("rank8", outputs / "tinker_rl_ablations" / "rank8", outputs / "eval_tinker_ablations" / "rank8"),
        ("rank16", outputs / "tinker_rl_ablations" / "rank16", outputs / "eval_tinker_ablations" / "rank16"),
    ]
    if not ablation_specs[0][2].exists() and (outputs / "eval_tinker").exists():
        ablation_specs[0] = ("main-rank32-gaussian", outputs / "tinker_rl", outputs / "eval_tinker")
    seen = {name for name, _, _ in ablation_specs}
    for run_meta in sorted((outputs / "tinker_rl_ablations").glob("*/run_meta.json")):
        name = run_meta.parent.name
        if name not in seen:
            ablation_specs.append((name, run_meta.parent, outputs / "eval_tinker_ablations" / name))

    ablation_rows = []
    for run_name, run_dir, eval_dir in ablation_specs:
        meta = _read_json(run_dir / "run_meta.json")
        config = _read_mapping(run_dir / "train_config.yaml")
        eval_metrics = _read_json(eval_dir / "policy_metrics.json")
        if not meta and not eval_metrics:
            continue
        transform = (config.get("reward_transform") or {}).get("type")
        notes = []
        if transform == "raw":
            notes.append("raw reward; high loss scale risk")
        if transform == "band_target":
            notes.append("band target; watch degenerate groups")
        if run_name in {"rank8", "rank16"}:
            notes.append("LoRA rank ablation")
        if run_name == "imbalanced_data":
            notes.append("data balance ablation")
        if run_name == "no_polarity_flip":
            notes.append("real-prompt polarity inverted")
        if run_name == "completion_only":
            notes.append("reward model input ablation")
        if run_name == "scalar_linear":
            notes.append("reward head ablation")
        ablation_rows.append(
            {
                "run": run_name,
                "base_model": meta.get("base_model"),
                "rank": meta.get("rank"),
                "reward_transform": transform,
                "fake_confabulates": eval_metrics.get("fake_fake_confabulates"),
                "real_false_refusal": eval_metrics.get("real_real_false_refusal"),
                "fake_refuses_or_flags": eval_metrics.get("fake_fake_refuses_or_flags"),
                "fake_hedges_but_engages": eval_metrics.get("fake_fake_hedges_but_engages"),
                "real_answers_substantively": eval_metrics.get("real_real_answers_substantively"),
                "reward_mean": eval_metrics.get("reward_mean"),
                "training_note": _training_note(run_dir / "metrics.jsonl"),
                "notes": "; ".join(notes),
                "sampler_path": meta.get("final_sampler_path"),
            }
        )
    ablation_df = pd.DataFrame(ablation_rows)
    ablation_csv = report / "ablation_table.csv"
    ablation_df.to_csv(ablation_csv, index=False)
    _write_markdown_table(ablation_df, report / "ablation_table.md")
    written["ablation_table"] = str(ablation_csv)

    return written
