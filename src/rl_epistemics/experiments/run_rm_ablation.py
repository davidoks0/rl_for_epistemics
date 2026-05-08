from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from rl_epistemics.data.schema import read_jsonl
from rl_epistemics.utils.config import deep_update, load_config, save_config


DEFAULT_VARIANTS = [
    {
        "name": "last_final_token_matrix_sum",
        "updates": {
            "hidden_layer": "last",
            "pool": "final_token",
            "rm_input": "prompt_completion",
            "head": {"type": "matrix_sum", "scale": 0.01},
        },
    },
    {
        "name": "last_mean_completion_matrix_sum",
        "updates": {
            "hidden_layer": "last",
            "pool": "mean_completion",
            "rm_input": "prompt_completion",
            "head": {"type": "matrix_sum", "scale": 0.01},
        },
    },
    {
        "name": "middle_final_token_matrix_sum",
        "updates": {
            "hidden_layer": "middle",
            "pool": "final_token",
            "rm_input": "prompt_completion",
            "head": {"type": "matrix_sum", "scale": 0.01},
        },
    },
    {
        "name": "last_completion_only_matrix_sum",
        "updates": {
            "hidden_layer": "last",
            "pool": "final_token",
            "rm_input": "completion_only",
            "head": {"type": "matrix_sum", "scale": 0.01},
        },
    },
    {
        "name": "last_final_token_scalar_linear",
        "updates": {
            "hidden_layer": "last",
            "pool": "final_token",
            "rm_input": "prompt_completion",
            "head": {"type": "scalar_linear", "scale": 0.01},
        },
    },
    {
        "name": "last_final_token_mlp",
        "updates": {
            "hidden_layer": "last",
            "pool": "final_token",
            "rm_input": "prompt_completion",
            "head": {"type": "mlp", "scale": 0.01},
        },
    },
]


def _variant_name(variant: dict[str, Any], idx: int) -> str:
    return str(variant.get("name") or f"rm_variant_{idx:04d}")


def dataset_manifest_summary(path: str | Path) -> dict[str, Any]:
    dataset_path = Path(path)
    summary: dict[str, Any] = {"path": str(dataset_path), "exists": dataset_path.exists()}
    if not dataset_path.exists():
        return summary
    data = dataset_path.read_bytes()
    pairs = read_jsonl(dataset_path)
    summary.update(
        {
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": dataset_path.stat().st_size,
            "n": len(pairs),
            "unique_prompts": len({pair.prompt for pair in pairs}),
            "unique_seed_prompts": len(
                {str(pair.metadata.get("seed_prompt", pair.prompt)) for pair in pairs}
            ),
            "domains": sorted({pair.domain for pair in pairs}),
            "categories": sorted({str(pair.metadata.get("category", "")) for pair in pairs}),
        }
    )
    return summary


def _flatten_rm_summary(name: str, summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if any(key in summary for key in ("all", "real", "fake")):
        for group, values in summary.items():
            if isinstance(values, dict):
                rows.append({"run": name, "split": "eval", "group": group, **values})
        return rows
    for split, split_summary in summary.items():
        if not isinstance(split_summary, dict):
            continue
        for group, values in split_summary.items():
            if isinstance(values, dict):
                rows.append({"run": name, "split": split, "group": group, **values})
    return rows


def write_rm_ablation_summary(output_dir: str | Path) -> dict[str, Any]:
    output = Path(output_dir)
    rows: list[dict[str, Any]] = []
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else []
    variant_by_name = {row["name"]: row.get("variant", {}) for row in manifest}
    for summary_path in sorted(output.glob("*/eval/reward_summary.json")):
        name = summary_path.parent.parent.name
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
        for row in _flatten_rm_summary(name, summary):
            variant = variant_by_name.get(name, {})
            row.update(
                {
                    "hidden_layer": variant.get("hidden_layer"),
                    "pool": variant.get("pool"),
                    "rm_input": variant.get("rm_input"),
                    "head_type": (variant.get("head") or {}).get("type"),
                }
            )
            rows.append(row)
    csv_path = output / "rm_design_summary.csv"
    columns = [
        "run",
        "split",
        "group",
        "hidden_layer",
        "pool",
        "rm_input",
        "head_type",
        "n",
        "accuracy",
        "margin_mean",
        "chosen_mean",
        "rejected_mean",
    ]
    pd.DataFrame(rows, columns=columns if not rows else None).to_csv(csv_path, index=False)
    return {"summary_csv": str(csv_path), "rows": len(rows)}


def _run_config_summary(rm_cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_id": rm_cfg.get("model_id"),
        "dataset_path": rm_cfg.get("dataset_path"),
        "hidden_layer": rm_cfg.get("hidden_layer"),
        "pool": rm_cfg.get("pool"),
        "rm_input": rm_cfg.get("rm_input"),
        "head": rm_cfg.get("head"),
        "score_calibration": rm_cfg.get("score_calibration"),
        "limit_train_examples": rm_cfg.get("limit_train_examples"),
        "limit_eval_examples": rm_cfg.get("limit_eval_examples"),
        "max_length": rm_cfg.get("max_length"),
        "epochs": rm_cfg.get("epochs"),
    }


def run_rm_ablation(config: dict[str, Any]) -> list[dict[str, Any]]:
    base_rm = load_config(config["base_reward_model_config"])
    output_dir = Path(config.get("output_dir", "outputs/rm_design_ablations"))
    output_dir.mkdir(parents=True, exist_ok=True)

    variants = list(config.get("variants") or DEFAULT_VARIANTS)
    if config.get("max_runs") is not None:
        variants = variants[: int(config["max_runs"])]

    manifest: list[dict[str, Any]] = []
    for idx, variant in enumerate(variants):
        name = _variant_name(variant, idx)
        run_dir = output_dir / name
        run_dir.mkdir(parents=True, exist_ok=True)
        rm_cfg = deep_update(base_rm, variant.get("updates") or {})
        if config.get("dataset_path"):
            rm_cfg["dataset_path"] = str(config["dataset_path"])
        for key in (
            "model_id",
            "device",
            "torch_dtype",
            "max_length",
            "epochs",
            "limit_train_examples",
            "limit_eval_examples",
        ):
            if config.get(key) is not None:
                rm_cfg[key] = config[key]
        rm_cfg["output_dir"] = str(run_dir / "reward_model")
        rm_cfg["eval_output_dir"] = str(run_dir / "eval")
        dataset_summary = dataset_manifest_summary(rm_cfg["dataset_path"])

        config_path = run_dir / "reward_model.yaml"
        save_config(rm_cfg, config_path)
        commands = [
            [sys.executable, "scripts/train_reward_model.py", "--config", str(config_path)],
            [sys.executable, "scripts/eval_reward_model.py", "--config", str(config_path)],
        ]
        record = {
            "name": name,
            "run_dir": str(run_dir),
            "variant": variant.get("updates") or {},
            "dataset_summary": dataset_summary,
            "run_config": _run_config_summary(rm_cfg),
            "commands": commands,
        }
        manifest.append(record)
        if not config.get("dry_run", True):
            for command in commands:
                subprocess.run(command, check=True)

    with (output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    write_rm_ablation_summary(output_dir)
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate or run focused reward-model ablations.")
    parser.add_argument("--config", default="configs/rm_design_ablation.yaml")
    parser.add_argument("--run", action="store_true", help="Actually run commands instead of dry-run manifest.")
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--torch-dtype", default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--limit-train-examples", type=int, default=None)
    parser.add_argument("--limit-eval-examples", type=int, default=None)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.run:
        config["dry_run"] = False
    if args.max_runs is not None:
        config["max_runs"] = args.max_runs
    for key in (
        "model_id",
        "dataset_path",
        "output_dir",
        "device",
        "torch_dtype",
        "max_length",
        "epochs",
        "limit_train_examples",
        "limit_eval_examples",
    ):
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    manifest = run_rm_ablation(config)
    print(json.dumps({"runs": len(manifest), "manifest": config.get("output_dir")}, indent=2))


if __name__ == "__main__":
    main()
