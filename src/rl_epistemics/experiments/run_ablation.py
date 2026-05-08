from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from rl_epistemics.utils.config import deep_update, load_config, save_config


def iter_grid(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = list(grid)
    values = [grid[key] for key in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def variant_to_updates(variant: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    data_updates: dict[str, Any] = {}
    rm_updates: dict[str, Any] = {}
    rl_updates: dict[str, Any] = {}
    if "data_balance" in variant:
        data_updates["data_balance"] = variant["data_balance"]
    if "prefill" in variant:
        data_updates["use_prefill"] = variant["prefill"]
    if "polarity_flip" in variant:
        data_updates["polarity_flip"] = variant["polarity_flip"]
    if "rm_input" in variant:
        rm_updates["rm_input"] = variant["rm_input"]
    if "hidden_layer" in variant:
        rm_updates["hidden_layer"] = variant["hidden_layer"]
    if "pool" in variant:
        rm_updates["pool"] = variant["pool"]
    if "head" in variant:
        rm_updates["head"] = {"type": variant["head"], "scale": 0.01}
    if "reward_transform" in variant:
        rl_updates["reward_transform"] = {"type": variant["reward_transform"], "std_floor": 0.1}
    if "rl_beta_or_kl" in variant:
        rl_updates["beta"] = variant["rl_beta_or_kl"]
    if "lora_rank" in variant:
        rl_updates["lora"] = {"r": variant["lora_rank"], "alpha": 2 * int(variant["lora_rank"])}
    return data_updates, rm_updates, rl_updates


def run_ablation(config: dict[str, Any]) -> list[dict[str, Any]]:
    base_data = load_config(config["base_data_config"])
    base_rm = load_config(config["base_reward_model_config"])
    base_rl = load_config(config["base_rl_config"])
    output_dir = Path(config.get("output_dir", "outputs/ablations"))
    output_dir.mkdir(parents=True, exist_ok=True)
    variants = iter_grid(config.get("grid") or {})
    if config.get("max_runs") is not None:
        variants = variants[: int(config["max_runs"])]

    manifest = []
    for idx, variant in enumerate(variants):
        run_dir = output_dir / f"run_{idx:04d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        data_updates, rm_updates, rl_updates = variant_to_updates(variant)
        data_cfg = deep_update(base_data, data_updates)
        rm_cfg = deep_update(base_rm, rm_updates)
        rl_cfg = deep_update(base_rl, rl_updates)

        data_cfg["output_path"] = str(run_dir / "pairs.jsonl")
        rm_cfg["dataset_path"] = data_cfg["output_path"]
        rm_cfg["output_dir"] = str(run_dir / "reward_model")
        rl_cfg["dataset_path"] = data_cfg["output_path"]
        rl_cfg["reward_model_dir"] = rm_cfg["output_dir"]
        rl_cfg["output_dir"] = str(run_dir / "rl")

        data_path = run_dir / "data.yaml"
        rm_path = run_dir / "reward_model.yaml"
        rl_path = run_dir / "rl.yaml"
        save_config(data_cfg, data_path)
        save_config(rm_cfg, rm_path)
        save_config(rl_cfg, rl_path)

        commands = [
            [sys.executable, "scripts/make_dataset.py", "--config", str(data_path)],
            [sys.executable, "scripts/train_reward_model.py", "--config", str(rm_path)],
            [sys.executable, "scripts/train_rl.py", "--config", str(rl_path)],
        ]
        record = {"run_dir": str(run_dir), "variant": variant, "commands": commands}
        manifest.append(record)
        if not config.get("dry_run", True):
            for command in commands:
                subprocess.run(command, check=True)

    with (output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate or run YAML-driven ablation variants.")
    parser.add_argument("--config", default="configs/ablations.yaml")
    parser.add_argument("--run", action="store_true", help="Actually run commands instead of dry-run manifest.")
    parser.add_argument("--max-runs", type=int, default=None)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.run:
        config["dry_run"] = False
    if args.max_runs is not None:
        config["max_runs"] = args.max_runs
    manifest = run_ablation(config)
    print(json.dumps({"runs": len(manifest), "manifest": config.get("output_dir", "outputs/ablations")}, indent=2))


if __name__ == "__main__":
    main()
