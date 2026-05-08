from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rl_epistemics.data.hard_eval import write_hard_eval_dataset
from rl_epistemics.data.verified_dataset import write_verified_dataset
from rl_epistemics.rl.train_tinker import eval_policy_tinker
from rl_epistemics.utils.config import deep_update, load_config


def _load_optional_config(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    return load_config(target) if target.exists() else {}


def _post_gate_eval_config(
    train_config: dict[str, Any],
    *,
    dataset_path: str,
    output_dir: str,
    manual_review_output: str,
    tinker_api_key_file: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = dict(train_config)
    config.update(
        {
            "dataset_path": dataset_path,
            "split": "ood",
            "run_meta_path": "outputs/tinker_rl_post_gate/run_meta.json",
            "output_dir": output_dir,
            "manual_review_output": manual_review_output,
            "include_reward_scores": True,
        }
    )
    if tinker_api_key_file:
        config["tinker_api_key_file"] = tinker_api_key_file
    if overrides:
        config = deep_update(config, overrides)
    return config


def eval_post_gate_tinker(config: dict[str, Any] | None = None) -> dict[str, Any]:
    config = config or {}
    train_config = load_config(config.get("train_config", "configs/tinker_rl_post_gate.yaml"))
    tinker_api_key_file = config.get("tinker_api_key_file")
    eval_overrides = config.get("eval_overrides") or {}

    if not config.get("skip_data_refresh", False):
        verified_data = write_verified_dataset(
            deep_update(
                _load_optional_config(config.get("verified_config", "configs/verified_data.yaml")),
                config.get("verified_overrides") or {},
            )
        )
        hard_data = write_hard_eval_dataset(
            deep_update(
                _load_optional_config(config.get("hard_config", "configs/hard_eval.yaml")),
                config.get("hard_overrides") or {},
            )
        )
    else:
        verified_data = {
            "output_path": str(config.get("verified_dataset_path", "data/verified_eval/pairs.jsonl"))
        }
        hard_data = {"output_path": str(config.get("hard_dataset_path", "data/hard_eval_pairs.jsonl"))}

    verified_eval = eval_policy_tinker(
        _post_gate_eval_config(
            train_config,
            dataset_path=str(verified_data["output_path"]),
            output_dir="outputs/eval_tinker_verified_post_gate",
            manual_review_output="outputs/blog/manual_review_samples_verified_post_gate.md",
            tinker_api_key_file=tinker_api_key_file,
            overrides=eval_overrides,
        )
    )
    hard_eval = eval_policy_tinker(
        _post_gate_eval_config(
            train_config,
            dataset_path=str(hard_data["output_path"]),
            output_dir="outputs/eval_tinker_hard_post_gate",
            manual_review_output="outputs/blog/manual_review_samples_hard_post_gate.md",
            tinker_api_key_file=tinker_api_key_file,
            overrides=eval_overrides,
        )
    )
    return {
        "verified_dataset": verified_data,
        "hard_dataset": hard_data,
        "verified_post_gate": verified_eval,
        "hard_post_gate": hard_eval,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate the local post-gate Tinker sampler.")
    parser.add_argument("--train-config", default="configs/tinker_rl_post_gate.yaml")
    parser.add_argument("--tinker-api-key-file", default=None)
    parser.add_argument("--skip-data-refresh", action="store_true")
    args = parser.parse_args(argv)
    print(
        json.dumps(
            eval_post_gate_tinker(
                {
                    "train_config": args.train_config,
                    "tinker_api_key_file": args.tinker_api_key_file,
                    "skip_data_refresh": args.skip_data_refresh,
                }
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
