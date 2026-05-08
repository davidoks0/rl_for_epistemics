from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rl_epistemics.data.schema import read_jsonl
from rl_epistemics.models.qwen_io import normalize_completion
from rl_epistemics.models.reward_scoring import RewardScorer
from rl_epistemics.rl.reward_transforms import (
    build_reward_transform,
    load_score_stats,
    resolve_reward_transform_config,
)
from rl_epistemics.utils.config import load_config, save_config
from rl_epistemics.utils.seed import set_seed


def balanced_prompt_rows(dataset_path: str, split: str = "train") -> list[dict[str, Any]]:
    pairs = [pair for pair in read_jsonl(dataset_path) if pair.split == split]
    real = [pair for pair in pairs if pair.prompt_type == "real"]
    fake = [pair for pair in pairs if pair.prompt_type == "fake"]
    n = min(len(real), len(fake))
    if n == 0:
        selected = pairs
    else:
        selected = real[:n] + fake[:n]
    return [
        {
            "prompt": pair.prompt,
            "prompt_type": pair.prompt_type,
            "id": pair.id,
            "domain": pair.domain,
            "entity_name": pair.entity_name,
        }
        for pair in selected
    ]


def train_grpo(config: dict[str, Any]) -> None:
    try:
        from datasets import Dataset
        from peft import LoraConfig
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:
        raise ImportError(
            "GRPO training requires datasets, peft, and trl. Install with `uv pip install -e .`."
        ) from exc

    set_seed(int(config.get("seed", 0)))
    output_dir = Path(config.get("output_dir", "outputs/rl_grpo"))
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, output_dir / "train_config.yaml")

    rows = balanced_prompt_rows(config["dataset_path"], split="train")
    train_dataset = Dataset.from_list(rows)

    reward_model_dir = Path(config["reward_model_dir"])
    transform_config = dict(config.get("reward_transform") or {})
    std_floor = float(transform_config.get("std_floor", 0.1))
    stats = load_score_stats(reward_model_dir, std_floor=std_floor)
    transform_config = resolve_reward_transform_config(
        transform_config,
        reward_model_dir,
        std_floor=std_floor,
    )
    transform = build_reward_transform(transform_config, stats)
    scorer = RewardScorer(reward_model_dir)
    reward_log_path = output_dir / "reward_samples.jsonl"

    def reward_func(prompts, completions, prompt_type=None, **kwargs):
        rewards = []
        prompt_types = prompt_type or kwargs.get("prompt_type") or [None] * len(prompts)
        with reward_log_path.open("a", encoding="utf-8") as log_f:
            for prompt, completion, ptype in zip(prompts, completions, prompt_types):
                completion_text = normalize_completion(completion)
                raw_score = scorer.score_many(
                    [{"prompt": str(prompt), "completion": completion_text, "prompt_type": ptype}]
                )[0]
                reward = transform(raw_score, ptype)
                rewards.append(reward)
                log_f.write(
                    json.dumps(
                        {
                            "prompt_type": ptype,
                            "raw_score": raw_score,
                            "reward": reward,
                            "completion_length": len(completion_text),
                        }
                    )
                    + "\n"
                )
        return rewards

    lora_cfg = config.get("lora") or {}
    peft_config = LoraConfig(
        r=int(lora_cfg.get("r", 16)),
        lora_alpha=int(lora_cfg.get("alpha", 32)),
        lora_dropout=float(lora_cfg.get("dropout", 0.05)),
        target_modules=lora_cfg.get("target_modules"),
        task_type="CAUSAL_LM",
    )

    training_args = GRPOConfig(
        output_dir=str(output_dir),
        max_steps=int(config.get("max_steps", 20)),
        per_device_train_batch_size=int(config.get("per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(config.get("gradient_accumulation_steps", 4)),
        num_generations=int(config.get("num_generations", 4)),
        max_prompt_length=int(config.get("max_prompt_length", 512)),
        max_completion_length=int(config.get("max_completion_length", 160)),
        learning_rate=float(config.get("learning_rate", 5.0e-6)),
        beta=float(config.get("beta", 0.02)),
        temperature=float(config.get("temperature", 0.7)),
        top_p=float(config.get("top_p", 0.8)),
        top_k=int(config.get("top_k", 20)),
        logging_steps=int(config.get("logging_steps", 1)),
        save_steps=int(config.get("save_steps", 25)),
        report_to=config.get("report_to", "none"),
        use_vllm=bool(config.get("use_vllm", False)),
    )

    trainer = GRPOTrainer(
        model=config["model_id"],
        reward_funcs=reward_func,
        args=training_args,
        train_dataset=train_dataset,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(str(output_dir / "final"))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train policy with TRL GRPO and a learned RM reward.")
    parser.add_argument("--config", default="configs/rl.yaml")
    args = parser.parse_args(argv)
    train_grpo(load_config(args.config))


if __name__ == "__main__":
    main()
