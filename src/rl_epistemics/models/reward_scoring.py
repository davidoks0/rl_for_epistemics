from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from tqdm import tqdm

from rl_epistemics.data.schema import read_jsonl
from rl_epistemics.models.bt_trainer import (
    extract_hiddens_batch as extract_reward_hiddens_batch,
    summarize_scores,
)
from rl_epistemics.models.device import model_device_map, model_load_dtype, select_torch_device
from rl_epistemics.models.qwen_io import load_causal_lm, load_tokenizer
from rl_epistemics.models.reward_calibration import apply_score_calibration, load_score_calibration
from rl_epistemics.models.reward_head import RewardHead
from rl_epistemics.utils.config import load_config


class RewardScorer:
    def __init__(self, reward_model_dir: str | Path, device: str | None = None):
        self.reward_model_dir = Path(reward_model_dir)
        requested_device = device or "auto"
        required = [
            self.reward_model_dir / "model_meta.json",
            self.reward_model_dir / "reward_head_config.json",
            self.reward_model_dir / "reward_head.pt",
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Reward model artifacts are missing. Train the reward model first. "
                f"Missing: {missing}"
            )
        with (self.reward_model_dir / "model_meta.json").open("r", encoding="utf-8") as f:
            self.meta = json.load(f)
        self.device = select_torch_device(requested_device)
        self.tokenizer = load_tokenizer(self.meta["model_id"])
        self.model = load_causal_lm(
            self.meta["model_id"],
            torch_dtype=model_load_dtype("auto", self.device),
            device_map=model_device_map(requested_device, self.device),
        )
        if not hasattr(self.model, "hf_device_map"):
            self.model.to(self.device)
        self.model.eval()
        self.head = RewardHead.load(self.reward_model_dir, map_location=self.device).to(self.device)
        self.head.eval()
        self.score_calibration = load_score_calibration(self.reward_model_dir)

    @torch.inference_mode()
    def score(self, prompt: str, completion: str, prompt_type: str | None = None) -> float:
        return self.score_many(
            [{"prompt": prompt, "completion": completion, "prompt_type": prompt_type}]
        )[0]

    @torch.inference_mode()
    def score_many(
        self,
        prompt_completion_rows: list[dict[str, Any]],
        *,
        batch_size: int = 8,
    ) -> list[float]:
        if not prompt_completion_rows:
            return []

        scores: list[float] = []
        for start in range(0, len(prompt_completion_rows), batch_size):
            batch = prompt_completion_rows[start : start + batch_size]
            hiddens = self._extract_hiddens_batch(batch).to(self.device)
            batch_scores = self.head(hiddens).detach().float().cpu().tolist()
            scores.extend(
                apply_score_calibration(
                    float(score),
                    self.score_calibration,
                    str(row.get("prompt_type", "")) or None,
                )
                for score, row in zip(batch_scores, batch)
            )
        return scores

    def _extract_hiddens_batch(self, rows: list[dict[str, Any]]) -> torch.Tensor:
        max_length = int(self.meta.get("max_length", 1024))
        hidden_layer = self.meta.get("hidden_layer", "last")
        pool = self.meta.get("pool", "final_token")
        rm_input = self.meta.get("rm_input", "prompt_completion")
        return extract_reward_hiddens_batch(
            self.model,
            self.tokenizer,
            [
                {"prompt": str(row["prompt"]), "completion": str(row["completion"])}
                for row in rows
            ],
            max_length=max_length,
            hidden_layer=hidden_layer,
            pool=pool,
            rm_input=rm_input,
            device=self.device,
        )


def eval_reward_model(config: dict[str, Any]) -> dict[str, Any]:
    dataset_path = config.get("dataset_path", "data/pairs.jsonl")
    reward_model_dir = Path(config.get("reward_model_dir") or config.get("output_dir", "outputs/reward_model"))
    output_dir = Path(config.get("eval_output_dir", reward_model_dir / "eval"))
    output_dir.mkdir(parents=True, exist_ok=True)

    scorer = RewardScorer(reward_model_dir, device=config.get("device"))
    pairs = read_jsonl(dataset_path)
    if config.get("limit_eval_examples"):
        pairs = pairs[: int(config["limit_eval_examples"])]

    rows = []
    for pair in tqdm(pairs, desc="eval_rm"):
        chosen_reward = scorer.score_many(
            [{"prompt": pair.prompt, "completion": pair.chosen, "prompt_type": pair.prompt_type}]
        )[0]
        rejected_reward = scorer.score_many(
            [{"prompt": pair.prompt, "completion": pair.rejected, "prompt_type": pair.prompt_type}]
        )[0]
        rows.append(
            {
                "id": pair.id,
                "split": pair.split,
                "prompt_type": pair.prompt_type,
                "domain": pair.domain,
                "category": str(pair.metadata.get("category", "")),
                "chosen_reward": chosen_reward,
                "rejected_reward": rejected_reward,
                "margin": chosen_reward - rejected_reward,
                "correct": chosen_reward > rejected_reward,
            }
        )
    pd.DataFrame(rows).to_csv(output_dir / "reward_scores.csv", index=False)
    summary = summarize_scores(rows)
    with (output_dir / "reward_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained reward head on JSONL pairs.")
    parser.add_argument("--config", default="configs/reward_model.yaml")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    print(json.dumps(eval_reward_model(config), indent=2))


if __name__ == "__main__":
    main()
