from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from rl_epistemics.data.schema import KnowledgePair, read_jsonl
from rl_epistemics.models.device import model_device_map, model_load_dtype, select_torch_device
from rl_epistemics.models.qwen_io import apply_chat_template, load_causal_lm, load_tokenizer
from rl_epistemics.models.reward_calibration import (
    apply_score_calibration_to_rows,
    chosen_score_stats_by_prompt_type,
    fit_score_calibration,
)
from rl_epistemics.models.reward_head import RewardHead, RewardHeadConfig
from rl_epistemics.utils.config import load_config, save_config
from rl_epistemics.utils.logging import setup_logging
from rl_epistemics.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)


def bt_loss(chosen_rewards: torch.Tensor, rejected_rewards: torch.Tensor) -> torch.Tensor:
    return -F.logsigmoid(chosen_rewards - rejected_rewards).mean()


def layer_index(hidden_states: tuple[torch.Tensor, ...], setting: str | int) -> int:
    if isinstance(setting, int):
        return setting
    if str(setting).isdigit() or (isinstance(setting, str) and setting.startswith("-")):
        return int(setting)
    n = len(hidden_states)
    if setting == "last":
        return -1
    if setting == "middle":
        return n // 2
    if setting == "early":
        return max(1, n // 4)
    raise ValueError(f"Unknown hidden_layer setting: {setting}")


def build_reward_text(
    tokenizer: Any,
    prompt: str,
    completion: str,
    rm_input: str = "prompt_completion",
) -> tuple[str, str | None]:
    if rm_input == "completion_only":
        return completion, None
    if rm_input != "prompt_completion":
        raise ValueError(f"Unknown rm_input: {rm_input}")
    prompt_text = apply_chat_template(tokenizer, prompt, None, add_generation_prompt=True)
    full_text = apply_chat_template(tokenizer, prompt, completion, add_generation_prompt=False)
    return full_text, prompt_text


def _selected_token_states(model: Any, encoded: dict[str, torch.Tensor], hidden_layer: str | int) -> torch.Tensor:
    if hidden_layer == "last" and hasattr(model, "model"):
        outputs = model.model(**encoded, use_cache=False)
        return outputs.last_hidden_state
    outputs = model(**encoded, output_hidden_states=True, use_cache=False)
    return outputs.hidden_states[layer_index(outputs.hidden_states, hidden_layer)]


@torch.no_grad()
def extract_hidden(
    model: Any,
    tokenizer: Any,
    prompt: str,
    completion: str,
    *,
    max_length: int,
    hidden_layer: str | int,
    pool: str,
    rm_input: str,
    device: torch.device,
) -> torch.Tensor:
    full_text, prompt_text = build_reward_text(tokenizer, prompt, completion, rm_input)
    encoded = tokenizer(
        full_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=False,
    ).to(device)
    states = _selected_token_states(model, encoded, hidden_layer)[0]
    attention = encoded["attention_mask"][0].bool()

    if pool == "final_token":
        final_idx = attention.nonzero()[-1].item()
        return states[final_idx].detach().float().cpu()

    if pool == "mean_completion":
        if prompt_text is not None:
            prompt_ids = tokenizer(
                prompt_text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=False,
            )["input_ids"][0]
            start = min(prompt_ids.shape[0], states.shape[0] - 1)
            mask = torch.zeros_like(attention)
            mask[start:] = attention[start:]
        else:
            mask = attention
        if mask.sum().item() == 0:
            mask = attention
        return states[mask].mean(dim=0).detach().float().cpu()

    raise ValueError(f"Unknown pool: {pool}")


@torch.no_grad()
def extract_hiddens_batch(
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, str]],
    *,
    max_length: int,
    hidden_layer: str | int,
    pool: str,
    rm_input: str,
    device: torch.device,
) -> torch.Tensor:
    texts: list[str] = []
    prompt_texts: list[str | None] = []
    for row in rows:
        full_text, prompt_text = build_reward_text(
            tokenizer,
            row["prompt"],
            row["completion"],
            rm_input,
        )
        texts.append(full_text)
        prompt_texts.append(prompt_text)

    encoded = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=True,
    ).to(device)
    states = _selected_token_states(model, encoded, hidden_layer)
    attention = encoded["attention_mask"].bool()

    if pool == "final_token":
        final_indices = attention.long().sum(dim=1).sub(1).clamp(min=0)
        batch_indices = torch.arange(states.shape[0], device=states.device)
        return states[batch_indices, final_indices].detach().float().cpu()

    if pool == "mean_completion":
        pooled: list[torch.Tensor] = []
        for idx, prompt_text in enumerate(prompt_texts):
            row_attention = attention[idx]
            mask = row_attention.clone()
            if prompt_text is not None:
                prompt_ids = tokenizer(
                    prompt_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_length,
                    padding=False,
                )["input_ids"][0]
                start = min(int(prompt_ids.shape[0]), int(row_attention.shape[0]) - 1)
                mask = torch.zeros_like(row_attention)
                mask[start:] = row_attention[start:]
                if mask.sum().item() == 0:
                    mask = row_attention
            pooled.append(states[idx][mask].mean(dim=0))
        return torch.stack(pooled).detach().float().cpu()

    raise ValueError(f"Unknown pool: {pool}")


class PairHiddenDataset(torch.utils.data.Dataset[dict[str, Any]]):
    def __init__(self, pairs: list[KnowledgePair]):
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        pair = self.pairs[idx]
        return {
            "id": pair.id,
            "prompt_type": pair.prompt_type,
            "prompt": pair.prompt,
            "chosen": pair.chosen,
            "rejected": pair.rejected,
        }


def collate_identity(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return batch


def split_pairs(pairs: list[KnowledgePair], split: str, limit: int | None = None) -> list[KnowledgePair]:
    filtered = [pair for pair in pairs if pair.split == split]
    if limit is not None:
        filtered = filtered[:limit]
    return filtered


def score_pairs(
    model: Any,
    tokenizer: Any,
    head: RewardHead,
    pairs: list[KnowledgePair],
    config: dict[str, Any],
    device: torch.device,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    head.eval()
    for pair in tqdm(pairs, desc="score", leave=False):
        chosen_hidden = extract_hidden(
            model,
            tokenizer,
            pair.prompt,
            pair.chosen,
            max_length=int(config.get("max_length", 1024)),
            hidden_layer=config.get("hidden_layer", "last"),
            pool=config.get("pool", "final_token"),
            rm_input=config.get("rm_input", "prompt_completion"),
            device=device,
        ).to(device)
        rejected_hidden = extract_hidden(
            model,
            tokenizer,
            pair.prompt,
            pair.rejected,
            max_length=int(config.get("max_length", 1024)),
            hidden_layer=config.get("hidden_layer", "last"),
            pool=config.get("pool", "final_token"),
            rm_input=config.get("rm_input", "prompt_completion"),
            device=device,
        ).to(device)
        with torch.no_grad():
            chosen_reward = head(chosen_hidden.unsqueeze(0)).item()
            rejected_reward = head(rejected_hidden.unsqueeze(0)).item()
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
    return rows


def score_cached_pairs(
    head: RewardHead,
    pairs: list[KnowledgePair],
    chosen_hiddens: torch.Tensor,
    rejected_hiddens: torch.Tensor,
    device: torch.device,
    *,
    batch_size: int = 32,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    head.eval()
    chosen_scores: list[float] = []
    rejected_scores: list[float] = []
    with torch.no_grad():
        for start in range(0, len(pairs), batch_size):
            chosen = chosen_hiddens[start : start + batch_size].to(device)
            rejected = rejected_hiddens[start : start + batch_size].to(device)
            chosen_scores.extend(head(chosen).detach().float().cpu().tolist())
            rejected_scores.extend(head(rejected).detach().float().cpu().tolist())
    for pair, chosen_reward, rejected_reward in zip(pairs, chosen_scores, rejected_scores):
        rows.append(
            {
                "id": pair.id,
                "split": pair.split,
                "prompt_type": pair.prompt_type,
                "domain": pair.domain,
                "category": str(pair.metadata.get("category", "")),
                "chosen_reward": float(chosen_reward),
                "rejected_reward": float(rejected_reward),
                "margin": float(chosen_reward - rejected_reward),
                "correct": chosen_reward > rejected_reward,
            }
        )
    return rows


def _hidden_cache_path(output_dir: Path, split: str, side: str) -> Path:
    return output_dir / "hidden_cache" / f"{split}_{side}.pt"


def _load_hidden_cache(output_dir: Path, split: str, pairs: list[KnowledgePair]) -> tuple[torch.Tensor, torch.Tensor] | None:
    chosen_path = _hidden_cache_path(output_dir, split, "chosen")
    rejected_path = _hidden_cache_path(output_dir, split, "rejected")
    if not chosen_path.exists() or not rejected_path.exists():
        return None
    chosen_payload = torch.load(chosen_path, map_location="cpu")
    rejected_payload = torch.load(rejected_path, map_location="cpu")
    ids = [pair.id for pair in pairs]
    if chosen_payload.get("ids") != ids or rejected_payload.get("ids") != ids:
        return None
    return chosen_payload["tensor"], rejected_payload["tensor"]


def _save_hidden_cache(
    output_dir: Path,
    split: str,
    pairs: list[KnowledgePair],
    chosen: torch.Tensor,
    rejected: torch.Tensor,
) -> None:
    cache_dir = output_dir / "hidden_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    ids = [pair.id for pair in pairs]
    torch.save({"ids": ids, "tensor": chosen.cpu()}, _hidden_cache_path(output_dir, split, "chosen"))
    torch.save({"ids": ids, "tensor": rejected.cpu()}, _hidden_cache_path(output_dir, split, "rejected"))


def precompute_pair_hiddens(
    model: Any,
    tokenizer: Any,
    pairs: list[KnowledgePair],
    config: dict[str, Any],
    device: torch.device,
    *,
    output_dir: Path,
    split_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if bool(config.get("reuse_hidden_cache", True)):
        cached = _load_hidden_cache(output_dir, split_name, pairs)
        if cached is not None:
            LOGGER.info("loaded %s hidden cache with %s pairs", split_name, len(pairs))
            return cached

    kwargs = {
        "max_length": int(config.get("max_length", 1024)),
        "hidden_layer": config.get("hidden_layer", "last"),
        "pool": config.get("pool", "final_token"),
        "rm_input": config.get("rm_input", "prompt_completion"),
        "device": device,
    }
    hidden_batch_size = int(config.get("hidden_batch_size", max(1, int(config.get("batch_size", 1)))))
    chosen_chunks: list[torch.Tensor] = []
    rejected_chunks: list[torch.Tensor] = []
    for start in tqdm(range(0, len(pairs), hidden_batch_size), desc=f"precompute {split_name}"):
        batch = pairs[start : start + hidden_batch_size]
        chosen_chunks.append(
            extract_hiddens_batch(
                model,
                tokenizer,
                [{"prompt": pair.prompt, "completion": pair.chosen} for pair in batch],
                **kwargs,
            )
        )
        rejected_chunks.append(
            extract_hiddens_batch(
                model,
                tokenizer,
                [{"prompt": pair.prompt, "completion": pair.rejected} for pair in batch],
                **kwargs,
            )
        )
    chosen = torch.cat(chosen_chunks, dim=0) if chosen_chunks else torch.empty(0)
    rejected = torch.cat(rejected_chunks, dim=0) if rejected_chunks else torch.empty(0)
    if bool(config.get("save_hidden_cache", True)):
        _save_hidden_cache(output_dir, split_name, pairs, chosen, rejected)
    return chosen, rejected


def summarize_scores(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    summary: dict[str, Any] = {"n": len(rows)}
    for key in ("all", "real", "fake"):
        subset = rows if key == "all" else [row for row in rows if row["prompt_type"] == key]
        if not subset:
            continue
        summary[key] = {
            "n": len(subset),
            "chosen_mean": float(np.mean([row["chosen_reward"] for row in subset])),
            "rejected_mean": float(np.mean([row["rejected_reward"] for row in subset])),
            "margin_mean": float(np.mean([row["margin"] for row in subset])),
            "accuracy": float(np.mean([row["correct"] for row in subset])),
        }
    return summary


def chosen_distribution(rows: list[dict[str, Any]]) -> dict[str, float]:
    values = np.array([row["chosen_reward"] for row in rows], dtype=np.float64)
    if values.size == 0:
        return {"mean": 0.0, "std": 1.0, "p10": -1.0, "p90": 1.0}
    return {
        "mean": float(values.mean()),
        "std": float(max(values.std(), 1.0e-6)),
        "p10": float(np.quantile(values, 0.1)),
        "p90": float(np.quantile(values, 0.9)),
    }


def train_reward_model(config: dict[str, Any]) -> dict[str, Any]:
    set_seed(int(config.get("seed", 0)))
    output_dir = Path(config.get("output_dir", "outputs/reward_model"))
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, output_dir / "train_config.yaml")

    tokenizer = load_tokenizer(config["model_id"])
    requested_device = config.get("device", "auto")
    device = select_torch_device(requested_device)
    model = load_causal_lm(
        config["model_id"],
        torch_dtype=model_load_dtype(config.get("torch_dtype", "auto"), device),
        device_map=model_device_map(requested_device, device),
        output_hidden_states=True,
    )
    if not hasattr(model, "hf_device_map"):
        model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    hidden_size = int(model.config.hidden_size)
    head_cfg_raw = config.get("head") or {}
    head = RewardHead(
        RewardHeadConfig(
            hidden_size=hidden_size,
            head_type=head_cfg_raw.get("type", head_cfg_raw.get("head_type", "matrix_sum")),
            scale=float(head_cfg_raw.get("scale", 0.01)),
        )
    ).to(device)

    pairs = read_jsonl(config["dataset_path"])
    train_pairs = split_pairs(pairs, "train", config.get("limit_train_examples"))
    val_pairs = split_pairs(pairs, "val", config.get("limit_eval_examples"))
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=float(config.get("learning_rate", 1.0e-4)),
        weight_decay=float(config.get("weight_decay", 0.0)),
    )

    use_precomputed = bool(config.get("precompute_hiddens", False))
    eval_pairs = val_pairs or train_pairs[: min(16, len(train_pairs))]
    train_chosen: torch.Tensor | None = None
    train_rejected: torch.Tensor | None = None
    eval_chosen: torch.Tensor | None = None
    eval_rejected: torch.Tensor | None = None

    if use_precomputed:
        train_chosen, train_rejected = precompute_pair_hiddens(
            model,
            tokenizer,
            train_pairs,
            config,
            device,
            output_dir=output_dir,
            split_name="train",
        )
        eval_chosen, eval_rejected = precompute_pair_hiddens(
            model,
            tokenizer,
            eval_pairs,
            config,
            device,
            output_dir=output_dir,
            split_name="eval",
        )
        if hasattr(torch, "mps") and device.type == "mps":
            torch.mps.empty_cache()
        loader = DataLoader(
            TensorDataset(train_chosen, train_rejected),
            batch_size=int(config.get("batch_size", 1)),
            shuffle=True,
        )
        step = 0
        for epoch in range(int(config.get("epochs", 1))):
            head.train()
            for chosen_cpu, rejected_cpu in tqdm(loader, desc=f"epoch {epoch + 1}"):
                chosen = chosen_cpu.to(device)
                rejected = rejected_cpu.to(device)
                loss = bt_loss(head(chosen), head(rejected))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if config.get("grad_clip_norm") is not None:
                    torch.nn.utils.clip_grad_norm_(head.parameters(), float(config["grad_clip_norm"]))
                optimizer.step()
                step += 1
                if step % int(config.get("eval_every_steps", 25)) == 0:
                    LOGGER.info("step=%s loss=%.4f", step, loss.item())
    else:
        loader = DataLoader(
            PairHiddenDataset(train_pairs),
            batch_size=int(config.get("batch_size", 1)),
            shuffle=True,
            collate_fn=collate_identity,
        )
        step = 0
        for epoch in range(int(config.get("epochs", 1))):
            head.train()
            for batch in tqdm(loader, desc=f"epoch {epoch + 1}"):
                chosen_hiddens = []
                rejected_hiddens = []
                for row in batch:
                    kwargs = {
                        "max_length": int(config.get("max_length", 1024)),
                        "hidden_layer": config.get("hidden_layer", "last"),
                        "pool": config.get("pool", "final_token"),
                        "rm_input": config.get("rm_input", "prompt_completion"),
                        "device": device,
                    }
                    chosen_hiddens.append(
                        extract_hidden(model, tokenizer, row["prompt"], row["chosen"], **kwargs)
                    )
                    rejected_hiddens.append(
                        extract_hidden(model, tokenizer, row["prompt"], row["rejected"], **kwargs)
                    )
                chosen = torch.stack(chosen_hiddens).to(device)
                rejected = torch.stack(rejected_hiddens).to(device)
                loss = bt_loss(head(chosen), head(rejected))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if config.get("grad_clip_norm") is not None:
                    torch.nn.utils.clip_grad_norm_(head.parameters(), float(config["grad_clip_norm"]))
                optimizer.step()
                step += 1
                if step % int(config.get("eval_every_steps", 25)) == 0:
                    LOGGER.info("step=%s loss=%.4f", step, loss.item())

    head.save(output_dir)
    if use_precomputed:
        assert train_chosen is not None and train_rejected is not None
        assert eval_chosen is not None and eval_rejected is not None
        train_rows = score_cached_pairs(head, train_pairs, train_chosen, train_rejected, device)
        val_rows = score_cached_pairs(head, eval_pairs, eval_chosen, eval_rejected, device)
    else:
        train_rows = score_pairs(model, tokenizer, head, train_pairs, config, device)
        val_rows = score_pairs(model, tokenizer, head, eval_pairs, config, device)
    raw_summary = {"train": summarize_scores(train_rows), "val": summarize_scores(val_rows)}
    calibration_cfg = config.get("score_calibration") or {}
    calibration_path = output_dir / "score_calibration.json"
    if bool(calibration_cfg.get("enabled", False)):
        calibration = fit_score_calibration(train_rows, calibration_cfg)
        with calibration_path.open("w", encoding="utf-8") as f:
            json.dump(calibration, f, indent=2)
        with (output_dir / "raw_score_summary.json").open("w", encoding="utf-8") as f:
            json.dump(raw_summary, f, indent=2)
        with (output_dir / "raw_chosen_score_stats.json").open("w", encoding="utf-8") as f:
            json.dump(chosen_distribution(train_rows), f, indent=2)
        train_rows = apply_score_calibration_to_rows(train_rows, calibration, keep_raw=True)
        val_rows = apply_score_calibration_to_rows(val_rows, calibration, keep_raw=True)
    elif calibration_path.exists():
        calibration_path.unlink()

    summary = {"train": summarize_scores(train_rows), "val": summarize_scores(val_rows)}
    stats = chosen_distribution(train_rows)
    with (output_dir / "score_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with (output_dir / "chosen_score_stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    with (output_dir / "chosen_score_stats_by_prompt_type.json").open("w", encoding="utf-8") as f:
        json.dump(chosen_score_stats_by_prompt_type(train_rows), f, indent=2)
    with (output_dir / "model_meta.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "model_id": config["model_id"],
                "hidden_layer": config.get("hidden_layer", "last"),
                "pool": config.get("pool", "final_token"),
                "rm_input": config.get("rm_input", "prompt_completion"),
                "max_length": int(config.get("max_length", 1024)),
            },
            f,
            indent=2,
        )
    LOGGER.info("saved reward model to %s", output_dir)
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train BT reward head on frozen LM hidden states.")
    parser.add_argument("--config", default="configs/reward_model.yaml")
    args = parser.parse_args(argv)
    setup_logging()
    config = load_config(args.config)
    summary = train_reward_model(config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
