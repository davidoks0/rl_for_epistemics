from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from tqdm import tqdm

from rl_epistemics.data.schema import read_jsonl
from rl_epistemics.eval.llm_judge import maybe_judge_rows
from rl_epistemics.eval.metrics import aggregate_metrics, classify_rows
from rl_epistemics.eval.plots import write_metric_plots
from rl_epistemics.models.qwen_io import apply_chat_template, load_causal_lm, load_tokenizer
from rl_epistemics.models.reward_scoring import RewardScorer
from rl_epistemics.utils.config import load_config
from rl_epistemics.utils.seed import set_seed


def _verification_fields(metadata: dict[str, Any]) -> dict[str, str]:
    verification = metadata.get("verification") if isinstance(metadata, dict) else None
    if not isinstance(verification, dict):
        return {"verification_status": "", "verification_evidence": ""}
    evidence_parts = []
    for item in verification.get("evidence") or []:
        if not isinstance(item, dict):
            continue
        source = item.get("source", "")
        query = item.get("query", "")
        note = item.get("note", "")
        evidence_parts.append(": ".join(str(part) for part in [source, query, note] if part))
    for item in verification.get("online_checks") or []:
        if not isinstance(item, dict):
            continue
        provider = item.get("provider", "")
        query = item.get("query", "")
        exists = item.get("exists")
        evidence_parts.append(f"{provider}: {query}: exists={exists}")
    return {
        "verification_status": str(verification.get("status", "")),
        "verification_evidence": " | ".join(evidence_parts[:6]),
    }


def load_eval_prompts(dataset_path: str, split: str, limit: int | None = None) -> list[dict[str, Any]]:
    pairs = [pair for pair in read_jsonl(dataset_path) if pair.split == split]
    if limit is not None:
        pairs = pairs[:limit]
    rows = []
    for pair in pairs:
        verification = _verification_fields(pair.metadata)
        rows.append(
            {
                "id": pair.id,
                "prompt_type": pair.prompt_type,
                "domain": pair.domain,
                "entity_name": pair.entity_name,
                "prompt": pair.prompt,
                "metadata_json": json.dumps(pair.metadata, ensure_ascii=False, sort_keys=True),
                **verification,
            }
        )
    return rows


def generate_completion(model: Any, tokenizer: Any, prompt: str, config: dict[str, Any]) -> str:
    text = apply_chat_template(tokenizer, prompt, add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=int(config.get("max_new_tokens", 180)),
            do_sample=True,
            temperature=float(config.get("temperature", 0.7)),
            top_p=float(config.get("top_p", 0.8)),
            top_k=int(config.get("top_k", 20)),
            pad_token_id=tokenizer.pad_token_id,
        )
    new_tokens = output_ids[0, inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def eval_policy(config: dict[str, Any]) -> dict[str, Any]:
    set_seed(int(config.get("seed", 0)))
    output_dir = Path(config.get("output_dir", "outputs/eval"))
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(config["model_id"])
    model = load_causal_lm(config["model_id"], torch_dtype="auto", device_map="auto")
    if config.get("adapter_path"):
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise ImportError("peft is required to load adapter_path.") from exc
        model = PeftModel.from_pretrained(model, config["adapter_path"])
    model.eval()

    scorer = None
    if config.get("include_reward_scores") and config.get("reward_model_dir"):
        scorer = RewardScorer(config["reward_model_dir"])

    rows = []
    for row in tqdm(
        load_eval_prompts(config["dataset_path"], config.get("split", "test"), config.get("limit")),
        desc="eval_policy",
    ):
        completion = generate_completion(model, tokenizer, row["prompt"], config)
        record = {**row, "completion": completion}
        if scorer is not None:
            record["reward"] = scorer.score(row["prompt"], completion)
        rows.append(record)

    classified = maybe_judge_rows(classify_rows(rows), config)
    pd.DataFrame(classified).to_csv(output_dir / "policy_outputs.csv", index=False)
    summary = aggregate_metrics(classified)
    with (output_dir / "policy_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    write_metric_plots(output_dir / "policy_outputs.csv", output_dir)
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate and classify policy outputs.")
    parser.add_argument("--config", default="configs/eval.yaml")
    args = parser.parse_args(argv)
    print(json.dumps(eval_policy(load_config(args.config)), indent=2))


if __name__ == "__main__":
    main()
