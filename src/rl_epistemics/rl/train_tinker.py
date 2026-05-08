from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import shutil
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from rl_epistemics.data.schema import read_jsonl
from rl_epistemics.eval.classify_outputs import classify_output
from rl_epistemics.eval.generate_eval_outputs import _verification_fields
from rl_epistemics.eval.llm_judge import maybe_judge_rows
from rl_epistemics.eval.metrics import aggregate_metrics, kalomaze_metric_aliases
from rl_epistemics.eval.plots import write_metric_plots
from rl_epistemics.models.qwen_io import apply_chat_template
from rl_epistemics.models.reward_scoring import RewardScorer
from rl_epistemics.rl.reward_transforms import (
    build_reward_transform,
    load_score_stats,
    resolve_reward_transform_config,
)
from rl_epistemics.utils.config import load_config, save_config
from rl_epistemics.utils.seed import set_seed


def _require_tinker() -> Any:
    try:
        import tinker
    except ImportError as exc:
        raise ImportError("Tinker RL requires `tinker`. Install with `uv pip install tinker`.") from exc
    return tinker


def _ensure_tinker_api_key(config: dict[str, Any]) -> str:
    if os.environ.get("TINKER_API_KEY"):
        return "env:TINKER_API_KEY"
    key_file = config.get("tinker_api_key_file") or os.environ.get("TINKER_API_KEY_FILE")
    if key_file:
        path = Path(str(key_file)).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"tinker_api_key_file does not exist: {path}")
        key = path.read_text(encoding="utf-8").strip()
        if not key:
            raise ValueError(f"tinker_api_key_file is empty: {path}")
        os.environ["TINKER_API_KEY"] = key
        return f"file:{path}"
    raise EnvironmentError(
        "TINKER_API_KEY is not set. Put it in the Modal secret for Modal runs, "
        "export it locally, or pass --tinker-api-key-file for a local fallback."
    )


def _tinker_type(tinker: Any, name: str) -> Any:
    if hasattr(tinker, name):
        return getattr(tinker, name)
    if hasattr(tinker, "types") and hasattr(tinker.types, name):
        return getattr(tinker.types, name)
    raise AttributeError(f"Tinker SDK does not expose {name}.")


def _tensor_data(tinker: Any, values: list[int] | list[float], dtype: torch.dtype) -> Any:
    tensor_data = _tinker_type(tinker, "TensorData")
    return tensor_data.from_torch(torch.tensor(values, dtype=dtype))


def _model_input_from_ints(tinker: Any, tokens: list[int]) -> Any:
    model_input = _tinker_type(tinker, "ModelInput")
    try:
        return model_input.from_ints(tokens=tokens)
    except TypeError:
        return model_input.from_ints(tokens)


async def _await_api_future(value: Any) -> Any:
    if hasattr(value, "result_async"):
        return await value.result_async()
    if hasattr(value, "result"):
        return value.result()
    return value


async def _maybe_await(value: Any) -> Any:
    if hasattr(value, "__await__"):
        return await value
    return value


def balanced_prompt_rows(dataset_path: str | Path, split: str = "train") -> list[dict[str, Any]]:
    pairs = [pair for pair in read_jsonl(dataset_path) if pair.split == split]
    real = [pair for pair in pairs if pair.prompt_type == "real"]
    fake = [pair for pair in pairs if pair.prompt_type == "fake"]
    n = min(len(real), len(fake))
    selected = pairs if n == 0 else real[:n] + fake[:n]
    return [
        {
            "id": pair.id,
            "prompt": pair.prompt,
            "prompt_type": pair.prompt_type,
            "domain": pair.domain,
            "entity_name": pair.entity_name,
        }
        for pair in selected
    ]


def _sample_balanced_batch(
    rows: list[dict[str, Any]],
    step: int,
    batch_size: int,
) -> list[dict[str, Any]]:
    if not rows:
        return []
    real = [row for row in rows if row.get("prompt_type") == "real"]
    fake = [row for row in rows if row.get("prompt_type") == "fake"]
    if not real or not fake or batch_size < 2:
        return [rows[(step * batch_size + offset) % len(rows)] for offset in range(batch_size)]
    half = batch_size // 2
    batch = [real[(step * half + i) % len(real)] for i in range(half)]
    batch.extend(fake[(step * (batch_size - half) + i) % len(fake)] for i in range(batch_size - half))
    return batch


async def _resolve_base_model(
    service_client: Any,
    requested: str,
    fallbacks: list[str],
    output_dir: Path,
) -> tuple[str, list[str]]:
    capabilities = await _maybe_await(service_client.get_server_capabilities_async())
    supported_raw = getattr(capabilities, "supported_models", []) or []
    supported = [
        str(getattr(model, "model_name", getattr(model, "name", model))) for model in supported_raw
    ]
    with (output_dir / "tinker_supported_models.json").open("w", encoding="utf-8") as f:
        json.dump({"supported_models": supported}, f, indent=2)
    if requested in supported:
        return requested, supported
    for fallback in fallbacks:
        if fallback in supported:
            return fallback, supported
    raise ValueError(
        f"Tinker does not report support for {requested!r}. "
        f"Checked fallbacks {fallbacks!r}. See {output_dir / 'tinker_supported_models.json'}."
    )


def _build_renderer(config: dict[str, Any], tokenizer: Any, base_model: str) -> tuple[Any | None, Any | None]:
    renderer_name = config.get("renderer") or "qwen3_disable_thinking"
    try:
        from tinker_cookbook import renderers
        from tinker_cookbook.renderers import get_text_content

        try:
            renderer = renderers.get_renderer(renderer_name, tokenizer, model_name=base_model)
        except TypeError:
            renderer = renderers.get_renderer(renderer_name, tokenizer)
        return renderer, get_text_content
    except Exception:
        return None, None


def _prompt_messages(prompt: str, system_prompt: str | None = None) -> list[dict[str, str]]:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return messages


def _build_prompt_input(
    tinker: Any,
    tokenizer: Any,
    renderer: Any | None,
    prompt: str,
    system_prompt: str | None,
) -> Any:
    messages = _prompt_messages(prompt, system_prompt)
    if renderer is not None:
        return renderer.build_generation_prompt(messages)
    text = apply_chat_template(tokenizer, prompt, add_generation_prompt=True, enable_thinking=False)
    return _model_input_from_ints(tinker, tokenizer.encode(text))


def _prompt_tokens(prompt_input: Any) -> list[int]:
    if hasattr(prompt_input, "to_ints"):
        return list(prompt_input.to_ints())
    if hasattr(prompt_input, "tokens"):
        return list(prompt_input.tokens)
    raise TypeError("Cannot recover token IDs from Tinker ModelInput.")


def _stop_sequences(renderer: Any | None, tokenizer: Any, config: dict[str, Any]) -> list[Any]:
    if config.get("stop") is not None:
        return list(config["stop"])
    if renderer is not None and hasattr(renderer, "get_stop_sequences"):
        return list(renderer.get_stop_sequences())
    stops = ["<|im_end|>"]
    if getattr(tokenizer, "eos_token", None):
        stops.append(tokenizer.eos_token)
    return stops


def _decode_completion(
    tokenizer: Any,
    renderer: Any | None,
    get_text_content: Any | None,
    tokens: list[int],
) -> str:
    if renderer is not None and get_text_content is not None and hasattr(renderer, "parse_response"):
        try:
            parsed, _ = renderer.parse_response(tokens)
            return str(get_text_content(parsed)).strip()
        except Exception:
            pass
    return tokenizer.decode(tokens, skip_special_tokens=True).replace("<|im_end|>", "").strip()


async def _completion_logprobs(
    tinker: Any,
    sampling_client: Any,
    prompt_tokens: list[int],
    completion_tokens: list[int],
    sequence: Any,
    compute_if_missing: bool,
) -> tuple[list[float], bool]:
    seq_logprobs = getattr(sequence, "logprobs", None)
    if seq_logprobs is not None and len(seq_logprobs) == len(completion_tokens):
        return [float(value) for value in seq_logprobs], False
    if compute_if_missing:
        model_input = _model_input_from_ints(tinker, prompt_tokens + completion_tokens)
        logprobs = await _maybe_await(sampling_client.compute_logprobs_async(model_input))
        values = [0.0 if value is None else float(value) for value in logprobs]
        return values[-len(completion_tokens) :], True
    return [0.0] * len(completion_tokens), True


def _make_adam_params(tinker: Any, config: dict[str, Any]) -> Any:
    adam_params = _tinker_type(tinker, "AdamParams")
    kwargs = {
        "learning_rate": float(config.get("learning_rate", 5.0e-6)),
        "weight_decay": float(config.get("weight_decay", 0.0)),
    }
    for key in ("beta1", "beta2", "eps"):
        if key in config:
            kwargs[key] = float(config[key])
    try:
        return adam_params(**kwargs)
    except TypeError:
        return adam_params(
            learning_rate=kwargs["learning_rate"],
            weight_decay=kwargs["weight_decay"],
        )


async def _save_sampler(
    tinker: Any,
    service_client: Any,
    training_client: Any,
    name: str,
) -> tuple[str | None, Any]:
    if hasattr(training_client, "save_weights_for_sampler_async"):
        save_future = await _maybe_await(training_client.save_weights_for_sampler_async(name))
        save_result = await _await_api_future(save_future)
        path = getattr(save_result, "path", None)
        if path:
            sampling_client = await _maybe_await(service_client.create_sampling_client_async(model_path=path))
            return str(path), sampling_client
    sampling_client = await _maybe_await(training_client.save_weights_and_get_sampling_client_async(name))
    return None, sampling_client


def _safe_mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _prompt_rows_for_split(
    dataset_path: str | Path,
    *,
    split: str,
    limit: int | None = None,
    balanced: bool = False,
) -> list[dict[str, Any]]:
    pairs = [pair for pair in read_jsonl(dataset_path) if pair.split == split]
    if balanced:
        real = [pair for pair in pairs if pair.prompt_type == "real"]
        fake = [pair for pair in pairs if pair.prompt_type == "fake"]
        n = min(len(real), len(fake))
        if n:
            pairs = real[:n] + fake[:n]
    if limit is not None:
        pairs = pairs[:limit]
    return [
        {
            "id": pair.id,
            "prompt": pair.prompt,
            "prompt_type": pair.prompt_type,
            "domain": pair.domain,
            "entity_name": pair.entity_name,
            "metadata": pair.metadata,
        }
        for pair in pairs
    ]


def _heldout_eval_enabled(config: dict[str, Any]) -> bool:
    return bool((config.get("heldout_eval") or {}).get("enabled", False))


def _should_run_heldout_eval(step: int, config: dict[str, Any]) -> bool:
    heldout = config.get("heldout_eval") or {}
    if not heldout.get("enabled", False):
        return False
    if step == 0 and bool(heldout.get("at_step_zero", True)):
        return True
    every = int(heldout.get("every_steps", 0) or 0)
    return every > 0 and step > 0 and step % every == 0


async def _run_heldout_eval(
    *,
    tinker: Any,
    tokenizer: Any,
    renderer: Any | None,
    get_text_content: Any | None,
    sampling_client: Any,
    sampling_params: Any,
    scorer: RewardScorer | None,
    transform: Any | None,
    rows: list[dict[str, Any]],
    output_dir: Path,
    step: int,
    split: str,
    sampler_path: str | None,
    system_prompt: str | None,
    reward_batch_size: int,
    include_samples: bool,
) -> dict[str, Any]:
    if not rows:
        return {"step": step, "split": split, "n": 0, "sampler_path": sampler_path}
    prompt_inputs = [
        _build_prompt_input(tinker, tokenizer, renderer, row["prompt"], system_prompt)
        for row in rows
    ]
    sample_results = await asyncio.gather(
        *[
            sampling_client.sample_async(
                prompt=prompt_input,
                num_samples=1,
                sampling_params=sampling_params,
            )
            for prompt_input in prompt_inputs
        ]
    )
    sampled_rows: list[dict[str, Any]] = []
    for prompt_row, sample_result in zip(rows, sample_results):
        sequence = sample_result.sequences[0]
        tokens = list(getattr(sequence, "tokens", []) or [])
        completion = _decode_completion(tokenizer, renderer, get_text_content, tokens)
        sampled_rows.append(
            {
                "step": step,
                "split": split,
                "id": prompt_row["id"],
                "prompt_type": prompt_row["prompt_type"],
                "domain": prompt_row["domain"],
                "entity_name": prompt_row["entity_name"],
                "prompt": prompt_row["prompt"],
                "completion": completion,
                "completion_tokens": tokens,
                "stop_reason": str(getattr(sequence, "stop_reason", "")),
                "sampler_path": sampler_path,
            }
        )
    if scorer is not None:
        raw_scores = scorer.score_many(
            [
                {
                    "prompt": row["prompt"],
                    "completion": row["completion"],
                    "prompt_type": row.get("prompt_type"),
                }
                for row in sampled_rows
            ],
            batch_size=reward_batch_size,
        )
        for row, raw_score in zip(sampled_rows, raw_scores):
            row["raw_reward"] = raw_score
            row["reward"] = raw_score
            if transform is not None:
                row["transformed_reward"] = transform(raw_score, row.get("prompt_type"))

    classified = [
        {
            **row,
            **classify_output(
                row["prompt_type"],
                row["completion"],
                prompt=row.get("prompt"),
                domain=row.get("domain"),
            ),
        }
        for row in sampled_rows
    ]
    summary = aggregate_metrics(classified)
    aliases = kalomaze_metric_aliases(summary)
    metric_row = {
        "step": step,
        "split": split,
        "n": summary.get("n", 0),
        "sampler_path": sampler_path,
        **aliases,
        "raw_reward_mean": summary.get("reward_mean"),
        "fake_n": summary.get("fake_n", 0),
        "real_n": summary.get("real_n", 0),
    }
    _write_jsonl(output_dir / "heldout_metrics.jsonl", [metric_row])
    if include_samples:
        _write_jsonl(output_dir / "heldout_samples.jsonl", classified)
    return metric_row


def _enforce_eval_quality_gate(config: dict[str, Any]) -> dict[str, Any]:
    gate = config.get("eval_quality_gate") or {}
    max_steps = int(config.get("max_steps", 0))
    threshold = gate.get("required_for_max_steps_over")
    auto_required = threshold is not None and max_steps > int(threshold)
    if not gate.get("enabled") and not auto_required:
        return {"enabled": False, "max_steps": max_steps, "auto_required": False}
    path = Path(gate.get("adjudication_path", "outputs/blog/manual_review_queue.csv"))
    if not path.exists():
        raise FileNotFoundError(
            "eval_quality_gate is required, but the adjudication file is missing: "
            f"{path}. Run policy eval plus manual or LLM adjudication before longer RL."
    )
    df = pd.read_csv(path)
    label_columns = [str(col) for col in (gate.get("label_columns") or [])]
    label_col = str(gate.get("label_column", "manual_label"))
    if label_col not in label_columns:
        label_columns.insert(0, label_col)
    present_label_columns = [col for col in label_columns if col in df.columns]
    if not present_label_columns:
        raise ValueError(
            f"Adjudication file {path} does not have any configured label columns: {label_columns!r}."
        )
    label_series = df[present_label_columns[0]].fillna("").astype(str).str.strip()
    for alternate_col in present_label_columns[1:]:
        alternate = df[alternate_col].fillna("").astype(str).str.strip()
        label_series = label_series.where(
            (label_series != "") & (label_series != "TODO"),
            alternate,
        )
    labels = label_series
    reviewed = labels[(labels != "") & (labels != "TODO")]
    min_reviewed = int(gate.get("min_reviewed", 30))
    if len(reviewed) < min_reviewed:
        raise ValueError(
            f"eval_quality_gate requires at least {min_reviewed} reviewed labels; found {len(reviewed)}."
        )
    allowed = set(gate.get("allowed_labels") or [])
    invalid = sorted({label for label in reviewed if allowed and label not in allowed})
    if invalid:
        raise ValueError(f"Adjudication labels outside allowed taxonomy: {invalid}")
    reviewed_df = df.loc[reviewed.index].copy()
    prompt_label_allowed = {
        "fake": {
            "clean_uncertainty",
            "false_premise_correction",
            "hedged_confabulation",
            "confident_confabulation",
            "unclear",
        },
        "real": {"real_substantive_answer", "false_refusal", "unclear"},
    }
    if "prompt_type" in reviewed_df.columns:
        context_invalid = []
        for idx, row in reviewed_df.iterrows():
            prompt_type = str(row.get("prompt_type", ""))
            label = str(labels.loc[idx])
            if prompt_type in prompt_label_allowed and label not in prompt_label_allowed[prompt_type]:
                context_invalid.append({"id": row.get("id", idx), "prompt_type": prompt_type, "label": label})
        if context_invalid:
            raise ValueError(
                "eval_quality_gate found labels incompatible with prompt_type: "
                f"{context_invalid[:10]}"
            )
    required_prompt_type_counts = gate.get("required_prompt_type_counts") or {}
    if required_prompt_type_counts:
        if "prompt_type" not in reviewed_df.columns:
            raise ValueError("eval_quality_gate.required_prompt_type_counts needs a prompt_type column.")
        prompt_counts = reviewed_df["prompt_type"].fillna("").astype(str).value_counts().to_dict()
        missing_counts = {
            prompt_type: int(required) - int(prompt_counts.get(prompt_type, 0))
            for prompt_type, required in required_prompt_type_counts.items()
            if int(prompt_counts.get(prompt_type, 0)) < int(required)
        }
        if missing_counts:
            raise ValueError(
                "eval_quality_gate reviewed labels lack required prompt_type coverage: "
                f"{missing_counts}. Current counts: {prompt_counts}."
            )
    return {
        "enabled": True,
        "auto_required": auto_required,
        "max_steps": max_steps,
        "required_for_max_steps_over": threshold,
        "adjudication_path": str(path),
        "reviewed": int(len(reviewed)),
        "label_column": present_label_columns[0],
        "label_columns": present_label_columns,
        "required_prompt_type_counts": required_prompt_type_counts,
    }


async def train_tinker_rl_async(config: dict[str, Any]) -> dict[str, Any]:
    api_key_source = _ensure_tinker_api_key(config)

    tinker = _require_tinker()
    set_seed(int(config.get("seed", 0)))
    quality_gate = _enforce_eval_quality_gate(config)

    output_dir = Path(config.get("output_dir", "outputs/tinker_rl"))
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, output_dir / "train_config.yaml")
    dataset_path = Path(config["dataset_path"])
    if dataset_path.exists():
        copied_dataset = output_dir / "pairs.jsonl"
        if dataset_path.resolve() != copied_dataset.resolve():
            shutil.copy2(dataset_path, copied_dataset)

    requested_model = str(config.get("base_model", config.get("model_id", "Qwen/Qwen3-8B")))
    fallbacks = list(config.get("model_fallbacks") or ["Qwen/Qwen3-8B", "Qwen/Qwen3-8B-Base"])
    rank = int(config.get("rank", (config.get("lora") or {}).get("r", 32)))
    run_name = str(config.get("run_name") or f"tinker-rl-{int(time.time())}")

    service_client = tinker.ServiceClient(user_metadata={"repo": "rl_for_epistemics", "run": run_name})
    base_model, supported_models = await _resolve_base_model(
        service_client, requested_model, fallbacks, output_dir
    )
    training_client = await _maybe_await(
        service_client.create_lora_training_client_async(
            base_model=base_model,
            rank=rank,
            seed=int(config.get("seed", 0)),
        )
    )
    tokenizer = training_client.get_tokenizer()
    renderer, get_text_content = _build_renderer(config, tokenizer, base_model)

    rows = balanced_prompt_rows(dataset_path, split=str(config.get("split", "train")))
    limit_prompts = config.get("limit_prompts")
    if limit_prompts is not None:
        rows = rows[: int(limit_prompts)]
    if not rows:
        raise ValueError(f"No training prompts found in {dataset_path}.")

    heldout_cfg = config.get("heldout_eval") or {}
    heldout_rows: list[dict[str, Any]] = []
    heldout_split = str(heldout_cfg.get("split", "test"))
    if _heldout_eval_enabled(config):
        heldout_rows = _prompt_rows_for_split(
            dataset_path,
            split=heldout_split,
            limit=heldout_cfg.get("limit"),
            balanced=bool(heldout_cfg.get("balanced", True)),
        )
        if not heldout_rows:
            raise ValueError(
                f"heldout_eval is enabled, but no prompts were found for split {heldout_split!r} "
                f"in {dataset_path}."
            )

    reward_model_dir = Path(config["reward_model_dir"])
    transform_config = dict(config.get("reward_transform") or {})
    std_floor = float(transform_config.get("std_floor", 0.1))
    stats = load_score_stats(
        reward_model_dir,
        std_floor=std_floor,
    )
    transform_config = resolve_reward_transform_config(
        transform_config,
        reward_model_dir,
        std_floor=std_floor,
    )
    transform = build_reward_transform(transform_config, stats)
    scorer = RewardScorer(reward_model_dir, device=config.get("reward_device"))

    sampling_params = _tinker_type(tinker, "SamplingParams")(
        max_tokens=int(config.get("max_completion_tokens", config.get("max_new_tokens", 160))),
        temperature=float(config.get("temperature", 0.7)),
        top_p=float(config.get("top_p", 0.8)),
        top_k=int(config.get("top_k", -1)),
        stop=_stop_sequences(renderer, tokenizer, config),
    )
    adam_params = _make_adam_params(tinker, config)

    metrics_path = output_dir / "metrics.jsonl"
    samples_path = output_dir / "sample_completions.jsonl"
    if not bool(config.get("append_metrics", False)):
        stale_paths = [metrics_path, samples_path]
        if _heldout_eval_enabled(config):
            stale_paths.extend([output_dir / "heldout_metrics.jsonl", output_dir / "heldout_samples.jsonl"])
        for stale_path in stale_paths:
            if stale_path.exists():
                stale_path.unlink()
    n_steps = int(config.get("max_steps", 20))
    batch_size = int(config.get("batch_size", 4))
    group_size = int(config.get("group_size", config.get("num_generations", 4)))
    degenerate_eps = float(config.get("degenerate_eps", 1.0e-8))
    compute_missing_logprobs = bool(config.get("compute_logprobs_if_missing", True))
    include_loss_weights = bool(config.get("include_loss_weights", False))

    last_sampler_path: str | None = None
    for step in range(n_steps):
        batch_rows = _sample_balanced_batch(rows, step, batch_size)
        sampler_name = f"{run_name}-step-{step:04d}"
        sampler_path, sampling_client = await _save_sampler(
            tinker, service_client, training_client, sampler_name
        )
        last_sampler_path = sampler_path or last_sampler_path

        if _should_run_heldout_eval(step, config):
            heldout_metric_row = await _run_heldout_eval(
                tinker=tinker,
                tokenizer=tokenizer,
                renderer=renderer,
                get_text_content=get_text_content,
                sampling_client=sampling_client,
                sampling_params=sampling_params,
                scorer=scorer if bool(config.get("include_reward_scores", True)) else None,
                transform=transform,
                rows=heldout_rows,
                output_dir=output_dir,
                step=step,
                split=heldout_split,
                sampler_path=sampler_path or last_sampler_path,
                system_prompt=config.get("system_prompt"),
                reward_batch_size=int(config.get("reward_batch_size", 4)),
                include_samples=bool(heldout_cfg.get("include_samples", True)),
            )
            print(json.dumps({"heldout_eval": heldout_metric_row}, sort_keys=True))

        prompts = [
            _build_prompt_input(
                tinker,
                tokenizer,
                renderer,
                row["prompt"],
                config.get("system_prompt"),
            )
            for row in batch_rows
        ]
        sample_results = await asyncio.gather(
            *[
                sampling_client.sample_async(
                    prompt=prompt,
                    num_samples=group_size,
                    sampling_params=sampling_params,
                )
                for prompt in prompts
            ]
        )

        scored_rows: list[dict[str, Any]] = []
        prompt_tokens_by_idx = [_prompt_tokens(prompt) for prompt in prompts]
        for prompt_idx, (sample_result, prompt_row) in enumerate(zip(sample_results, batch_rows)):
            for group_idx, sequence in enumerate(sample_result.sequences):
                tokens = list(getattr(sequence, "tokens", []) or [])
                completion = _decode_completion(tokenizer, renderer, get_text_content, tokens)
                scored_rows.append(
                    {
                        "step": step,
                        "prompt_index": prompt_idx,
                        "group_index": group_idx,
                        "prompt_id": prompt_row["id"],
                        "prompt_type": prompt_row["prompt_type"],
                        "prompt": prompt_row["prompt"],
                        "completion": completion,
                        "completion_tokens": tokens,
                        "stop_reason": str(getattr(sequence, "stop_reason", "")),
                    }
                )

        raw_scores = scorer.score_many(
            [
                {
                    "prompt": row["prompt"],
                    "completion": row["completion"],
                    "prompt_type": row.get("prompt_type"),
                }
                for row in scored_rows
            ],
            batch_size=int(config.get("reward_batch_size", 4)),
        )
        for row, raw_score in zip(scored_rows, raw_scores):
            row["raw_reward"] = raw_score
            row["transformed_reward"] = transform(raw_score, row.get("prompt_type"))
            row.update(classify_output(row["prompt_type"], row["completion"]))

        datums: list[Any] = []
        group_rewards: list[float] = []
        group_transformed: list[float] = []
        n_degenerate = 0
        missing_logprob_groups = 0
        completion_logprob_means: list[float] = []
        for prompt_idx, prompt_row in enumerate(batch_rows):
            group = [row for row in scored_rows if row["prompt_index"] == prompt_idx]
            rewards = [float(row["transformed_reward"]) for row in group]
            raw_rewards = [float(row["raw_reward"]) for row in group]
            group_rewards.append(_safe_mean(raw_rewards))
            group_transformed.append(_safe_mean(rewards))
            if not group or max(rewards) - min(rewards) <= degenerate_eps:
                n_degenerate += 1
                continue
            mean_reward = _safe_mean(rewards)
            advantages = [reward - mean_reward for reward in rewards]
            prompt_tokens = prompt_tokens_by_idx[prompt_idx]
            ob_len = max(0, len(prompt_tokens) - 1)
            for row, advantage in zip(group, advantages):
                completion_tokens = [int(tok) for tok in row["completion_tokens"]]
                if not completion_tokens or len(prompt_tokens) < 2:
                    continue
                sequence = sample_results[prompt_idx].sequences[int(row["group_index"])]
                logprobs, was_missing = await _completion_logprobs(
                    tinker,
                    sampling_client,
                    prompt_tokens,
                    completion_tokens,
                    sequence,
                    compute_missing_logprobs,
                )
                if was_missing:
                    missing_logprob_groups += 1
                completion_logprob_means.extend(logprobs)
                model_input = _model_input_from_ints(tinker, prompt_tokens + completion_tokens[:-1])
                target_tokens = [0] * ob_len + completion_tokens
                padded_logprobs = [0.0] * ob_len + logprobs
                padded_advantages = [0.0] * ob_len + [float(advantage)] * len(completion_tokens)
                loss_inputs = {
                    "target_tokens": _tensor_data(tinker, target_tokens, torch.long),
                    "logprobs": _tensor_data(tinker, padded_logprobs, torch.float32),
                    "advantages": _tensor_data(tinker, padded_advantages, torch.float32),
                }
                if include_loss_weights:
                    weights = [0.0] * ob_len + [1.0] * len(completion_tokens)
                    loss_inputs["weights"] = _tensor_data(tinker, weights, torch.float32)
                datum = _tinker_type(tinker, "Datum")(
                    model_input=model_input,
                    loss_fn_inputs=loss_inputs,
                )
                datums.append(datum)

        if datums:
            fwdbwd_future = await _maybe_await(
                training_client.forward_backward_async(datums, loss_fn="importance_sampling")
            )
            optim_future = await _maybe_await(training_client.optim_step_async(adam_params))
            fwdbwd_result = await _await_api_future(fwdbwd_future)
            optim_result = await _await_api_future(optim_future)
            loss = float(getattr(fwdbwd_result, "loss", math.nan))
            result_metrics = getattr(fwdbwd_result, "metrics", {}) or {}
            if not math.isfinite(loss) and "loss:sum" in result_metrics:
                loss = float(result_metrics["loss:sum"])
        else:
            optim_result = None
            loss = math.nan
            result_metrics = {}

        classified_metrics = aggregate_metrics(scored_rows)
        kalomaze_aliases = kalomaze_metric_aliases(classified_metrics)
        metric_row = {
            "step": step,
            "base_model": base_model,
            "rank": rank,
            "sampler_path": sampler_path,
            **kalomaze_aliases,
            "raw_reward_mean": _safe_mean([float(row["raw_reward"]) for row in scored_rows]),
            "transformed_reward_mean": _safe_mean(
                [float(row["transformed_reward"]) for row in scored_rows]
            ),
            "group_raw_reward_mean": _safe_mean(group_rewards),
            "group_transformed_reward_mean": _safe_mean(group_transformed),
            "fake_confabulates": classified_metrics.get("fake_fake_confabulates", 0.0),
            "real_false_refusal": classified_metrics.get("real_real_false_refusal", 0.0),
            "fake_refuses_or_flags": classified_metrics.get("fake_fake_refuses_or_flags", 0.0),
            "fake_hedges_but_engages": classified_metrics.get(
                "fake_fake_hedges_but_engages", 0.0
            ),
            "real_answers_substantively": classified_metrics.get(
                "real_real_answers_substantively", 0.0
            ),
            "mean_completion_logprob": _safe_mean(completion_logprob_means),
            "mean_completion_tokens": _safe_mean(
                [float(len(row["completion_tokens"])) for row in scored_rows]
            ),
            "degenerate_group_fraction": n_degenerate / max(len(batch_rows), 1),
            "missing_logprob_count": missing_logprob_groups,
            "datums": len(datums),
            "loss": loss,
            "tinker_metrics": result_metrics,
            "optim_result": repr(optim_result) if optim_result is not None else None,
        }
        _write_jsonl(metrics_path, [metric_row])
        _write_jsonl(samples_path, scored_rows)
        print(json.dumps(metric_row, sort_keys=True))

    final_state_path: str | None = None
    if hasattr(training_client, "save_state_async"):
        save_future = await _maybe_await(training_client.save_state_async(f"{run_name}-final"))
        save_result = await _await_api_future(save_future)
        final_state_path = getattr(save_result, "path", None)
    final_sampler_path, final_sampling_client = await _save_sampler(
        tinker, service_client, training_client, f"{run_name}-final-sampler"
    )
    if final_sampler_path:
        last_sampler_path = final_sampler_path

    if _heldout_eval_enabled(config) and bool(heldout_cfg.get("final", True)):
        heldout_metric_row = await _run_heldout_eval(
            tinker=tinker,
            tokenizer=tokenizer,
            renderer=renderer,
            get_text_content=get_text_content,
            sampling_client=final_sampling_client,
            sampling_params=sampling_params,
            scorer=scorer if bool(config.get("include_reward_scores", True)) else None,
            transform=transform,
            rows=heldout_rows,
            output_dir=output_dir,
            step=n_steps,
            split=heldout_split,
            sampler_path=last_sampler_path,
            system_prompt=config.get("system_prompt"),
            reward_batch_size=int(config.get("reward_batch_size", 4)),
            include_samples=bool(heldout_cfg.get("include_samples", True)),
        )
        print(json.dumps({"heldout_eval": heldout_metric_row}, sort_keys=True))

    meta = {
        "run_name": run_name,
        "requested_base_model": requested_model,
        "base_model": base_model,
        "rank": rank,
        "output_dir": str(output_dir),
        "quality_gate": quality_gate,
        "api_key_source": api_key_source,
        "final_state_path": final_state_path,
        "final_sampler_path": last_sampler_path,
        "supported_models_checked": supported_models,
        "metrics_path": str(metrics_path),
        "samples_path": str(samples_path),
        "heldout_metrics_path": str(output_dir / "heldout_metrics.jsonl")
        if _heldout_eval_enabled(config)
        else None,
        "heldout_samples_path": str(output_dir / "heldout_samples.jsonl")
        if _heldout_eval_enabled(config)
        else None,
    }
    with (output_dir / "run_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta


def train_tinker_rl(config: dict[str, Any]) -> dict[str, Any]:
    return asyncio.run(train_tinker_rl_async(config))


async def eval_policy_tinker_async(config: dict[str, Any]) -> dict[str, Any]:
    _ensure_tinker_api_key(config)

    tinker = _require_tinker()
    set_seed(int(config.get("seed", 0)))
    output_dir = Path(config.get("output_dir", "outputs/eval_tinker"))
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, output_dir / "eval_config.yaml")

    service_client = tinker.ServiceClient(user_metadata={"repo": "rl_for_epistemics", "mode": "eval"})
    model_path = config.get("model_path")
    if model_path is None and config.get("run_meta_path"):
        with Path(config["run_meta_path"]).open("r", encoding="utf-8") as f:
            model_path = json.load(f).get("final_sampler_path")
    base_model = str(config.get("base_model", config.get("model_id", "Qwen/Qwen3-8B")))

    if model_path:
        sampling_client = await _maybe_await(service_client.create_sampling_client_async(model_path=model_path))
    else:
        fallbacks = list(config.get("model_fallbacks") or ["Qwen/Qwen3-8B", "Qwen/Qwen3-8B-Base"])
        base_model, _ = await _resolve_base_model(service_client, base_model, fallbacks, output_dir)
        sampling_client = await _maybe_await(service_client.create_sampling_client_async(base_model=base_model))

    tokenizer = sampling_client.get_tokenizer()
    renderer, get_text_content = _build_renderer(config, tokenizer, base_model)
    sampling_params = _tinker_type(tinker, "SamplingParams")(
        max_tokens=int(config.get("max_new_tokens", config.get("max_completion_tokens", 180))),
        temperature=float(config.get("temperature", 0.7)),
        top_p=float(config.get("top_p", 0.8)),
        top_k=int(config.get("top_k", -1)),
        stop=_stop_sequences(renderer, tokenizer, config),
    )

    prompts = [
        pair
        for pair in read_jsonl(config["dataset_path"])
        if pair.split == str(config.get("split", "test"))
    ]
    if not prompts and config.get("split") == "test":
        prompts = [pair for pair in read_jsonl(config["dataset_path"]) if pair.split == "ood"]
    if config.get("limit"):
        prompts = prompts[: int(config["limit"])]

    prompt_inputs = [
        _build_prompt_input(tinker, tokenizer, renderer, pair.prompt, config.get("system_prompt"))
        for pair in prompts
    ]
    sample_results = await asyncio.gather(
        *[
            sampling_client.sample_async(
                prompt=prompt_input,
                num_samples=1,
                sampling_params=sampling_params,
            )
            for prompt_input in prompt_inputs
        ]
    )
    rows: list[dict[str, Any]] = []
    for pair, sample_result in zip(prompts, sample_results):
        sequence = sample_result.sequences[0]
        tokens = list(sequence.tokens)
        completion = _decode_completion(tokenizer, renderer, get_text_content, tokens)
        verification = _verification_fields(pair.metadata)
        rows.append(
            {
                "id": pair.id,
                "split": pair.split,
                "prompt_type": pair.prompt_type,
                "domain": pair.domain,
                "entity_name": pair.entity_name,
                "prompt": pair.prompt,
                "metadata_json": json.dumps(pair.metadata, ensure_ascii=False, sort_keys=True),
                **verification,
                "completion": completion,
                "stop_reason": str(getattr(sequence, "stop_reason", "")),
                "model_path": model_path,
                "base_model": base_model,
            }
        )

    if config.get("include_reward_scores") and config.get("reward_model_dir"):
        scorer = RewardScorer(config["reward_model_dir"], device=config.get("reward_device"))
        rewards = scorer.score_many(
            [
                {
                    "prompt": row["prompt"],
                    "completion": row["completion"],
                    "prompt_type": row.get("prompt_type"),
                }
                for row in rows
            ],
            batch_size=int(config.get("reward_batch_size", 4)),
        )
        for row, reward in zip(rows, rewards):
            row["reward"] = reward

    classified = [
        {
            **row,
            **classify_output(
                row["prompt_type"],
                row["completion"],
                prompt=row.get("prompt"),
                domain=row.get("domain"),
            ),
        }
        for row in rows
    ]
    classified = maybe_judge_rows(classified, config)
    pd.DataFrame(classified).to_csv(output_dir / "policy_outputs.csv", index=False)
    summary = aggregate_metrics(classified)
    with (output_dir / "policy_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    write_metric_plots(output_dir / "policy_outputs.csv", output_dir)
    _write_manual_review(
        classified,
        Path(config.get("manual_review_output", "outputs/blog/manual_review_samples.md")),
        limit=int(config.get("manual_review_limit", 50)),
    )
    return summary


def eval_policy_tinker(config: dict[str, Any]) -> dict[str, Any]:
    return asyncio.run(eval_policy_tinker_async(config))


def _write_manual_review(rows: list[dict[str, Any]], path: Path, limit: int = 50) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# Manual Review Samples\n\n")
        f.write(
            "Review labels for fake-name artifacts, refusal-style artifacts, overrefusal, "
            "and false-premise versus fake-entity confusion.\n\n"
        )
        for idx, row in enumerate(rows[:limit], start=1):
            f.write(f"## {idx}. {row.get('id', '')} [{row.get('prompt_type', '')}]\n\n")
            f.write(f"Prompt: {row.get('prompt', '')}\n\n")
            f.write(f"Heuristic label: {row.get('label', '')}\n\n")
            f.write(f"Completion:\n\n{row.get('completion', '')}\n\n")
            f.write("Manual notes: TODO\n\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train or evaluate a Tinker LoRA RL policy.")
    parser.add_argument("--config", default="configs/tinker_rl.yaml")
    parser.add_argument("--eval", action="store_true", help="Evaluate a Tinker sampler instead of training.")
    parser.add_argument(
        "--tinker-api-key-file",
        help="Read TINKER_API_KEY from this local file for non-Modal fallback runs.",
    )
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.tinker_api_key_file:
        config["tinker_api_key_file"] = args.tinker_api_key_file
    result = eval_policy_tinker(config) if args.eval else train_tinker_rl(config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
