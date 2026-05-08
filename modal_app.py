from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

import modal

app = modal.App("rl-for-epistemics")

ARTIFACTS = modal.Volume.from_name("rl-epistemics-artifacts", create_if_missing=True)
CACHE = modal.Volume.from_name("rl-epistemics-hf-cache", create_if_missing=True)
TINKER_SECRET = modal.Secret.from_name("tinker-api-key", required_keys=["TINKER_API_KEY"])

REPO_DIR = Path("/root/rl_for_epistemics")
ARTIFACT_DIR = Path("/artifacts")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install_from_pyproject("pyproject.toml", optional_dependencies=["modal", "tinker"])
    .pip_install("hf-transfer")
    .add_local_dir("src", str(REPO_DIR / "src"), copy=True)
    .add_local_dir("configs", str(REPO_DIR / "configs"), copy=True)
    .add_local_dir("scripts", str(REPO_DIR / "scripts"), copy=True)
    .add_local_file("pyproject.toml", str(REPO_DIR / "pyproject.toml"), copy=True)
    .env(
        {
            "PYTHONPATH": str(REPO_DIR / "src"),
            "HF_HOME": "/cache/huggingface",
            "TRANSFORMERS_CACHE": "/cache/huggingface/transformers",
            "HF_DATASETS_CACHE": "/cache/huggingface/datasets",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "TINKER_SUBPROCESS_SAMPLING": "1",
        }
    )
)

COMMON_KWARGS = {
    "image": image,
    "volumes": {"/artifacts": ARTIFACTS, "/cache": CACHE},
}
POST_GATE_GPU = "H100"


def _prepare() -> None:
    os.chdir(REPO_DIR)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / "data").mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / "outputs").mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / "configs").mkdir(parents=True, exist_ok=True)


def _load_config(path: str) -> dict[str, Any]:
    from rl_epistemics.utils.config import load_config

    return load_config(REPO_DIR / path)


def _merge_config(config: dict[str, Any], overrides: Optional[dict[str, Any]]) -> dict[str, Any]:
    from rl_epistemics.utils.config import deep_update

    if isinstance(overrides, str):
        overrides = json.loads(overrides) if overrides.strip() else {}
    return deep_update(config, overrides or {})


def _commit() -> None:
    ARTIFACTS.commit()
    CACHE.commit()


def _make_dataset_impl(config_overrides: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    from rl_epistemics.data.generate_pairs import generate_pairs, write_eval_prompts
    from rl_epistemics.data.schema import write_jsonl
    from rl_epistemics.utils.config import save_config

    config = _load_config("configs/data.yaml")
    defaults = {
        "output_path": str(ARTIFACT_DIR / "data" / "pairs.jsonl"),
        "eval_output_path": str(ARTIFACT_DIR / "data" / "eval_prompts.jsonl"),
    }
    config = _merge_config(config, defaults)
    config = _merge_config(config, config_overrides)
    pairs = generate_pairs(config)
    write_jsonl(pairs, config["output_path"])
    if config.get("eval_output_path"):
        write_eval_prompts(pairs, config["eval_output_path"])
    save_config(config, ARTIFACT_DIR / "configs" / "data.yaml")
    return {"pairs": len(pairs), "output_path": config["output_path"]}


def _make_hard_eval_impl(config_overrides: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    from rl_epistemics.data.hard_eval import write_hard_eval_dataset
    from rl_epistemics.utils.config import save_config

    config = _load_config("configs/hard_eval.yaml")
    defaults = {
        "output_path": str(ARTIFACT_DIR / "data" / "hard_eval" / "pairs.jsonl"),
        "eval_output_path": str(ARTIFACT_DIR / "data" / "hard_eval" / "eval_prompts.jsonl"),
    }
    config = _merge_config(config, defaults)
    config = _merge_config(config, config_overrides)
    save_config(config, ARTIFACT_DIR / "configs" / "hard_eval.yaml")
    return write_hard_eval_dataset(config)


def _make_verified_dataset_impl(config_overrides: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    from rl_epistemics.data.verified_dataset import write_verified_dataset
    from rl_epistemics.utils.config import save_config

    config = _load_config("configs/verified_data.yaml")
    defaults = {
        "output_path": str(ARTIFACT_DIR / "data" / "verified_eval" / "pairs.jsonl"),
        "eval_output_path": str(ARTIFACT_DIR / "data" / "verified_eval" / "eval_prompts.jsonl"),
    }
    config = _merge_config(config, defaults)
    config = _merge_config(config, config_overrides)
    save_config(config, ARTIFACT_DIR / "configs" / "verified_data.yaml")
    return write_verified_dataset(config)


def _make_verified_train_dataset_impl(config_overrides: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    from rl_epistemics.data.verified_dataset import write_verified_dataset
    from rl_epistemics.utils.config import save_config

    config = _load_config("configs/verified_train.yaml")
    defaults = {
        "output_path": str(ARTIFACT_DIR / "data" / "verified_train" / "pairs.jsonl"),
        "eval_output_path": str(ARTIFACT_DIR / "data" / "verified_train" / "eval_prompts.jsonl"),
    }
    config = _merge_config(config, defaults)
    config = _merge_config(config, config_overrides)
    save_config(config, ARTIFACT_DIR / "configs" / "verified_train.yaml")
    return write_verified_dataset(config)


def _make_kalomaze_dataset_impl(config_overrides: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    from collections import Counter

    from rl_epistemics.data.kalomaze_replica import generate_pairs, write_eval_prompts
    from rl_epistemics.data.schema import write_jsonl
    from rl_epistemics.utils.config import save_config

    if isinstance(config_overrides, str):
        config_overrides = json.loads(config_overrides) if config_overrides.strip() else {}
    config_overrides = dict(config_overrides or {})
    config_path = str(config_overrides.pop("config_path", "configs/kalomaze_replica/data.yaml"))
    artifact_subdir = str(config_overrides.pop("artifact_subdir", "kalomaze_replica"))
    config = _load_config(config_path)
    defaults = {
        "output_path": str(ARTIFACT_DIR / "data" / artifact_subdir / "pairs.jsonl"),
        "eval_output_path": str(
            ARTIFACT_DIR / "data" / artifact_subdir / "eval_prompts.jsonl"
        ),
        "generation": {
            "backend": "tinker",
            "output_dir": str(ARTIFACT_DIR / "outputs" / artifact_subdir / "data"),
            "cache_path": str(
                ARTIFACT_DIR / "outputs" / artifact_subdir / "data" / "generation_cache.jsonl"
            ),
        },
    }
    config = _merge_config(config, defaults)
    config = _merge_config(config, config_overrides)
    pairs = generate_pairs(config)
    write_jsonl(pairs, config["output_path"])
    if config.get("eval_output_path"):
        write_eval_prompts(pairs, config["eval_output_path"])
    save_config(config, ARTIFACT_DIR / "configs" / artifact_subdir / "data.yaml")
    return {
        "pairs": len(pairs),
        "output_path": config["output_path"],
        "splits": dict(Counter(pair.split for pair in pairs)),
        "prompt_types": dict(Counter(pair.prompt_type for pair in pairs)),
    }


def _train_reward_model_impl(config_overrides: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    from rl_epistemics.models.bt_trainer import train_reward_model
    from rl_epistemics.utils.config import save_config

    config = _load_config("configs/reward_model.yaml")
    defaults = {
        "model_id": "Qwen/Qwen3-8B",
        "dataset_path": str(ARTIFACT_DIR / "data" / "pairs.jsonl"),
        "output_dir": str(ARTIFACT_DIR / "outputs" / "reward_model"),
        "device": "auto",
    }
    config = _merge_config(config, defaults)
    config = _merge_config(config, config_overrides)
    save_config(config, ARTIFACT_DIR / "configs" / "reward_model.yaml")
    return train_reward_model(config)


def _eval_reward_model_impl(config_overrides: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    from rl_epistemics.models.reward_scoring import eval_reward_model
    from rl_epistemics.utils.config import save_config

    config = _load_config("configs/reward_model.yaml")
    defaults = {
        "dataset_path": str(ARTIFACT_DIR / "data" / "pairs.jsonl"),
        "reward_model_dir": str(ARTIFACT_DIR / "outputs" / "reward_model"),
        "eval_output_dir": str(ARTIFACT_DIR / "outputs" / "reward_model" / "eval"),
        "device": "auto",
    }
    config = _merge_config(config, defaults)
    config = _merge_config(config, config_overrides)
    save_config(config, ARTIFACT_DIR / "configs" / "reward_model_eval.yaml")
    return eval_reward_model(config)


def _train_kalomaze_reward_model_impl(config_overrides: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    from rl_epistemics.models.bt_trainer import train_reward_model
    from rl_epistemics.utils.config import save_config

    config = _load_config("configs/kalomaze_replica/reward_model.yaml")
    defaults = {
        "dataset_path": str(ARTIFACT_DIR / "data" / "kalomaze_replica" / "pairs.jsonl"),
        "output_dir": str(ARTIFACT_DIR / "outputs" / "kalomaze_replica" / "reward_model"),
        "eval_output_dir": str(
            ARTIFACT_DIR / "outputs" / "kalomaze_replica" / "reward_model" / "eval"
        ),
        "device": "auto",
    }
    config = _merge_config(config, defaults)
    config = _merge_config(config, config_overrides)
    save_config(config, ARTIFACT_DIR / "configs" / "kalomaze_replica" / "reward_model.yaml")
    return train_reward_model(config)


def _eval_kalomaze_reward_model_impl(config_overrides: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    from rl_epistemics.models.reward_scoring import eval_reward_model
    from rl_epistemics.utils.config import save_config

    config = _load_config("configs/kalomaze_replica/reward_model.yaml")
    defaults = {
        "dataset_path": str(ARTIFACT_DIR / "data" / "kalomaze_replica" / "pairs.jsonl"),
        "reward_model_dir": str(ARTIFACT_DIR / "outputs" / "kalomaze_replica" / "reward_model"),
        "eval_output_dir": str(
            ARTIFACT_DIR / "outputs" / "kalomaze_replica" / "reward_model" / "eval"
        ),
        "device": "auto",
    }
    config = _merge_config(config, defaults)
    config = _merge_config(config, config_overrides)
    save_config(config, ARTIFACT_DIR / "configs" / "kalomaze_replica" / "reward_model_eval.yaml")
    return eval_reward_model(config)


def _train_tinker_impl(config_overrides: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    from rl_epistemics.rl.train_tinker import train_tinker_rl
    from rl_epistemics.utils.config import save_config

    config = _load_config("configs/tinker_rl.yaml")
    defaults = {
        "dataset_path": str(ARTIFACT_DIR / "data" / "pairs.jsonl"),
        "reward_model_dir": str(ARTIFACT_DIR / "outputs" / "reward_model"),
        "output_dir": str(ARTIFACT_DIR / "outputs" / "tinker_rl"),
        "manual_review_output": str(ARTIFACT_DIR / "outputs" / "blog" / "manual_review_samples.md"),
    }
    config = _merge_config(config, defaults)
    config = _merge_config(config, config_overrides)
    save_config(config, ARTIFACT_DIR / "configs" / "tinker_rl.yaml")
    return train_tinker_rl(config)


def _train_kalomaze_tinker_impl(
    reward_transform_name: str = "gaussian",
    config_overrides: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    from rl_epistemics.rl.train_tinker import train_tinker_rl
    from rl_epistemics.utils.config import save_config

    if reward_transform_name == "gaussian":
        config_path = "configs/kalomaze_replica/tinker_rl_gaussian.yaml"
        output_name = "tinker_rl_gaussian"
    elif reward_transform_name == "raw":
        config_path = "configs/kalomaze_replica/tinker_rl_raw.yaml"
        output_name = "tinker_rl_raw"
    else:
        raise ValueError("reward_transform_name must be 'gaussian' or 'raw'.")
    config = _load_config(config_path)
    defaults = {
        "dataset_path": str(ARTIFACT_DIR / "data" / "kalomaze_replica" / "pairs.jsonl"),
        "reward_model_dir": str(ARTIFACT_DIR / "outputs" / "kalomaze_replica" / "reward_model"),
        "output_dir": str(ARTIFACT_DIR / "outputs" / "kalomaze_replica" / output_name),
        "manual_review_output": str(
            ARTIFACT_DIR
            / "outputs"
            / "kalomaze_replica"
            / "report"
            / f"manual_review_samples_{reward_transform_name}.md"
        ),
    }
    config = _merge_config(config, defaults)
    config = _merge_config(config, config_overrides)
    save_config(config, ARTIFACT_DIR / "configs" / "kalomaze_replica" / f"{output_name}.yaml")
    return train_tinker_rl(config)


def _train_tinker_post_gate_impl(config_overrides: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    from rl_epistemics.rl.train_tinker import train_tinker_rl
    from rl_epistemics.utils.config import save_config

    config = _load_config("configs/tinker_rl_post_gate.yaml")
    defaults = {
        "dataset_path": str(ARTIFACT_DIR / "data" / "verified_train" / "pairs.jsonl"),
        "reward_model_dir": str(
            ARTIFACT_DIR
            / "outputs"
            / "rm_design_ablations"
            / "last_final_token_matrix_sum"
            / "reward_model"
        ),
        "output_dir": str(ARTIFACT_DIR / "outputs" / "tinker_rl_post_gate"),
        "manual_review_output": str(
            ARTIFACT_DIR / "outputs" / "blog" / "manual_review_samples_post_gate.md"
        ),
        "eval_quality_gate": {
            "adjudication_path": str(
                ARTIFACT_DIR / "outputs" / "blog" / "manual_review_queue_adjudicated_gpt55.csv"
            ),
        },
    }
    config = _merge_config(config, defaults)
    config = _merge_config(config, config_overrides)
    save_config(config, ARTIFACT_DIR / "configs" / "tinker_rl_post_gate.yaml")
    return train_tinker_rl(config)


def _eval_tinker_impl(config_overrides: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    from rl_epistemics.rl.train_tinker import eval_policy_tinker
    from rl_epistemics.utils.config import save_config

    run_meta = ARTIFACT_DIR / "outputs" / "tinker_rl" / "run_meta.json"
    config = _load_config("configs/tinker_rl.yaml")
    defaults = {
        "dataset_path": str(ARTIFACT_DIR / "data" / "pairs.jsonl"),
        "reward_model_dir": str(ARTIFACT_DIR / "outputs" / "reward_model"),
        "output_dir": str(ARTIFACT_DIR / "outputs" / "eval_tinker"),
        "run_meta_path": str(run_meta),
        "split": "test",
        "include_reward_scores": True,
        "manual_review_output": str(ARTIFACT_DIR / "outputs" / "blog" / "manual_review_samples.md"),
    }
    config = _merge_config(config, defaults)
    config = _merge_config(config, config_overrides)
    save_config(config, ARTIFACT_DIR / "configs" / "eval_tinker.yaml")
    return eval_policy_tinker(config)


def _eval_hard_tinker_impl(config_overrides: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    if isinstance(config_overrides, str):
        config_overrides = json.loads(config_overrides) if config_overrides.strip() else {}
    config_overrides = config_overrides or {}
    data_result = _make_hard_eval_impl(config_overrides.get("data"))
    hard_pairs = data_result["output_path"]
    eval_overrides = config_overrides.get("eval") or {}
    manual_dir = ARTIFACT_DIR / "outputs" / "blog"
    base = _eval_tinker_impl(
        {
            **eval_overrides,
            "dataset_path": hard_pairs,
            "split": "ood",
            "run_meta_path": None,
            "model_path": None,
            "output_dir": str(ARTIFACT_DIR / "outputs" / "eval_tinker_hard_base"),
            "manual_review_output": str(manual_dir / "manual_review_samples_hard_base.md"),
        }
    )
    main = _eval_tinker_impl(
        {
            **eval_overrides,
            "dataset_path": hard_pairs,
            "split": "ood",
            "run_meta_path": str(ARTIFACT_DIR / "outputs" / "tinker_rl" / "run_meta.json"),
            "output_dir": str(ARTIFACT_DIR / "outputs" / "eval_tinker_hard_main"),
            "manual_review_output": str(manual_dir / "manual_review_samples_hard_main.md"),
        }
    )
    return {"dataset": data_result, "base": base, "main": main}


def _eval_verified_tinker_impl(config_overrides: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    if isinstance(config_overrides, str):
        config_overrides = json.loads(config_overrides) if config_overrides.strip() else {}
    config_overrides = config_overrides or {}
    data_result = _make_verified_dataset_impl(config_overrides.get("data"))
    verified_pairs = data_result["output_path"]
    eval_overrides = config_overrides.get("eval") or {}
    manual_dir = ARTIFACT_DIR / "outputs" / "blog"
    base = _eval_tinker_impl(
        {
            **eval_overrides,
            "dataset_path": verified_pairs,
            "split": "ood",
            "run_meta_path": None,
            "model_path": None,
            "output_dir": str(ARTIFACT_DIR / "outputs" / "eval_tinker_verified_base"),
            "manual_review_output": str(manual_dir / "manual_review_samples_verified_base.md"),
        }
    )
    main = _eval_tinker_impl(
        {
            **eval_overrides,
            "dataset_path": verified_pairs,
            "split": "ood",
            "run_meta_path": str(ARTIFACT_DIR / "outputs" / "tinker_rl" / "run_meta.json"),
            "output_dir": str(ARTIFACT_DIR / "outputs" / "eval_tinker_verified_main"),
            "manual_review_output": str(manual_dir / "manual_review_samples_verified_main.md"),
        }
    )
    return {"dataset": data_result, "base": base, "main": main}


def _eval_post_gate_tinker_impl(config_overrides: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    if isinstance(config_overrides, str):
        config_overrides = json.loads(config_overrides) if config_overrides.strip() else {}
    config_overrides = config_overrides or {}
    data_overrides = config_overrides.get("data") or {}
    eval_overrides = config_overrides.get("eval") or {}
    manual_dir = ARTIFACT_DIR / "outputs" / "blog"
    post_gate_run_meta = ARTIFACT_DIR / "outputs" / "tinker_rl_post_gate" / "run_meta.json"

    verified_data = _make_verified_dataset_impl(data_overrides.get("verified"))
    hard_data = _make_hard_eval_impl(data_overrides.get("hard"))

    verified = _eval_tinker_impl(
        {
            **eval_overrides,
            "dataset_path": verified_data["output_path"],
            "split": "ood",
            "run_meta_path": str(post_gate_run_meta),
            "output_dir": str(ARTIFACT_DIR / "outputs" / "eval_tinker_verified_post_gate"),
            "manual_review_output": str(manual_dir / "manual_review_samples_verified_post_gate.md"),
        }
    )
    hard = _eval_tinker_impl(
        {
            **eval_overrides,
            "dataset_path": hard_data["output_path"],
            "split": "ood",
            "run_meta_path": str(post_gate_run_meta),
            "output_dir": str(ARTIFACT_DIR / "outputs" / "eval_tinker_hard_post_gate"),
            "manual_review_output": str(manual_dir / "manual_review_samples_hard_post_gate.md"),
        }
    )
    return {
        "verified_dataset": verified_data,
        "hard_dataset": hard_data,
        "verified_post_gate": verified,
        "hard_post_gate": hard,
    }


def _run_rm_design_ablation_suite(config_overrides: Optional[dict[str, Any]] = None) -> list[dict[str, Any]]:
    from rl_epistemics.experiments.run_rm_ablation import (
        DEFAULT_VARIANTS,
        dataset_manifest_summary,
        _run_config_summary,
        write_rm_ablation_summary,
    )
    from rl_epistemics.models.bt_trainer import train_reward_model
    from rl_epistemics.models.reward_scoring import eval_reward_model
    from rl_epistemics.utils.config import deep_update, save_config

    if isinstance(config_overrides, str):
        config_overrides = json.loads(config_overrides) if config_overrides.strip() else {}
    config_overrides = config_overrides or {}
    dataset_override = config_overrides.get("dataset_path")
    dataset_path = Path(
        str(dataset_override or ARTIFACT_DIR / "data" / "verified_train" / "pairs.jsonl")
    )
    refresh_dataset = bool(config_overrides.get("refresh_dataset", dataset_override is None))
    if refresh_dataset or not dataset_path.exists():
        _make_verified_train_dataset_impl(
            {
                "output_path": str(dataset_path),
                "eval_output_path": str(dataset_path.parent / "eval_prompts.jsonl"),
            }
        )
    base_config = _load_config("configs/reward_model.yaml")
    variants = list(config_overrides.get("variants") or DEFAULT_VARIANTS)
    max_runs = config_overrides.get("max_runs")
    if max_runs is not None:
        variants = variants[: int(max_runs)]
    output_dir = ARTIFACT_DIR / "outputs" / "rm_design_ablations"
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    for idx, variant in enumerate(variants):
        name = str(variant.get("name") or f"rm_variant_{idx:04d}")
        run_dir = output_dir / name
        rm_dir = run_dir / "reward_model"
        eval_dir = run_dir / "eval"
        rm_config = deep_update(base_config, variant.get("updates") or {})
        rm_config = _merge_config(
            rm_config,
            {
                "model_id": config_overrides.get("model_id", "Qwen/Qwen3-8B"),
                "dataset_path": str(dataset_path),
                "output_dir": str(rm_dir),
                "eval_output_dir": str(eval_dir),
                "device": "auto",
            },
        )
        dataset_summary = dataset_manifest_summary(dataset_path)
        if config_overrides.get("limit_train_examples") is not None:
            rm_config["limit_train_examples"] = int(config_overrides["limit_train_examples"])
        if config_overrides.get("limit_eval_examples") is not None:
            rm_config["limit_eval_examples"] = int(config_overrides["limit_eval_examples"])
        run_dir.mkdir(parents=True, exist_ok=True)
        save_config(rm_config, run_dir / "reward_model.yaml")
        reward_model = train_reward_model(rm_config)
        reward_eval = eval_reward_model(
            {
                **rm_config,
                "reward_model_dir": str(rm_dir),
                "eval_output_dir": str(eval_dir),
            }
        )
        record = {
            "name": name,
            "run_dir": str(run_dir),
            "variant": variant.get("updates") or {},
            "run_config": _run_config_summary(rm_config),
            "reward_model": reward_model,
            "reward_eval": reward_eval,
        }
        manifest.append(
            {k: record[k] for k in ("name", "run_dir", "variant", "run_config")}
            | {"dataset_summary": dataset_summary}
        )
        results.append(record)
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    write_rm_ablation_summary(output_dir)
    return results


def _collect_results_impl() -> dict[str, Any]:
    from rl_epistemics.eval.reclassify_outputs import refresh_policy_metrics
    from rl_epistemics.experiments.reward_transform_audit import write_reward_transform_audit
    from rl_epistemics.reporting.failures import write_failure_reports
    from rl_epistemics.reporting.plots import write_report_plots
    from rl_epistemics.reporting.tables import write_report_tables

    report_dir = ARTIFACT_DIR / "outputs" / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    refreshed = refresh_policy_metrics(ARTIFACT_DIR / "outputs")
    tables = write_report_tables(ARTIFACT_DIR / "outputs", report_dir)
    plots = write_report_plots(ARTIFACT_DIR / "outputs", report_dir)
    failures = write_failure_reports(ARTIFACT_DIR / "outputs", report_dir, ARTIFACT_DIR / "outputs" / "blog")
    transform_audit = None
    ablation_scores = sorted(
        (ARTIFACT_DIR / "outputs" / "rm_design_ablations").glob("*/eval/reward_scores.csv")
    )
    scores_path = ARTIFACT_DIR / "outputs" / "reward_model" / "eval" / "reward_scores.csv"
    stats_path = ARTIFACT_DIR / "outputs" / "reward_model" / "chosen_score_stats.json"
    if ablation_scores:
        transform_audit = write_reward_transform_audit(
            {
                "scores_glob": str(
                    ARTIFACT_DIR / "outputs" / "rm_design_ablations" / "*" / "eval" / "reward_scores.csv"
                ),
                "output_dir": str(report_dir),
            }
        )
    elif scores_path.exists() and stats_path.exists():
        transform_audit = write_reward_transform_audit(
            {
                "scores_path": str(scores_path),
                "stats_path": str(stats_path),
                "output_dir": str(report_dir),
            }
        )
    return {
        "report_dir": str(report_dir),
        "refreshed_policy_metrics": refreshed,
        "tables": tables,
        "plots": plots,
        "failures": failures,
        "reward_transform_audit": transform_audit,
    }


def _write_blog_impl() -> dict[str, Any]:
    from rl_epistemics.eval.reclassify_outputs import refresh_policy_metrics
    from rl_epistemics.reporting.write_blog import write_blog

    refresh_policy_metrics(ARTIFACT_DIR / "outputs")
    blog_path = ARTIFACT_DIR / "blog" / "rl_for_epistemic_humility.md"
    return write_blog(ARTIFACT_DIR / "outputs", blog_path)


def _write_kalomaze_report_impl() -> str:
    from rl_epistemics.reporting.kalomaze_replica import write_report

    return str(
        write_report(
            output_root=ARTIFACT_DIR / "outputs" / "kalomaze_replica",
            report_path=ARTIFACT_DIR / "outputs" / "kalomaze_replica" / "report.md",
        )
    )


@app.function(**COMMON_KWARGS, secrets=[TINKER_SECRET], timeout=60 * 20)
def modal_smoke() -> dict[str, Any]:
    _prepare()
    try:
        has_key = bool(os.environ.get("TINKER_API_KEY"))
        if not has_key:
            raise EnvironmentError("Modal secret `tinker-api-key` lacks TINKER_API_KEY.")
        import tinker

        service_client = tinker.ServiceClient()
        capabilities = service_client.get_server_capabilities()
        supported = [
            str(getattr(model, "model_name", getattr(model, "name", model)))
            for model in getattr(capabilities, "supported_models", [])
        ]
        result = _make_dataset_impl(
            {
                "smoke": True,
                "output_path": str(ARTIFACT_DIR / "data" / "smoke" / "pairs.jsonl"),
                "eval_output_path": str(ARTIFACT_DIR / "data" / "smoke" / "eval_prompts.jsonl"),
            }
        )
        smoke_result = {
            "secret_has_tinker_api_key": has_key,
            "qwen3_8b_supported": "Qwen/Qwen3-8B" in supported,
            "supported_model_count": len(supported),
            "dataset": result,
        }
        with (ARTIFACT_DIR / "outputs" / "modal_smoke.json").open("w", encoding="utf-8") as f:
            json.dump(smoke_result, f, indent=2)
        print(json.dumps(smoke_result, sort_keys=True))
        return smoke_result
    finally:
        _commit()


@app.function(**COMMON_KWARGS, timeout=60 * 30)
def modal_make_dataset(config_overrides=None) -> dict[str, Any]:
    _prepare()
    try:
        return _make_dataset_impl(config_overrides)
    finally:
        _commit()


@app.function(**COMMON_KWARGS, timeout=60 * 30)
def modal_make_hard_eval(config_overrides=None) -> dict[str, Any]:
    _prepare()
    try:
        return _make_hard_eval_impl(config_overrides)
    finally:
        _commit()


@app.function(**COMMON_KWARGS, timeout=60 * 30)
def modal_make_verified_dataset(config_overrides=None) -> dict[str, Any]:
    _prepare()
    try:
        return _make_verified_dataset_impl(config_overrides)
    finally:
        _commit()


@app.function(**COMMON_KWARGS, timeout=60 * 30)
def modal_make_verified_train_dataset(config_overrides=None) -> dict[str, Any]:
    _prepare()
    try:
        return _make_verified_train_dataset_impl(config_overrides)
    finally:
        _commit()


@app.function(**COMMON_KWARGS, secrets=[TINKER_SECRET], timeout=60 * 60)
def modal_make_kalomaze_dataset(config_overrides=None) -> dict[str, Any]:
    _prepare()
    try:
        return _make_kalomaze_dataset_impl(config_overrides)
    finally:
        _commit()


@app.function(**COMMON_KWARGS, gpu="A100-80GB", timeout=60 * 60 * 10)
def modal_train_reward_model(config_overrides=None) -> dict[str, Any]:
    _prepare()
    try:
        return _train_reward_model_impl(config_overrides)
    finally:
        _commit()


@app.function(**COMMON_KWARGS, gpu="A100-80GB", timeout=60 * 60 * 4)
def modal_eval_reward_model(config_overrides=None) -> dict[str, Any]:
    _prepare()
    try:
        return _eval_reward_model_impl(config_overrides)
    finally:
        _commit()


@app.function(**COMMON_KWARGS, gpu="A100-80GB", timeout=60 * 60 * 10)
def modal_train_kalomaze_reward_model(config_overrides=None) -> dict[str, Any]:
    _prepare()
    try:
        return _train_kalomaze_reward_model_impl(config_overrides)
    finally:
        _commit()


@app.function(**COMMON_KWARGS, gpu="A100-80GB", timeout=60 * 60 * 4)
def modal_eval_kalomaze_reward_model(config_overrides=None) -> dict[str, Any]:
    _prepare()
    try:
        return _eval_kalomaze_reward_model_impl(config_overrides)
    finally:
        _commit()


@app.function(**COMMON_KWARGS, secrets=[TINKER_SECRET], gpu="A100-80GB", timeout=60 * 60 * 12)
def modal_train_rl_tinker(config_overrides=None) -> dict[str, Any]:
    _prepare()
    try:
        return _train_tinker_impl(config_overrides)
    finally:
        _commit()


@app.function(**COMMON_KWARGS, secrets=[TINKER_SECRET], gpu=POST_GATE_GPU, timeout=60 * 60 * 24)
def modal_train_post_gate_tinker(config_overrides=None) -> dict[str, Any]:
    _prepare()
    try:
        return _train_tinker_post_gate_impl(config_overrides)
    finally:
        _commit()


@app.local_entrypoint()
def run_post_gate_tinker(config_overrides: str = "") -> None:
    overrides = json.loads(config_overrides) if config_overrides.strip() else None
    print("Submitting modal_train_post_gate_tinker.remote(...)")
    result = modal_train_post_gate_tinker.remote(overrides)
    print(json.dumps(result, indent=2, sort_keys=True))


@app.function(**COMMON_KWARGS, secrets=[TINKER_SECRET], gpu=POST_GATE_GPU, timeout=60 * 60 * 24)
def modal_train_kalomaze_tinker_gaussian(config_overrides=None) -> dict[str, Any]:
    _prepare()
    try:
        return _train_kalomaze_tinker_impl("gaussian", config_overrides)
    finally:
        _commit()


@app.function(**COMMON_KWARGS, secrets=[TINKER_SECRET], gpu=POST_GATE_GPU, timeout=60 * 60 * 24)
def modal_train_kalomaze_tinker_raw(config_overrides=None) -> dict[str, Any]:
    _prepare()
    try:
        return _train_kalomaze_tinker_impl("raw", config_overrides)
    finally:
        _commit()


@app.function(**COMMON_KWARGS, secrets=[TINKER_SECRET], gpu=POST_GATE_GPU, timeout=60 * 60 * 24)
def modal_run_kalomaze_replica(config_overrides=None) -> dict[str, Any]:
    if isinstance(config_overrides, str):
        config_overrides = json.loads(config_overrides) if config_overrides.strip() else {}
    config_overrides = config_overrides or {}
    _prepare()
    try:
        results = {
            "dataset": _make_kalomaze_dataset_impl(config_overrides.get("data")),
            "reward_model": _train_kalomaze_reward_model_impl(config_overrides.get("reward_model")),
            "reward_eval": _eval_kalomaze_reward_model_impl(config_overrides.get("reward_eval")),
        }
        if config_overrides.get("skip_gaussian") is not True:
            results["tinker_gaussian"] = _train_kalomaze_tinker_impl(
                "gaussian", config_overrides.get("tinker_gaussian")
            )
        if config_overrides.get("skip_raw") is not True:
            results["tinker_raw"] = _train_kalomaze_tinker_impl(
                "raw", config_overrides.get("tinker_raw")
            )
        results["report"] = _write_kalomaze_report_impl()
        with (ARTIFACT_DIR / "outputs" / "kalomaze_replica" / "suite_result.json").open(
            "w", encoding="utf-8"
        ) as f:
            json.dump(results, f, indent=2)
        return results
    finally:
        _commit()


@app.local_entrypoint()
def run_kalomaze_replica(config_overrides: str = "") -> None:
    overrides = json.loads(config_overrides) if config_overrides.strip() else None
    print("Submitting modal_run_kalomaze_replica.remote(...)")
    result = modal_run_kalomaze_replica.remote(overrides)
    print(json.dumps(result, indent=2, sort_keys=True))


@app.function(**COMMON_KWARGS, secrets=[TINKER_SECRET], gpu="A100-80GB", timeout=60 * 60 * 4)
def modal_eval_policy_tinker(config_overrides=None) -> dict[str, Any]:
    _prepare()
    try:
        return _eval_tinker_impl(config_overrides)
    finally:
        _commit()


@app.function(**COMMON_KWARGS, secrets=[TINKER_SECRET], gpu="A100-80GB", timeout=60 * 60 * 4)
def modal_eval_hard_tinker(config_overrides=None) -> dict[str, Any]:
    _prepare()
    try:
        return _eval_hard_tinker_impl(config_overrides)
    finally:
        _commit()


@app.function(**COMMON_KWARGS, secrets=[TINKER_SECRET], gpu="A100-80GB", timeout=60 * 60 * 4)
def modal_eval_verified_tinker(config_overrides=None) -> dict[str, Any]:
    _prepare()
    try:
        return _eval_verified_tinker_impl(config_overrides)
    finally:
        _commit()


@app.function(**COMMON_KWARGS, secrets=[TINKER_SECRET], gpu=POST_GATE_GPU, timeout=60 * 60 * 8)
def modal_eval_post_gate_tinker(config_overrides=None) -> dict[str, Any]:
    _prepare()
    try:
        return _eval_post_gate_tinker_impl(config_overrides)
    finally:
        _commit()


@app.local_entrypoint()
def run_post_gate_eval_tinker(config_overrides: str = "") -> None:
    overrides = json.loads(config_overrides) if config_overrides.strip() else None
    print("Submitting modal_eval_post_gate_tinker.remote(...)")
    result = modal_eval_post_gate_tinker.remote(overrides)
    print(json.dumps(result, indent=2, sort_keys=True))


@app.function(**COMMON_KWARGS, gpu="A100-80GB", timeout=60 * 60 * 24)
def modal_run_rm_design_ablations(config_overrides=None) -> list[dict[str, Any]]:
    _prepare()
    try:
        return _run_rm_design_ablation_suite(config_overrides)
    finally:
        _commit()


@app.function(**COMMON_KWARGS, secrets=[TINKER_SECRET], gpu=POST_GATE_GPU, timeout=60 * 60 * 24)
def modal_run_experiment_suite(suite_name: str = "main") -> dict[str, Any]:
    _prepare()
    try:
        results: dict[str, Any] = {"suite_name": suite_name}
        if suite_name == "smoke":
            smoke_data = ARTIFACT_DIR / "data" / "smoke"
            smoke_outputs = ARTIFACT_DIR / "outputs" / "smoke"
            smoke_pairs = smoke_data / "pairs.jsonl"
            smoke_rm = smoke_outputs / "reward_model"
            smoke_tinker = smoke_outputs / "tinker_rl"
            smoke_eval = smoke_outputs / "eval_tinker"
            results["dataset"] = _make_dataset_impl(
                {
                    "smoke": True,
                    "output_path": str(smoke_pairs),
                    "eval_output_path": str(smoke_data / "eval_prompts.jsonl"),
                }
            )
            results["reward_model"] = _train_reward_model_impl(
                {
                    "model_id": "Qwen/Qwen3-0.6B",
                    "limit_train_examples": 8,
                    "limit_eval_examples": 4,
                    "dataset_path": str(smoke_pairs),
                    "output_dir": str(smoke_rm),
                }
            )
            results["reward_eval"] = _eval_reward_model_impl(
                {
                    "dataset_path": str(smoke_pairs),
                    "reward_model_dir": str(smoke_rm),
                    "eval_output_dir": str(smoke_rm / "eval"),
                    "limit_eval_examples": 8,
                }
            )
            results["tinker_rl"] = _train_tinker_impl(
                {
                    "base_model": "Qwen/Qwen3-4B-Instruct-2507",
                    "rank": 8,
                    "max_steps": 1,
                    "batch_size": 2,
                    "group_size": 2,
                    "max_completion_tokens": 48,
                    "limit_prompts": 4,
                    "run_name": "smoke",
                    "dataset_path": str(smoke_pairs),
                    "reward_model_dir": str(smoke_rm),
                    "output_dir": str(smoke_tinker),
                    "manual_review_output": str(smoke_outputs / "manual_review_samples.md"),
                }
            )
            results["policy_eval"] = _eval_tinker_impl(
                {
                    "base_model": "Qwen/Qwen3-4B-Instruct-2507",
                    "limit": 4,
                    "max_new_tokens": 64,
                    "dataset_path": str(smoke_pairs),
                    "reward_model_dir": str(smoke_rm),
                    "run_meta_path": str(smoke_tinker / "run_meta.json"),
                    "output_dir": str(smoke_eval),
                    "manual_review_output": str(smoke_outputs / "manual_review_samples.md"),
                }
            )
        elif suite_name == "main":
            results["dataset"] = _make_dataset_impl()
            results["reward_model"] = _train_reward_model_impl()
            results["reward_eval"] = _eval_reward_model_impl()
            results["tinker_rl"] = _train_tinker_impl({"run_name": "main-rank32-gaussian"})
            results["policy_eval"] = _eval_tinker_impl()
        elif suite_name == "ablations":
            results["ablations"] = _run_ablation_suite()
        elif suite_name == "critical_ablations":
            results["critical_ablations"] = _run_critical_ablation_suite()
        elif suite_name == "hard_eval":
            results["hard_eval"] = _eval_hard_tinker_impl()
        elif suite_name == "verified_eval":
            results["verified_eval"] = _eval_verified_tinker_impl()
        elif suite_name == "rm_design_ablations":
            results["rm_design_ablations"] = _run_rm_design_ablation_suite()
        elif suite_name == "second_stage_data":
            results["verified_dataset"] = _make_verified_dataset_impl()
            results["verified_train_dataset"] = _make_verified_train_dataset_impl()
            results["hard_eval_dataset"] = _make_hard_eval_impl()
        elif suite_name == "post_gate_long_rl":
            results["post_gate_tinker_rl"] = _train_tinker_post_gate_impl()
        elif suite_name == "post_gate_eval":
            results["post_gate_eval"] = _eval_post_gate_tinker_impl()
        else:
            raise ValueError(
                "suite_name must be one of: smoke, main, ablations, critical_ablations, "
                "hard_eval, verified_eval, rm_design_ablations, second_stage_data, "
                "post_gate_long_rl, post_gate_eval"
            )
        results["collect"] = _collect_results_impl()
        results["blog"] = _write_blog_impl()
        with (ARTIFACT_DIR / "outputs" / "suite_result.json").open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        return results
    finally:
        _commit()


def _run_ablation_suite() -> list[dict[str, Any]]:
    ablations = [
        ("raw_reward", {"reward_transform": {"type": "raw", "std_floor": 0.1}, "run_name": "raw"}),
        ("band_target", {"reward_transform": {"type": "band_target", "std_floor": 0.1}, "run_name": "band"}),
        ("rank8", {"rank": 8, "run_name": "rank8"}),
        ("rank16", {"rank": 16, "run_name": "rank16"}),
    ]
    results = []
    for name, overrides in ablations:
        run_output = ARTIFACT_DIR / "outputs" / "tinker_rl_ablations" / name
        merged = {"output_dir": str(run_output), **overrides}
        train_meta = _train_tinker_impl(merged)
        eval_summary = _eval_tinker_impl(
            {
                "run_meta_path": str(run_output / "run_meta.json"),
                "output_dir": str(ARTIFACT_DIR / "outputs" / "eval_tinker_ablations" / name),
            }
        )
        results.append({"name": name, "train": train_meta, "eval": eval_summary})
    return results


def _run_critical_ablation_suite() -> list[dict[str, Any]]:
    ablations = [
        (
            "imbalanced_data",
            {"data_balance": "imbalanced"},
            {},
            {"run_name": "imbalanced-data"},
        ),
        (
            "no_polarity_flip",
            {"polarity_flip": False},
            {},
            {"run_name": "no-polarity-flip"},
        ),
        (
            "completion_only",
            {},
            {"rm_input": "completion_only"},
            {"run_name": "completion-only"},
        ),
        (
            "scalar_linear",
            {},
            {"head": {"type": "scalar_linear", "scale": 0.01}},
            {"run_name": "scalar-linear"},
        ),
    ]
    results = []
    for name, data_overrides, rm_overrides, rl_overrides in ablations:
        data_dir = ARTIFACT_DIR / "data" / "ablations" / name
        rm_dir = ARTIFACT_DIR / "outputs" / "reward_model_ablations" / name
        tinker_dir = ARTIFACT_DIR / "outputs" / "tinker_rl_ablations" / name
        eval_dir = ARTIFACT_DIR / "outputs" / "eval_tinker_ablations" / name
        dataset_path = data_dir / "pairs.jsonl"
        manual_review_path = ARTIFACT_DIR / "outputs" / "blog" / f"manual_review_samples_{name}.md"

        dataset = _make_dataset_impl(
            {
                **data_overrides,
                "output_path": str(dataset_path),
                "eval_output_path": str(data_dir / "eval_prompts.jsonl"),
            }
        )
        reward_model = _train_reward_model_impl(
            {
                **rm_overrides,
                "dataset_path": str(dataset_path),
                "output_dir": str(rm_dir),
            }
        )
        reward_eval = _eval_reward_model_impl(
            {
                "dataset_path": str(dataset_path),
                "reward_model_dir": str(rm_dir),
                "eval_output_dir": str(rm_dir / "eval"),
            }
        )
        train_meta = _train_tinker_impl(
            {
                **rl_overrides,
                "dataset_path": str(dataset_path),
                "reward_model_dir": str(rm_dir),
                "output_dir": str(tinker_dir),
                "manual_review_output": str(manual_review_path),
            }
        )
        eval_summary = _eval_tinker_impl(
            {
                "dataset_path": str(dataset_path),
                "reward_model_dir": str(rm_dir),
                "run_meta_path": str(tinker_dir / "run_meta.json"),
                "output_dir": str(eval_dir),
                "manual_review_output": str(manual_review_path),
            }
        )
        results.append(
            {
                "name": name,
                "dataset": dataset,
                "reward_model": reward_model,
                "reward_eval": reward_eval,
                "train": train_meta,
                "eval": eval_summary,
            }
        )
    return results


@app.function(**COMMON_KWARGS, timeout=60 * 30)
def modal_collect_results() -> dict[str, Any]:
    _prepare()
    try:
        return _collect_results_impl()
    finally:
        _commit()


@app.function(**COMMON_KWARGS, timeout=60 * 30)
def modal_write_blog() -> dict[str, Any]:
    _prepare()
    try:
        return _write_blog_impl()
    finally:
        _commit()
