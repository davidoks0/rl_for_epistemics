from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from rl_epistemics.data.schema import read_jsonl
from rl_epistemics.utils.config import load_config


REQUIRED_LABELS = {
    "clean_uncertainty",
    "false_premise_correction",
    "hedged_confabulation",
    "confident_confabulation",
    "real_substantive_answer",
    "false_refusal",
    "unclear",
}

MIN_VERIFIED_SEED_PROMPTS = 150


REQUIREMENT_SPECS = [
    {
        "id": "retrieval_verified_eval_data",
        "requirement": "Build retrieval-verified eval data for real/fake entities, documents/APIs, and false premises.",
        "checks": ["verified_dataset", "verified_retrieval_audit", "domain_coverage_beyond_researchers"],
        "stage": "local_wiring",
        "blocks_publishable_claims": True,
    },
    {
        "id": "verified_train_val_test_data",
        "requirement": "Provide split-aware verified train/val/test data for second-stage RM experiments.",
        "checks": ["verified_train_val_test_dataset"],
        "stage": "local_wiring",
        "blocks_publishable_claims": True,
    },
    {
        "id": "adjudication_tooling",
        "requirement": "Add human or optional LLM-judge adjudication for sampled policy outputs.",
        "checks": ["adjudication_tooling"],
        "stage": "local_wiring",
        "blocks_publishable_claims": True,
    },
    {
        "id": "actual_adjudicated_labels",
        "requirement": "Collect enough non-heuristic adjudicated labels to evaluate sampled outputs and unlock longer RL.",
        "checks": [
            "actual_human_or_llm_adjudication",
            "adjudication_gate_progress",
            "adjudicated_failure_taxonomy",
        ],
        "stage": "external_execution",
        "blocks_publishable_claims": True,
    },
    {
        "id": "richer_eval_labels",
        "requirement": (
            "Distinguish clean uncertainty, false-premise correction, hedged/confident confabulation, "
            "real substantive answer, false refusal, and unclear outputs."
        ),
        "checks": ["eval_taxonomy"],
        "stage": "local_wiring",
        "blocks_publishable_claims": True,
    },
    {
        "id": "defer_longer_rl_until_eval_fixed",
        "requirement": "Gate longer Tinker RL until adjudication quality is adequate.",
        "checks": ["longer_rl_quality_gate"],
        "stage": "local_wiring",
        "blocks_publishable_claims": True,
    },
    {
        "id": "rm_design_comparisons",
        "requirement": (
            "Compare RM design choices: pooling, layer, prompt+completion versus completion-only, and head type."
        ),
        "checks": ["rm_design_ablation_surface", "actual_rm_design_eval_results"],
        "stage": "mixed",
        "blocks_publishable_claims": True,
    },
    {
        "id": "reward_transform_comparisons",
        "requirement": "Test raw reward against bounded target-distribution reward transforms on chosen/rejected RM pairs.",
        "checks": ["reward_transform_audit_surface", "chosen_rejected_reward_transform_audit"],
        "stage": "mixed",
        "blocks_publishable_claims": True,
    },
    {
        "id": "modal_second_stage_execution_surface",
        "requirement": "Expose the second-stage data, eval, RM-ablation, collection, and blog steps through Modal.",
        "checks": ["modal_second_stage_surface"],
        "stage": "local_wiring",
        "blocks_publishable_claims": False,
    },
    {
        "id": "honest_report_and_blog",
        "requirement": "Update the report/blog to tell the honest mixed-to-negative story with hard-OOD results.",
        "checks": ["hard_ood_policy_results_ingested", "adjudicated_failure_taxonomy", "honest_blog_report"],
        "stage": "local_wiring",
        "blocks_publishable_claims": True,
    },
    {
        "id": "post_gate_longer_rl",
        "requirement": "Run longer Tinker RL only after the adjudication quality gate passes.",
        "checks": ["post_gate_recovery_surface", "post_gate_longer_rl_run"],
        "stage": "external_execution",
        "blocks_publishable_claims": False,
    },
]


def _exists(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def _status(ok: bool, evidence: str, *, required: bool = True) -> dict[str, Any]:
    return {"ok": bool(ok), "required": bool(required), "evidence": evidence}


def _dataset_file_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    data = path.read_bytes()
    pairs = read_jsonl(path)
    return {
        "path": str(path),
        "exists": True,
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": path.stat().st_size,
        "n": len(pairs),
        "unique_prompts": len({pair.prompt for pair in pairs}),
        "unique_seed_prompts": len({str(pair.metadata.get("seed_prompt", pair.prompt)) for pair in pairs}),
    }


def _read_jsonl_dicts(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _is_reviewed_label(value: Any) -> bool:
    if value is None or pd.isna(value):
        return False
    label = str(value).strip()
    return label not in {"", "TODO", "todo", "nan", "None", "none", "null"}


def _verified_dataset_status(root: Path) -> dict[str, Any]:
    path = root / "data" / "verified_eval" / "pairs.jsonl"
    if not path.exists():
        return _status(False, f"missing {path}")
    pairs = read_jsonl(path)
    domains = sorted({pair.domain for pair in pairs})
    categories = sorted({str(pair.metadata.get("category", "")) for pair in pairs})
    prompt_types = sorted({pair.prompt_type for pair in pairs})
    all_have_evidence = all(pair.metadata.get("verification", {}).get("evidence") for pair in pairs)
    all_have_checks = all(
        pair.metadata.get("verification", {}).get("online_checks")
        or pair.metadata.get("verification", {}).get("evidence")
        for pair in pairs
    )
    unique_seed_prompts = len({str(pair.metadata.get("seed_prompt", pair.prompt)) for pair in pairs})
    errors = [
        (pair.id, check.get("error"))
        for pair in pairs
        for check in pair.metadata.get("verification", {}).get("online_checks", [])
        if check.get("error")
    ]
    ok = (
        len(pairs) >= 30
        and len({pair.prompt for pair in pairs}) == len(pairs)
        and unique_seed_prompts >= MIN_VERIFIED_SEED_PROMPTS
        and {"real", "fake"} <= set(prompt_types)
        and len(domains) >= 6
        and {"real_entity", "fake_entity", "fake_paper", "fake_api", "false_premise"} <= set(categories)
        and all_have_evidence
        and all_have_checks
        and not errors
    )
    return _status(
        ok,
        json.dumps(
            {
                "path": str(path),
                "n": len(pairs),
                "unique_prompts": len({pair.prompt for pair in pairs}),
                "unique_seed_prompts": unique_seed_prompts,
                "min_unique_seed_prompts": MIN_VERIFIED_SEED_PROMPTS,
                "domains": domains,
                "categories": categories,
                "prompt_types": prompt_types,
                "all_have_seeded_evidence": all_have_evidence,
                "all_have_online_or_seeded_checks": all_have_checks,
                "online_errors": len(errors),
            },
            sort_keys=True,
        ),
    )


def _verified_train_dataset_status(root: Path) -> dict[str, Any]:
    path = root / "data" / "verified_train" / "pairs.jsonl"
    if not path.exists():
        return _status(False, f"missing {path}")
    pairs = read_jsonl(path)
    splits = {split: sum(pair.split == split for pair in pairs) for split in ["train", "val", "test"]}
    all_have_evidence = all(pair.metadata.get("verification", {}).get("evidence") for pair in pairs)
    train_prompts = {pair.prompt for pair in pairs}
    unique_seed_prompts = len({str(pair.metadata.get("seed_prompt", pair.prompt)) for pair in pairs})
    eval_path = root / "data" / "verified_eval" / "pairs.jsonl"
    eval_prompts = {pair.prompt for pair in read_jsonl(eval_path)} if eval_path.exists() else set()
    prompt_overlap_with_eval = len(train_prompts & eval_prompts)
    ok = (
        len(pairs) >= 30
        and len(train_prompts) == len(pairs)
        and unique_seed_prompts >= MIN_VERIFIED_SEED_PROMPTS
        and all(count > 0 for count in splits.values())
        and all_have_evidence
        and prompt_overlap_with_eval == 0
    )
    return _status(
        ok,
        json.dumps(
            {
                "path": str(path),
                "n": len(pairs),
                "unique_prompts": len(train_prompts),
                "unique_seed_prompts": unique_seed_prompts,
                "min_unique_seed_prompts": MIN_VERIFIED_SEED_PROMPTS,
                "splits": splits,
                "prompt_overlap_with_eval": prompt_overlap_with_eval,
                "domains": sorted({pair.domain for pair in pairs}),
                "categories": sorted({str(pair.metadata.get("category", "")) for pair in pairs}),
                "all_have_seeded_evidence": all_have_evidence,
            },
            sort_keys=True,
        ),
    )


def _domain_coverage_status(root: Path) -> dict[str, Any]:
    paths = [
        root / "data" / "verified_eval" / "pairs.jsonl",
        root / "data" / "verified_train" / "pairs.jsonl",
    ]
    pairs = []
    for path in paths:
        if path.exists():
            pairs.extend(read_jsonl(path))
    domains = sorted({pair.domain for pair in pairs})
    categories = sorted({str(pair.metadata.get("category", "")) for pair in pairs})
    non_researcher_domains = sorted(domain for domain in domains if domain not in {"researcher", "paper"})
    expected_domains = {
        "biography",
        "software_api",
        "medical_claim",
        "legal_claim",
        "finance_claim",
        "literary_claim",
        "historical_claim",
    }
    expected_categories = {
        "real_entity",
        "fake_entity",
        "real_paper",
        "fake_paper",
        "real_api",
        "fake_api",
        "false_premise",
    }
    ok = expected_domains <= set(domains) and expected_categories <= set(categories)
    return _status(
        ok,
        json.dumps(
            {
                "paths": [str(path) for path in paths],
                "domains": domains,
                "non_researcher_domains": non_researcher_domains,
                "missing_domains": sorted(expected_domains - set(domains)),
                "categories": categories,
                "missing_categories": sorted(expected_categories - set(categories)),
            },
            sort_keys=True,
        ),
    )


def _verified_retrieval_audit_status(root: Path) -> dict[str, Any]:
    json_path = root / "outputs" / "report" / "verified_retrieval_audit.json"
    csv_path = root / "outputs" / "report" / "verified_retrieval_audit.csv"
    summary = _read_json(json_path)
    input_paths = {
        str(item.get("path"))
        for item in summary.get("input_summaries", [])
        if isinstance(item, dict)
    }
    required_inputs = {"data/verified_train/pairs.jsonl", "data/verified_eval/pairs.jsonl"}
    total_pairs = int(summary.get("total_pairs", 0) or 0)
    live_check_pairs = int(summary.get("live_check_pairs", 0) or 0)
    fail_pairs = int(summary.get("fail_pairs", 0) or 0)
    seeded_only = int(summary.get("seeded_evidence_only_pairs", 0) or 0)
    ok = (
        _exists(json_path)
        and _exists(csv_path)
        and required_inputs <= input_paths
        and total_pairs >= 2016
        and live_check_pairs >= 1000
        and seeded_only > 0
        and fail_pairs == 0
    )
    return _status(
        ok,
        json.dumps(
            {
                "json": str(json_path),
                "csv": str(csv_path),
                "required_inputs": sorted(required_inputs),
                "input_paths": sorted(input_paths),
                "total_pairs": total_pairs,
                "live_check_pairs": live_check_pairs,
                "seeded_evidence_only_pairs": seeded_only,
                "fail_pairs": fail_pairs,
                "inconclusive_pairs": int(summary.get("inconclusive_pairs", 0) or 0),
                "expected_result_counts": summary.get("expected_result_counts", {}),
                "live_provider_counts": summary.get("live_provider_counts", {}),
            },
            sort_keys=True,
        ),
    )


def _eval_taxonomy_status(root: Path) -> dict[str, Any]:
    path = root / "src" / "rl_epistemics" / "eval" / "classify_outputs.py"
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    found = {label for label in REQUIRED_LABELS if label in text}
    return _status(found == REQUIRED_LABELS, f"{path}; labels={sorted(found)}")


def _adjudication_status(root: Path) -> dict[str, Any]:
    script = root / "scripts" / "adjudicate_outputs.py"
    config = root / "configs" / "adjudication.yaml"
    queue = root / "outputs" / "blog" / "manual_review_queue.csv"
    metrics = list((root / "outputs").glob("eval*/adjudicated_metrics.json")) + list(
        (root / "outputs").glob("eval*adjudicated_metrics.json")
    ) + list(
        (root / "outputs" / "blog").glob("*adjudicated_metrics.json")
    )
    ok = _exists(script) and _exists(config) and _exists(queue)
    return _status(
        ok,
        json.dumps(
            {
                "script": str(script),
                "config": str(config),
                "manual_queue": str(queue),
                "adjudicated_metrics_count": len(metrics),
                "adjudicated_metrics": [str(path) for path in metrics],
            },
            sort_keys=True,
        ),
    )


def _actual_adjudication_status(root: Path) -> dict[str, Any]:
    paths = list((root / "outputs").glob("eval*adjudicated_metrics.json")) + list(
        (root / "outputs" / "blog").glob("*adjudicated_metrics.json")
    )
    reviewed_by_source: dict[str, set[tuple[str, str, str]]] = {}

    def add_reviewed(source: str, row: dict[str, Any], csv_path: Path, row_idx: int) -> None:
        source = str(source)
        run = str(row.get("run") or row.get("run_name") or "")
        row_id = str(row.get("id") or "")
        if not row_id:
            row_id = str(row.get("prompt") or row_idx)
        reviewed_by_source.setdefault(source, set()).add((source, run, row_id))

    for csv_path in list((root / "outputs").glob("eval*adjudicated*.csv")) + list(
        (root / "outputs").glob("eval*/policy_outputs_adjudicated.csv")
    ) + list((root / "outputs" / "blog").glob("*adjudicated*.csv")) + list(
        (root / "outputs" / "blog").glob("*llm_judged*.csv")
    ):
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        has_source_col = "adjudicated_source" in df
        if has_source_col:
            for row_idx, row in enumerate(df.to_dict(orient="records")):
                source = row.get("adjudicated_source")
                if source is not None and not pd.isna(source):
                    add_reviewed(str(source), row, csv_path, row_idx)
        for source, col in [("manual", "manual_label"), ("llm_judge", "llm_judge_label")]:
            if col in df and not has_source_col:
                for row_idx, row in enumerate(df.to_dict(orient="records")):
                    label = row.get(col)
                    if _is_reviewed_label(label):
                        add_reviewed(source, row, csv_path, row_idx)
    reviewed_sources = {source: len(keys) for source, keys in reviewed_by_source.items()}
    human_or_llm = sum(
        count for source, count in reviewed_sources.items() if source in {"manual", "llm_judge"}
    )
    return _status(
        human_or_llm > 0,
        json.dumps(
            {
                "required_for_claims": True,
                "adjudicated_metrics": [str(path) for path in paths],
                "adjudicated_sources": reviewed_sources,
                "manual_or_llm_labeled_rows": human_or_llm,
                "note": "Heuristic fallback metrics can exercise the pipeline but are not human/LLM adjudication.",
            },
            sort_keys=True,
        ),
        required=False,
    )


def _adjudication_gate_progress_status(root: Path) -> dict[str, Any]:
    config_path = root / "configs" / "tinker_rl.yaml"
    try:
        import yaml

        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except Exception:
        config = {}
    gate = config.get("eval_quality_gate") or {}
    queue_path = root / str(gate.get("adjudication_path", "outputs/blog/manual_review_queue.csv"))
    label_columns = [str(col) for col in (gate.get("label_columns") or [])]
    label_col = str(gate.get("label_column", "manual_label"))
    if label_col not in label_columns:
        label_columns.insert(0, label_col)
    min_reviewed = int(gate.get("min_reviewed", 0))
    required_prompt_type_counts = {
        str(key): int(value) for key, value in (gate.get("required_prompt_type_counts") or {}).items()
    }
    reviewed = pd.DataFrame()
    counts: dict[str, int] = {}
    available_counts: dict[str, int] = {}
    if queue_path.exists():
        queue = pd.read_csv(queue_path)
        if "prompt_type" in queue:
            available_counts = {str(k): int(v) for k, v in queue["prompt_type"].value_counts().to_dict().items()}
        present_label_columns = [col for col in label_columns if col in queue.columns]
        if present_label_columns:
            labels = queue[present_label_columns[0]].fillna("").astype(str).str.strip()
            for alternate_col in present_label_columns[1:]:
                alternate = queue[alternate_col].fillna("").astype(str).str.strip()
                labels = labels.where((labels != "") & (labels != "TODO"), alternate)
            mask = (labels != "") & (labels != "TODO")
            reviewed = queue[mask]
            if "prompt_type" in reviewed:
                counts = {str(k): int(v) for k, v in reviewed["prompt_type"].value_counts().to_dict().items()}
    else:
        present_label_columns = []
    missing_total = max(0, min_reviewed - len(reviewed))
    missing_by_prompt_type = {
        prompt_type: max(0, required - counts.get(prompt_type, 0))
        for prompt_type, required in required_prompt_type_counts.items()
    }
    ok = missing_total == 0 and all(value == 0 for value in missing_by_prompt_type.values())
    return _status(
        ok,
        json.dumps(
            {
                "required_for_long_rl": True,
                "queue_path": str(queue_path),
                "label_columns": label_columns,
                "present_label_columns": present_label_columns,
                "reviewed_total": int(len(reviewed)),
                "min_reviewed": min_reviewed,
                "missing_total": missing_total,
                "available_by_prompt_type": available_counts,
                "reviewed_by_prompt_type": counts,
                "required_prompt_type_counts": required_prompt_type_counts,
                "missing_by_prompt_type": missing_by_prompt_type,
            },
            sort_keys=True,
        ),
        required=False,
    )


def _adjudicated_failure_taxonomy_status(root: Path) -> dict[str, Any]:
    csv_path = root / "outputs" / "report" / "adjudication_failure_taxonomy.csv"
    report_md = root / "outputs" / "report" / "adjudication_failure_taxonomy.md"
    blog_md = root / "outputs" / "blog" / "adjudication_failure_taxonomy.md"
    blog_path = root / "blog" / "rl_for_epistemic_humility.md"
    required_columns = {
        "source_path",
        "run",
        "domain",
        "prompt_type",
        "n",
        "clean_uncertainty",
        "false_premise_correction",
        "hedged_confabulation",
        "confident_confabulation",
        "real_substantive_answer",
        "false_refusal",
        "unclear",
        "fake_good_handling",
        "fake_bad_confabulation",
        "real_good_answer",
        "real_false_refusal",
    }
    evidence: dict[str, Any] = {
        "csv": str(csv_path),
        "report_md": str(report_md),
        "blog_md": str(blog_md),
        "blog": str(blog_path),
    }
    if not csv_path.exists():
        evidence.update({"missing": [str(csv_path)]})
        return _status(False, json.dumps(evidence, sort_keys=True), required=False)
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        evidence.update({"csv_read_error": str(exc)})
        return _status(False, json.dumps(evidence, sort_keys=True), required=False)

    missing_columns = sorted(required_columns - set(df.columns))
    if "n" in df:
        n_values = pd.to_numeric(df["n"], errors="coerce").fillna(0)
    else:
        n_values = pd.Series(dtype=float)
    if "prompt_type" in df:
        prompt_types = df["prompt_type"].fillna("").astype(str)
        fake_n = int(n_values[prompt_types == "fake"].sum())
        real_n = int(n_values[prompt_types == "real"].sum())
    else:
        prompt_types = pd.Series(dtype=str)
        fake_n = 0
        real_n = 0
    total_n = int(n_values.sum()) if not n_values.empty else 0

    source_paths = sorted({str(path) for path in df.get("source_path", pd.Series(dtype=str)).dropna().unique() if str(path)})
    source_labeled_rows = 0
    source_counts: dict[str, int] = {}
    source_missing: list[str] = []
    for raw_path in source_paths:
        source_path = Path(raw_path)
        if not source_path.is_absolute():
            source_path = root / source_path
        if not source_path.exists():
            source_missing.append(str(source_path))
            continue
        try:
            source_df = pd.read_csv(source_path)
        except Exception:
            source_missing.append(str(source_path))
            continue
        label_col = next(
            (col for col in ["adjudicated_label", "manual_label", "llm_judge_label", "judge_label"] if col in source_df),
            None,
        )
        if not label_col:
            continue
        labels = source_df[label_col].fillna("").astype(str).str.strip()
        label_mask = labels.isin(REQUIRED_LABELS)
        if "adjudicated_source" in source_df:
            sources = source_df["adjudicated_source"].fillna("").astype(str).str.strip()
            label_mask &= sources.str.lower().ne("heuristic")
            for source, count in sources[label_mask].value_counts().to_dict().items():
                source_counts[str(source)] = source_counts.get(str(source), 0) + int(count)
        source_labeled_rows += int(label_mask.sum())

    fake_bad = pd.to_numeric(df.get("fake_bad_confabulation", pd.Series(dtype=float)), errors="coerce")
    real_answer = pd.to_numeric(df.get("real_good_answer", pd.Series(dtype=float)), errors="coerce")
    blog_text = blog_path.read_text(encoding="utf-8") if blog_path.exists() else ""
    ok = (
        not df.empty
        and not missing_columns
        and _exists(report_md)
        and _exists(blog_md)
        and "Adjudicated Failure Taxonomy" in blog_text
        and total_n >= 30
        and fake_n >= 20
        and real_n >= 10
        and source_labeled_rows >= total_n
        and not source_missing
        and fake_bad.notna().any()
        and real_answer.notna().any()
    )
    evidence.update(
        {
            "rows": int(len(df)),
            "total_reviewed_n": total_n,
            "fake_n": fake_n,
            "real_n": real_n,
            "missing_columns": missing_columns,
            "source_paths": source_paths,
            "source_missing": source_missing,
            "source_labeled_rows": source_labeled_rows,
            "source_counts": source_counts,
            "blog_section_present": "Adjudicated Failure Taxonomy" in blog_text,
            "max_fake_bad_confabulation": float(fake_bad.max()) if fake_bad.notna().any() else None,
            "max_real_answer": float(real_answer.max()) if real_answer.notna().any() else None,
        }
    )
    return _status(ok, json.dumps(evidence, sort_keys=True), required=False)


def _quality_gate_status(root: Path) -> dict[str, Any]:
    train_tinker = root / "src" / "rl_epistemics" / "rl" / "train_tinker.py"
    config = root / "configs" / "tinker_rl.yaml"
    text = train_tinker.read_text(encoding="utf-8") if train_tinker.exists() else ""
    cfg = config.read_text(encoding="utf-8") if config.exists() else ""
    ok = "_enforce_eval_quality_gate" in text and "eval_quality_gate" in cfg
    return _status(ok, f"{train_tinker}; {config}")


def _rm_design_status(root: Path) -> dict[str, Any]:
    manifest = root / "outputs" / "rm_design_ablations" / "manifest.json"
    summary = root / "outputs" / "rm_design_ablations" / "rm_design_summary.csv"
    if not manifest.exists():
        return _status(False, f"missing {manifest}")
    variants = json.loads(manifest.read_text(encoding="utf-8"))
    names = {row.get("name") for row in variants}
    expected = {
        "last_final_token_matrix_sum",
        "last_mean_completion_matrix_sum",
        "middle_final_token_matrix_sum",
        "last_completion_only_matrix_sum",
        "last_final_token_scalar_linear",
        "last_final_token_mlp",
    }
    expected_dataset = str(root / "data" / "verified_train" / "pairs.jsonl")
    expected_dataset_rel = "data/verified_train/pairs.jsonl"
    expected_dataset_modal = "/artifacts/data/verified_train/pairs.jsonl"
    configured_datasets: dict[str, str] = {}
    stale_dataset_configs: list[str] = []
    for row in variants:
        name = str(row.get("name", ""))
        config_path = root / "outputs" / "rm_design_ablations" / name / "reward_model.yaml"
        if not config_path.exists() and row.get("run_dir"):
            config_path = root / str(row.get("run_dir", "")) / "reward_model.yaml"
        if not config_path.exists() and row.get("run_dir"):
            config_path = Path(str(row["run_dir"])) / "reward_model.yaml"
        if not config_path.exists():
            stale_dataset_configs.append(f"{name}:missing_config")
            continue
        try:
            dataset_path = str(load_config(config_path).get("dataset_path", ""))
        except Exception:
            dataset_path = ""
        configured_datasets[name] = dataset_path
        if dataset_path not in {expected_dataset_rel, expected_dataset, expected_dataset_modal}:
            stale_dataset_configs.append(f"{name}:{dataset_path}")
    eval_summaries = list((root / "outputs" / "rm_design_ablations").glob("*/eval/reward_summary.json"))
    ok = expected <= names and _exists(summary) and not stale_dataset_configs
    return _status(
        ok,
        json.dumps(
            {
                "manifest": str(manifest),
                "summary": str(summary),
                "variant_count": len(variants),
                "expected_present": sorted(expected <= names and expected or names),
                "expected_dataset_path": expected_dataset_rel,
                "configured_datasets": configured_datasets,
                "stale_dataset_configs": stale_dataset_configs,
                "eval_summary_count": len(eval_summaries),
            },
            sort_keys=True,
        ),
    )


def _rm_ablation_config_path(root: Path, row: dict[str, Any]) -> Path:
    name = str(row.get("name", ""))
    path = root / "outputs" / "rm_design_ablations" / name / "reward_model.yaml"
    if path.exists():
        return path
    if row.get("run_dir"):
        candidate = root / str(row.get("run_dir", "")) / "reward_model.yaml"
        if candidate.exists():
            return candidate
        candidate = Path(str(row["run_dir"])) / "reward_model.yaml"
        if candidate.exists():
            return candidate
    return path


def _has_limit(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() not in {"", "none", "null"}
    return True


def _actual_rm_design_results_status(root: Path) -> dict[str, Any]:
    manifest = root / "outputs" / "rm_design_ablations" / "manifest.json"
    variants = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else []
    expected = {str(row.get("name")) for row in variants if row.get("name")}
    base_dir = root / "outputs" / "rm_design_ablations"
    eval_summaries = sorted(base_dir.glob("*/eval/reward_summary.json"))
    reward_scores = sorted(base_dir.glob("*/eval/reward_scores.csv"))
    summary_variants = {path.parent.parent.name for path in eval_summaries}
    score_variants = {path.parent.parent.name for path in reward_scores}
    missing_summary_variants = sorted(expected - summary_variants)
    missing_score_variants = sorted(expected - score_variants)
    current_dataset = _dataset_file_summary(root / "data" / "verified_train" / "pairs.jsonl")
    manifest_dataset_summaries = {
        str(row.get("name")): row.get("dataset_summary") or {}
        for row in variants
        if row.get("name")
    }
    missing_dataset_summary_variants = sorted(
        name
        for name in expected
        if not manifest_dataset_summaries.get(name, {}).get("sha256")
    )
    stale_dataset_variants = sorted(
        name
        for name, summary in manifest_dataset_summaries.items()
        if summary.get("sha256")
        and current_dataset.get("sha256")
        and summary.get("sha256") != current_dataset.get("sha256")
    )
    low_seed_dataset_variants = sorted(
        name
        for name, summary in manifest_dataset_summaries.items()
        if summary.get("sha256")
        and int(summary.get("unique_seed_prompts") or 0) < MIN_VERIFIED_SEED_PROMPTS
    )
    missing_config_variants: list[str] = []
    limited_dataset_variants: list[str] = []
    for row in variants:
        name = str(row.get("name", ""))
        if not name:
            continue
        run_config = row.get("run_config") or {}
        config_path = _rm_ablation_config_path(root, row)
        file_config: dict[str, Any] = {}
        if config_path.exists():
            try:
                file_config = load_config(config_path)
            except Exception:
                file_config = {}
        else:
            missing_config_variants.append(name)
        limit_train = run_config.get("limit_train_examples", file_config.get("limit_train_examples"))
        limit_eval = run_config.get("limit_eval_examples", file_config.get("limit_eval_examples"))
        if _has_limit(limit_train) or _has_limit(limit_eval):
            limited_dataset_variants.append(name)
    limited_dataset_variants = sorted(set(limited_dataset_variants))
    missing_config_variants = sorted(set(missing_config_variants))
    ok = (
        bool(expected)
        and not missing_summary_variants
        and not missing_score_variants
        and not missing_dataset_summary_variants
        and not stale_dataset_variants
        and not low_seed_dataset_variants
        and not missing_config_variants
        and not limited_dataset_variants
    )
    return _status(
        ok,
        json.dumps(
            {
                "required_for_claims": True,
                "manifest": str(manifest),
                "expected_variant_count": len(expected),
                "expected_variants": sorted(expected),
                "eval_summary_count": len(eval_summaries),
                "reward_scores_count": len(reward_scores),
                "current_dataset_summary": current_dataset,
                "manifest_dataset_summaries": manifest_dataset_summaries,
                "eval_summaries": [str(path) for path in eval_summaries],
                "reward_scores": [str(path) for path in reward_scores],
                "missing_summary_variants": missing_summary_variants,
                "missing_score_variants": missing_score_variants,
                "missing_dataset_summary_variants": missing_dataset_summary_variants,
                "stale_dataset_variants": stale_dataset_variants,
                "low_seed_dataset_variants": low_seed_dataset_variants,
                "missing_config_variants": missing_config_variants,
                "limited_dataset_variants": limited_dataset_variants,
            },
            sort_keys=True,
        ),
        required=False,
    )


def _reward_transform_status(root: Path) -> dict[str, Any]:
    script = root / "scripts" / "audit_reward_transforms.py"
    config = root / "configs" / "reward_transform_audit.yaml"
    audit = root / "outputs" / "report" / "reward_transform_audit.csv"
    scores = root / "outputs" / "reward_model" / "eval" / "reward_scores.csv"
    ablation_scores = list((root / "outputs" / "rm_design_ablations").glob("*/eval/reward_scores.csv"))
    ok = _exists(script) and _exists(config)
    return _status(
        ok,
        json.dumps(
            {
                "script": str(script),
                "config": str(config),
                "audit_exists": _exists(audit),
                "reward_scores_exist": _exists(scores),
                "rm_ablation_reward_scores_count": len(ablation_scores),
            },
            sort_keys=True,
        ),
    )


def _chosen_rejected_reward_audit_status(root: Path) -> dict[str, Any]:
    scores = root / "outputs" / "reward_model" / "eval" / "reward_scores.csv"
    ablation_scores = list((root / "outputs" / "rm_design_ablations").glob("*/eval/reward_scores.csv"))
    manifest = root / "outputs" / "rm_design_ablations" / "manifest.json"
    variants = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else []
    expected_variants = {str(row.get("name")) for row in variants if row.get("name")}
    score_variants = {path.parent.parent.name for path in ablation_scores}
    audit = root / "outputs" / "report" / "reward_transform_audit.csv"
    audit_kind = None
    audit_runs: list[str] = []
    if audit.exists():
        try:
            df = pd.read_csv(audit)
            if "audit_kind" in df:
                audit_kind = sorted(df["audit_kind"].dropna().unique().tolist())
            if "run" in df:
                audit_runs = sorted(str(run) for run in df["run"].dropna().unique().tolist())
        except Exception:
            audit_kind = None
    has_scores = _exists(scores) or bool(ablation_scores)
    missing_score_variants = sorted(expected_variants - score_variants)
    missing_audit_runs = sorted(expected_variants - set(audit_runs)) if expected_variants else []
    full_ablation_audit = (
        bool(expected_variants)
        and not missing_score_variants
        and not missing_audit_runs
    )
    return _status(
        has_scores
        and audit_kind is not None
        and "chosen_rejected_pairs" in audit_kind
        and (full_ablation_audit or _exists(scores)),
        json.dumps(
            {
                "required_for_claims": True,
                "expected_variant_count": len(expected_variants),
                "expected_variants": sorted(expected_variants),
                "reward_scores_exist": _exists(scores),
                "rm_ablation_reward_scores_count": len(ablation_scores),
                "rm_ablation_score_variants": sorted(score_variants),
                "missing_score_variants": missing_score_variants,
                "audit_path": str(audit),
                "audit_kind": audit_kind,
                "audit_runs": audit_runs,
                "missing_audit_runs": missing_audit_runs,
                "note": "Policy-output reward audit is useful but does not replace chosen/rejected RM-pair transform testing.",
            },
            sort_keys=True,
        ),
        required=False,
    )


def _post_gate_long_rl_status(root: Path) -> dict[str, Any]:
    config_path = root / "configs" / "tinker_rl_post_gate.yaml"
    config = load_config(config_path) if config_path.exists() else {}
    modal_preflight_path = root / "outputs" / "report" / "modal_preflight.json"
    modal_preflight = _read_json(modal_preflight_path)
    output_dir_value = config.get("output_dir", "outputs/tinker_rl_post_gate")
    output_dir = Path(output_dir_value)
    if not output_dir.is_absolute():
        output_dir = root / output_dir

    metrics = output_dir / "metrics.jsonl"
    run_meta = output_dir / "run_meta.json"
    train_config_path = output_dir / "train_config.yaml"
    meta = json.loads(run_meta.read_text(encoding="utf-8")) if run_meta.exists() else {}
    train_config = load_config(train_config_path) if train_config_path.exists() else {}
    rows = _read_jsonl_dicts(metrics)

    gate = train_config.get("eval_quality_gate") or config.get("eval_quality_gate") or {}
    threshold = gate.get("required_for_max_steps_over", 20)
    configured_steps = int(train_config.get("max_steps", config.get("max_steps", 0)) or 0)
    last_step = max((int(row.get("step", -1)) for row in rows), default=-1)
    completed_steps = len({int(row.get("step", -1)) for row in rows if row.get("step") is not None})
    quality_gate = meta.get("quality_gate") or {}
    dataset_path = str(train_config.get("dataset_path", config.get("dataset_path", "")))
    reward_model_dir = train_config.get("reward_model_dir", config.get("reward_model_dir", ""))
    reward_model_path = Path(str(reward_model_dir))
    if reward_model_dir and not reward_model_path.is_absolute():
        reward_model_path = root / reward_model_path
    expected_dataset = str(root / "data" / "verified_train" / "pairs.jsonl")
    dataset_ok = dataset_path in {"data/verified_train/pairs.jsonl", expected_dataset}
    reward_model_ok = bool(reward_model_dir) and _exists(reward_model_path / "reward_head.pt")
    no_prompt_limit = train_config.get("limit_prompts", config.get("limit_prompts")) in (None, "")
    longer_than_gate = configured_steps > int(threshold)
    completed_configured_steps = configured_steps > 0 and completed_steps >= configured_steps and last_step >= configured_steps - 1
    gate_enabled = quality_gate.get("enabled") is True and quality_gate.get("reviewed", 0) >= int(gate.get("min_reviewed", 30))
    ok = (
        _exists(metrics)
        and _exists(run_meta)
        and _exists(train_config_path)
        and config_path.exists()
        and longer_than_gate
        and completed_configured_steps
        and gate_enabled
        and dataset_ok
        and reward_model_ok
        and no_prompt_limit
    )
    return _status(
        ok,
        json.dumps(
            {
                "required_for_claims": True,
                "config": str(config_path),
                "output_dir": str(output_dir),
                "metrics": str(metrics),
                "run_meta": str(run_meta),
                "train_config": str(train_config_path),
                "configured_steps": configured_steps,
                "completed_steps": completed_steps,
                "last_step": last_step,
                "required_for_max_steps_over": threshold,
                "longer_than_gate": longer_than_gate,
                "completed_configured_steps": completed_configured_steps,
                "quality_gate": quality_gate,
                "configured_quality_gate": gate,
                "gate_enabled": gate_enabled,
                "dataset_path": dataset_path,
                "dataset_ok": dataset_ok,
                "reward_model_dir": str(reward_model_dir),
                "reward_model_ok": reward_model_ok,
                "no_prompt_limit": no_prompt_limit,
                "credential_env_present": {
                    "TINKER_API_KEY": bool(os.environ.get("TINKER_API_KEY")),
                    "MODAL_TOKEN_ID": bool(os.environ.get("MODAL_TOKEN_ID")),
                    "MODAL_TOKEN_SECRET": bool(os.environ.get("MODAL_TOKEN_SECRET")),
                },
                "modal_preflight": {
                    "path": str(modal_preflight_path),
                    "exists": bool(modal_preflight),
                    "created_at_utc": modal_preflight.get("created_at_utc"),
                    "summary": modal_preflight.get("summary", {}),
                },
                "note": "Longer RL should run only after manual/LLM adjudication quality gate passes.",
            },
            sort_keys=True,
        ),
        required=False,
    )


def _policy_results_status(root: Path) -> dict[str, Any]:
    paths = [
        root / "outputs" / "eval_tinker_hard_base_policy_metrics.json",
        root / "outputs" / "eval_tinker_hard_main_policy_metrics.json",
        root / "outputs" / "report" / "policy_domain_breakdown.csv",
        root / "outputs" / "report" / "policy_metrics_table.csv",
    ]
    ok = all(_exists(path) for path in paths)
    evidence: dict[str, Any] = {"paths": [str(path) for path in paths]}
    metrics_path = root / "outputs" / "eval_tinker_hard_main_policy_metrics.json"
    if metrics_path.exists():
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
        evidence["rl_hard_fake_confabulates"] = data.get("fake_fake_confabulates")
        evidence["rl_hard_real_false_refusal"] = data.get("real_real_false_refusal")
    return _status(ok, json.dumps(evidence, sort_keys=True))


def _blog_status(root: Path) -> dict[str, Any]:
    path = root / "blog" / "rl_for_epistemic_humility.md"
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    required_phrases = [
        "mixed-to-negative",
        "does not show robust transfer",
        "Hard OOD fake confabulation",
        "Adjudicated Failure Taxonomy",
        "verified train/eval seed coverage",
        "RM design current-data eval results",
        "second-stage publishable claims:",
    ]
    ok = path.exists() and all(phrase in text for phrase in required_phrases)
    return _status(ok, f"{path}; phrases_present={[phrase for phrase in required_phrases if phrase in text]}")


def _modal_surface_status(root: Path) -> dict[str, Any]:
    path = root / "modal_app.py"
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    required_tokens = [
        "def modal_make_verified_dataset",
        "def modal_make_verified_train_dataset",
        "def modal_eval_verified_tinker",
        "def modal_eval_post_gate_tinker",
        "def modal_run_rm_design_ablations",
        "def modal_train_post_gate_tinker",
        "def run_post_gate_tinker",
        "def run_post_gate_eval_tinker",
        "def modal_collect_results",
        "def modal_write_blog",
        'suite_name == "second_stage_data"',
        'suite_name == "post_gate_long_rl"',
        'suite_name == "post_gate_eval"',
        'results["verified_train_dataset"]',
    ]
    present = [token for token in required_tokens if token in text]
    missing = [token for token in required_tokens if token not in text]
    return _status(
        not missing,
        json.dumps(
            {
                "path": str(path),
                "present": present,
                "missing": missing,
            },
            sort_keys=True,
        ),
    )


def _post_gate_recovery_surface_status(root: Path) -> dict[str, Any]:
    runner = root / "src" / "rl_epistemics" / "experiments" / "run_post_gate_pipeline.py"
    runner_script = root / "scripts" / "run_post_gate_pipeline.py"
    train_script = root / "scripts" / "train_tinker_rl.py"
    eval_script = root / "scripts" / "eval_post_gate_tinker.py"
    key_helper = root / "scripts" / "write_tinker_key_file.py"
    config = root / "configs" / "tinker_rl_post_gate.yaml"
    text = runner.read_text(encoding="utf-8") if runner.exists() else ""
    required_tokens = [
        "build_post_gate_pipeline_steps",
        "build_local_post_gate_pipeline_steps",
        "wait_for_modal_status",
        "fetch_modal_status",
        "--local-tinker",
        "--tinker-api-key-file",
        "scripts/eval_post_gate_tinker.py",
        "scripts/train_tinker_rl.py",
        "scripts/write_tinker_key_file.py",
        "Local Tinker key file is missing or empty",
        "refresh-second-stage-audit",
    ]
    present = [token for token in required_tokens if token in text]
    missing = [token for token in required_tokens if token not in text]
    paths = {
        "runner_module": str(runner),
        "runner_script": str(runner_script),
        "train_script": str(train_script),
        "eval_script": str(eval_script),
        "key_helper": str(key_helper),
        "config": str(config),
    }
    missing_paths = [path for path in paths.values() if not _exists(Path(path))]
    return _status(
        not missing and not missing_paths,
        json.dumps(
            {
                "paths": paths,
                "missing_paths": missing_paths,
                "present": present,
                "missing": missing,
                "modal_guarded": "Modal status page reports an active incident; refusing launch" in text,
                "local_key_guarded": "Local Tinker key file is missing or empty" in text,
            },
            sort_keys=True,
        ),
        required=False,
    )


def _requirement_matrix(checks: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in REQUIREMENT_SPECS:
        check_names = list(spec["checks"])
        missing = [
            check_name
            for check_name in check_names
            if not checks.get(check_name, {"ok": False}).get("ok", False)
        ]
        rows.append(
            {
                "id": spec["id"],
                "requirement": spec["requirement"],
                "checks": check_names,
                "ok": not missing,
                "missing_checks": missing,
                "stage": spec["stage"],
                "blocks_publishable_claims": bool(spec["blocks_publishable_claims"]),
            }
        )
    return rows


def _markdown_escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _check_summary(check: dict[str, Any]) -> str:
    status = "present" if check.get("ok") else "missing/incomplete"
    return f"{status}; required={str(check.get('required', True)).lower()}"


def _write_pipeline_checklist(root: Path, path: Path, result: dict[str, Any]) -> None:
    post_gate_complete = result["checks"].get("post_gate_longer_rl_run", {}).get("ok", False)
    lines = [
        "# Second-Stage Pipeline Checklist",
        "",
        "Objective restated as deliverables: build a stronger second-stage research pipeline for RL-based knowledge awareness by improving the data, adjudication, eval taxonomy, domain coverage, RL gating, RM comparisons, reward-transform checks, Modal execution surface, and honest report/blog framing.",
        "",
        "This is a prompt-to-artifact completion audit. Passing tests or a green manifest is not treated as enough by itself; each row maps an explicit requirement to concrete files, commands, gates, and audit evidence.",
        "",
        "## Current Verdict",
        "",
        f"- Local pipeline surfaces ready: `{result['ok']}`.",
        f"- Scoped publishable claims ready: `{result['publishable_claims_ready']}`.",
        f"- Executed post-gate longer RL run present: `{post_gate_complete}`.",
        "",
        "## Prompt-to-Artifact Checklist",
        "",
    ]

    checks = result["checks"]
    for row in result["requirements"]:
        status = "present" if row["ok"] else "missing/incomplete"
        lines.extend(
            [
                f"### {row['id']}",
                "",
                f"- Requirement: {row['requirement']}",
                f"- Status: {status}.",
                f"- Stage: {row['stage']}.",
                f"- Blocks scoped publishable claims: {str(row['blocks_publishable_claims']).lower()}.",
                f"- Missing checks: {', '.join(row['missing_checks']) or '-'}.",
                "- Evidence:",
            ]
        )
        for check_name in row["checks"]:
            check = checks.get(check_name, {"ok": False, "required": True, "evidence": "missing check implementation"})
            lines.append(f"  - `{check_name}`: {_check_summary(check)}. {check['evidence']}")
        lines.append("")

    lines.extend(
        [
            "## Verification Commands",
            "",
            "- Full test suite: `PYTHONPATH=src python3 -m pytest -q`.",
            "- Compile check: `PYTHONPATH=src python3 -m compileall -q src scripts modal_app.py`.",
            "- Readiness audit: `PYTHONPATH=src python3 scripts/audit_second_stage.py`.",
            "- Modal preflight: `PYTHONPATH=src python3 scripts/modal_preflight.py`.",
            "- Failure-report refresh: `PYTHONPATH=src python3 -c \"from rl_epistemics.reporting.failures import write_failure_reports; write_failure_reports('outputs', 'outputs/report', 'outputs/blog')\"`.",
            "- Reward-transform audit refresh: `PYTHONPATH=src python3 scripts/audit_reward_transforms.py --scores-glob 'outputs/rm_design_ablations/*/eval/reward_scores.csv' --output-dir outputs/report`.",
            "- Blog refresh: `PYTHONPATH=src python3 -c \"from rl_epistemics.reporting.write_blog import write_blog; write_blog('outputs', 'blog/rl_for_epistemic_humility.md')\"`.",
            "",
            "## Post-Gate RL Commands",
            "",
            "- Full post-gate recovery pipeline: `PYTHONPATH=src python3 scripts/run_post_gate_pipeline.py`.",
            "- Wait for Modal recovery, then run pipeline: `PYTHONPATH=src python3 scripts/run_post_gate_pipeline.py --wait-for-modal --poll-seconds 60`.",
            "- Dry-run the recovery pipeline: `PYTHONPATH=src python3 scripts/run_post_gate_pipeline.py --dry-run --skip-status-check`.",
            "- Local Tinker key-file helper: `python3 scripts/write_tinker_key_file.py`.",
            "- Local post-gate train/eval/report wrapper: `PYTHONPATH=src python3 scripts/run_post_gate_pipeline.py --local-tinker --tinker-api-key-file /tmp/rl_epistemics_tinker_key`.",
            "- Local Tinker train-only command: `PYTHONPATH=src python3 scripts/train_tinker_rl.py --config configs/tinker_rl_post_gate.yaml --tinker-api-key-file /tmp/rl_epistemics_tinker_key`.",
            "- Local post-gate eval-only command: `PYTHONPATH=src python3 scripts/eval_post_gate_tinker.py --tinker-api-key-file /tmp/rl_epistemics_tinker_key`.",
            "- Modal post-gate training entrypoint: `python3 -m modal run modal_app.py::run_post_gate_tinker`.",
            "- Modal Tinker run: `python3 -m modal run modal_app.py::modal_train_post_gate_tinker`.",
            "- Modal suite run: `python3 -m modal run modal_app.py::modal_run_experiment_suite --suite-name post_gate_long_rl`.",
            "- Modal post-gate eval entrypoint: `python3 -m modal run modal_app.py::run_post_gate_eval_tinker`.",
            "- Modal post-gate eval suite: `python3 -m modal run modal_app.py::modal_run_experiment_suite --suite-name post_gate_eval`.",
            "",
            "The post-gate run needs Tinker/Modal credentials and must write `outputs/tinker_rl_post_gate/metrics.jsonl`, `run_meta.json`, and `train_config.yaml` with all configured steps completed. The audit intentionally does not accept first-pass `outputs/tinker_rl` metrics as satisfying this requirement.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_second_stage_audit(root_dir: str | Path = ".") -> dict[str, Any]:
    root = Path(root_dir).resolve()
    checks = {
        "verified_dataset": _verified_dataset_status(root),
        "verified_train_val_test_dataset": _verified_train_dataset_status(root),
        "domain_coverage_beyond_researchers": _domain_coverage_status(root),
        "verified_retrieval_audit": _verified_retrieval_audit_status(root),
        "eval_taxonomy": _eval_taxonomy_status(root),
        "adjudication_tooling": _adjudication_status(root),
        "actual_human_or_llm_adjudication": _actual_adjudication_status(root),
        "adjudication_gate_progress": _adjudication_gate_progress_status(root),
        "adjudicated_failure_taxonomy": _adjudicated_failure_taxonomy_status(root),
        "longer_rl_quality_gate": _quality_gate_status(root),
        "rm_design_ablation_surface": _rm_design_status(root),
        "actual_rm_design_eval_results": _actual_rm_design_results_status(root),
        "reward_transform_audit_surface": _reward_transform_status(root),
        "chosen_rejected_reward_transform_audit": _chosen_rejected_reward_audit_status(root),
        "hard_ood_policy_results_ingested": _policy_results_status(root),
        "post_gate_recovery_surface": _post_gate_recovery_surface_status(root),
        "post_gate_longer_rl_run": _post_gate_long_rl_status(root),
        "honest_blog_report": _blog_status(root),
        "modal_second_stage_surface": _modal_surface_status(root),
    }
    required_ok = all(check["ok"] for check in checks.values() if check.get("required", True))
    requirements = _requirement_matrix(checks)
    publishable_claims_ready = all(row["ok"] for row in requirements if row["blocks_publishable_claims"])
    return {
        "ok": required_ok,
        "publishable_claims_ready": publishable_claims_ready,
        "requirements": requirements,
        "checks": checks,
    }


def write_second_stage_audit(config: dict[str, Any]) -> dict[str, Any]:
    root = Path(config.get("root_dir", ".")).resolve()
    output_json = root / config.get("output_json", "outputs/report/second_stage_audit.json")
    output_md = root / config.get("output_md", "outputs/report/second_stage_audit.md")
    output_checklist = root / config.get("output_checklist", "outputs/report/second_stage_pipeline_checklist.md")
    result = run_second_stage_audit(root)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = [
        "# Second-Stage Audit",
        "",
        f"Local pipeline surfaces ready: `{result['ok']}`",
        f"Publishable claims ready: `{result['publishable_claims_ready']}`",
        "",
        "Checks marked missing/incomplete with `required=false` are external execution steps, not local wiring failures.",
        "Publishable claims are not ready until all blocking requirements in the matrix are present.",
        "",
        "## Requirement Matrix",
        "",
        "| Requirement | Status | Stage | Checks | Missing | Blocks publishable claims |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in result["requirements"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    _markdown_escape(row["requirement"]),
                    "present" if row["ok"] else "missing/incomplete",
                    _markdown_escape(row["stage"]),
                    _markdown_escape(", ".join(row["checks"])),
                    _markdown_escape(", ".join(row["missing_checks"]) or "-"),
                    str(row["blocks_publishable_claims"]).lower(),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Check Evidence",
            "",
        ]
    )
    for name, check in result["checks"].items():
        status = "present" if check["ok"] else "missing/incomplete"
        lines.append(
            f"- `{name}`: {status}; required={str(check.get('required', True)).lower()}. {check['evidence']}"
        )
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_pipeline_checklist(root, output_checklist, result)
    return {
        "output_json": str(output_json),
        "output_md": str(output_md),
        "output_checklist": str(output_checklist),
        **result,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Audit second-stage pipeline readiness.")
    parser.add_argument("--root-dir", default=".")
    parser.add_argument("--output-json", default="outputs/report/second_stage_audit.json")
    parser.add_argument("--output-md", default="outputs/report/second_stage_audit.md")
    parser.add_argument("--output-checklist", default="outputs/report/second_stage_pipeline_checklist.md")
    args = parser.parse_args(argv)
    print(
        json.dumps(
            write_second_stage_audit(
                {
                    "root_dir": args.root_dir,
                    "output_json": args.output_json,
                    "output_md": args.output_md,
                    "output_checklist": args.output_checklist,
                }
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
