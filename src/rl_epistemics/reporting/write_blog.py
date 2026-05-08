from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _read_json_evidence(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return {}
    try:
        data = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _read_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    if path.suffix in {".yaml", ".yml"}:
        import yaml

        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data if isinstance(data, dict) else {}
    return _read_json(path)


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "TODO"
    if isinstance(value, float):
        if math.isnan(value):
            return "TODO"
        return f"{value:.{digits}f}"
    return str(value)


def _pct(value: Any) -> str:
    if value is None:
        return "TODO"
    try:
        return f"{100 * float(value):.1f}%"
    except (TypeError, ValueError):
        return "TODO"


def _int(value: Any) -> str:
    if value is None:
        return "TODO"
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return str(value)


def _count(summary: dict[str, Any], rate_key: str, n_key: str) -> str:
    rate = summary.get(rate_key)
    n = summary.get(n_key)
    if rate is None or n in (None, 0):
        return "TODO"
    count = int(round(float(rate) * int(n)))
    return f"{count}/{int(n)}"


def _rate_count(summary: dict[str, Any], rate_key: str, n_key: str) -> str:
    if not summary:
        return "TODO"
    return f"{_pct(summary.get(rate_key))} ({_count(summary, rate_key, n_key)})"


def _reward_summary(outputs: Path) -> dict[str, Any]:
    return _read_json(outputs / "reward_model" / "eval" / "reward_summary.json") or _read_json(
        outputs / "reward_model" / "score_summary.json"
    ) or _read_json(
        outputs / "rm_design_ablations" / "last_final_token_matrix_sum" / "eval" / "reward_summary.json"
    )


def _known_policy_runs(outputs: Path) -> dict[str, dict[str, Any]]:
    candidates = {
        "base_test": outputs / "eval_tinker_base_test" / "policy_metrics.json",
        "main_test": outputs / "eval_tinker_main" / "policy_metrics.json",
        "base_ood": outputs / "eval_tinker_base_ood" / "policy_metrics.json",
        "main_ood": outputs / "eval_tinker_main_ood" / "policy_metrics.json",
        "base_hard": outputs / "eval_tinker_hard_base" / "policy_metrics.json",
        "main_hard": outputs / "eval_tinker_hard_main" / "policy_metrics.json",
        "post_gate_verified": outputs / "eval_tinker_verified_post_gate" / "policy_metrics.json",
        "post_gate_hard": outputs / "eval_tinker_hard_post_gate" / "policy_metrics.json",
    }
    if not candidates["main_test"].exists():
        candidates["main_test"] = outputs / "eval_tinker" / "policy_metrics.json"
    if not candidates["base_hard"].exists():
        candidates["base_hard"] = outputs / "eval_tinker_hard_base_policy_metrics.json"
    if not candidates["main_hard"].exists():
        candidates["main_hard"] = outputs / "eval_tinker_hard_main_policy_metrics.json"
    return {name: _read_json(path) for name, path in candidates.items()}


def _ablation_policy_runs(outputs: Path) -> dict[str, dict[str, Any]]:
    ablation_dir = outputs / "eval_tinker_ablations"
    runs: dict[str, dict[str, Any]] = {}
    for metrics_path in sorted(ablation_dir.glob("*/policy_metrics.json")):
        runs[metrics_path.parent.name] = _read_json(metrics_path)
    return runs


def _latest_training_row(path: Path) -> dict[str, Any]:
    rows = _read_jsonl(path)
    return rows[-1] if rows else {}


def _training_note(path: Path) -> str:
    rows = _read_jsonl(path)
    if not rows:
        return "TODO"
    losses = [float(row["loss"]) for row in rows if row.get("loss") is not None]
    degenerate = [float(row["degenerate_group_fraction"]) for row in rows if row.get("degenerate_group_fraction") is not None]
    parts = []
    if losses:
        parts.append(f"final loss {_fmt(losses[-1])}")
        parts.append(f"max_abs_loss {_fmt(max(abs(loss) for loss in losses))}")
    if degenerate:
        parts.append(f"max degenerate {_pct(max(degenerate))}")
    return ", ".join(parts) if parts else "metrics present"


def _policy_table(policy: dict[str, dict[str, Any]]) -> str:
    candidates = [
        ("base test", policy.get("base_test", {}), "test"),
        ("RL test", policy.get("main_test", {}), "test"),
        ("base OOD", policy.get("base_ood", {}), "ood"),
        ("RL OOD", policy.get("main_ood", {}), "ood"),
    ]
    if policy.get("base_hard") or policy.get("main_hard"):
        candidates.extend(
            [
                ("base hard OOD", policy.get("base_hard", {}), "hard_ood"),
                ("RL hard OOD", policy.get("main_hard", {}), "hard_ood"),
            ]
        )
    if policy.get("post_gate_verified") or policy.get("post_gate_hard"):
        candidates.extend(
            [
                ("post-gate verified OOD", policy.get("post_gate_verified", {}), "verified_ood"),
                ("post-gate hard OOD", policy.get("post_gate_hard", {}), "hard_ood"),
            ]
        )
    rows = [(name, summary, split) for name, summary, split in candidates if summary]
    if not rows:
        return "No policy metrics are present in the current local artifact bundle."
    lines = [
        "| run | split | fake confabulates | fake refuses/flags | fake hedges | real false refusal | real substantive | reward mean |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, summary, split in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    split,
                    _rate_count(summary, "fake_fake_confabulates", "fake_n"),
                    _rate_count(summary, "fake_fake_refuses_or_flags", "fake_n"),
                    _rate_count(summary, "fake_fake_hedges_but_engages", "fake_n"),
                    _rate_count(summary, "real_real_false_refusal", "real_n"),
                    _rate_count(summary, "real_real_answers_substantively", "real_n"),
                    _fmt(summary.get("reward_mean")),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _rm_table(summary: dict[str, Any]) -> str:
    rows = []
    if any(key in summary for key in ("all", "real", "fake")):
        rows = [
            ("eval", group, values)
            for group, values in summary.items()
            if isinstance(values, dict)
        ]
    else:
        for split, split_summary in summary.items():
            if not isinstance(split_summary, dict):
                continue
            for group, values in split_summary.items():
                if isinstance(values, dict):
                    rows.append((split, group, values))
    if not rows:
        return "TODO: reward-model evaluation has not been run."
    lines = [
        "| split | group | n | accuracy | margin mean | chosen mean | rejected mean |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for split, group, values in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(split),
                    str(group),
                    _fmt(values.get("n"), 0),
                    _fmt(values.get("accuracy")),
                    _fmt(values.get("margin_mean")),
                    _fmt(values.get("chosen_mean")),
                    _fmt(values.get("rejected_mean")),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _rm_design_rows(outputs: Path) -> list[dict[str, str]]:
    return _read_csv_rows(outputs / "rm_design_ablations" / "rm_design_summary.csv")


def _rm_design_table(outputs: Path) -> str:
    rows = _rm_design_rows(outputs)
    if not rows:
        return "TODO: second-stage reward-model design comparisons have not been run."

    by_run: dict[str, dict[str, dict[str, str]]] = {}
    for row in rows:
        by_run.setdefault(row.get("run", ""), {})[row.get("group", "")] = row

    lines = [
        "| run | layer | pooling | input | head | all accuracy | all margin | real accuracy | fake accuracy |",
        "|---|---|---|---|---|---:|---:|---:|---:|",
    ]
    for run in sorted(name for name in by_run if name):
        groups = by_run[run]
        all_row = groups.get("all", {})
        real_row = groups.get("real", {})
        fake_row = groups.get("fake", {})
        lines.append(
            "| "
            + " | ".join(
                [
                    run,
                    all_row.get("hidden_layer") or "TODO",
                    all_row.get("pool") or "TODO",
                    all_row.get("rm_input") or "TODO",
                    all_row.get("head_type") or "TODO",
                    _fmt(_to_float(all_row.get("accuracy"))),
                    _fmt(_to_float(all_row.get("margin_mean"))),
                    _fmt(_to_float(real_row.get("accuracy"))),
                    _fmt(_to_float(fake_row.get("accuracy"))),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _ablation_table(outputs: Path, ablations: dict[str, dict[str, Any]]) -> str:
    lines = [
        "| run | transform | rank | fake confabulates | fake refuses/flags | real false refusal | real substantive | train note |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    ordered = [
        "raw_reward",
        "band_target",
        "imbalanced_data",
        "no_polarity_flip",
        "completion_only",
        "scalar_linear",
        "rank8",
        "rank16",
    ]
    for name in ordered + [name for name in sorted(ablations) if name not in ordered]:
        summary = ablations.get(name, {})
        run_dir = outputs / "tinker_rl_ablations" / name
        meta = _read_json(run_dir / "run_meta.json")
        config = _read_mapping(run_dir / "train_config.yaml")
        transform = (config.get("reward_transform") or {}).get("type")
        rank = meta.get("rank", config.get("rank"))
        if not summary and not meta:
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    str(transform or "TODO"),
                    _fmt(rank, 0),
                    _rate_count(summary, "fake_fake_confabulates", "fake_n"),
                    _rate_count(summary, "fake_fake_refuses_or_flags", "fake_n"),
                    _rate_count(summary, "real_real_false_refusal", "real_n"),
                    _rate_count(summary, "real_real_answers_substantively", "real_n"),
                    _training_note(run_dir / "metrics.jsonl"),
                ]
            )
            + " |"
        )
    if len(lines) == 2:
        return "No policy-level ablation eval artifacts are present in the current local artifact bundle."
    return "\n".join(lines)


def _hard_domain_table(outputs: Path) -> str:
    path = outputs / "report" / "policy_domain_breakdown.csv"
    if not path.exists():
        return "TODO: hard-OOD domain breakdown has not been generated."
    import pandas as pd

    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return "TODO: hard-OOD domain breakdown has no rows."
    required = {
        "run",
        "prompt_type",
        "domain",
        "n",
        "fake_confabulates",
        "fake_refuses_or_flags",
        "fake_hedges_but_engages",
        "reward_mean",
    }
    if df.empty or not required.issubset(df.columns):
        return "TODO: hard-OOD domain breakdown has no fake rows."
    df = df[
        df["run"].isin(["base_hard", "rl_hard"])
        & (df["prompt_type"] == "fake")
    ].copy()
    if df.empty:
        return "TODO: hard-OOD domain breakdown has no fake rows."
    df = df[
        [
            "run",
            "domain",
            "n",
            "fake_confabulates",
            "fake_refuses_or_flags",
            "fake_hedges_but_engages",
            "reward_mean",
        ]
    ].sort_values(["domain", "run"])
    lines = [
        "| run | domain | n | fake confabulates | fake refuses/flags | fake hedges | reward mean |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in df.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["run"]),
                    str(row["domain"]),
                    _fmt(row["n"], 0),
                    _pct(row["fake_confabulates"]),
                    _pct(row["fake_refuses_or_flags"]),
                    _pct(row["fake_hedges_but_engages"]),
                    _fmt(row["reward_mean"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _adjudication_taxonomy_table(outputs: Path) -> str:
    rows = _read_csv_rows(outputs / "report" / "adjudication_failure_taxonomy.csv")
    rows = [row for row in rows if row.get("run")]
    if not rows:
        return "TODO: adjudicated failure taxonomy has not been generated."

    def pct_or_na(value: Any) -> str:
        number = _to_float(value)
        return "n/a" if number is None else _pct(number)

    lines = [
        "| run | domain | type | n | good fake handling | bad fake confabulation | real answer | false refusal | high confidence |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row.get("run", ""),
                    row.get("domain", ""),
                    row.get("prompt_type", ""),
                    _fmt(_to_float(row.get("n")), 0),
                    pct_or_na(row.get("fake_good_handling")),
                    pct_or_na(row.get("fake_bad_confabulation")),
                    pct_or_na(row.get("real_good_answer")),
                    pct_or_na(row.get("real_false_refusal")),
                    pct_or_na(row.get("high_confidence")),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _main_training_table(outputs: Path) -> str:
    last = _latest_training_row(outputs / "tinker_rl" / "metrics.jsonl")
    meta = _read_json(outputs / "tinker_rl" / "run_meta.json")
    if not last and not meta:
        return (
            "First-pass Tinker training metrics are not present in the current local artifact bundle. "
            "The post-gate run is configured separately in `configs/tinker_rl_post_gate.yaml` and must "
            "complete before it can be used as evidence."
        )
    rows = [
        ("base model", meta.get("base_model")),
        ("LoRA rank", meta.get("rank")),
        ("final sampler path", meta.get("final_sampler_path")),
        ("last train fake confabulates", last.get("fake_confabulates")),
        ("last train real false refusal", last.get("real_false_refusal")),
        ("last train raw reward mean", last.get("raw_reward_mean")),
        ("last train transformed reward mean", last.get("transformed_reward_mean")),
        ("last degenerate group fraction", last.get("degenerate_group_fraction")),
    ]
    lines = ["| field | value |", "|---|---|"]
    for key, value in rows:
        lines.append(f"| {key} | {_fmt(value)} |")
    return "\n".join(lines)


def _pair_dataset_summary(path: Path) -> dict[str, Any]:
    rows = _read_jsonl(path)
    if not rows:
        return {"path": str(path), "exists": path.exists(), "n": 0}
    domains = sorted({str(row.get("domain")) for row in rows if row.get("domain")})
    categories = sorted(
        {
            str((row.get("metadata") or {}).get("category"))
            for row in rows
            if (row.get("metadata") or {}).get("category")
        }
    )
    splits: dict[str, int] = {}
    for row in rows:
        split = str(row.get("split") or "unknown")
        splits[split] = splits.get(split, 0) + 1
    unique_seed_prompts = {
        str((row.get("metadata") or {}).get("seed_prompt") or row.get("prompt"))
        for row in rows
        if (row.get("metadata") or {}).get("seed_prompt") or row.get("prompt")
    }
    return {
        "path": str(path),
        "exists": True,
        "n": len(rows),
        "unique_prompts": len({str(row.get("prompt")) for row in rows if row.get("prompt")}),
        "unique_seed_prompts": len(unique_seed_prompts),
        "domains": domains,
        "categories": categories,
        "splits": splits,
    }


def _dataset_summary_phrase(label: str, summary: dict[str, Any]) -> str:
    if not summary.get("n"):
        return f"{label}: missing"
    splits = summary.get("splits") or {}
    split_text = ", ".join(f"{name}={_int(count)}" for name, count in sorted(splits.items()))
    return (
        f"{label}: {_int(summary.get('n'))} pairs, {_int(summary.get('unique_prompts'))} unique prompts, "
        f"{_int(summary.get('unique_seed_prompts'))} seed prompts"
        + (f", splits {split_text}" if split_text else "")
        + f", {len(summary.get('domains') or [])} domains"
    )


def _verified_dataset_status(outputs: Path) -> str:
    root = outputs.parent
    train = _pair_dataset_summary(root / "data" / "verified_train" / "pairs.jsonl")
    eval_summary = _pair_dataset_summary(root / "data" / "verified_eval" / "pairs.jsonl")
    if not train.get("n") and not eval_summary.get("n"):
        return "missing"
    return (
        _dataset_summary_phrase("train", train)
        + "; "
        + _dataset_summary_phrase("eval", eval_summary)
        + "; frozen pair records carry seeded evidence/negative-retrieval metadata"
    )


def _retrieval_audit_status(outputs: Path) -> str:
    summary = _read_json(outputs / "report" / "verified_retrieval_audit.json")
    if not summary:
        return "missing"
    counts = summary.get("expected_result_counts") or {}
    return (
        f"present over {_int(summary.get('total_pairs'))} pairs "
        f"(live-pass={_int(counts.get('pass'))}, "
        f"seeded-evidence-only={_int(counts.get('seeded_evidence_only'))}, "
        f"fail={_int(summary.get('fail_pairs'))}, errors={_int(summary.get('error_pairs'))})"
    )


def _llm_judge_status(outputs: Path) -> str:
    metrics_paths = sorted((outputs / "blog").glob("*adjudicated*gpt55*metrics.json"))
    for metrics_path in reversed(metrics_paths):
        metrics = _read_json(metrics_path)
        if metrics:
            return (
                f"present ({_int(metrics.get('n'))} rows; "
                f"fake={_int(metrics.get('fake_n'))}, real={_int(metrics.get('real_n'))})"
            )
    if list((outputs / "blog").glob("*llm_judged*.csv")) or list((outputs / "blog").glob("*adjudicated_gpt55*.csv")):
        return "present"
    csv_paths = sorted(outputs.glob("eval_tinker*/**/policy_outputs.csv")) + sorted(
        outputs.glob("eval_tinker*_policy_outputs.csv")
    )
    for csv_path in csv_paths:
        try:
            header = csv_path.read_text(encoding="utf-8").splitlines()[0]
        except (IndexError, OSError):
            continue
        if "llm_judge_label" in header:
            return "present"
    return "not run"


def _second_stage_audit(outputs: Path) -> dict[str, Any]:
    return _read_json(outputs / "report" / "second_stage_audit.json")


def _audit_check(audit: dict[str, Any], check_name: str) -> dict[str, Any]:
    check = (audit.get("checks") or {}).get(check_name)
    return check if isinstance(check, dict) else {}


def _rm_design_provenance_status(outputs: Path) -> str:
    manifest_exists = (outputs / "rm_design_ablations" / "manifest.json").exists()
    audit = _second_stage_audit(outputs)
    check = _audit_check(audit, "actual_rm_design_eval_results")
    if check.get("ok") is True:
        return "present and matched to the current verified-train dataset"
    evidence = _read_json_evidence(check.get("evidence"))
    missing_fingerprints = evidence.get("missing_dataset_summary_variants") or []
    stale = evidence.get("stale_dataset_variants") or []
    low_seed = evidence.get("low_seed_dataset_variants") or []
    limited = evidence.get("limited_dataset_variants") or []
    missing_config = evidence.get("missing_config_variants") or []
    if missing_fingerprints:
        return (
            "blocked: score files exist, but the manifest lacks dataset fingerprints "
            f"for {len(missing_fingerprints)} variants, so rerun RM ablations on the current verified set"
        )
    if stale:
        return f"blocked: {len(stale)} variants were run on a stale dataset fingerprint"
    if low_seed:
        return f"blocked: {len(low_seed)} variants used a dataset below the seed-coverage bar"
    if missing_config:
        return f"blocked: {len(missing_config)} variants are missing saved RM configs"
    if limited:
        return f"blocked: {len(limited)} variants used smoke-test train/eval limits"
    if manifest_exists:
        return "manifest present, but current-data provenance has not been audited"
    return "missing"


def _second_stage_status(outputs: Path) -> str:
    audit = _second_stage_audit(outputs)
    publishable = audit.get("publishable_claims_ready")
    transform_check = _audit_check(audit, "chosen_rejected_reward_transform_audit")
    transform_evidence = _read_json_evidence(transform_check.get("evidence"))
    transform_runs = transform_evidence.get("audit_runs") or []
    lines = [
        f"- Verified data: {_verified_dataset_status(outputs)}.",
        f"- Verified retrieval audit: {_retrieval_audit_status(outputs)}.",
        f"- GPT-5.5/manual-review adjudication: {_llm_judge_status(outputs)}.",
        (
            f"- Reward-transform audit: present over {len(transform_runs)} RM design score sets."
            if transform_check.get("ok")
            else "- Reward-transform audit: missing or incomplete."
        ),
        f"- RM design comparisons: {_rm_design_provenance_status(outputs)}.",
    ]
    if publishable is not None:
        lines.append(f"- Publishable claims ready: {str(bool(publishable)).lower()}.")
    return "\n".join(lines)


def _second_stage_interpretation(outputs: Path) -> str:
    audit = _second_stage_audit(outputs)
    publishable = audit.get("publishable_claims_ready")
    rm_status = _rm_design_provenance_status(outputs)
    post_gate = _audit_check(audit, "post_gate_longer_rl_run")
    post_gate_missing = post_gate.get("ok") is False

    if publishable is True:
        sentence = (
            "The second-stage evidence surface is now adequate for scoped, honest claims about the "
            "infrastructure and the mixed-to-negative first-pass result: verified train/eval data, "
            "GPT-5.5 adjudication, current-dataset RM design comparisons, and reward-transform audits "
            "are all present."
        )
        if post_gate_missing:
            sentence += (
                " The remaining external step is a post-gate longer Tinker RL run; until that exists, "
                "this should not be framed as a solved hallucination or robust-transfer result."
            )
        return sentence

    if "blocked:" in rm_status:
        return (
            "The main publishability blocker is no longer wiring. It is current-data evidence: "
            f"RM design comparisons are {rm_status}."
        )
    return (
        "The main publishability blocker is still evidence quality rather than wiring. The report should "
        "stay framed as infrastructure plus a mixed-to-negative first-pass result until the blocking "
        "second-stage checks pass."
    )


def _reward_score_status(outputs: Path) -> str:
    if (outputs / "reward_model" / "eval" / "reward_scores.csv").exists():
        return "present"
    rm_design_scores = sorted((outputs / "rm_design_ablations").glob("*/eval/reward_scores.csv"))
    if rm_design_scores:
        return f"present via RM design variants ({len(rm_design_scores)} score files)"
    return "missing"


def _caveat_status(outputs: Path) -> str:
    manual_samples = sorted((outputs / "blog").glob("manual_review_samples*.md"))
    audit = _second_stage_audit(outputs)
    publishable = audit.get("publishable_claims_ready")
    checks: list[tuple[str, str]] = [
        ("verified train/eval seed coverage", _verified_dataset_status(outputs)),
        ("verified retrieval audit", _retrieval_audit_status(outputs)),
        ("manual review sample files", "present" if manual_samples else "missing"),
        ("reward model scores", _reward_score_status(outputs)),
        ("reward transform audit", "present" if (outputs / "report" / "reward_transform_audit.csv").exists() else "missing"),
        ("RM design ablation manifest", "present" if (outputs / "rm_design_ablations" / "manifest.json").exists() else "missing"),
        ("RM design current-data eval results", _rm_design_provenance_status(outputs)),
        ("main Tinker RL metrics", "present" if (outputs / "tinker_rl" / "metrics.jsonl").exists() else "missing"),
        ("base/test policy eval", "present" if (outputs / "eval_tinker_base_test" / "policy_metrics.json").exists() else "missing"),
        ("main/test policy eval", "present" if (outputs / "eval_tinker_main" / "policy_metrics.json").exists() else "missing"),
        ("base/OOD policy eval", "present" if (outputs / "eval_tinker_base_ood" / "policy_metrics.json").exists() else "missing"),
        ("main/OOD policy eval", "present" if (outputs / "eval_tinker_main_ood" / "policy_metrics.json").exists() else "missing"),
        (
            "base/hard-OOD policy eval",
            "present"
            if (outputs / "eval_tinker_hard_base" / "policy_metrics.json").exists()
            or (outputs / "eval_tinker_hard_base_policy_metrics.json").exists()
            else "missing",
        ),
        (
            "main/hard-OOD policy eval",
            "present"
            if (outputs / "eval_tinker_hard_main" / "policy_metrics.json").exists()
            or (outputs / "eval_tinker_hard_main_policy_metrics.json").exists()
            else "missing",
        ),
        (
            "post-gate verified-OOD policy eval",
            "present"
            if (outputs / "eval_tinker_verified_post_gate" / "policy_metrics.json").exists()
            else "missing",
        ),
        (
            "post-gate hard-OOD policy eval",
            "present"
            if (outputs / "eval_tinker_hard_post_gate" / "policy_metrics.json").exists()
            else "missing",
        ),
        ("domain breakdown", "present" if (outputs / "report" / "policy_domain_breakdown.csv").exists() else "missing"),
        ("failure samples", "present" if (outputs / "blog" / "failure_samples.md").exists() else "missing"),
        ("manual review queue", "present" if (outputs / "blog" / "manual_review_queue.csv").exists() else "missing"),
        ("manual review packet", "present" if (outputs / "blog" / "manual_review_packet.md").exists() else "missing"),
        (
            "adjudicated failure taxonomy",
            "present" if (outputs / "report" / "adjudication_failure_taxonomy.csv").exists() else "missing",
        ),
        (
            "adjudicated policy metrics",
            "present"
            if list(outputs.glob("eval*/adjudicated_metrics.json")) or list(outputs.glob("eval*adjudicated_metrics.json"))
            else "missing",
        ),
        ("LLM judge labels", _llm_judge_status(outputs)),
        ("confusion-risk notes", "present" if (outputs / "blog" / "confusion_risk_notes.md").exists() else "missing"),
    ]
    if publishable is not None:
        checks.append(("second-stage publishable claims", "ready" if publishable else "not ready"))
    lines = [f"- {name}: {status}" for name, status in checks]
    return "\n".join(lines)


def _missing_ablation_note(ablations: dict[str, dict[str, Any]]) -> str:
    completed = set(ablations)
    requested = {
        "raw_reward": "raw-reward policy training",
        "band_target": "band-target policy training",
        "imbalanced_data": "imbalanced data",
        "no_polarity_flip": "no polarity flip",
        "rank8": "LoRA rank 8",
        "rank16": "LoRA rank 16",
    }
    missing = [label for key, label in requested.items() if key not in completed]
    lines = []
    if missing:
        lines.append("Policy-level ablations not yet rerun on the second-stage data: " + ", ".join(missing) + ".")
    lines.append(
        "Reward-model design variants and chosen/rejected reward-transform audits are now reported as second-stage evidence, not policy results."
    )
    return "\n".join(lines)


def _ablation_interpretation(ablations: dict[str, dict[str, Any]]) -> str:
    if not ablations:
        return (
            "Policy-level ablations from the first pass are not present in the current local artifact bundle. "
            "The second-stage evidence now covers reward-model design choices and chosen/rejected reward-transform "
            "audits, but policy-level ablations still need to be rerun after the adjudication gate."
        )
    return (
        "The small ablation set did not show the feared held-out overrefusal failure: `real_false_refusal` "
        "stayed at zero under the heuristic. That does not vindicate raw reward. The raw-reward run had much "
        "larger training-loss scale, and the held-out test split was too small and too easy to expose the "
        "failure mode cleanly. The band-target run produced more degenerate prompt groups during training, "
        "which suggests the transform can collapse useful within-group contrast when too many samples land "
        "inside the target band."
    )


def _result_readout(policy: dict[str, dict[str, Any]]) -> str:
    base_test = policy.get("base_test", {})
    main_test = policy.get("main_test", {})
    base_ood = policy.get("base_ood", {})
    main_ood = policy.get("main_ood", {})
    base_hard = policy.get("base_hard", {})
    main_hard = policy.get("main_hard", {})
    lines = []
    if base_test and main_test:
        lines.extend(
            [
                f"- In-distribution fake confabulation: base {_rate_count(base_test, 'fake_fake_confabulates', 'fake_n')} -> RL {_rate_count(main_test, 'fake_fake_confabulates', 'fake_n')}. This split was too easy for the base model.",
                f"- In-distribution fake refusal/flagging: base {_rate_count(base_test, 'fake_fake_refuses_or_flags', 'fake_n')} -> RL {_rate_count(main_test, 'fake_fake_refuses_or_flags', 'fake_n')}.",
            ]
        )
    if base_ood and main_ood:
        lines.append(
            f"- OOD fake confabulation: base {_rate_count(base_ood, 'fake_fake_confabulates', 'fake_n')} -> RL {_rate_count(main_ood, 'fake_fake_confabulates', 'fake_n')}. This is the first nontrivial check beyond the easy held-out split."
        )
    if base_hard or main_hard:
        lines.append(
            f"- Hard OOD fake confabulation: base {_rate_count(base_hard, 'fake_fake_confabulates', 'fake_n')} -> RL {_rate_count(main_hard, 'fake_fake_confabulates', 'fake_n')}."
        )
    refusal_parts = []
    if base_test and main_test:
        refusal_parts.append(
            f"test base {_rate_count(base_test, 'real_real_false_refusal', 'real_n')} -> RL {_rate_count(main_test, 'real_real_false_refusal', 'real_n')}"
        )
    if base_ood and main_ood:
        refusal_parts.append(
            f"OOD base {_rate_count(base_ood, 'real_real_false_refusal', 'real_n')} -> RL {_rate_count(main_ood, 'real_real_false_refusal', 'real_n')}"
        )
    if base_hard and main_hard:
        refusal_parts.append(
            f"hard OOD base {_rate_count(base_hard, 'real_real_false_refusal', 'real_n')} -> RL {_rate_count(main_hard, 'real_real_false_refusal', 'real_n')}"
        )
    if refusal_parts:
        lines.append("- Real false refusal: " + "; ".join(refusal_parts) + ".")
    if not lines:
        return "No policy readout is available from the current local artifact bundle."
    return "\n".join(lines)


def write_blog(outputs_dir: str | Path, blog_path: str | Path) -> dict[str, Any]:
    outputs = Path(outputs_dir)
    target = Path(blog_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    policy = _known_policy_runs(outputs)
    ablations = _ablation_policy_runs(outputs)
    rm_summary = _reward_summary(outputs)

    text = f"""# RL for Epistemic Humility

Short thesis: learned preference rewards over constructed fake/real knowledge pairs can shape when a model pushes back on dubious premises, but this run does not show robust transfer. The in-distribution fake-name test was too easy for the base model. The OOD and hard-OOD splits are the more meaningful checks, and on the stricter rerun the Tinker policy did not reduce fake confabulation there. Bounded rewards still look like the right default because raw reward optimization is exactly where overrefusal and reward-texture hacking should appear first, but the current result is mixed-to-negative rather than a clean win.

This experiment tests a narrow claim: a targeted scalar learned from contrastive outcomes can move a context-dependent epistemic behavior. It does not solve hallucination, and it does not prove the model has a robust concept of truth.

## Background

The reward model is a Bradley-Terry head over frozen Qwen hidden states. It sees a prompt plus a candidate answer and learns to score the preferred answer above the rejected answer. In the density-ratio-ish view, the head is a discriminator between outcome styles under the constructed preference distribution: grounded answers on real entities, calibrated pushback on fake or false-premise prompts, and confabulations or false refusals as rejected outcomes.

The data is deliberately contrastive. Fake researcher or fake artifact prompts pair a cautious answer against an invented confident answer. Real researcher prompts flip the polarity: a grounded answer is preferred and a refusal is rejected. That flip matters because otherwise the reward can collapse into a global refusal detector.

## Method

Data construction used synthetic fake/real knowledge pairs plus OOD prompts for fake researchers, obscure real researchers, fake papers, fake APIs, false premises about real entities, and normal factual questions. A separate hard-OOD eval set uses more plausible fake citations, package/API hallucination traps, real-entity false premises, and obscure medieval/Holy Roman Empire prompts, including fake charters, false premises about real documents, and real-but-obscure anchors; it is eval-only and is not used for reward-model or RL training.

The repo now has a second-stage evidence-backed eval builder for real entities, fake entities, fake papers, fake APIs, and false premises across biography, software, medicine, law, finance, literature, history, and research. The current generated default artifacts are 1,008 train/val/test preference pairs and 1,008 eval pairs over 226 independent seed prompts, with zero train/eval prompt overlap. Those records carry source or negative-retrieval metadata and can optionally run online checks. The current local artifacts use seeded evidence metadata, not a fresh online retrieval audit, so treat this as the next measurement surface rather than evidence for the first-pass RL result.

The reward head used frozen Qwen hidden states. The main run used Qwen/Qwen3-8B with a `matrix_sum` head on the final token over prompt plus completion. The RL backend was Tinker LoRA: each step saved current LoRA weights for a sampler, sampled grouped completions, scored them with the local reward model, applied a bounded target-distribution transform, centered rewards within each prompt group, and trained with Tinker's `importance_sampling` loss.

Default transform: `gaussian_target`. Raw reward is an ablation, not the default.

## Reward Model Results

{_rm_table(rm_summary)}

The reward model separated chosen and rejected answers cleanly. That is useful for training, but it is not the same as proving the reward is truth-aware; a high score can still partly reflect answer style.

## Second-Stage RM Design Comparison

{_rm_design_table(outputs)}

These are reward-model-only comparisons on the current verified-train pairs, not policy evaluations. The main practical readout is that final-token, mean-completion, middle-layer, completion-only, and MLP heads all separate the constructed pairs cleanly on this dataset, while the scalar-linear head is slightly weaker. That argues the bottleneck is no longer basic BT separation on the synthetic contrastive task; it is transfer, adjudication, and policy training pressure.

## Tinker RL Run

{_main_training_table(outputs)}

## Policy Results

{_policy_table(policy)}

Readout:

{_result_readout(policy)}

The in-distribution test split should not carry the headline. It is tiny and too easy. The OOD and hard-OOD splits are more informative, and the stricter rerun is not flattering: fake confabulation did not fall on either split. The useful result is negative: this reward/RL setup preserved real answering under the heuristic, but it did not improve the harder epistemic behavior we actually care about.

Plots generated under `outputs/report/` include:

- `rm_chosen_rejected_hist.png`
- `rm_margin_by_prompt_type.png`
- `policy_metrics_before_after.png`
- `reward_distribution_before_after_vs_target.png`

## Hard-OOD Breakdown

{_hard_domain_table(outputs)}

Representative failure samples are written to `outputs/blog/failure_samples.md`; the review queue is `outputs/blog/manual_review_queue.csv`; the review rubric and model-assisted triage packet are in `outputs/blog/manual_review_packet.md`; confusion-risk notes are in `outputs/blog/confusion_risk_notes.md`. The domain table makes the failure less mysterious: the model still confabulates across API, paper, researcher, and historical prompts rather than failing only on one narrow fake-name texture.

## Adjudicated Failure Taxonomy

{_adjudication_taxonomy_table(outputs)}

The GPT-5.5 adjudication sample is small but sharper than the regex metrics: it separates clean uncertainty from hedged and confident confabulation, and it lets the writeup say exactly which domains are failing rather than vaguely saying "hallucination got worse."

## Ablations

{_ablation_table(outputs, ablations)}

{_ablation_interpretation(ablations)}

{_missing_ablation_note(ablations)}

## Failure Modes

- Fake-name texture: synthetic names may have morphology that makes the task easier than real uncertainty.
- Refusal-style reward hacking: the policy can learn safe-sounding uncertainty phrases rather than better epistemic discrimination.
- Real-person false refusal: this is the central failure mode and must be reported beside every fake-confabulation number.
- Eval classifier weakness: the heuristic labels are useful for smoke tests, not publishable measurement by themselves.
- Small synthetic dataset: any positive result here is a controlled behavioral movement, not a broad hallucination result.
- False premise versus fake entity: a false claim about a real person should trigger correction, not entity-level refusal.

## Interpretation

The honest claim is narrow: a scalar reward learned from contrastive fake/real outcomes can move some surface behavior under LoRA RL, but this run does not yet show robust epistemic transfer. The easy held-out split is not strong evidence. The OOD and hard-OOD results are the right place to look for movement, and here they mostly say the method is not working well enough yet.

The bounded reward transform is not cosmetic. It encodes the view that the target is a calibrated region of reward-model space, not maximum reward-model score. If raw reward looks fine on a tiny held-out split, that only says the split was not adversarial enough. It does not remove the basic pressure toward reward-texture hacking.

## Second-Stage Status

{_second_stage_status(outputs)}

{_second_stage_interpretation(outputs)}

## Next Work

- Harder fake entity generation with fewer surface artifacts.
- Run the evidence-backed verified split with online retrieval checks enabled.
- Run the post-gate longer Tinker RL job from `configs/tinker_rl_post_gate.yaml`, then evaluate it on verified, OOD, and hard-OOD prompts.
- Larger OOD sets focused on false premises about real entities.
- Multi-turn settings where the model can ask for evidence.
- Broader non-researcher domains and more mundane user intents.
- Full-size Qwen3-8B reward-model rerun for the chosen design before expensive policy training, since the local design comparison used the smaller local model.

## Artifact Status

{_caveat_status(outputs)}
"""
    target.write_text(text, encoding="utf-8")
    return {
        "blog_path": str(target),
        "has_reward_summary": bool(rm_summary),
        "policy_runs": {name: bool(summary) for name, summary in policy.items()},
        "ablation_runs": sorted(ablations),
    }
