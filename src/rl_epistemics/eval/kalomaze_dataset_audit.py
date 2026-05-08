from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from rl_epistemics.data.schema import read_jsonl
from rl_epistemics.eval.classify_outputs import classify_output
from rl_epistemics.eval.llm_judge import maybe_judge_rows


FAKE_REFUSAL_LABELS = {"clean_uncertainty", "false_premise_correction"}
FAKE_CONFAB_LABELS = {"hedged_confabulation", "confident_confabulation"}
REAL_ANSWER_LABELS = {"real_substantive_answer", "false_premise_correction"}
REAL_REFUSAL_LABELS = {"false_refusal"}


def pair_side_rows(dataset_path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pair in read_jsonl(dataset_path):
        for side, completion, source in [
            ("chosen", pair.chosen, pair.chosen_source),
            ("rejected", pair.rejected, pair.rejected_source),
        ]:
            label = classify_output(
                pair.prompt_type,
                completion,
                prompt=pair.prompt,
                domain=pair.domain,
                metadata=pair.metadata,
            )
            rows.append(
                {
                    "id": f"{pair.id}:{side}",
                    "pair_id": pair.id,
                    "side": side,
                    "source": source,
                    "split": pair.split,
                    "prompt_type": pair.prompt_type,
                    "domain": pair.domain,
                    "entity_name": pair.entity_name,
                    "prompt": pair.prompt,
                    "completion": completion,
                    "verification_status": pair.metadata.get("source"),
                    "verification_evidence": _verification_evidence(pair.metadata),
                    "label": label["label"],
                    **{f"heuristic_{key}": value for key, value in label.items()},
                    "metadata_json": json.dumps(pair.metadata, sort_keys=True),
                }
            )
    return rows


def _verification_evidence(metadata: dict[str, Any]) -> str:
    if metadata.get("source") == "synthetic_nonexistent_researcher":
        title = metadata.get("paper_title")
        return (
            "Synthetic fake researcher"
            + (f"; fake paper title: {title}" if title else "")
            + f"; wikipedia verification: {metadata.get('verification', 'not_checked')}"
        )
    if metadata.get("source") == "wikipedia_pageview_candidate":
        return (
            f"Wikipedia title: {metadata.get('wikipedia_title')}; "
            f"pageviews_total: {metadata.get('pageviews_total')}; "
            f"extract: {str(metadata.get('wikipedia_extract', ''))[:300]}"
        )
    return json.dumps(metadata, sort_keys=True)[:500]


def _active_label(row: dict[str, Any], prefer_judge: bool = True) -> str:
    if prefer_judge and row.get("llm_judge_label"):
        return str(row["llm_judge_label"])
    if row.get("judge_label"):
        return str(row["judge_label"])
    return str(row.get("label", "unclear"))


def _expected_ok(row: dict[str, Any], label: str) -> bool:
    prompt_type = str(row.get("prompt_type"))
    side = str(row.get("side"))
    if prompt_type == "fake" and side == "chosen":
        return label in FAKE_REFUSAL_LABELS
    if prompt_type == "fake" and side == "rejected":
        return label in FAKE_CONFAB_LABELS
    if prompt_type == "real" and side == "chosen":
        return label in REAL_ANSWER_LABELS
    if prompt_type == "real" and side == "rejected":
        return label in REAL_REFUSAL_LABELS
    return False


def adjudicate_dataset_rows(
    rows: list[dict[str, Any]],
    *,
    llm_judge: bool,
    judge_config: dict[str, Any],
) -> list[dict[str, Any]]:
    judged = maybe_judge_rows(rows, {"llm_judge": judge_config}) if llm_judge else rows
    output = []
    for row in judged:
        label = _active_label(row)
        output.append(
            {
                **row,
                "audit_label": label,
                "audit_expected_ok": _expected_ok(row, label),
            }
        )
    return output


def _rate(rows: list[dict[str, Any]], predicate) -> float:
    if not rows:
        return 0.0
    return sum(bool(predicate(row)) for row in rows) / len(rows)


def aggregate_dataset_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fake_chosen = [row for row in rows if row["prompt_type"] == "fake" and row["side"] == "chosen"]
    fake_rejected = [row for row in rows if row["prompt_type"] == "fake" and row["side"] == "rejected"]
    real_chosen = [row for row in rows if row["prompt_type"] == "real" and row["side"] == "chosen"]
    real_rejected = [row for row in rows if row["prompt_type"] == "real" and row["side"] == "rejected"]
    pair_ids = sorted({str(row["pair_id"]) for row in rows})
    pair_ok = []
    by_pair = {pair_id: [row for row in rows if row["pair_id"] == pair_id] for pair_id in pair_ids}
    for pair_id, pair_rows in by_pair.items():
        pair_ok.append(all(bool(row.get("audit_expected_ok")) for row in pair_rows))

    return {
        "rows": len(rows),
        "pairs": len(pair_ids),
        "pair_expected_ok": sum(pair_ok) / len(pair_ok) if pair_ok else 0.0,
        "fake_chosen_refusal": _rate(fake_chosen, lambda row: row["audit_label"] in FAKE_REFUSAL_LABELS),
        "fake_rejected_confabulation": _rate(
            fake_rejected, lambda row: row["audit_label"] in FAKE_CONFAB_LABELS
        ),
        "fake_rejected_confident_confabulation": _rate(
            fake_rejected, lambda row: row["audit_label"] == "confident_confabulation"
        ),
        "real_chosen_answer": _rate(real_chosen, lambda row: row["audit_label"] in REAL_ANSWER_LABELS),
        "real_rejected_refusal": _rate(real_rejected, lambda row: row["audit_label"] in REAL_REFUSAL_LABELS),
        "fake_pairs": len(fake_chosen),
        "real_pairs": len(real_chosen),
        "label_counts": {
            label: sum(str(row.get("audit_label")) == label for row in rows)
            for label in sorted({str(row.get("audit_label")) for row in rows})
        },
        "by_prompt_type_side": _group_rates(rows),
    }


def _group_rates(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for prompt_type in sorted({str(row["prompt_type"]) for row in rows}):
        for side in sorted({str(row["side"]) for row in rows}):
            group = [
                row for row in rows if str(row["prompt_type"]) == prompt_type and str(row["side"]) == side
            ]
            if not group:
                continue
            key = f"{prompt_type}_{side}"
            output[key] = {
                "n": len(group),
                "expected_ok": _rate(group, lambda row: row.get("audit_expected_ok")),
                "label_counts": {
                    label: sum(str(row.get("audit_label")) == label for row in group)
                    for label in sorted({str(row.get("audit_label")) for row in group})
                },
            }
    return output


def audit_dataset(config: dict[str, Any]) -> dict[str, Any]:
    dataset_path = Path(config.get("dataset_path", "data/kalomaze_replica/pairs.jsonl"))
    output_dir = Path(config.get("output_dir", dataset_path.parent / "audit"))
    rows = pair_side_rows(dataset_path)
    judge_cfg = dict(config.get("llm_judge") or {})
    judged = adjudicate_dataset_rows(
        rows,
        llm_judge=bool(judge_cfg.get("enabled")),
        judge_config=judge_cfg,
    )
    metrics = aggregate_dataset_audit(judged)

    output_dir.mkdir(parents=True, exist_ok=True)
    rows_csv = output_dir / "pair_side_audit.csv"
    metrics_json = output_dir / "dataset_audit_metrics.json"
    pd.DataFrame(judged).to_csv(rows_csv, index=False)
    with metrics_json.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    return {
        "dataset_path": str(dataset_path),
        "rows_csv": str(rows_csv),
        "metrics_json": str(metrics_json),
        "metrics": metrics,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Audit Kalomaze replica pair-side labels.")
    parser.add_argument("--dataset-path", default="data/kalomaze_replica/pairs.jsonl")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--llm-judge", action="store_true")
    parser.add_argument("--provider", choices=["openai", "anthropic"], default="openai")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--api-key-file", default=None)
    parser.add_argument("--endpoint", default="https://api.openai.com/v1/responses")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=45.0)
    args = parser.parse_args(argv)

    config: dict[str, Any] = {
        "dataset_path": args.dataset_path,
        "llm_judge": {
            "enabled": args.llm_judge,
            "provider": args.provider,
            "model": args.model,
            "api_key_env": args.api_key_env,
            "api_key_file": args.api_key_file,
            "endpoint": args.endpoint,
            "limit": args.limit,
            "max_tokens": args.max_tokens,
            "timeout": args.timeout,
        },
    }
    if args.output_dir:
        config["output_dir"] = args.output_dir
    print(json.dumps(audit_dataset(config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
