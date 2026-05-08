from __future__ import annotations

from collections import defaultdict
from typing import Any

import pandas as pd

from rl_epistemics.eval.classify_outputs import classify_output


METRIC_COLUMNS = [
    "clean_uncertainty",
    "false_premise_correction",
    "hedged_confabulation",
    "confident_confabulation",
    "real_substantive_answer",
    "false_refusal",
    "fake_refuses_or_flags",
    "fake_hedges_but_engages",
    "fake_confabulates",
    "real_answers_substantively",
    "real_false_refusal",
]


def kalomaze_metric_aliases(summary: dict[str, Any]) -> dict[str, float]:
    """Return the four metric names used in Kalomaze's knowledge-awareness post."""
    refuses_fake = float(summary.get("fake_clean_uncertainty", 0.0)) + float(
        summary.get("fake_false_premise_correction", 0.0)
    )
    return {
        "refuses_fake": min(refuses_fake, 1.0),
        "fake_hedge": float(
            summary.get(
                "fake_hedged_confabulation",
                summary.get("fake_fake_hedges_but_engages", 0.0),
            )
        ),
        "fake_substantive": float(
            summary.get(
                "fake_confident_confabulation",
                summary.get("fake_fake_confabulates", 0.0),
            )
        ),
        "answers_real": float(
            summary.get(
                "real_real_substantive_answer",
                summary.get("real_real_answers_substantively", 0.0),
            )
        ),
    }


def classify_rows(rows: list[dict[str, Any]], text_key: str = "completion") -> list[dict[str, Any]]:
    output = []
    for row in rows:
        classified = classify_output(
            row["prompt_type"],
            row[text_key],
            prompt=row.get("prompt"),
            domain=row.get("domain"),
            metadata=row.get("metadata") if isinstance(row.get("metadata"), dict) else None,
        )
        output.append({**row, **classified})
    return output


def aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0}
    result: dict[str, Any] = {"n": len(rows)}
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    groups["all"] = rows
    for row in rows:
        groups[row.get("prompt_type", "unknown")].append(row)

    for group_name, group_rows in groups.items():
        prefix = "" if group_name == "all" else f"{group_name}_"
        result[f"{prefix}n"] = len(group_rows)
        for col in METRIC_COLUMNS:
            result[f"{prefix}{col}"] = sum(bool(row.get(col)) for row in group_rows) / len(group_rows)
        result[f"{prefix}answer_length"] = sum(int(row.get("answer_length", 0)) for row in group_rows) / len(
            group_rows
        )
        if any("reward" in row for row in group_rows):
            rewards = [float(row["reward"]) for row in group_rows if row.get("reward") is not None]
            if rewards:
                result[f"{prefix}reward_mean"] = sum(rewards) / len(rewards)
    result.update(kalomaze_metric_aliases(result))
    return result


def write_metrics(rows: list[dict[str, Any]], output_csv: str, output_summary_json: str) -> dict[str, Any]:
    import json

    classified = classify_rows(rows)
    pd.DataFrame(classified).to_csv(output_csv, index=False)
    summary = aggregate_metrics(classified)
    with open(output_summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary
