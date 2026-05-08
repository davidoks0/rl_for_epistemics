from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from rl_epistemics.eval.classify_outputs import classify_output
from rl_epistemics.eval.metrics import METRIC_COLUMNS, aggregate_metrics


CLASSIFICATION_COLUMNS = set(METRIC_COLUMNS) | {"label", "answer_length"}


def _metrics_path_for_outputs(csv_path: Path) -> Path:
    if csv_path.name.endswith("_policy_outputs.csv"):
        return csv_path.with_name(csv_path.name.replace("_policy_outputs.csv", "_policy_metrics.json"))
    return csv_path.parent / "policy_metrics.json"


def reclassify_policy_outputs(csv_path: str | Path) -> dict[str, Any]:
    path = Path(csv_path)
    df = pd.read_csv(path)
    rows = df.to_dict(orient="records")
    refreshed = []
    for row in rows:
        base = {key: value for key, value in row.items() if key not in CLASSIFICATION_COLUMNS}
        prompt_type = str(base.get("prompt_type", "fake"))
        completion = "" if pd.isna(base.get("completion")) else str(base.get("completion", ""))
        refreshed.append(
            {
                **base,
                **classify_output(
                    prompt_type,  # type: ignore[arg-type]
                    completion,
                    prompt=None if pd.isna(base.get("prompt")) else str(base.get("prompt", "")),
                    domain=None if pd.isna(base.get("domain")) else str(base.get("domain", "")),
                ),
            }
        )
    pd.DataFrame(refreshed).to_csv(path, index=False)
    summary = aggregate_metrics(refreshed)
    with _metrics_path_for_outputs(path).open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def refresh_policy_metrics(outputs_dir: str | Path) -> dict[str, dict[str, Any]]:
    outputs = Path(outputs_dir)
    refreshed: dict[str, dict[str, Any]] = {}
    for csv_path in sorted(outputs.glob("eval_tinker*/**/policy_outputs.csv")):
        refreshed[str(csv_path.parent)] = reclassify_policy_outputs(csv_path)
    for csv_path in sorted(outputs.glob("eval_tinker*_policy_outputs.csv")):
        refreshed[str(csv_path)] = reclassify_policy_outputs(csv_path)
    return refreshed
