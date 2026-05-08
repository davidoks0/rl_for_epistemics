from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from rl_epistemics.eval.adjudicate_outputs import adjudicate_rows
from rl_epistemics.eval.llm_judge import maybe_judge_rows


RUN_DIRS = {
    "gaussian": "tinker_rl_gaussian",
    "raw": "tinker_rl_raw",
    "per_class_gaussian_calibrated": "tinker_rl_per_class_gaussian_calibrated",
    "raw_calibrated": "tinker_rl_raw_calibrated",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def _parse_steps(values: list[str] | None) -> set[int] | None:
    if not values:
        return None
    steps: set[int] = set()
    for value in values:
        for part in str(value).split(","):
            part = part.strip()
            if part:
                steps.add(int(part))
    return steps


def load_kalomaze_heldout_samples(
    output_root: str | Path,
    *,
    runs: list[str] | None = None,
    steps: set[int] | None = None,
) -> list[dict[str, Any]]:
    root = Path(output_root)
    active_runs = runs or list(RUN_DIRS)
    rows: list[dict[str, Any]] = []
    for run in active_runs:
        run_dir = RUN_DIRS.get(run, run)
        for row in _read_jsonl(root / run_dir / "heldout_samples.jsonl"):
            step = int(row.get("step", 0))
            if steps is not None and step not in steps:
                continue
            original_id = str(row.get("id", ""))
            prompt_type = str(row.get("prompt_type", ""))
            enriched = {
                **row,
                "run": run,
                "original_id": original_id,
                "id": f"{run}-step-{step:04d}-{original_id}",
                "verification_status": (
                    "synthetic_nonexistent_researcher"
                    if prompt_type == "fake"
                    else "wikipedia_pageview_grounded_real_researcher"
                ),
                "verification_evidence": (
                    "Name was generated for the fake researcher subset and should not be treated "
                    "as a known public researcher."
                    if prompt_type == "fake"
                    else "Real researcher prompt selected from the Wikipedia/pageview grounded subset."
                ),
            }
            rows.append(enriched)
    rows.sort(key=lambda row: (str(row.get("run", "")), int(row.get("step", 0)), str(row.get("original_id", ""))))
    return rows


def _rate(rows: list[dict[str, Any]], label: str) -> float:
    if not rows:
        return 0.0
    return sum(str(row.get("adjudicated_label")) == label for row in rows) / len(rows)


def aggregate_kalomaze_adjudicated_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row.get("run", "")), int(row.get("step", 0))), []).append(row)

    metrics: list[dict[str, Any]] = []
    for (run, step), group in sorted(groups.items()):
        fake = [row for row in group if row.get("prompt_type") == "fake"]
        real = [row for row in group if row.get("prompt_type") == "real"]
        fake_refuses = _rate(fake, "clean_uncertainty") + _rate(fake, "false_premise_correction")
        needs_review = sum(bool(row.get("llm_judge_needs_human_review")) for row in group)
        metrics.append(
            {
                "run": run,
                "step": step,
                "n": len(group),
                "fake_n": len(fake),
                "real_n": len(real),
                "refuses_fake": min(float(fake_refuses), 1.0),
                "fake_hedge": _rate(fake, "hedged_confabulation"),
                "fake_substantive": _rate(fake, "confident_confabulation"),
                "fake_unclear": _rate(fake, "unclear"),
                "answers_real": _rate(real, "real_substantive_answer"),
                "real_false_refusal": _rate(real, "false_refusal"),
                "real_unclear": _rate(real, "unclear"),
                "needs_human_review": needs_review / len(group) if group else 0.0,
                "label_counts": {
                    label: sum(str(row.get("adjudicated_label")) == label for row in group)
                    for label in sorted({str(row.get("adjudicated_label")) for row in group})
                },
            }
        )
    return metrics


def _judge_config(
    *,
    provider: str,
    model: str,
    api_key_env: str | None,
    api_key_file: str | None,
    endpoint: str | None,
    limit: int | None,
    max_tokens: int,
    reasoning_effort: str | None,
    timeout: float,
) -> dict[str, Any]:
    config: dict[str, Any] = {
        "enabled": True,
        "provider": provider,
        "model": model,
        "limit": limit,
        "max_tokens": max_tokens,
        "timeout": timeout,
    }
    if api_key_env:
        config["api_key_env"] = api_key_env
    if api_key_file:
        config["api_key_file"] = api_key_file
    if endpoint:
        config["endpoint"] = endpoint
    if reasoning_effort:
        config["reasoning_effort"] = reasoning_effort
    return config


def _ensure_judged(rows: list[dict[str, Any]], judge_config: dict[str, Any]) -> list[dict[str, Any]]:
    judged = maybe_judge_rows(rows, {"llm_judge": judge_config})
    if rows and not any(row.get("llm_judge_label") for row in judged):
        notes = next((str(row.get("llm_judge_notes", "")) for row in judged if row.get("llm_judge_notes")), "")
        raise RuntimeError(f"LLM judge produced no labels. {notes}")
    return judged


def adjudicate_kalomaze_heldout(
    *,
    output_root: str | Path,
    output_dir: str | Path | None = None,
    runs: list[str] | None = None,
    steps: set[int] | None = None,
    provider: str = "openai",
    model: str = "gpt-4o-mini",
    api_key_env: str | None = "OPENAI_API_KEY",
    api_key_file: str | None = None,
    endpoint: str | None = "https://api.openai.com/v1/responses",
    limit: int | None = None,
    max_tokens: int = 512,
    reasoning_effort: str | None = None,
    timeout: float = 45.0,
    llm_judge: bool = True,
) -> dict[str, Any]:
    root = Path(output_root)
    target = Path(output_dir) if output_dir else root / "report" / "adjudication"
    rows = load_kalomaze_heldout_samples(root, runs=runs, steps=steps)
    if llm_judge:
        rows = _ensure_judged(
            rows,
            _judge_config(
                provider=provider,
                model=model,
                api_key_env=api_key_env,
                api_key_file=api_key_file,
                endpoint=endpoint,
                limit=limit,
                max_tokens=max_tokens,
                reasoning_effort=reasoning_effort,
                timeout=timeout,
            ),
        )
    adjudicated = adjudicate_rows(rows)
    metrics = aggregate_kalomaze_adjudicated_metrics(adjudicated)

    target.mkdir(parents=True, exist_ok=True)
    input_csv = target / "kalomaze_heldout_for_judge.csv"
    judged_csv = target / "kalomaze_heldout_judged.csv"
    adjudicated_csv = target / "kalomaze_heldout_adjudicated.csv"
    metrics_jsonl = target / "kalomaze_heldout_adjudicated_metrics.jsonl"
    metrics_csv = target / "kalomaze_heldout_adjudicated_metrics.csv"
    summary_json = target / "kalomaze_heldout_adjudication_summary.json"

    pd.DataFrame(load_kalomaze_heldout_samples(root, runs=runs, steps=steps)).to_csv(input_csv, index=False)
    pd.DataFrame(rows).to_csv(judged_csv, index=False)
    pd.DataFrame(adjudicated).to_csv(adjudicated_csv, index=False)
    _write_jsonl(metrics_jsonl, metrics)
    pd.DataFrame(metrics).to_csv(metrics_csv, index=False)
    summary = {
        "rows": len(adjudicated),
        "runs": sorted({str(row.get("run")) for row in adjudicated}),
        "steps": sorted({int(row.get("step", 0)) for row in adjudicated}),
        "llm_judge": llm_judge,
        "provider": provider if llm_judge else None,
        "model": model if llm_judge else None,
        "input_csv": str(input_csv),
        "judged_csv": str(judged_csv),
        "adjudicated_csv": str(adjudicated_csv),
        "metrics_jsonl": str(metrics_jsonl),
        "metrics_csv": str(metrics_csv),
    }
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    return {**summary, "summary_json": str(summary_json), "metrics": metrics}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="LLM-adjudicate Kalomaze replica heldout samples.")
    parser.add_argument("--output-root", default="outputs/kalomaze_replica")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run", action="append", choices=sorted(RUN_DIRS), default=None)
    parser.add_argument("--step", action="append", default=None, help="Step or comma-separated steps to adjudicate.")
    parser.add_argument("--provider", choices=["openai", "anthropic"], default="openai")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--api-key-file", default=os.environ.get("RL_EPISTEMICS_OPENAI_KEY_FILE"))
    parser.add_argument("--endpoint", default="https://api.openai.com/v1/responses")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--reasoning-effort", default=None)
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--no-llm-judge", action="store_true")
    args = parser.parse_args(argv)

    result = adjudicate_kalomaze_heldout(
        output_root=args.output_root,
        output_dir=args.output_dir,
        runs=args.run,
        steps=_parse_steps(args.step),
        provider=args.provider,
        model=args.model,
        api_key_env=args.api_key_env,
        api_key_file=args.api_key_file,
        endpoint=args.endpoint,
        limit=args.limit,
        max_tokens=args.max_tokens,
        reasoning_effort=args.reasoning_effort,
        timeout=args.timeout,
        llm_judge=not args.no_llm_judge,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
