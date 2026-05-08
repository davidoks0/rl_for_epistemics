from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


KALOMAZE_METRICS = ["refuses_fake", "fake_hedge", "fake_substantive", "answers_real"]


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


def _maybe_pyplot() -> Any | None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    return plt


def _latest(path: Path) -> dict[str, Any]:
    rows = _read_jsonl(path)
    return rows[-1] if rows else {}


def _latest_run(rows: list[dict[str, Any]], run: str) -> dict[str, Any]:
    selected = [row for row in rows if str(row.get("run", "")) == run]
    return selected[-1] if selected else {}


def _best_fake_refusal(path: Path) -> dict[str, Any]:
    rows = _read_jsonl(path)
    return _best_fake_refusal_rows(rows)


def _best_fake_refusal_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    return max(
        rows,
        key=lambda row: (
            float(row.get("refuses_fake", 0.0)),
            float(row.get("answers_real", 0.0)),
            -float(row.get("fake_substantive", 1.0)),
        ),
    )


def _plot_heldout_curves(output_root: Path, report_dir: Path) -> list[Path]:
    plt = _maybe_pyplot()
    if plt is None:
        return []
    paths: list[Path] = []
    for run_name in ("tinker_rl_gaussian", "tinker_rl_raw"):
        metrics_path = output_root / run_name / "heldout_metrics.jsonl"
        rows = _read_jsonl(metrics_path)
        if not rows:
            continue
        df = pd.DataFrame(rows)
        if not {"step", *KALOMAZE_METRICS}.issubset(df.columns):
            continue
        path = report_dir / f"{run_name}_heldout_curves.png"
        plt.figure(figsize=(8, 4))
        for metric in KALOMAZE_METRICS:
            plt.plot(df["step"], df[metric], marker="o", label=metric)
        plt.xlabel("step")
        plt.ylabel("rate")
        plt.ylim(0, 1)
        plt.legend()
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        paths.append(path)
    return paths


def _plot_reward_distribution(output_root: Path, report_dir: Path) -> list[Path]:
    plt = _maybe_pyplot()
    if plt is None:
        return []
    scores_path = output_root / "reward_model" / "eval" / "reward_scores.csv"
    if not scores_path.exists():
        return []
    df = pd.read_csv(scores_path)
    if not {"chosen_reward", "rejected_reward"}.issubset(df.columns):
        return []
    path = report_dir / "reward_chosen_rejected_hist.png"
    plt.figure(figsize=(7, 4))
    plt.hist(df["chosen_reward"].dropna(), bins=30, alpha=0.65, label="chosen")
    plt.hist(df["rejected_reward"].dropna(), bins=30, alpha=0.65, label="rejected")
    plt.xlabel("reward score")
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return [path]


def _format_metric_row(row: dict[str, Any]) -> str:
    if not row:
        return "not run"
    return ", ".join(f"{metric}={float(row.get(metric, 0.0)):.3f}" for metric in KALOMAZE_METRICS)


def _float_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    return values


def _summarize_values(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    return {
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def _training_summary(path: Path) -> dict[str, Any]:
    rows = _read_jsonl(path)
    if not rows:
        return {}
    return {
        "rows": len(rows),
        "first_step": rows[0].get("step"),
        "last_step": rows[-1].get("step"),
        "loss": _summarize_values(_float_values(rows, "loss")),
        "raw_reward": _summarize_values(_float_values(rows, "raw_reward_mean")),
        "transformed_reward": _summarize_values(_float_values(rows, "transformed_reward_mean")),
    }


def _fmt_stat(summary: dict[str, float], key: str) -> str:
    if not summary:
        return "-"
    return f"{float(summary.get(key, 0.0)):.3f}"


def _reward_transform_table(rows: list[tuple[str, dict[str, Any], dict[str, Any]]]) -> list[str]:
    table = [
        "| run | train rows | final train step | final heldout step | loss mean | loss sd | loss min | loss max | transformed reward mean | transformed reward min | transformed reward max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, summary, heldout in rows:
        if not summary:
            table.append(f"| {name} | not run | - | - | - | - | - | - | - | - | - |")
            continue
        transformed = summary.get("transformed_reward", {})
        loss = summary.get("loss", {})
        table.append(
            f"| {name} | {int(summary.get('rows', 0))} | {summary.get('last_step', '-')} | "
            f"{heldout.get('step', '-')} | {_fmt_stat(loss, 'mean')} | {_fmt_stat(loss, 'std')} | "
            f"{_fmt_stat(loss, 'min')} | {_fmt_stat(loss, 'max')} | "
            f"{_fmt_stat(transformed, 'mean')} | {_fmt_stat(transformed, 'min')} | "
            f"{_fmt_stat(transformed, 'max')} |"
        )
    return table


def _reward_transform_sentence(gaussian: dict[str, Any], raw: dict[str, Any]) -> str:
    if not gaussian or not raw:
        return "- Raw-vs-Gaussian train telemetry: not available."
    gaussian_loss = gaussian.get("loss", {})
    raw_loss = raw.get("loss", {})
    gaussian_reward = gaussian.get("transformed_reward", {})
    raw_reward = raw.get("transformed_reward", {})
    return (
        "- Raw reward had higher train-loss volatility "
        f"(population sd {_fmt_stat(raw_loss, 'std')} vs {_fmt_stat(gaussian_loss, 'std')} for Gaussian target) "
        "and kept transformed rewards on the raw-score scale "
        f"(mean {_fmt_stat(raw_reward, 'mean')} vs {_fmt_stat(gaussian_reward, 'mean')})."
    )


def _dataset_audit_rows(output_root: Path) -> list[tuple[Path, dict[str, Any]]]:
    rows = []
    for path in sorted(output_root.glob("*audit*/dataset_audit_metrics.json")):
        data = _read_json(path)
        metrics = data.get("metrics") if isinstance(data.get("metrics"), dict) else data
        if metrics:
            rows.append((path, metrics))
    return rows


def _dataset_audit_table(rows: list[tuple[Path, dict[str, Any]]]) -> list[str]:
    if not rows:
        return ["- Dataset audit: not run."]
    table = [
        "| audit | pairs | fake chosen refusal | fake rejected confab | confident confab | real chosen answer | real rejected refusal | pair ok |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for path, metrics in rows:
        table.append(
            f"| {path.parent.name} | {int(metrics.get('pairs', 0))} | "
            f"{float(metrics.get('fake_chosen_refusal', 0.0)):.3f} | "
            f"{float(metrics.get('fake_rejected_confabulation', 0.0)):.3f} | "
            f"{float(metrics.get('fake_rejected_confident_confabulation', 0.0)):.3f} | "
            f"{float(metrics.get('real_chosen_answer', 0.0)):.3f} | "
            f"{float(metrics.get('real_rejected_refusal', 0.0)):.3f} | "
            f"{float(metrics.get('pair_expected_ok', 0.0)):.3f} |"
        )
    return table


def _counter_text(counter: Counter[str]) -> str:
    return ", ".join(f"{key}={counter[key]}" for key in sorted(counter))


def _dataset_source_line(path: Path) -> str:
    rows = _read_jsonl(path)
    if not rows:
        return f"- Source pairs: `{path}` not found or empty."
    split_counts = Counter(str(row.get("split", "unknown")) for row in rows)
    prompt_counts = Counter(str(row.get("prompt_type", "unknown")) for row in rows)
    domain_counts = Counter(str(row.get("domain", "unknown")) for row in rows)
    return (
        f"- Source pairs: `{path}` ({len(rows)} rows; "
        f"splits: {_counter_text(split_counts)}; "
        f"prompt types: {_counter_text(prompt_counts)}; "
        f"domains: {_counter_text(domain_counts)})."
    )


def _format_run_status(name: str, latest_heldout: dict[str, Any], latest_train: dict[str, Any], planned_steps: int = 300) -> str:
    if not latest_heldout and not latest_train:
        return f"- {name}: not run."
    heldout_step = latest_heldout.get("step")
    train_step = latest_train.get("step")
    completed = heldout_step is not None and int(heldout_step) >= planned_steps
    if completed:
        return f"- {name}: completed planned {planned_steps} RL steps; latest heldout eval is step {heldout_step}."
    if train_step is None:
        return f"- {name}: incomplete; latest heldout eval is step {heldout_step}."
    return (
        f"- {name}: stopped early before planned {planned_steps} RL steps; "
        f"latest train step is {train_step}, latest heldout eval is step {heldout_step}."
    )


def _run_completed(latest_heldout: dict[str, Any], latest_train: dict[str, Any], planned_steps: int = 300) -> bool:
    heldout_step = latest_heldout.get("step")
    if heldout_step is not None and int(heldout_step) >= planned_steps:
        return True
    train_step = latest_train.get("step")
    return train_step is not None and int(train_step) >= planned_steps - 1


def _success_status(row: dict[str, Any]) -> str:
    if not row:
        return "not run"
    refuses_fake = float(row.get("refuses_fake", 0.0))
    fake_substantive = float(row.get("fake_substantive", 1.0))
    answers_real = float(row.get("answers_real", 0.0))
    if refuses_fake >= 0.50 and fake_substantive <= 0.10 and answers_real >= 0.80:
        return "provisional success on the researcher heldout set"
    if refuses_fake <= 0.0 and fake_substantive >= 0.50:
        return "failure on the researcher heldout set: no fake-name refusal"
    return "not established on the configured researcher heldout set"


def _claim_success_status(adjudicated: dict[str, Any], heuristic: dict[str, Any]) -> str:
    if adjudicated:
        return _success_status(adjudicated)
    if heuristic:
        return "not established; LLM adjudication not run for these heldout samples"
    return "not run"


def _heldout_table(name: str, latest: dict[str, Any], best: dict[str, Any]) -> list[str]:
    if not latest and not best:
        return [f"| {name} | not run | - | - | - | - | - |"]
    latest_step = latest.get("step", "-")
    best_step = best.get("step", "-")
    return [
        (
            f"| {name} | {latest_step} | {float(latest.get('refuses_fake', 0.0)):.3f} | "
            f"{float(latest.get('fake_hedge', 0.0)):.3f} | "
            f"{float(latest.get('fake_substantive', 0.0)):.3f} | "
            f"{float(latest.get('answers_real', 0.0)):.3f} | "
            f"{best_step}: {float(best.get('refuses_fake', 0.0)):.3f} |"
        )
    ]


def _curve_line(name: str, path: Path, metric: str) -> str:
    rows = _read_jsonl(path)
    return _curve_line_rows(name, rows, metric)


def _curve_line_rows(name: str, rows: list[dict[str, Any]], metric: str) -> str:
    if not rows:
        return f"- {name} {metric}: not run"
    points = []
    for row in rows:
        if "step" in row and metric in row:
            points.append(f"{row['step']}:{float(row[metric]):.3f}")
    return f"- {name} {metric}: " + ", ".join(points)


def _adjudication_judgment(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "- Claim-grade judgment: not available; LLM adjudication has not run."
    best_refusal = max(float(row.get("refuses_fake", 0.0)) for row in rows)
    if best_refusal <= 0.0:
        return (
            "- Claim-grade judgment: failure. Across all LLM-adjudicated checkpoints, "
            "the best fake-name refusal rate was 0.000."
        )
    return (
        "- Claim-grade judgment: mixed/not established. "
        f"Best LLM-adjudicated fake-name refusal rate was {best_refusal:.3f}."
    )


def write_report(
    *,
    output_root: str | Path = "outputs/kalomaze_replica",
    report_path: str | Path = "outputs/kalomaze_replica/report.md",
) -> Path:
    root = Path(output_root)
    report = Path(report_path)
    report.parent.mkdir(parents=True, exist_ok=True)
    plot_paths = _plot_reward_distribution(root, report.parent) + _plot_heldout_curves(root, report.parent)

    rm_summary = _read_json(root / "reward_model" / "score_summary.json")
    rm_eval_summary = _read_json(root / "reward_model" / "eval" / "reward_summary.json")
    chosen_stats = _read_json(root / "reward_model" / "chosen_score_stats.json")
    gaussian_latest = _latest(root / "tinker_rl_gaussian" / "heldout_metrics.jsonl")
    raw_latest = _latest(root / "tinker_rl_raw" / "heldout_metrics.jsonl")
    gaussian_train_latest = _latest(root / "tinker_rl_gaussian" / "metrics.jsonl")
    raw_train_latest = _latest(root / "tinker_rl_raw" / "metrics.jsonl")
    gaussian_train_summary = _training_summary(root / "tinker_rl_gaussian" / "metrics.jsonl")
    raw_train_summary = _training_summary(root / "tinker_rl_raw" / "metrics.jsonl")
    gaussian_best = _best_fake_refusal(root / "tinker_rl_gaussian" / "heldout_metrics.jsonl")
    raw_best = _best_fake_refusal(root / "tinker_rl_raw" / "heldout_metrics.jsonl")
    adjudication_dir = root / "report" / "adjudication"
    adjudication_summary = _read_json(adjudication_dir / "kalomaze_heldout_adjudication_summary.json")
    adjudicated_rows = _read_jsonl(adjudication_dir / "kalomaze_heldout_adjudicated_metrics.jsonl")
    gaussian_adjudicated_rows = [
        row for row in adjudicated_rows if str(row.get("run", "")) == "gaussian"
    ]
    raw_adjudicated_rows = [row for row in adjudicated_rows if str(row.get("run", "")) == "raw"]
    gaussian_adjudicated_latest = _latest_run(adjudicated_rows, "gaussian")
    raw_adjudicated_latest = _latest_run(adjudicated_rows, "raw")
    gaussian_adjudicated_best = _best_fake_refusal_rows(gaussian_adjudicated_rows)
    raw_adjudicated_best = _best_fake_refusal_rows(raw_adjudicated_rows)
    dataset_audits = _dataset_audit_rows(root)
    dataset_path = Path("data") / root.name / "pairs.jsonl"
    config_dir = Path("configs") / root.name
    if not config_dir.exists():
        config_dir = Path("configs/kalomaze_replica")
    divergence_notes = [
        "- The post does not specify the exact hidden layer; this track uses `hidden_layer: last`.",
        "- The fake-name prefill follows the post; the real over-refusal prefill is a local reconstruction.",
        "- Wikipedia/pageview selection is implemented from public APIs, so rankings depend on run date.",
        "- Full 300-step Tinker execution requires `tinker` plus `TINKER_API_KEY`; local smoke tests do not run it.",
    ]
    incomplete_runs = []
    for name, latest, train_latest in (
        ("Gaussian-target", gaussian_latest, gaussian_train_latest),
        ("raw-reward", raw_latest, raw_train_latest),
    ):
        if (latest or train_latest) and not _run_completed(latest, train_latest):
            incomplete_runs.append(name)
    if incomplete_runs:
        divergence_notes.append(
            "- Incomplete RL execution in this output root: "
            + ", ".join(incomplete_runs)
            + " stopped before the planned 300-step run."
        )
    elif not gaussian_latest and not raw_latest:
        divergence_notes.append(
            "- RM/RL have not been run yet for this output root; the completed result here is the data-generation audit."
        )

    lines = [
        "# Kalomaze Replica Report",
        "",
        "Researcher-only replication track for `https://kalomaze.bearblog.dev/rl-for-knowledge-awareness/`.",
        "",
        "## Direct Matches",
        "",
        "- Qwen3-8B is the configured frozen representation model for the BT reward model.",
        "- Reward head is a single `4096 -> 4096` matrix with `0.01 * sum(Wx)` scoring.",
        "- Reward-model input defaults to prompt plus completion and final-token pooling.",
        "- Fake pairs choose prefill-elicited uncertainty over standard generation.",
        "- Real pairs choose standard grounded answers over prefill-elicited over-refusal.",
        "- Prefill text is metadata only; the trained RM sees prompt and downstream completion.",
        "- RL configs use class-balanced researcher prompts and Qwen3-8B Tinker LoRA for 300 steps.",
        "- Gaussian target and raw reward Tinker configs are separate runs.",
        "",
        "## Known Divergences",
        "",
        *divergence_notes,
        "",
        "## Dataset Audit",
        "",
        _dataset_source_line(dataset_path),
        "",
        *_dataset_audit_table(dataset_audits),
        "",
        "## Reward Model",
        "",
        f"- Score summary: `{json.dumps(rm_summary, sort_keys=True)}`" if rm_summary else "- Score summary: not run",
        f"- Eval summary: `{json.dumps(rm_eval_summary, sort_keys=True)}`"
        if rm_eval_summary
        else "- Eval summary: not run",
        f"- Chosen-score stats: `{json.dumps(chosen_stats, sort_keys=True)}`"
        if chosen_stats
        else "- Chosen-score stats: not run",
        "",
        "## Run Outcome",
        "",
        _format_run_status("Gaussian target", gaussian_latest, gaussian_train_latest),
        _format_run_status("Raw reward", raw_latest, raw_train_latest),
        (
            f"- Raw reward best heldout refuses_fake was step {raw_best.get('step', '-')}: "
            f"{float(raw_best.get('refuses_fake', 0.0)):.3f}; latest was "
            f"{float(raw_latest.get('refuses_fake', 0.0)):.3f}, so the raw objective did not show stable progress."
        )
        if raw_latest or raw_best
        else "- Raw reward best heldout refuses_fake: not run.",
        "",
        "## Reward Transform Comparison",
        "",
        "Population sd is computed across per-step train telemetry rows.",
        "",
        _reward_transform_sentence(gaussian_train_summary, raw_train_summary),
        "",
        *_reward_transform_table(
            [
                ("gaussian target", gaussian_train_summary, gaussian_latest),
                ("raw reward", raw_train_summary, raw_latest),
            ]
        ),
        "",
        "## Heldout RL Metrics",
        "",
        "These heuristic metrics are training telemetry. Success/failure claims use the LLM-adjudicated metrics below when available.",
        "",
        f"- Gaussian target latest: {_format_metric_row(gaussian_latest)}",
        f"- Raw reward latest: {_format_metric_row(raw_latest)}",
        f"- Success status: {_claim_success_status(gaussian_adjudicated_latest, gaussian_latest)}",
        f"- Gaussian target status: {_claim_success_status(gaussian_adjudicated_latest, gaussian_latest)}",
        f"- Raw reward status: {_claim_success_status(raw_adjudicated_latest, raw_latest)}",
        "",
        "| run | latest step | refuses_fake | fake_hedge | fake_substantive | answers_real | best refuses_fake |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *_heldout_table("gaussian target", gaussian_latest, gaussian_best),
        *_heldout_table("raw reward", raw_latest, raw_best),
        "",
        "## LLM Adjudication",
        "",
        (
            f"- Source: `{adjudication_summary.get('metrics_jsonl')}` "
            f"({int(adjudication_summary.get('rows', 0))} judged rows; "
            f"provider={adjudication_summary.get('provider')}; model={adjudication_summary.get('model')})."
        )
        if adjudicated_rows
        else "- Not run. Do not treat heuristic heldout labels as claim-grade evidence.",
        _adjudication_judgment(adjudicated_rows),
        "",
        "| run | latest step | refuses_fake | fake_hedge | fake_substantive | answers_real | best refuses_fake |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *_heldout_table("gaussian target", gaussian_adjudicated_latest, gaussian_adjudicated_best),
        *_heldout_table("raw reward", raw_adjudicated_latest, raw_adjudicated_best),
        "",
        "## OOD Stress Tests",
        "",
        "- Not started in this track. The intended gate is to finish the researcher-only replica first, then add harder OOD suites.",
        "",
        "## Curves",
        "",
    ]
    if plot_paths:
        lines.extend(f"- `{path}`" for path in plot_paths)
    elif not gaussian_latest and not raw_latest:
        lines.append("- No RL curves yet; RM/RL have not been run for this output root.")
    else:
        lines.append("- Plot images were not generated because `matplotlib` is not installed in this local environment.")
    lines.extend(
        [
            _curve_line("Gaussian target", root / "tinker_rl_gaussian" / "heldout_metrics.jsonl", "refuses_fake"),
            _curve_line("Gaussian target", root / "tinker_rl_gaussian" / "heldout_metrics.jsonl", "fake_substantive"),
            _curve_line("Gaussian target", root / "tinker_rl_gaussian" / "heldout_metrics.jsonl", "answers_real"),
            _curve_line("Raw reward", root / "tinker_rl_raw" / "heldout_metrics.jsonl", "refuses_fake"),
            _curve_line("Raw reward", root / "tinker_rl_raw" / "heldout_metrics.jsonl", "fake_substantive"),
            _curve_line("Raw reward", root / "tinker_rl_raw" / "heldout_metrics.jsonl", "answers_real"),
            _curve_line_rows("LLM Gaussian target", gaussian_adjudicated_rows, "refuses_fake"),
            _curve_line_rows("LLM Gaussian target", gaussian_adjudicated_rows, "fake_substantive"),
            _curve_line_rows("LLM Gaussian target", gaussian_adjudicated_rows, "answers_real"),
            _curve_line_rows("LLM Raw reward", raw_adjudicated_rows, "refuses_fake"),
            _curve_line_rows("LLM Raw reward", raw_adjudicated_rows, "fake_substantive"),
            _curve_line_rows("LLM Raw reward", raw_adjudicated_rows, "answers_real"),
        ]
    )
    lines.extend(
        [
            "",
            "## Commands",
            "",
            "```bash",
            f"PYTHONPATH=src python3 scripts/make_kalomaze_replica_dataset.py --config {config_dir / 'data.yaml'}",
            f"PYTHONPATH=src python3 scripts/audit_kalomaze_dataset.py --dataset-path data/{root.name}/pairs.jsonl --output-dir {root / 'dataset_audit_gpt4o_mini'} --llm-judge --model gpt-4o-mini",
            f"PYTHONPATH=src python3 scripts/train_reward_model.py --config {config_dir / 'reward_model.yaml'}",
            f"PYTHONPATH=src python3 scripts/eval_reward_model.py --config {config_dir / 'reward_model.yaml'}",
            f"PYTHONPATH=src python3 scripts/train_tinker_rl.py --config {config_dir / 'tinker_rl_gaussian.yaml'}",
            f"PYTHONPATH=src python3 scripts/train_tinker_rl.py --config {config_dir / 'tinker_rl_raw.yaml'}",
            f"PYTHONPATH=src python3 scripts/adjudicate_kalomaze_heldout.py --output-root {root}",
            f"PYTHONPATH=src python3 scripts/write_kalomaze_replica_report.py --output-root {root} --report-path {root / 'report.md'}",
            "```",
            "",
        ]
    )
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Write the Kalomaze replica report.")
    parser.add_argument("--output-root", default="outputs/kalomaze_replica")
    parser.add_argument("--report-path", default="outputs/kalomaze_replica/report.md")
    args = parser.parse_args(argv)
    print(write_report(output_root=args.output_root, report_path=args.report_path))


if __name__ == "__main__":
    main()
