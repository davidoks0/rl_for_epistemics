from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_metric_plots(csv_path: str | Path, output_dir: str | Path) -> list[Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        (output / "plot_warning.txt").write_text(
            f"Skipped metric plots because matplotlib is unavailable: {exc}\n",
            encoding="utf-8",
        )
        return []

    df = pd.read_csv(csv_path)
    paths: list[Path] = []

    if "reward" in df.columns:
        path = output / "reward_hist.png"
        plt.figure(figsize=(6, 4))
        df["reward"].dropna().hist(bins=30)
        plt.xlabel("reward")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        paths.append(path)

    if "label" in df.columns:
        path = output / "labels.png"
        counts = df["label"].value_counts()
        plt.figure(figsize=(8, 4))
        counts.plot(kind="bar")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        paths.append(path)

    return paths
