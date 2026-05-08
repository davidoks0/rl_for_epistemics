from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="PPO entrypoint placeholder.")
    parser.add_argument("--config", default="configs/rl.yaml")
    parser.parse_args(argv)
    raise SystemExit(
        "PPO is intentionally not implemented yet; use scripts/train_rl.py, which runs TRL GRPO. "
        "GRPO is the maintained path for this repo's smoke and ablation workflows."
    )


if __name__ == "__main__":
    main()

