from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rl_epistemics.models.reward_calibration import write_calibrated_reward_model
from rl_epistemics.utils.config import load_config


def calibrate_reward_model(config: dict[str, Any]) -> dict[str, Any]:
    return write_calibrated_reward_model(config)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Write post-hoc affine calibration artifacts for a reward model.")
    parser.add_argument("--config", default="configs/calibrate_reward_model.yaml")
    parser.add_argument("--source-reward-model-dir", default=None)
    parser.add_argument("--output-reward-model-dir", default=None)
    parser.add_argument("--scores-path", default=None)
    parser.add_argument("--fit-split", default=None)
    args = parser.parse_args(argv)

    config = load_config(args.config) if Path(args.config).exists() else {}
    if args.source_reward_model_dir:
        config["source_reward_model_dir"] = args.source_reward_model_dir
    if args.output_reward_model_dir:
        config["output_reward_model_dir"] = args.output_reward_model_dir
    if args.scores_path:
        config["scores_path"] = args.scores_path
    if args.fit_split is not None:
        config["fit_split"] = args.fit_split

    print(json.dumps(calibrate_reward_model(config), indent=2))


if __name__ == "__main__":
    main()
