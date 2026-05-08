#!/usr/bin/env bash
set -euo pipefail

python3 scripts/make_dataset.py --config configs/data.yaml --smoke --output data/smoke_pairs.jsonl
python3 scripts/run_ablation.py --config configs/ablations.yaml --max-runs 4

cat <<'MSG'
Smoke data and dry-run ablation manifest completed.
Heavy model smoke commands:
  python3 scripts/train_reward_model.py --config configs/reward_model.yaml
  python3 scripts/eval_reward_model.py --config configs/reward_model.yaml
  python3 scripts/train_rl.py --config configs/rl.yaml
MSG
