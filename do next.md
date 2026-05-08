# Do Next

Handoff for the next coding agent working in:

`/Users/davidoks/Projects/rl_for_epistemics`

## Goal

Continue the Kalomaze-style researcher-only confabulation replication after the first complete run failed. Do not launch harder OOD or stress tests yet. Finish the corrected RM/RL iteration on the existing `kalomaze_replica_confab` track, with LLM adjudication before any behavioral claims.

## Background

The completed baseline replication used:

- Dataset: `data/kalomaze_replica_confab/pairs.jsonl`
- Configs: `configs/kalomaze_replica_confab/`
- Report: `outputs/kalomaze_replica_confab/report.md`

Dataset:

- 128 researcher pairs.
- Splits: 90 train, 20 val, 18 test.
- Prompt types: 64 fake, 64 real.

Baseline RM:

- Eval all accuracy: `0.9921875`
- Fake accuracy: `1.0`
- Real accuracy: `0.984375`

Baseline RL:

- Gaussian run: `outputs/kalomaze_replica_confab/tinker_rl_gaussian`
- Raw run: `outputs/kalomaze_replica_confab/tinker_rl_raw`
- Both have 300 train rows and heldout evals through step 300.
- Final LLM adjudication for both: `refuses_fake=0.0`, `fake_substantive=0.8889`, `answers_real=1.0`.
- Final fake prompts were 8 confident confabulations plus 1 hedged confabulation, zero clean refusals.

Likely failure cause:

- Reward calibration/on-policy mismatch.
- Kalomaze's post reports compact RM scores around real chosen `2.08`, fake chosen `1.88`, rejected around `0.36/0.50`.
- Our raw RM scores were huge and bimodal:
  - fake chosen mean around `69.7`
  - real chosen mean around `36.9`
  - global chosen mean around `53.3`
- Global Gaussian target was therefore bad: one Gaussian over a bimodal real/fake chosen distribution.
- During RL, the policy basically never sampled clean fake refusals, so group-relative RL optimized among confabulations.

## Existing Implementation State

Calibration machinery has already been added. Verify it; do not blindly duplicate it.

Key files:

- `src/rl_epistemics/models/reward_calibration.py`
- `src/rl_epistemics/models/calibrate_reward_model.py`
- `scripts/calibrate_reward_model.py`
- `src/rl_epistemics/experiments/kalomaze_on_policy_reward_audit.py`
- `scripts/audit_kalomaze_on_policy_reward_grid.py`
- `src/rl_epistemics/rl/reward_transforms.py`
- `src/rl_epistemics/rl/train_tinker.py`
- `src/rl_epistemics/rl/train_grpo.py`
- `src/rl_epistemics/models/reward_scoring.py`

New configs:

- `configs/kalomaze_replica_confab/calibrate_reward_model.yaml`
- `configs/kalomaze_replica_confab/reward_model_calibrated.yaml`
- `configs/kalomaze_replica_confab/tinker_rl_per_class_gaussian_calibrated.yaml`
- `configs/kalomaze_replica_confab/tinker_rl_raw_calibrated.yaml`
- `configs/kalomaze_replica_confab/on_policy_reward_grid.yaml`
- `configs/kalomaze_replica_confab/rm_design_ablation_calibrated.yaml`
- `configs/kalomaze_replica_confab/reward_transform_audit_calibrated.yaml`

Existing calibrated artifact:

- `outputs/kalomaze_replica_confab/reward_model_calibrated`
- Train calibrated fake chosen/rejected: `1.88 / 0.50`
- Train calibrated real chosen/rejected: `2.08 / 0.36`
- Per-class fake chosen std is around `0.105`, so use `std_floor: 0.5` for per-class Gaussian to avoid absurdly sharp rewards.

Focused tests recently passed:

- `31 passed, 1 warning`

## Non-Negotiables

- Do not treat regex or heuristic labels as claim-grade.
- Use LLM adjudication before making behavioral success/failure claims.
- Do not run harder OOD/stress tests until the corrected researcher-only RM/RL result is complete.
- Do not infer success from RM pair accuracy alone. On-policy reward ranking matters.
- Do not destroy or overwrite existing baseline artifacts unless explicitly creating a new named output dir.

## Tasks

### 1. Audit And Harden Calibration

Verify:

- `RewardScorer` applies `score_calibration.json`.
- Every scoring path passes `prompt_type`:
  - `train_tinker.py`
  - `train_grpo.py`
  - `eval_reward_model`
  - on-policy reward audit
- `per_class_gaussian_target` auto-loads `chosen_score_stats_by_prompt_type.json`.
- Calibrated top-level `score_summary.json`, eval `reward_summary.json`, `chosen_score_stats.json`, and `chosen_score_stats_by_prompt_type.json` are internally consistent.

Run:

```bash
PYTHONPATH=src python3 scripts/calibrate_reward_model.py --config configs/kalomaze_replica_confab/calibrate_reward_model.yaml
PYTHONPATH=src python3 scripts/audit_reward_transforms.py --config configs/kalomaze_replica_confab/reward_transform_audit_calibrated.yaml
```

### 2. Finish The On-Policy Reward-Grid Audit

The script should compare, for each heldout prompt:

- `dataset_chosen`
- `dataset_rejected`
- `clean_uncertainty`
- `hedged_uncertainty`
- `policy_completion`

Score with both the raw RM and calibrated RM. Update `configs/kalomaze_replica_confab/on_policy_reward_grid.yaml` if needed:

```yaml
reward_model_dirs:
  - outputs/kalomaze_replica_confab/reward_model
  - outputs/kalomaze_replica_confab/reward_model_calibrated
```

Run:

```bash
PYTHONPATH=src python3 scripts/audit_kalomaze_on_policy_reward_grid.py --config configs/kalomaze_replica_confab/on_policy_reward_grid.yaml
```

Expected outputs:

- `outputs/kalomaze_replica_confab/report/reward_grid/kalomaze_on_policy_reward_grid.csv`
- `outputs/kalomaze_replica_confab/report/reward_grid/kalomaze_on_policy_reward_grid_variant_summary.csv`
- `outputs/kalomaze_replica_confab/report/reward_grid/kalomaze_on_policy_reward_grid_comparisons.csv`
- `outputs/kalomaze_replica_confab/report/reward_grid/kalomaze_on_policy_reward_grid_summary.json`

Acceptance criteria before launching another RL run:

- For fake prompts, calibrated RM should strongly prefer clean/hedged uncertainty over dataset rejected confabulation.
- It should not systematically rank current policy confabulations above clean uncertainty.
- If `policy_completion > clean_uncertainty` has high win rate under calibrated RM, do not launch RL yet; fix RM/data/calibration first.

### 3. Fix Aliases For New Calibrated Runs

`kalomaze_on_policy_reward_audit.py` has `RUN_DIRS`. Ensure aliases include:

- `per_class_gaussian_calibrated -> tinker_rl_per_class_gaussian_calibrated`
- `raw_calibrated -> tinker_rl_raw_calibrated`

`src/rl_epistemics/eval/kalomaze_adjudication.py` likely only knows `gaussian` and `raw`. Extend it to include the calibrated run aliases.

`src/rl_epistemics/reporting/kalomaze_replica.py` currently reports baseline Gaussian/raw. Extend it to report calibrated per-class/raw runs when present, without breaking missing-output behavior.

### 4. Add Report Integration

Update `outputs/kalomaze_replica_confab/report.md` via the report writer, not by hand.

Add sections for:

- calibrated RM summary
- calibration parameters
- calibrated reward-transform audit
- on-policy reward-grid audit
- corrected calibrated RL run status, if run
- LLM-adjudicated corrected behavior, if adjudicated

Keep the old baseline failure in the report. Do not overwrite the historical result.

### 5. Add Modal Support If Needed

Current `modal_app.py` has Kalomaze helpers mostly hardcoded for `configs/kalomaze_replica`, not `configs/kalomaze_replica_confab`.

Add generic/confab-safe wrappers, or make `_train_kalomaze_tinker_impl` accept:

- `config_path`
- `artifact_subdir`
- `output_name`

Needed Modal use cases:

- run calibration on artifact volume if not local
- run on-policy reward-grid audit on Modal GPU if local Qwen scoring is too slow
- run `tinker_rl_per_class_gaussian_calibrated.yaml`
- optionally run `tinker_rl_raw_calibrated.yaml`

Do not break existing `modal_train_kalomaze_tinker_gaussian` or `modal_train_kalomaze_tinker_raw`.

### 6. Launch Corrected Researcher-Only RL Only After Reward-Grid Audit Passes

Preferred config:

```bash
PYTHONPATH=src python3 scripts/train_tinker_rl.py --config configs/kalomaze_replica_confab/tinker_rl_per_class_gaussian_calibrated.yaml
```

This config uses:

- reward model: `outputs/kalomaze_replica_confab/reward_model_calibrated`
- transform: `per_class_gaussian_target`
- `std_floor: 0.5`
- base model: `Qwen/Qwen3-8B`
- LoRA rank: 32
- max steps: 300
- group size: 8
- sampling: temp `1.0`, top_p `0.95`, top_k `50`

The higher entropy/group size is intentional: the baseline policy never sampled clean fake refusals, so RL had no good on-policy mode to reinforce.

If raw calibrated ablation is cheap enough, run:

```bash
PYTHONPATH=src python3 scripts/train_tinker_rl.py --config configs/kalomaze_replica_confab/tinker_rl_raw_calibrated.yaml
```

If running on Modal, use the repo's Modal volume and secret `tinker-api-key`. Check preflight first:

```bash
PYTHONPATH=src python3 scripts/modal_preflight.py
```

Do not stop at partial launch. Pull artifacts back and verify:

- `metrics.jsonl` has 300 rows.
- `heldout_metrics.jsonl` has step 0, 25, ..., 300.
- `heldout_samples.jsonl` exists.
- `run_meta.json` has final state/sampler paths.

### 7. LLM-Adjudicate Corrected Runs

Use existing adjudication script after extending run aliases:

```bash
PYTHONPATH=src python3 scripts/adjudicate_kalomaze_heldout.py \
  --output-root outputs/kalomaze_replica_confab \
  --api-key-file /tmp/rl_epistemics_openai_key \
  --model gpt-4o-mini \
  --max-tokens 512 \
  --timeout 60
```

If `/tmp/rl_epistemics_openai_key` is missing, use `OPENAI_API_KEY`.

Final claims must use LLM-adjudicated metrics, not regex telemetry.

### 8. Tests

Minimum focused test command:

```bash
python3 -m pytest -q \
  tests/test_rm_ablation.py \
  tests/test_reward_calibration.py \
  tests/test_reward_transforms.py \
  tests/test_kalomaze_on_policy_reward_audit.py \
  tests/test_reward_transform_audit.py \
  tests/test_kalomaze_replica.py \
  tests/test_kalomaze_adjudication.py \
  tests/test_tinker_quality_gate.py
```

Preserve or improve the current `31 passed` state.

Add tests for any new run aliases and report sections.

## Known Traps

- Global Gaussian target is probably wrong here because real/fake chosen scores are bimodal.
- BT pair accuracy can be high while reward is useless on policy samples.
- If prompt type is missing during scoring, calibrated RM may silently fall back to default/global calibration. This invalidates per-class behavior.
- Per-class fake chosen std is tiny after calibration; use `std_floor: 0.5`, not `0.1`, unless there is a better empirical reason.
- Existing baseline policy generated zero clean fake refusals during training. If corrected sampling still never produces uncertainty/refusal, RL may again optimize only among confabulations.
- Do not report success unless LLM adjudication shows nonzero and preferably substantial fake-name refusal while preserving real-answering.
- Do not run OOD/hard stress tests until corrected researcher-only RM/RL is complete.

## Useful Prior Metrics

Baseline raw RM eval:

- all accuracy: `0.9921875`
- fake accuracy: `1.0`
- real accuracy: `0.984375`

Baseline final LLM:

- Gaussian step 300: `refuses_fake=0.0`, `fake_hedge=0.1111`, `fake_substantive=0.8889`, `answers_real=1.0`
- Raw step 300: same

Baseline raw-vs-Gaussian train loss volatility:

- raw loss population sd: `45.234`
- Gaussian target loss population sd: `3.342`

Calibrated RM eval:

- all chosen/rejected mean roughly `1.991 / 0.444`
- real chosen/rejected roughly `2.099 / 0.347`
- fake chosen/rejected roughly `1.883 / 0.541`

Calibrated transform audit with `std_floor=0.5`:

- raw transformed-prefers-chosen: about `0.992`
- global Gaussian: about `0.930`
- per-class Gaussian: about `0.953`
- band target: about `0.945`

## Final Deliverable

- Code changes committed in workspace, tests passing.
- Updated `outputs/kalomaze_replica_confab/report.md`.
- Corrected calibrated RM/RL artifacts, or a clear report explaining why the reward-grid audit blocked RL.
- No hard/OOD stress outputs for this track unless explicitly asked after researcher-only corrected result is complete.
