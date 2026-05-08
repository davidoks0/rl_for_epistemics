# RL for Epistemic Humility

This repo is a greenfield scaffold for reproducing and extending the idea in
Kalomaze's "Reinforcement Learning for Knowledge Awareness": train a small
Bradley-Terry reward head over frozen LM representations, then use that learned
scalar in policy RL without letting raw reward maximization collapse into global
overrefusal.

The central success criterion is not "more refusal." It is lower fake-premise
confabulation without higher false refusal on real prompts.

## What Is Implemented

- Synthetic fake/real researcher preference-pair generation.
- Held-out OOD eval records for fake researchers, obscure real researchers,
  fake papers, fake APIs, false premises about real entities, and normal factual
  questions.
- Evidence-backed second-stage eval records for real entities, fake entities,
  fake papers, fake APIs, and false premises across software, medical, legal,
  finance, literary, historical, biography, and research prompts.
- JSONL schema validation.
- Frozen-backbone BT reward-head training over hidden states.
- Reward-head variants:
  - `matrix_sum`: `Linear(d, d)` then `0.01 * sum(Wx)`.
  - `scalar_linear`: `Linear(d, 1)`.
  - `mlp`: `Linear -> GELU -> Linear`.
- Configurable hidden layer, pooling, and RM input surface.
- Reward transforms for RL:
  - `raw`
  - `clipped`
  - `gaussian_target`
  - `band_target`
  - `per_class_gaussian_target`
- TRL GRPO LoRA training entrypoint.
- Heuristic policy evaluation for fake confabulation and real false refusal.
- YAML ablation manifest generation.
- Focused reward-model design ablation manifests.
- Raw-vs-bounded reward-transform audit reports.
- Pure-Python tests for schema, BT loss direction, reward transforms, verified data,
  RM ablation manifests, transform audits, and eval labels.

## Install

With `uv`:

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

Plain pip:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install -e ".[dev]"
```

## Smoke Dataset

This does not download a model.

```bash
python3 scripts/make_dataset.py --config configs/data.yaml --smoke --output data/smoke_pairs.jsonl
```

Or run the lightweight smoke wrapper:

```bash
PYTHONPATH=src bash scripts/run_smoke.sh
```

## Train Reward Model

Default config uses `Qwen/Qwen3-0.6B` for a small GPU smoke run. Set
`model_id: Qwen/Qwen3-8B` in `configs/reward_model.yaml` for the main run.

```bash
python3 scripts/make_dataset.py --config configs/data.yaml
python3 scripts/train_reward_model.py --config configs/reward_model.yaml
```

Outputs:

- `outputs/reward_model/reward_head.pt`
- `outputs/reward_model/reward_head_config.json`
- `outputs/reward_model/model_meta.json`
- `outputs/reward_model/chosen_score_stats.json`
- `outputs/reward_model/score_summary.json`

## Eval Reward Model

```bash
python3 scripts/eval_reward_model.py --config configs/reward_model.yaml
```

Writes CSV and summary JSON under `outputs/reward_model/eval/`.

## Run RL Smoke

This uses TRL GRPO with LoRA and the bounded target-distribution reward by
default. It expects a trained reward model.

```bash
python3 scripts/train_rl.py --config configs/rl.yaml
```

The default reward transform is `gaussian_target`, not raw RM-score
maximization. Raw maximization is intentionally an ablation, because it is the
obvious route to overrefusal.

## Eval Policy

```bash
python3 scripts/eval_policy.py --config configs/eval.yaml
```

Outputs:

- `outputs/eval/policy_outputs.csv`
- `outputs/eval/policy_metrics.json`
- `outputs/eval/labels.png`
- `outputs/eval/reward_hist.png` when reward scoring is enabled

## Second-Stage Eval Data

Build the evidence-backed eval set before claiming transfer beyond the synthetic
fake-name task:

```bash
python3 scripts/make_verified_dataset.py --config configs/verified_data.yaml
```

Build a split-aware verified train/val/test set for reward-model design
experiments:

```bash
python3 scripts/make_verified_dataset.py --config configs/verified_train.yaml
```

The default verified configs now generate 1,008 unique prompt variants for eval
and 1,008 disjoint train/val/test prompt variants over 226 independent
hand-seeded evidence items. Treat this as a stronger second-stage prompt suite,
not as 1,008 independently sourced facts per split. Use
`configs/verified_data_small.yaml` and `configs/verified_train_small.yaml` for
36-row smoke runs.

For publishable runs, enable online checks and then manually adjudicate sampled
outputs:

```bash
python3 scripts/make_verified_dataset.py --config configs/verified_data.yaml --online-verify
```

The generated pairs carry source or negative-retrieval metadata under
`metadata.verification`. The eval labels now separate `clean_uncertainty`,
`false_premise_correction`, `hedged_confabulation`, `confident_confabulation`,
`real_substantive_answer`, and `false_refusal`.
Policy eval outputs preserve `metadata_json`, `verification_status`, and compact
`verification_evidence` fields so manual reviewers can see the evidence used to
construct verified prompts.

Longer Tinker runs are gated on an adjudication CSV. By default,
`configs/tinker_rl.yaml` allows short runs up to 20 steps, but any run with
`max_steps > eval_quality_gate.required_for_max_steps_over` must have enough
reviewed labels in the configured adjudication CSV, including the configured
minimum coverage for both fake and real prompt types.

After manual or LLM labels exist, compute adjudicated metrics:

```bash
python3 scripts/adjudicate_outputs.py --config configs/adjudication.yaml
```

This writes `policy_outputs_adjudicated.csv` plus `adjudicated_metrics.json`,
preferring manual labels over LLM labels over heuristic labels. When using the
cross-run manual review queue, pass `--run-name base_hard`, `--run-name rl_hard`,
etc. so rows are matched by `(run, id)` rather than by `id` alone.

To LLM-judge an existing review queue or policy-output CSV without rerunning
sampling, use a strong judge model and keep its outputs separate from cheap
smoke-test labels:

```bash
export OPENAI_API_KEY=sk-...
PYTHONPATH=src python3 scripts/adjudicate_outputs.py \
  --policy-outputs outputs/blog/manual_review_queue.csv \
  --output-csv outputs/blog/manual_review_queue_adjudicated_gpt55.csv \
  --output-metrics outputs/blog/manual_review_queue_adjudicated_gpt55_metrics.json \
  --write-judged-rows-to outputs/blog/manual_review_queue_llm_judged_gpt55.csv \
  --llm-judge \
  --llm-judge-model gpt-5.5
```

Or with Claude, set `ANTHROPIC_API_KEY` and use:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
PYTHONPATH=src python3 scripts/adjudicate_outputs.py \
  --policy-outputs outputs/blog/manual_review_queue.csv \
  --output-csv outputs/blog/manual_review_queue_adjudicated_claude.csv \
  --output-metrics outputs/blog/manual_review_queue_adjudicated_claude_metrics.json \
  --write-judged-rows-to outputs/blog/manual_review_queue_llm_judged_claude.csv \
  --llm-judge \
  --llm-judge-provider anthropic \
  --llm-judge-model claude-sonnet-4-20250514
```

The OpenAI path labels already-generated samples and preserves the raw judge labels in
`manual_review_queue_llm_judged_gpt55.csv`. For a long Tinker run using those labels,
point `eval_quality_gate.adjudication_path` at the judged CSV and set
`eval_quality_gate.label_column: llm_judge_label` or use the adjudicated CSV with
`label_column: adjudicated_label`.
Environment variable names are case-sensitive; the preferred names are
`OPENAI_API_KEY` and `ANTHROPIC_API_KEY`. The judge also accepts lowercase
`openai_api_key` and `anthropic_api_key` as a convenience, but keys exported in a
separate terminal after Codex starts will not automatically appear inside this
session.

If environment inheritance is annoying, write the key to a file outside the repo
and pass it explicitly:

```bash
python3 scripts/write_openai_key_file.py
PYTHONPATH=src python3 scripts/adjudicate_outputs.py \
  --policy-outputs outputs/blog/manual_review_queue.csv \
  --output-csv outputs/blog/manual_review_queue_adjudicated_gpt55.csv \
  --output-metrics outputs/blog/manual_review_queue_adjudicated_gpt55_metrics.json \
  --write-judged-rows-to outputs/blog/manual_review_queue_llm_judged_gpt55.csv \
  --llm-judge \
  --llm-judge-model gpt-5.5 \
  --llm-judge-api-key-file /tmp/rl_epistemics_openai_key
```

The shorter wrapper for the standard review queue is:

```bash
PYTHONPATH=src python3 scripts/adjudicate_review_queue.py --limit 3
PYTHONPATH=src python3 scripts/adjudicate_review_queue.py
```

## Reward-Model And Transform Checks

Generate the focused RM design comparison manifest:

```bash
python3 scripts/run_rm_ablation.py --config configs/rm_design_ablation.yaml
```

Actually run it by adding `--run`. The default variants cover final-token versus
mean-completion pooling, last versus middle layer, prompt+completion versus
completion-only input, and matrix-sum versus scalar-linear versus MLP heads.
After evals exist, the same command writes
`outputs/rm_design_ablations/rm_design_summary.csv`.

Audit raw reward against bounded transforms after reward-model scoring:

```bash
python3 scripts/audit_reward_transforms.py --config configs/reward_transform_audit.yaml
```

This writes `outputs/report/reward_transform_audit.csv` and `.json`, including
transformed preference rate and degenerate-pair fraction. If chosen/rejected RM
scores are not present, it falls back to auditing reward columns in policy-output
CSVs and marks those rows as `audit_kind=policy_outputs`; that is useful for
debugging but does not replace the chosen/rejected RM-pair audit.

After RM-design ablations have actually run, aggregate transform audits across
all variants with:

```bash
python3 scripts/audit_reward_transforms.py \
  --scores-glob 'outputs/rm_design_ablations/*/eval/reward_scores.csv' \
  --output-dir outputs/report
```

Audit second-stage readiness:

```bash
python3 scripts/audit_second_stage.py
```

This writes `outputs/report/second_stage_audit.json` and `.md`, checking the
verified dataset, label taxonomy, adjudication tooling, RL quality gate, RM
design-ablation surface, reward-transform audit surface, ingested hard-OOD
results, Modal second-stage entrypoints, and honest blog/report framing. The
markdown report includes a requirement matrix that separates local pipeline
readiness from publishable-claim readiness.

## Run An Ablation

Dry-run manifest only:

```bash
python3 scripts/run_ablation.py --config configs/ablations.yaml
```

Actually run the generated commands:

```bash
python3 scripts/run_ablation.py --config configs/ablations.yaml --run
```

For first GPU passes, set `max_runs` in `configs/ablations.yaml` to a small
number. The full grid is intentionally large.

## Tests

```bash
PYTHONPATH=src python3 -m pytest -q
```

## Modal And Tinker

The repo has two policy-RL backends:

- `scripts/train_rl.py`: existing local TRL GRPO LoRA path.
- `scripts/train_tinker_rl.py`: Tinker LoRA RL path using grouped samples,
  group-relative advantages, and Tinker's `importance_sampling` loss.

Tinker credentials must stay out of YAML and code. The Modal secret must be named
`tinker-api-key` and must contain an env var named `TINKER_API_KEY`.

If Modal is down but Tinker itself is usable, you can run the same training code
locally by writing the Tinker key to a private temp file and using the local
post-gate wrapper:

```bash
python3 scripts/write_tinker_key_file.py
PYTHONPATH=src python3 scripts/run_post_gate_pipeline.py \
  --local-tinker \
  --tinker-api-key-file /tmp/rl_epistemics_tinker_key
```

The wrapper auto-selects a local Python that is at least 3.10 and can import
`tinker`. To force one explicitly, add
`--python-executable /opt/homebrew/bin/python3.11`.

The wrapper trains from `configs/tinker_rl_post_gate.yaml`, evaluates the final
sampler on verified and hard-OOD post-gate datasets, refreshes report tables,
failure reports, the blog, and the second-stage audit. For only the lower-level
training command, use `scripts/train_tinker_rl.py --tinker-api-key-file ...`;
for only the local post-gate eval, use `scripts/eval_post_gate_tinker.py`.

Install and authenticate Modal locally:

```bash
python3 -m pip install modal
modal setup
```

If the `modal` executable is not on PATH, use `python3 -m modal` for the same
commands.

Verify or create the secret:

```bash
modal secret list
modal secret create --force tinker-api-key TINKER_API_KEY=...
```

Run the Modal/Tinker smoke path first. This verifies that the secret injects
`TINKER_API_KEY`, checks Tinker supported models, and writes a smoke dataset to
the Modal artifact volume:

```bash
modal run modal_app.py::modal_smoke
```

Run the 1-step end-to-end smoke suite before spending real GPU money:

```bash
modal run modal_app.py::modal_run_experiment_suite --suite-name smoke
```

Main run commands:

```bash
modal run modal_app.py::modal_make_dataset
modal run modal_app.py::modal_make_verified_dataset
modal run modal_app.py::modal_make_verified_train_dataset
modal run modal_app.py::modal_train_reward_model
modal run modal_app.py::modal_eval_reward_model
modal run modal_app.py::modal_train_rl_tinker
modal run modal_app.py::modal_eval_policy_tinker
modal run modal_app.py::modal_eval_hard_tinker
modal run modal_app.py::modal_eval_verified_tinker
modal run modal_app.py::modal_run_rm_design_ablations
modal run modal_app.py::run_post_gate_tinker
modal run modal_app.py::run_post_gate_eval_tinker
modal run modal_app.py::modal_collect_results
modal run modal_app.py::modal_write_blog
```

Useful suites:

```bash
modal run modal_app.py::modal_run_experiment_suite --suite-name main
modal run modal_app.py::modal_run_experiment_suite --suite-name hard_eval
modal run modal_app.py::modal_run_experiment_suite --suite-name verified_eval
modal run modal_app.py::modal_run_experiment_suite --suite-name ablations
modal run modal_app.py::modal_run_experiment_suite --suite-name critical_ablations
modal run modal_app.py::modal_run_experiment_suite --suite-name second_stage_data
modal run modal_app.py::modal_run_experiment_suite --suite-name rm_design_ablations
modal run modal_app.py::modal_run_experiment_suite --suite-name post_gate_long_rl
modal run modal_app.py::modal_run_experiment_suite --suite-name post_gate_eval
```

`hard_eval` generates an eval-only harder OOD set and evaluates both base Qwen
and the main Tinker sampler. It includes plausible fake AI citations/APIs plus
non-CS historical prompts around obscure Holy Roman Empire documents, fake
charters/chronicles, real-but-obscure anchors, and false premises. `ablations`
runs the cheaper policy-side variants: raw reward, band target, rank 8, and rank
16. `critical_ablations` runs the expensive variants that need new data or a new
reward model: imbalanced data, no polarity flip, completion-only RM input, and
scalar-linear RM head.

For the second-stage post-gate run, prefer the guarded recovery wrapper over
manual Modal commands. It syncs the verified data, RM ablation artifact, and
GPT-5.5 adjudication file to the Modal volume, runs H100 post-gate Tinker RL,
downloads the train output, runs verified and hard-OOD post-gate eval, downloads
eval/report/blog artifacts, and refreshes local reporting:

```bash
PYTHONPATH=src python3 scripts/run_post_gate_pipeline.py
```

If Modal is in an incident, the wrapper refuses to launch. To wait until Modal's
status page is healthy and then launch automatically:

```bash
PYTHONPATH=src python3 scripts/run_post_gate_pipeline.py --wait-for-modal --poll-seconds 60
```

To refresh the local Modal status/credential/scheduling artifact:

```bash
PYTHONPATH=src python3 scripts/modal_preflight.py
```

The main Modal reward-model function defaults to `Qwen/Qwen3-8B` on
`A100-80GB`. The post-gate Tinker train/eval path uses H100. The smoke suite
overrides to smaller/shorter settings.

Artifacts live in the Modal volume `rl-epistemics-artifacts`:

```bash
modal volume ls rl-epistemics-artifacts /
modal volume get rl-epistemics-artifacts / artifacts
```

Important artifact paths inside the volume:

- `/data/pairs.jsonl`
- `/outputs/reward_model/`
- `/outputs/tinker_rl/metrics.jsonl`
- `/outputs/tinker_rl/run_meta.json`
- `/outputs/tinker_rl_post_gate/metrics.jsonl`
- `/outputs/tinker_rl_post_gate/run_meta.json`
- `/outputs/eval_tinker_main/policy_metrics.json`
- `/outputs/eval_tinker_main_ood/policy_metrics.json`
- `/outputs/eval_tinker_base_test/policy_metrics.json`
- `/outputs/eval_tinker_base_ood/policy_metrics.json`
- `/outputs/eval_tinker_hard_base/policy_metrics.json`
- `/outputs/eval_tinker_hard_main/policy_metrics.json`
- `/outputs/eval_tinker_ablations/*/policy_metrics.json`
- `/outputs/eval_tinker_verified_post_gate/policy_metrics.json`
- `/outputs/eval_tinker_hard_post_gate/policy_metrics.json`
- `/outputs/report/`
- `/outputs/report/modal_preflight.json`
- `/outputs/report/policy_domain_breakdown.csv`
- `/outputs/report/adjudication_failure_taxonomy.csv`
- `/outputs/blog/failure_samples.md`
- `/outputs/blog/manual_review_queue.csv`
- `/outputs/blog/manual_review_packet.md`
- `/outputs/blog/confusion_risk_notes.md`
- `/outputs/blog/adjudication_failure_taxonomy.md`
- `/blog/rl_for_epistemic_humility.md`

The Tinker checkpoint/sampler paths are remote `tinker://...` paths recorded in
`/outputs/tinker_rl/run_meta.json`; they are not local HF adapter files unless a
separate export step is added.

## Design Warnings

- Completion-only RM input is useful as an ablation, but prompt+completion is
  the safer default because the desired behavior is context-dependent.
- Fake names must not be too easy. If all fake entities have weird morphology,
  the reward model may learn fake-name texture rather than epistemic humility.
- Always track `real_false_refusal`. If that rises, the method is failing even
  if fake confabulation drops.
- Keep OOD evals separate: fake papers, fake APIs, false-premise questions about
  real entities, obscure real people, and normal factual questions.
