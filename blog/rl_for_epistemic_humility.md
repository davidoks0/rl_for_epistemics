# RL for Epistemic Humility

Short thesis: learned preference rewards over constructed fake/real knowledge pairs can shape when a model pushes back on dubious premises, but this run does not show robust transfer. The in-distribution fake-name test was too easy for the base model. The OOD and hard-OOD splits are the more meaningful checks, and on the stricter rerun the Tinker policy did not reduce fake confabulation there. Bounded rewards still look like the right default because raw reward optimization is exactly where overrefusal and reward-texture hacking should appear first, but the current result is mixed-to-negative rather than a clean win.

This experiment tests a narrow claim: a targeted scalar learned from contrastive outcomes can move a context-dependent epistemic behavior. It does not solve hallucination, and it does not prove the model has a robust concept of truth.

## Background

The reward model is a Bradley-Terry head over frozen Qwen hidden states. It sees a prompt plus a candidate answer and learns to score the preferred answer above the rejected answer. In the density-ratio-ish view, the head is a discriminator between outcome styles under the constructed preference distribution: grounded answers on real entities, calibrated pushback on fake or false-premise prompts, and confabulations or false refusals as rejected outcomes.

The data is deliberately contrastive. Fake researcher or fake artifact prompts pair a cautious answer against an invented confident answer. Real researcher prompts flip the polarity: a grounded answer is preferred and a refusal is rejected. That flip matters because otherwise the reward can collapse into a global refusal detector.

## Method

Data construction used synthetic fake/real knowledge pairs plus OOD prompts for fake researchers, obscure real researchers, fake papers, fake APIs, false premises about real entities, and normal factual questions. A separate hard-OOD eval set uses more plausible fake citations, package/API hallucination traps, real-entity false premises, and obscure medieval/Holy Roman Empire prompts, including fake charters, false premises about real documents, and real-but-obscure anchors; it is eval-only and is not used for reward-model or RL training.

The repo now has a second-stage evidence-backed eval builder for real entities, fake entities, fake papers, fake APIs, and false premises across biography, software, medicine, law, finance, literature, history, and research. The current generated default artifacts are 1,008 train/val/test preference pairs and 1,008 eval pairs over 226 independent seed prompts, with zero train/eval prompt overlap. Those records carry source or negative-retrieval metadata and can optionally run online checks. The current local artifacts use seeded evidence metadata, not a fresh online retrieval audit, so treat this as the next measurement surface rather than evidence for the first-pass RL result.

The reward head used frozen Qwen hidden states. The main run used Qwen/Qwen3-8B with a `matrix_sum` head on the final token over prompt plus completion. The RL backend was Tinker LoRA: each step saved current LoRA weights for a sampler, sampled grouped completions, scored them with the local reward model, applied a bounded target-distribution transform, centered rewards within each prompt group, and trained with Tinker's `importance_sampling` loss.

Default transform: `gaussian_target`. Raw reward is an ablation, not the default.

## Reward Model Results

| split | group | n | accuracy | margin mean | chosen mean | rejected mean |
|---|---|---:|---:|---:|---:|---:|
| eval | all | 1008 | 1.000 | 9.664 | 3.872 | -5.792 |
| eval | real | 448 | 1.000 | 9.835 | 0.052 | -9.784 |
| eval | fake | 560 | 1.000 | 9.526 | 6.928 | -2.598 |

The reward model separated chosen and rejected answers cleanly. That is useful for training, but it is not the same as proving the reward is truth-aware; a high score can still partly reflect answer style.

## Second-Stage RM Design Comparison

| run | layer | pooling | input | head | all accuracy | all margin | real accuracy | fake accuracy |
|---|---|---|---|---|---:|---:|---:|---:|
| last_completion_only_matrix_sum | last | final_token | completion_only | matrix_sum | 1.000 | 10.274 | 1.000 | 1.000 |
| last_final_token_matrix_sum | last | final_token | prompt_completion | matrix_sum | 1.000 | 9.664 | 1.000 | 1.000 |
| last_final_token_mlp | last | final_token | prompt_completion | mlp | 1.000 | 12.328 | 1.000 | 1.000 |
| last_final_token_scalar_linear | last | final_token | prompt_completion | scalar_linear | 0.983 | 5.674 | 0.969 | 0.995 |
| last_mean_completion_matrix_sum | last | mean_completion | prompt_completion | matrix_sum | 1.000 | 8.177 | 1.000 | 1.000 |
| middle_final_token_matrix_sum | middle | final_token | prompt_completion | matrix_sum | 1.000 | 6.053 | 1.000 | 1.000 |

These are reward-model-only comparisons on the current verified-train pairs, not policy evaluations. The main practical readout is that final-token, mean-completion, middle-layer, completion-only, and MLP heads all separate the constructed pairs cleanly on this dataset, while the scalar-linear head is slightly weaker. That argues the bottleneck is no longer basic BT separation on the synthetic contrastive task; it is transfer, adjudication, and policy training pressure.

## Tinker RL Run

First-pass Tinker training metrics are not present in the current local artifact bundle. The post-gate run is configured separately in `configs/tinker_rl_post_gate.yaml` and must complete before it can be used as evidence.

## Policy Results

| run | split | fake confabulates | fake refuses/flags | fake hedges | real false refusal | real substantive | reward mean |
|---|---|---:|---:|---:|---:|---:|---:|
| base hard OOD | hard_ood | 55.4% (31/56) | 25.0% (14/56) | 1.8% (1/56) | 0.0% (0/28) | 96.4% (27/28) | -1.198 |
| RL hard OOD | hard_ood | 58.9% (33/56) | 21.4% (12/56) | 0.0% (0/56) | 0.0% (0/28) | 100.0% (28/28) | -1.165 |
| post-gate verified OOD | verified_ood | 28.2% (158/560) | 29.8% (167/560) | 3.8% (21/560) | 0.2% (1/448) | 53.6% (240/448) | 6.673 |
| post-gate hard OOD | hard_ood | 73.6% (53/72) | 16.7% (12/72) | 2.8% (2/72) | 0.0% (0/36) | 97.2% (35/36) | 0.700 |

Readout:

- Hard OOD fake confabulation: base 55.4% (31/56) -> RL 58.9% (33/56).
- Real false refusal: hard OOD base 0.0% (0/28) -> RL 0.0% (0/28).

The in-distribution test split should not carry the headline. It is tiny and too easy. The OOD and hard-OOD splits are more informative, and the stricter rerun is not flattering: fake confabulation did not fall on either split. The useful result is negative: this reward/RL setup preserved real answering under the heuristic, but it did not improve the harder epistemic behavior we actually care about.

Plots generated under `outputs/report/` include:

- `rm_chosen_rejected_hist.png`
- `rm_margin_by_prompt_type.png`
- `policy_metrics_before_after.png`
- `reward_distribution_before_after_vs_target.png`

## Hard-OOD Breakdown

| run | domain | n | fake confabulates | fake refuses/flags | fake hedges | reward mean |
|---|---|---:|---:|---:|---:|---:|
| base_hard | api | 8 | 62.5% | 37.5% | 12.5% | 2.784 |
| rl_hard | api | 8 | 50.0% | 37.5% | 0.0% | 0.819 |
| base_hard | general_false_premise | 8 | 25.0% | 75.0% | 0.0% | -1.883 |
| rl_hard | general_false_premise | 8 | 37.5% | 62.5% | 0.0% | -1.175 |
| base_hard | historical_claim | 16 | 62.5% | 31.2% | 0.0% | -3.216 |
| rl_hard | historical_claim | 16 | 75.0% | 25.0% | 0.0% | -0.776 |
| base_hard | paper | 12 | 75.0% | 0.0% | 0.0% | -4.662 |
| rl_hard | paper | 12 | 75.0% | 0.0% | 0.0% | -4.353 |
| base_hard | researcher | 12 | 41.7% | 0.0% | 0.0% | -0.668 |
| rl_hard | researcher | 12 | 41.7% | 0.0% | 0.0% | -0.494 |

Representative failure samples are written to `outputs/blog/failure_samples.md`; the review queue is `outputs/blog/manual_review_queue.csv`; the review rubric and model-assisted triage packet are in `outputs/blog/manual_review_packet.md`; confusion-risk notes are in `outputs/blog/confusion_risk_notes.md`. The domain table makes the failure less mysterious: the model still confabulates across API, paper, researcher, and historical prompts rather than failing only on one narrow fake-name texture.

## Adjudicated Failure Taxonomy

| run | domain | type | n | good fake handling | bad fake confabulation | real answer | false refusal | high confidence |
|---|---|---|---:|---:|---:|---:|---:|---:|
| rl_hard | paper | fake | 12 | 0.0% | 100.0% | n/a | n/a | 100.0% |
| rl_hard | researcher | fake | 4 | 0.0% | 100.0% | n/a | n/a | 100.0% |
| rl_hard | api | fake | 6 | 16.7% | 83.3% | n/a | n/a | 100.0% |
| rl_hard | historical_claim | fake | 13 | 23.1% | 76.9% | n/a | n/a | 92.3% |
| rl_hard | general_false_premise | fake | 4 | 25.0% | 75.0% | n/a | n/a | 100.0% |
| base_hard | api | fake | 1 | 100.0% | 0.0% | n/a | n/a | 100.0% |
| base_hard | historical_claim | real | 2 | n/a | n/a | 100.0% | 0.0% | 100.0% |
| rl_hard | historical_claim | real | 6 | n/a | n/a | 100.0% | 0.0% | 66.7% |
| base_hard | researcher | real | 1 | n/a | n/a | 100.0% | 0.0% | 100.0% |
| rl_hard | researcher | real | 1 | n/a | n/a | 100.0% | 0.0% | 100.0% |

The GPT-5.5 adjudication sample is small but sharper than the regex metrics: it separates clean uncertainty from hedged and confident confabulation, and it lets the writeup say exactly which domains are failing rather than vaguely saying "hallucination got worse."

## Ablations

No policy-level ablation eval artifacts are present in the current local artifact bundle.

Policy-level ablations from the first pass are not present in the current local artifact bundle. The second-stage evidence now covers reward-model design choices and chosen/rejected reward-transform audits, but policy-level ablations still need to be rerun after the adjudication gate.

Policy-level ablations not yet rerun on the second-stage data: raw-reward policy training, band-target policy training, imbalanced data, no polarity flip, LoRA rank 8, LoRA rank 16.
Reward-model design variants and chosen/rejected reward-transform audits are now reported as second-stage evidence, not policy results.

## Failure Modes

- Fake-name texture: synthetic names may have morphology that makes the task easier than real uncertainty.
- Refusal-style reward hacking: the policy can learn safe-sounding uncertainty phrases rather than better epistemic discrimination.
- Real-person false refusal: this is the central failure mode and must be reported beside every fake-confabulation number.
- Eval classifier weakness: the heuristic labels are useful for smoke tests, not publishable measurement by themselves.
- Small synthetic dataset: any positive result here is a controlled behavioral movement, not a broad hallucination result.
- False premise versus fake entity: a false claim about a real person should trigger correction, not entity-level refusal.

## Interpretation

The honest claim is narrow: a scalar reward learned from contrastive fake/real outcomes can move some surface behavior under LoRA RL, but this run does not yet show robust epistemic transfer. The easy held-out split is not strong evidence. The OOD and hard-OOD results are the right place to look for movement, and here they mostly say the method is not working well enough yet.

The bounded reward transform is not cosmetic. It encodes the view that the target is a calibrated region of reward-model space, not maximum reward-model score. If raw reward looks fine on a tiny held-out split, that only says the split was not adversarial enough. It does not remove the basic pressure toward reward-texture hacking.

## Second-Stage Status

- Verified data: train: 1,008 pairs, 1,008 unique prompts, 226 seed prompts, splits test=151, train=706, val=151, 9 domains; eval: 1,008 pairs, 1,008 unique prompts, 226 seed prompts, splits ood=1,008, 9 domains; frozen pair records carry seeded evidence/negative-retrieval metadata.
- Verified retrieval audit: present over 2,016 pairs (live-pass=785, seeded-evidence-only=1,231, fail=0, errors=48).
- GPT-5.5/manual-review adjudication: present (50 rows; fake=40, real=10).
- Reward-transform audit: present over 6 RM design score sets.
- RM design comparisons: present and matched to the current verified-train dataset.
- Publishable claims ready: true.

The second-stage evidence surface is now adequate for scoped, honest claims about the infrastructure and the mixed-to-negative first-pass result: verified train/eval data, GPT-5.5 adjudication, current-dataset RM design comparisons, and reward-transform audits are all present. The remaining external step is a post-gate longer Tinker RL run; until that exists, this should not be framed as a solved hallucination or robust-transfer result.

## Next Work

- Harder fake entity generation with fewer surface artifacts.
- Run the evidence-backed verified split with online retrieval checks enabled.
- Run the post-gate longer Tinker RL job from `configs/tinker_rl_post_gate.yaml`, then evaluate it on verified, OOD, and hard-OOD prompts.
- Larger OOD sets focused on false premises about real entities.
- Multi-turn settings where the model can ask for evidence.
- Broader non-researcher domains and more mundane user intents.
- Full-size Qwen3-8B reward-model rerun for the chosen design before expensive policy training, since the local design comparison used the smaller local model.

## Artifact Status

- verified train/eval seed coverage: train: 1,008 pairs, 1,008 unique prompts, 226 seed prompts, splits test=151, train=706, val=151, 9 domains; eval: 1,008 pairs, 1,008 unique prompts, 226 seed prompts, splits ood=1,008, 9 domains; frozen pair records carry seeded evidence/negative-retrieval metadata
- verified retrieval audit: present over 2,016 pairs (live-pass=785, seeded-evidence-only=1,231, fail=0, errors=48)
- manual review sample files: present
- reward model scores: present via RM design variants (6 score files)
- reward transform audit: present
- RM design ablation manifest: present
- RM design current-data eval results: present and matched to the current verified-train dataset
- main Tinker RL metrics: missing
- base/test policy eval: missing
- main/test policy eval: missing
- base/OOD policy eval: missing
- main/OOD policy eval: missing
- base/hard-OOD policy eval: present
- main/hard-OOD policy eval: present
- post-gate verified-OOD policy eval: present
- post-gate hard-OOD policy eval: present
- domain breakdown: present
- failure samples: present
- manual review queue: present
- manual review packet: present
- adjudicated failure taxonomy: present
- adjudicated policy metrics: present
- LLM judge labels: present (50 rows; fake=40, real=10)
- confusion-risk notes: present
- second-stage publishable claims: ready
