from pathlib import Path

from rl_epistemics.experiments.eval_post_gate_tinker import _post_gate_eval_config


def test_post_gate_eval_config_points_at_post_gate_sampler_and_outputs():
    config = _post_gate_eval_config(
        {"base_model": "Qwen/Qwen3-8B", "reward_model_dir": "outputs/rm"},
        dataset_path="data/verified_eval/pairs.jsonl",
        output_dir="outputs/eval_tinker_verified_post_gate",
        manual_review_output="outputs/blog/manual_review_samples_verified_post_gate.md",
        tinker_api_key_file="/tmp/key",
        overrides={"limit": 3},
    )

    assert config["dataset_path"] == "data/verified_eval/pairs.jsonl"
    assert config["split"] == "ood"
    assert config["run_meta_path"] == "outputs/tinker_rl_post_gate/run_meta.json"
    assert config["output_dir"] == "outputs/eval_tinker_verified_post_gate"
    assert config["manual_review_output"] == "outputs/blog/manual_review_samples_verified_post_gate.md"
    assert config["tinker_api_key_file"] == "/tmp/key"
    assert config["include_reward_scores"] is True
    assert config["limit"] == 3


def test_eval_post_gate_script_exists():
    assert Path("scripts/eval_post_gate_tinker.py").exists()
