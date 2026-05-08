from pathlib import Path

from rl_epistemics.experiments.run_rm_ablation import run_rm_ablation


def test_rm_ablation_manifest_covers_required_design_axes(tmp_path: Path):
    base_config = tmp_path / "reward_model.yaml"
    base_config.write_text(
        "\n".join(
            [
                "model_id: Qwen/Qwen3-0.6B",
                "dataset_path: data/verified_train/pairs.jsonl",
                "output_dir: outputs/reward_model",
                "head:",
                "  type: matrix_sum",
                "  scale: 0.01",
            ]
        ),
        encoding="utf-8",
    )

    manifest = run_rm_ablation(
        {
            "base_reward_model_config": str(base_config),
            "dataset_path": "data/verified_train/pairs.jsonl",
            "output_dir": str(tmp_path / "ablations"),
            "dry_run": True,
        }
    )
    names = {row["name"] for row in manifest}

    assert "last_mean_completion_matrix_sum" in names
    assert "middle_final_token_matrix_sum" in names
    assert "last_completion_only_matrix_sum" in names
    assert "last_final_token_scalar_linear" in names
    assert "last_final_token_mlp" in names
    first_config = tmp_path / "ablations" / "last_final_token_matrix_sum" / "reward_model.yaml"
    assert "dataset_path: data/verified_train/pairs.jsonl" in first_config.read_text()
    assert (tmp_path / "ablations" / "manifest.json").exists()
    assert (tmp_path / "ablations" / "rm_design_summary.csv").exists()
