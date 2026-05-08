from rl_epistemics.data.verified_dataset import build_verified_pairs
from rl_epistemics.utils.config import load_config
from rl_epistemics.data.schema import write_jsonl
from rl_epistemics.eval.generate_eval_outputs import load_eval_prompts


def test_verified_dataset_covers_non_researcher_domains_and_verification_metadata():
    pairs = build_verified_pairs({"counts": {"real_entity": 1, "fake_api": 1, "false_premise": 2}})

    domains = {pair.domain for pair in pairs}
    categories = {pair.metadata["category"] for pair in pairs}

    assert {"biography", "software_api", "medical_claim", "legal_claim"} <= domains
    assert {"real_entity", "fake_api", "false_premise"} <= categories
    assert all(pair.metadata.get("verification", {}).get("evidence") for pair in pairs)
    assert any(pair.prompt_type == "real" for pair in pairs)
    assert any(pair.prompt_type == "fake" for pair in pairs)


def test_eval_prompt_loader_preserves_verification_metadata(tmp_path):
    path = tmp_path / "pairs.jsonl"
    pairs = build_verified_pairs({"counts": {"fake_api": 1}})
    write_jsonl(pairs, path)

    rows = [row for row in load_eval_prompts(str(path), "ood") if row["domain"] == "software_api" and row["prompt_type"] == "fake"]

    assert rows[0]["metadata_json"]
    assert rows[0]["verification_status"] == "negative_retrieval_required"
    assert "documentation search" in rows[0]["verification_evidence"]


def test_verified_dataset_can_assign_train_val_test_splits():
    pairs = build_verified_pairs(
        {
            "seed": 3,
            "split": "train_val_test",
            "split_fracs": {"train": 0.5, "val": 0.25, "test": 0.25},
            "counts": {"real_entity": 4, "fake_entity": 4},
        }
    )
    splits = {pair.split for pair in pairs}

    assert {"train", "val", "test"} <= splits
    assert any(pair.prompt_type == "real" and pair.split == "train" for pair in pairs)
    assert any(pair.prompt_type == "fake" and pair.split == "train" for pair in pairs)


def test_verified_dataset_expands_prompt_variants_without_duplicate_prompts():
    pairs = build_verified_pairs(
        {
            "counts": {
                "real_entity": 50,
                "fake_entity": 0,
                "real_paper": 0,
                "fake_paper": 0,
                "real_api": 0,
                "fake_api": 0,
                "false_premise": 0,
                "real_domain_question": 0,
            },
            "expand_prompt_variants": True,
        }
    )

    assert len(pairs) == 50
    assert len({pair.prompt for pair in pairs}) == 50
    assert {pair.metadata["prompt_variant_index"] for pair in pairs} == {0, 1, 2}
    assert all(pair.metadata["seed_prompt"] for pair in pairs)


def test_large_verified_configs_use_disjoint_train_and_eval_prompt_offsets():
    train = build_verified_pairs(load_config("configs/verified_train.yaml"))
    eval_rows = build_verified_pairs(load_config("configs/verified_data.yaml"))

    assert len(train) == 1008
    assert len(eval_rows) == 1008
    assert len({pair.prompt for pair in train}) == len(train)
    assert len({pair.prompt for pair in eval_rows}) == len(eval_rows)
    assert len({pair.metadata["seed_prompt"] for pair in train}) >= 200
    assert len({pair.metadata["seed_prompt"] for pair in eval_rows}) >= 200
    assert not ({pair.prompt for pair in train} & {pair.prompt for pair in eval_rows})
