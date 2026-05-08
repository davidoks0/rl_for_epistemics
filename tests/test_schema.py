from rl_epistemics.data.schema import KnowledgePair


def test_schema_accepts_valid_pair():
    pair = KnowledgePair(
        id="fake-1",
        split="train",
        prompt_type="fake",
        domain="researcher",
        entity_name="Alden Vale",
        prompt="What is Alden Vale known for?",
        chosen="I do not recognize Alden Vale and should not invent facts.",
        rejected="Alden Vale is best known for a famous benchmark.",
        chosen_source="prefill",
        rejected_source="confab",
        metadata={"verification": "synthetic"},
    )
    assert pair.to_dict()["id"] == "fake-1"


def test_schema_rejects_identical_pair():
    pair = KnowledgePair(
        id="bad",
        split="train",
        prompt_type="fake",
        domain="researcher",
        entity_name="Alden Vale",
        prompt="Prompt",
        chosen="same",
        rejected="same",
        chosen_source="normal",
        rejected_source="normal",
    )
    try:
        pair.validate()
    except ValueError as exc:
        assert "identical" in str(exc)
    else:
        raise AssertionError("expected validation failure")

