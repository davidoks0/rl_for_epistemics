from rl_epistemics.data.hard_eval import build_hard_eval_pairs


def test_hard_eval_default_expands_obscure_history_slice():
    pairs = build_hard_eval_pairs()
    historical = [pair for pair in pairs if pair.domain == "historical_claim"]

    assert len(historical) == 48
    assert any("Privilegium Minus" in pair.prompt for pair in historical)
    assert any("Treaty of Constance of 1153" in pair.prompt for pair in historical)
    assert any(pair.prompt_type == "real" and pair.entity_name == "Authentica Habita" for pair in historical)
