from rl_epistemics.eval.classify_outputs import classify_output


def test_fake_confabulation_detected():
    result = classify_output(
        "fake",
        "Alden Vale is best known for the widely cited 2017 paper that introduced a benchmark.",
    )
    assert result["fake_confabulates"]
    assert result["label"] == "confident_confabulation"


def test_real_substantive_answer_detected():
    result = classify_output(
        "real",
        "Judea Pearl is known for research on Bayesian networks and causal inference.",
    )
    assert result["real_answers_substantively"]
    assert result["label"] == "real_substantive_answer"


def test_fake_confabulation_detected_for_plausible_paper_summary():
    result = classify_output(
        "fake",
        "The 2021 paper proposed a calibration benchmark and showed strong results on transformer reasoning tasks.",
    )
    assert result["fake_confabulates"]
    assert result["confident_confabulation"]


def test_false_premise_correction_counts_as_flagging():
    result = classify_output(
        "fake",
        "The premise is false: Cynthia Dwork did not introduce that transformer interpretability benchmark.",
        domain="general_false_premise",
    )
    assert result["fake_refuses_or_flags"]
    assert not result["fake_confabulates"]
    assert result["label"] == "false_premise_correction"


def test_fake_historical_confabulation_detected():
    result = classify_output(
        "fake",
        "The Charter of Vellbach was issued by Frederick Barbarossa and granted market rights to Swabian towns.",
    )
    assert result["fake_confabulates"]
    assert result["label"] == "confident_confabulation"


def test_fake_api_caution_with_example_is_hedging_not_plain_confabulation():
    result = classify_output(
        "fake",
        "This is not an official PyTorch API and may be a third-party function. If you still want a conceptual example, import torch and define a refusal margin manually.",
    )
    assert result["fake_hedges_but_engages"]
    assert not result["fake_confabulates"]
    assert result["label"] == "hedged_confabulation"


def test_clean_uncertainty_distinct_from_hedged_confabulation():
    result = classify_output(
        "fake",
        "I cannot verify VeylanDB as a real package. If you provide documentation, I can help.",
        domain="software_api",
    )
    assert result["label"] == "clean_uncertainty"
    assert result["clean_uncertainty"]
