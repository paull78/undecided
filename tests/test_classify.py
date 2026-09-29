import math

import torch

from undecided.model import Undecided


def make_classifier(tokenizer, tiny_scorer):
    return Undecided(tiny_scorer, tokenizer, max_length=128, device=torch.device("cpu"))


def test_classify_returns_one_probability_per_option_in_the_callers_order(tokenizer, tiny_scorer):
    clf = make_classifier(tokenizer, tiny_scorer)

    probs = clf.classify(
        "My card was charged twice.",
        options=["billing", "bug report", "praise"],
        instruction="Classify the support ticket.",
    )

    assert list(probs) == ["billing", "bug report", "praise"]
    assert math.isclose(sum(probs.values()), 1.0, rel_tol=1e-6)
    assert all(0.0 < p < 1.0 for p in probs.values())


def test_a_saved_classifier_reloads_and_gives_identical_probabilities(tokenizer, tiny_scorer, tmp_path):
    clf = make_classifier(tokenizer, tiny_scorer)
    options = ["billing", "bug report", "praise", "refund request"]
    before = clf.classify("My card was charged twice.", options)

    clf.save_pretrained(tmp_path / "run")
    reloaded = Undecided.from_pretrained(tmp_path / "run", device=torch.device("cpu"))

    after = reloaded.classify("My card was charged twice.", options)
    assert after.keys() == before.keys()
    assert all(math.isclose(after[o], before[o], rel_tol=1e-5) for o in options)
    assert reloaded.max_length == 128


def test_duplicate_or_fewer_than_two_options_are_rejected(tokenizer, tiny_scorer):
    import pytest

    clf = make_classifier(tokenizer, tiny_scorer)

    with pytest.raises(ValueError, match="duplicate"):
        clf.classify("text", options=["billing", "praise", "billing"])
    with pytest.raises(ValueError, match="at least 2"):
        clf.classify("text", options=["billing"])


def test_more_options_than_fit_in_the_context_raise_a_clear_error(tokenizer, tiny_scorer):
    import pytest

    clf = make_classifier(tokenizer, tiny_scorer)  # max_length=128

    with pytest.raises(ValueError, match="max_length=128"):
        clf.classify("text", options=[f"option number {i}" for i in range(60)])
