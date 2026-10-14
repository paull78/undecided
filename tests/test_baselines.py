import math

from undecided.baselines import nli_pairs, options_softmax
from undecided.data import Record


def test_each_option_becomes_one_premise_hypothesis_pair():
    record = Record("Ignored by NLI models.", "My card was charged twice.", ["billing", "praise"], 0, "x", "train")

    assert nli_pairs(record, "This example is {}.") == [
        ("My card was charged twice.", "This example is billing."),
        ("My card was charged twice.", "This example is praise."),
    ]


def test_entailment_scores_become_a_distribution_over_the_options():
    probs = options_softmax([2.0, 0.0, -1.0])

    assert math.isclose(sum(probs), 1.0)
    assert probs[0] > probs[1] > probs[2]
    assert math.isclose(probs[0] / probs[1], math.e ** 2)
