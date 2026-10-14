import math
import random

from undecided.calibration import apply_temperature, brier_score, fit_temperature, log_loss
from undecided.data import Record


def rec(options, answer_idx):
    return Record("", "text", options, answer_idx, "test", "train")


def softmax(z):
    m = max(z)
    e = [math.exp(v - m) for v in z]
    return [v / sum(e) for v in e]


def test_fitting_recovers_the_temperature_that_generated_the_labels():
    # The model's probabilities are softmax(z), but the labels were drawn from softmax(z / 2):
    # the model is overconfident by a factor of 2, so the best temperature is about 2.
    rng = random.Random(0)
    records, probs = [], []
    for _ in range(20000):
        z = [rng.gauss(0, 3) for _ in range(4)]
        truth = softmax([v / 2 for v in z])
        answer = rng.choices(range(4), weights=truth)[0]
        records.append(rec(["a", "b", "c", "d"], answer))
        probs.append(softmax(z))

    assert math.isclose(fit_temperature(records, probs), 2.0, rel_tol=0.05)


def test_brier_score_and_log_loss_match_hand_computed_values():
    records = [rec(["a", "b"], 0), rec(["a", "b", "c"], 2)]
    probs = [[0.8, 0.2], [0.5, 0.25, 0.25]]

    # Brier: (0.2² + 0.2²) and (0.5² + 0.25² + 0.75²), averaged over the two examples.
    assert math.isclose(brier_score(records, probs), (0.08 + 0.875) / 2)
    assert math.isclose(log_loss(records, probs), (-math.log(0.8) - math.log(0.25)) / 2)


def test_temperature_one_changes_nothing_and_a_higher_one_flattens():
    probs = [[0.7, 0.2, 0.1]]

    assert all(math.isclose(a, b) for a, b in zip(apply_temperature(probs, 1.0)[0], probs[0]))
    flatter = apply_temperature(probs, 3.0)[0]
    assert max(flatter) < 0.7 and math.isclose(sum(flatter), 1.0)
    assert flatter.index(max(flatter)) == 0  # the ranking never changes
