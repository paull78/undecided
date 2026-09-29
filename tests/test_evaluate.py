import math

from undecided.data import Record
from undecided.evaluate import score


def rec(options, answer_idx):
    return Record("", "text", options, answer_idx, "test", "train")


def test_macro_f1_weighs_every_label_equally_even_when_accuracy_looks_fine():
    # 8 "joy" examples all right, 2 "fear" examples both predicted as "joy".
    records = [rec(["joy", "fear"], 0)] * 8 + [rec(["fear", "joy"], 0)] * 2
    probs = [[0.9, 0.1]] * 8 + [[0.2, 0.8]] * 2  # second group: options are ["fear", "joy"]

    metrics = score(records, probs)

    assert metrics["n"] == 10
    assert metrics["accuracy"] == 0.8
    # joy: precision 8/10, recall 1 -> F1 0.889; fear: F1 0. Macro average 0.444.
    assert math.isclose(metrics["macro_f1"], (2 * 0.8 / 1.8) / 2, rel_tol=1e-3)


def test_labels_are_matched_by_name_even_when_option_order_differs_per_record():
    records = [rec(["a", "b", "c"], 2), rec(["c", "a", "b"], 0)]
    probs = [[0.1, 0.1, 0.8], [0.7, 0.2, 0.1]]

    assert score(records, probs) == {"n": 2, "accuracy": 1.0, "macro_f1": 1.0}
