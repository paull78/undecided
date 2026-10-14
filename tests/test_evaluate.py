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

    metrics = score(records, probs)
    assert (metrics["accuracy"], metrics["macro_f1"]) == (1.0, 1.0)


def test_accuracy_is_broken_down_by_the_position_of_the_correct_option():
    # Correct option first: 2 of 2 right. Second: 1 of 2. Third: 0 of 1.
    records = [rec(["a", "b", "c"], 0), rec(["b", "a", "c"], 0),
               rec(["a", "b", "c"], 1), rec(["c", "b", "a"], 1),
               rec(["a", "b", "c"], 2)]
    probs = [[0.8, 0.1, 0.1], [0.6, 0.3, 0.1],
             [0.2, 0.7, 0.1], [0.5, 0.4, 0.1],
             [0.9, 0.05, 0.05]]

    by_position = score(records, probs)["accuracy_by_position"]

    assert by_position == {
        "1": {"n": 2, "accuracy": 1.0},
        "2": {"n": 2, "accuracy": 0.5},
        "3": {"n": 1, "accuracy": 0.0},
    }


def test_calibration_error_compares_confidence_with_accuracy_bin_by_bin():
    # Four predictions at 0.9 confidence, three right: that bin is off by |0.75 - 0.9| = 0.15.
    # Two predictions at 0.6 confidence, both right: off by |1.0 - 0.6| = 0.4.
    # ECE weighs bins by size: (4 * 0.15 + 2 * 0.4) / 6.
    records = [rec(["a", "b"], 0)] * 3 + [rec(["a", "b"], 1)] + [rec(["a", "b"], 0)] * 2
    probs = [[0.9, 0.1]] * 4 + [[0.6, 0.4]] * 2

    metrics = score(records, probs)

    assert math.isclose(metrics["ece"], (4 * 0.15 + 2 * 0.4) / 6)
    assert math.isclose(metrics["confidence"], (4 * 0.9 + 2 * 0.6) / 6)


def test_a_perfectly_calibrated_model_has_zero_calibration_error():
    records = [rec(["a", "b"], 0)] * 7 + [rec(["a", "b"], 1)] * 3
    probs = [[0.7, 0.3]] * 10  # always 70% sure of "a", right 70% of the time

    assert math.isclose(score(records, probs)["ece"], 0.0, abs_tol=1e-9)


def test_binary_datasets_report_ranking_auc_whatever_the_option_order():
    # P("spam") ranks all spam above all ham, though the model never crosses 0.5 for spam.
    records = [rec(["spam", "ham"], 0), rec(["ham", "spam"], 1), rec(["ham", "spam"], 0), rec(["spam", "ham"], 1)]
    probs = [[0.45, 0.55], [0.6, 0.4], [0.9, 0.1], [0.2, 0.8]]

    metrics = score(records, probs)

    assert metrics["accuracy"] == 0.5
    assert metrics["auc"] == 1.0


def test_auc_is_left_out_when_records_have_more_than_two_options():
    records = [rec(["a", "b", "c"], 0), rec(["a", "b", "c"], 1)]

    assert "auc" not in score(records, [[0.5, 0.3, 0.2], [0.2, 0.5, 0.3]])
