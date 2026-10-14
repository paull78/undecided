from undecided.data import Record
from undecided.probe import with_instruction
from undecided.tasks import INSTRUCTIONS


def rec(source, instruction):
    return Record(instruction, "text", ["a", "b"], 0, source, "train")


def test_instruction_variants_keep_remove_or_swap_in_another_datasets_instruction():
    sets = {"ag_news": [rec("ag_news", INSTRUCTIONS["ag_news"][0])],
            "sst2": [rec("sst2", INSTRUCTIONS["sst2"][0])],
            "tweet_irony": [rec("tweet_irony", INSTRUCTIONS["tweet_irony"][0])]}

    canonical = with_instruction(sets, "canonical")
    none = with_instruction(sets, "none")
    mismatched = with_instruction(sets, "mismatched")

    assert canonical == sets
    assert all(r.instruction == "" for records in none.values() for r in records)
    assert mismatched["ag_news"][0].instruction == INSTRUCTIONS["sst2"][0]       # the next dataset's
    assert mismatched["tweet_irony"][0].instruction == INSTRUCTIONS["ag_news"][0]  # wraps around
    assert all(m[0].text == s[0].text and m[0].options == s[0].options
               for m, s in zip(mismatched.values(), sets.values()))


def test_pair_tasks_are_left_out_of_the_variants_that_would_drop_their_second_text():
    sets = {"ag_news": [rec("ag_news", INSTRUCTIONS["ag_news"][0])],
            "mnli": [rec("mnli", 'Given the text, is this statement true: "It rained."')]}

    assert set(with_instruction(sets, "none")) == {"ag_news"}
    assert set(with_instruction(sets, "mismatched")) == {"ag_news"}
    assert set(with_instruction(sets, "canonical")) == {"ag_news", "mnli"}


def test_order_averaging_matches_options_by_name_and_returns_them_in_the_original_order():
    import math

    from undecided.probe import average_by_name

    base = Record("", "text", ["a", "b", "c"], 0, "test", "train")
    reordered = Record("", "text", ["c", "a", "b"], 1, "test", "train")
    probs_base = [[0.6, 0.3, 0.1]]
    probs_reordered = [[0.3, 0.2, 0.5]]  # c, a, b

    [averaged] = average_by_name([base], [[base], [reordered]], [probs_base, probs_reordered])

    assert all(math.isclose(x, y) for x, y in zip(averaged, [0.4, 0.4, 0.2]))  # a, b, c


def test_extra_orders_are_permutations_that_keep_the_answer():
    import random

    from undecided.probe import reorder

    base = Record("", "text", ["a", "b", "c", "d"], 2, "test", "train")
    rng = random.Random(0)
    orders = [reorder(base, rng) for _ in range(20)]

    assert all(sorted(r.options) == sorted(base.options) for r in orders)
    assert all(r.options[r.answer_idx] == "c" for r in orders)
    assert len({tuple(r.options) for r in orders}) > 5


def test_a_distractor_is_added_at_a_random_position_without_moving_the_answer():
    import random

    from undecided.probe import with_distractor

    base = Record("", "text", ["a", "b", "c"], 1, "test", "train")
    rng = random.Random(0)
    out = [with_distractor(base, "weather forecast", rng) for _ in range(30)]

    assert all(len(r.options) == 4 and "weather forecast" in r.options for r in out)
    assert all(r.options[r.answer_idx] == "b" for r in out)
    assert len({r.options.index("weather forecast") for r in out}) == 4  # every position used


def test_accuracy_is_split_by_whether_the_gold_label_has_a_training_counterpart():
    from undecided.probe import accuracy_by_overlap

    records = [Record("", "t", ["sports", "health"], 0, "y", "heldout_labels"),   # overlapping class, right
               Record("", "t", ["sports", "health"], 0, "y", "heldout_labels"),   # overlapping class, wrong
               Record("", "t", ["sports", "health"], 1, "y", "heldout_labels")]   # new class, right
    probs = [[0.9, 0.1], [0.2, 0.8], [0.3, 0.7]]

    split = accuracy_by_overlap(records, probs, overlapping={"sports"})

    assert split == {"overlapping": {"n": 2, "accuracy": 0.5}, "new": {"n": 1, "accuracy": 1.0}}
