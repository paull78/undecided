import random

from undecided.augment import augment
from undecided.data import Record

LABELS = [f"label {i}" for i in range(20)]


def make_record(answer_idx=7):
    return Record(
        instruction="",
        text="some text",
        options=list(LABELS),
        answer_idx=answer_idx,
        source="test",
        group="train",
    )


def test_augmented_options_always_contain_the_correct_label_at_answer_idx():
    rng = random.Random(0)
    for _ in range(500):
        out = augment(make_record(), rng, max_options=16)

        assert out.options[out.answer_idx] == "label 7"
        assert 2 <= len(out.options) <= 16
        assert len(set(out.options)) == len(out.options)
        assert set(out.options) <= set(LABELS)


def test_correct_option_lands_in_every_position_and_first_position_is_not_favoured():
    rng = random.Random(1)
    record = make_record(answer_idx=0)
    firsts = 0
    positions = set()
    trials = 2000
    for _ in range(trials):
        out = augment(record, rng, max_options=4)
        positions.add(out.answer_idx)
        firsts += out.answer_idx == 0

    assert positions == {0, 1, 2, 3}
    # with 2 to 4 options, the first slot should hold the answer roughly 36% of the time
    assert 0.30 < firsts / trials < 0.43


def test_shuffle_options_keeps_every_option_and_follows_the_answer():
    from undecided.augment import shuffle_options

    rng = random.Random(2)
    outs = [shuffle_options(make_record(answer_idx=3), rng) for _ in range(50)]

    for out in outs:
        assert sorted(out.options) == sorted(LABELS)
        assert out.options[out.answer_idx] == "label 3"
    assert len({tuple(o.options) for o in outs}) > 1
