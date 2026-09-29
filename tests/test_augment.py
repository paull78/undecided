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


def test_instruction_is_a_random_template_or_empty_about_a_fifth_of_the_time():
    from undecided.tasks import INSTRUCTIONS

    rng = random.Random(3)
    emotion = Record(INSTRUCTIONS["emotion"][0], "im so happy", ["sadness", "joy"], 1, "emotion", "train")
    outs = [augment(emotion, rng, max_options=16).instruction for _ in range(2000)]

    empty_rate = outs.count("") / len(outs)
    assert 0.16 < empty_rate < 0.24
    assert set(outs) - {""} <= set(INSTRUCTIONS["emotion"])
    assert len(set(outs)) > 5


def test_mnli_and_unknown_sources_keep_their_instruction():
    rng = random.Random(4)
    mnli = Record('Is "it rains" implied?', "Clouds.", ["true", "cannot tell", "false"], 0, "mnli", "train")
    other = Record("Custom instruction.", "text", ["a", "b"], 0, "test", "train")

    for _ in range(50):
        assert augment(mnli, rng, max_options=16).instruction == 'Is "it rains" implied?'
        assert augment(other, rng, max_options=16).instruction == "Custom instruction."


def test_small_label_sets_always_keep_every_label():
    rng = random.Random(5)
    nli = Record("Is it true?", "text", ["true", "cannot tell", "false"], 2, "test", "train")  # no paraphrases

    for _ in range(200):
        out = augment(nli, rng, max_options=16)
        assert sorted(out.options) == ["cannot tell", "false", "true"]
        assert out.options[out.answer_idx] == "false"


def test_options_are_sometimes_paraphrased_and_the_answer_follows_its_paraphrase():
    from undecided.tasks import PARAPHRASES

    rng = random.Random(6)
    labels = ["world", "sports", "business", "science and technology"]
    record = Record("", "Rover finds water on Mars", labels, 3, "ag_news", "train")
    accepted = {"science and technology", *PARAPHRASES["ag_news"]["science and technology"]}

    answers = []
    for _ in range(2000):
        out = augment(record, rng, max_options=4)
        assert out.options[out.answer_idx] in accepted
        assert len(set(out.options)) == len(out.options)
        answers.append(out.options[out.answer_idx])

    paraphrased = sum(a != "science and technology" for a in answers) / len(answers)
    assert 0.25 < paraphrased < 0.35
