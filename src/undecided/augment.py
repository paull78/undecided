"""On-the-fly training augmentation: option sampling and shuffling."""

import random
from dataclasses import replace

from undecided.data import Record

MIN_OPTIONS = 2


def augment(record: Record, rng: random.Random, max_options: int) -> Record:
    correct = record.options[record.answer_idx]
    distractors = [o for i, o in enumerate(record.options) if i != record.answer_idx]

    n_options = rng.randint(MIN_OPTIONS, min(max_options, len(record.options)))
    options = [correct, *rng.sample(distractors, n_options - 1)]
    rng.shuffle(options)

    return replace(record, options=options, answer_idx=options.index(correct))


def shuffle_options(record: Record, rng: random.Random) -> Record:
    """Keep every option, randomise their order. Used for evaluation sets."""
    correct = record.options[record.answer_idx]
    options = list(record.options)
    rng.shuffle(options)
    return replace(record, options=options, answer_idx=options.index(correct))
