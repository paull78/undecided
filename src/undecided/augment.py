"""On-the-fly training augmentation: instructions, option sampling and shuffling."""

import random
from dataclasses import replace

from undecided.data import Record
from undecided.tasks import INSTRUCTIONS, PARAPHRASES

MIN_OPTIONS = 2
KEEP_ALL_UP_TO = 3  # sentiment and NLI: dropping a label would change the task
NO_INSTRUCTION_RATE = 0.2
PARAPHRASE_RATE = 0.3
FIXED_INSTRUCTION_SOURCES = {"mnli"}  # the instruction carries the hypothesis


def augment(record: Record, rng: random.Random, max_options: int) -> Record:
    correct = record.options[record.answer_idx]
    distractors = [o for i, o in enumerate(record.options) if i != record.answer_idx]

    if len(record.options) <= KEEP_ALL_UP_TO:
        n_options = len(record.options)
    else:
        n_options = rng.randint(MIN_OPTIONS, min(max_options, len(record.options)))
    options = [correct, *rng.sample(distractors, n_options - 1)]
    rng.shuffle(options)
    answer_idx = options.index(correct)
    options = _paraphrase(options, PARAPHRASES.get(record.source, {}), rng)

    return replace(
        record,
        instruction=_instruction(record, rng),
        options=options,
        answer_idx=answer_idx,
    )


def _paraphrase(options: list[str], paraphrases: dict[str, list[str]], rng: random.Random) -> list[str]:
    """Swap some options for an alternative wording, position by position, never creating a duplicate."""
    out = list(options)
    for i, option in enumerate(options):
        if option in paraphrases and rng.random() < PARAPHRASE_RATE:
            alternative = rng.choice(paraphrases[option])
            if alternative not in out:
                out[i] = alternative
    return out


def _instruction(record: Record, rng: random.Random) -> str:
    if record.source in FIXED_INSTRUCTION_SOURCES or record.source not in INSTRUCTIONS:
        return record.instruction
    if rng.random() < NO_INSTRUCTION_RATE:
        return ""
    return rng.choice(INSTRUCTIONS[record.source])


def shuffle_options(record: Record, rng: random.Random) -> Record:
    """Keep every option, randomise their order. Used for evaluation sets."""
    correct = record.options[record.answer_idx]
    options = list(record.options)
    rng.shuffle(options)
    return replace(record, options=options, answer_idx=options.index(correct))
