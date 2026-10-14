"""On-the-fly training augmentation: instructions, option sampling, shuffling and yes/no recasting."""

import random
from dataclasses import replace

from undecided.data import Record
from undecided.tasks import INSTRUCTIONS, PARAPHRASES

MIN_OPTIONS = 2
KEEP_ALL_UP_TO = 3  # sentiment and NLI: dropping a label would change the task
NO_INSTRUCTION_RATE = 0.2
PARAPHRASE_RATE = 0.3
FIXED_INSTRUCTION_SOURCES = {"mnli", "qqp", "qnli", "boolq"}  # the instruction carries the second text


def augment(record: Record, rng: random.Random, max_options: int, recast_rate: float = 0.0) -> Record:
    if record.source not in FIXED_INSTRUCTION_SOURCES and rng.random() < recast_rate:
        return _recast(record, rng)
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


def _recast(record: Record, rng: random.Random) -> Record:
    """A check on one label, the shape of an unseen binary task ("ironic" or "not ironic"). The label
    asked about is the true one half the time, so neither answer is a shortcut."""
    correct = record.options[record.answer_idx]
    asked = correct if rng.random() < 0.5 else rng.choice([o for o in record.options if o != correct])
    is_correct = asked == correct
    asked = _paraphrase([asked], PARAPHRASES.get(record.source, {}), rng)[0]
    instruction = _instruction(record, rng)
    if asked.startswith("not ") or rng.random() < 0.5:
        question = f'Is the answer "{asked}"?' if instruction else f'Does "{asked}" describe this text?'
        instruction = f"{instruction} {question}".strip()
        options, answer = ["yes", "no"], "yes" if is_correct else "no"
    else:
        options, answer = [asked, f"not {asked}"], asked if is_correct else f"not {asked}"
    rng.shuffle(options)
    return replace(record, instruction=instruction, options=options, answer_idx=options.index(answer))


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
