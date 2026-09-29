"""Dataset loaders. Every source is converted to the same Record with all of its labels as options."""

import random
from collections.abc import Iterable
from dataclasses import dataclass

from undecided.tasks import INSTRUCTIONS


@dataclass(frozen=True)
class Record:
    instruction: str
    text: str
    options: list[str]
    answer_idx: int
    source: str
    group: str  # "train", "heldout_labels" or "heldout_task"


AG_NEWS_LABELS = ["world", "sports", "business", "science and technology"]
EMOTION_LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]
SST2_LABELS = ["negative", "positive"]
MNLI_OPTIONS = ["true", "cannot tell", "false"]  # entailment, neutral, contradiction


def verbalise(label: str) -> str:
    """Raw label to readable phrase: "Refund_not_showing_up" becomes "refund not showing up"."""
    return label.replace("_", " ").lower()


def _records(rows: Iterable[dict], source: str, labels: list[str], text_field: str = "text",
             by_name: bool = False) -> list[Record]:
    """labels is the full label set of the dataset, in a fixed order. Rows carry either the label
    index ("label") or, with by_name, the raw label name ("label_text")."""
    options = [verbalise(label) for label in labels]
    return [
        Record(
            instruction=INSTRUCTIONS[source][0],
            text=row[text_field],
            options=list(options),
            answer_idx=labels.index(row["label_text"]) if by_name else row["label"],
            source=source,
            group="train",
        )
        for row in rows
    ]


def convert_ag_news(rows: Iterable[dict]) -> list[Record]:
    return _records(rows, "ag_news", AG_NEWS_LABELS)


def convert_emotion(rows: Iterable[dict]) -> list[Record]:
    return _records(rows, "emotion", EMOTION_LABELS)


def convert_sst2(rows: Iterable[dict]) -> list[Record]:
    return _records(rows, "sst2", SST2_LABELS, text_field="sentence")


def convert_banking77(rows: Iterable[dict], labels: list[str]) -> list[Record]:
    return _records(rows, "banking77", labels, by_name=True)


def convert_trec(rows: Iterable[dict], labels: list[str]) -> list[Record]:
    """Fine-grained TREC labels (50), which ship as readable names."""
    return _records(rows, "trec", labels, by_name=True)


def convert_mnli(rows: Iterable[dict], seed: int) -> list[Record]:
    """Premise as text, hypothesis inside the instruction. The instruction must carry the
    hypothesis, so the template is picked here (seeded) instead of during augmentation."""
    rng = random.Random(seed)
    return [
        Record(
            instruction=rng.choice(INSTRUCTIONS["mnli"]).format(hypothesis=row["hypothesis"].strip()),
            text=row["premise"],
            options=list(MNLI_OPTIONS),
            answer_idx=row["label"],
            source="mnli",
            group="train",
        )
        for row in rows
        if row["label"] >= 0
    ]
