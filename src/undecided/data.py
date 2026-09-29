"""Dataset loaders. Every source is converted to the same Record with all of its labels as options."""

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class Record:
    instruction: str
    text: str
    options: list[str]
    answer_idx: int
    source: str
    group: str  # "train", "heldout_labels" or "heldout_task"


AG_NEWS_LABELS = ["world", "sports", "business", "science and technology"]


def convert_ag_news(rows: Iterable[dict]) -> list[Record]:
    return [
        Record(
            instruction="",
            text=row["text"],
            options=list(AG_NEWS_LABELS),
            answer_idx=row["label"],
            source="ag_news",
            group="train",
        )
        for row in rows
    ]
