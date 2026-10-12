"""Hugging Face sources for every dataset: where to download it and how to convert it."""

import random
from collections.abc import Callable
from dataclasses import dataclass

from datasets import load_dataset

from undecided import data
from undecided.data import Record


@dataclass(frozen=True)
class Source:
    path: str
    train_split: str
    eval_split: str
    convert: Callable[..., list[Record]]
    label_field: str | None = None  # datasets whose rows carry label names need the full label set
    seeded: bool = False  # converters that pick instruction templates need the seed
    config: str | None = None


SOURCES = {
    "ag_news": Source("fancyzhx/ag_news", "train", "test", data.convert_ag_news),
    "emotion": Source("dair-ai/emotion", "train", "test", data.convert_emotion),
    "banking77": Source("mteb/banking77", "train", "test", data.convert_banking77, label_field="label_text"),
    "trec": Source("SetFit/TREC-QC", "train", "test", data.convert_trec, label_field="label_text"),
    "sst2": Source("stanfordnlp/sst2", "train", "validation", data.convert_sst2),
    "mnli": Source("nyu-mll/multi_nli", "train", "validation_matched", data.convert_mnli, seeded=True),
    # Held out: never trained on. Their train splits are only for fitting a per-dataset temperature.
    "yahoo_answers": Source("community-datasets/yahoo_answers_topics", "train", "test", data.convert_yahoo_answers),
    "massive": Source("SetFit/amazon_massive_intent_en-US", "train", "test", data.convert_massive,
                      label_field="label_text"),
    "tweet_irony": Source("cardiffnlp/tweet_eval", "train", "test", data.convert_tweet_irony, config="irony"),
    # Added in Phase 4b. SMS spam has only a train split, so evaluation samples from it too.
    "sms_spam": Source("ucirvine/sms_spam", "train", "train", data.convert_sms_spam),
    "tweet_hate": Source("cardiffnlp/tweet_eval", "train", "test", data.convert_tweet_hate, config="hate"),
    "subjectivity": Source("SetFit/subj", "train", "test", data.convert_subjectivity),
}


def load_records(name: str, split: str, n: int, seed: int) -> list[Record]:
    """A fixed-seed sample of n examples from the "train" or "eval" split, the same on every call."""
    source = SOURCES[name]
    rows = load_dataset(source.path, source.config, split=source.train_split if split == "train" else source.eval_split)
    kwargs = {}
    if source.label_field:
        # The label set comes from the train split, so train and eval records share one option order.
        train_rows = load_dataset(source.path, source.config, split=source.train_split)
        kwargs["labels"] = sorted(set(train_rows[source.label_field]))
    if source.seeded:
        kwargs["seed"] = seed
    if n < len(rows):
        rows = rows.select(sorted(random.Random(seed).sample(range(len(rows)), n)))
    return source.convert(rows, **kwargs)
