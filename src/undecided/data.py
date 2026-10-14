"""Dataset loaders. Every source is converted to the same Record with all of its labels as options."""

import random
from collections.abc import Iterable
from dataclasses import dataclass

from undecided.tasks import INSTRUCTIONS, MASSIVE_NAMES


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
YAHOO_LABELS = ["Society & Culture", "Science & Mathematics", "Health", "Education & Reference",
                "Computers & Internet", "Sports", "Business & Finance", "Entertainment & Music",
                "Family & Relationships", "Politics & Government"]
MNLI_OPTIONS = ["true", "cannot tell", "false"]  # entailment, neutral, contradiction
IRONY_LABELS = ["not ironic", "ironic"]
SPAM_LABELS = ["not spam", "spam"]
HATE_LABELS = ["not hate speech", "hate speech"]
SUBJECTIVITY_LABELS = ["objective", "subjective"]
QQP_OPTIONS = ["not duplicate", "duplicate"]
QNLI_OPTIONS = ["answers it", "does not answer it"]  # entailment, not_entailment
BOOLQ_OPTIONS = ["no", "yes"]
COLA_LABELS = ["ungrammatical", "grammatical"]


def verbalise(label: str) -> str:
    """Raw label to readable phrase: "Refund_not_showing_up" becomes "refund not showing up"."""
    return label.replace("_", " ").lower()


def _records(rows: Iterable[dict], source: str, labels: list[str], text_field: str = "text",
             by_name: bool = False, group: str = "train", names: dict[str, str] | None = None) -> list[Record]:
    """labels is the full label set of the dataset, in a fixed order. Rows carry either the label
    index ("label") or, with by_name, the raw label name ("label_text"). names maps raw labels to
    option text where verbalise() isn't enough."""
    options = [names[label] if names else verbalise(label) for label in labels]
    return [
        Record(
            instruction=INSTRUCTIONS[source][0],
            text=row[text_field],
            options=list(options),
            answer_idx=labels.index(row["label_text"]) if by_name else row["label"],
            source=source,
            group=group,
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


def _pair_records(rows: Iterable[dict], source: str, seed: int, text: str, slot: str,
                  options: list[str], answer) -> list[Record]:
    """Tasks over two texts: one is the text, the other goes inside the instruction. The instruction
    must carry it, so the template is picked here (seeded) instead of during augmentation."""
    rng = random.Random(seed)
    return [
        Record(
            instruction=rng.choice(INSTRUCTIONS[source]).format(**{slot: row[slot].strip()}),
            text=row[text],
            options=list(options),
            answer_idx=answer(row),
            source=source,
            group="train",
        )
        for row in rows
        if answer(row) >= 0
    ]


def convert_mnli(rows: Iterable[dict], seed: int) -> list[Record]:
    """Premise as text, hypothesis inside the instruction. Rows without a gold label (-1) are dropped."""
    return _pair_records(rows, "mnli", seed, "premise", "hypothesis", MNLI_OPTIONS, lambda r: r["label"])


def convert_qqp(rows: Iterable[dict], seed: int) -> list[Record]:
    return _pair_records(rows, "qqp", seed, "question1", "question2", QQP_OPTIONS, lambda r: r["label"])


def convert_qnli(rows: Iterable[dict], seed: int) -> list[Record]:
    return _pair_records(rows, "qnli", seed, "sentence", "question", QNLI_OPTIONS, lambda r: r["label"])


def convert_boolq(rows: Iterable[dict], seed: int) -> list[Record]:
    return _pair_records(rows, "boolq", seed, "passage", "question", BOOLQ_OPTIONS, lambda r: int(r["answer"]))


def convert_cola(rows: Iterable[dict]) -> list[Record]:
    return _records(rows, "cola", COLA_LABELS, text_field="sentence")


def convert_yahoo_answers(rows: Iterable[dict]) -> list[Record]:
    """Question title and body as the text; the best answer is left out."""
    rows = ({"text": f"{r['question_title']} {r['question_content']}".strip(), "label": r["topic"]} for r in rows)
    return _records(rows, "yahoo_answers", YAHOO_LABELS, group="heldout_labels")


def convert_massive(rows: Iterable[dict], labels: list[str]) -> list[Record]:
    """Assistant intents with hand-written names: "iot_hue_lightoff" becomes "turn the lights off"."""
    return _records(rows, "massive", labels, by_name=True, group="heldout_labels", names=MASSIVE_NAMES)


def convert_tweet_irony(rows: Iterable[dict]) -> list[Record]:
    return _records(rows, "tweet_irony", IRONY_LABELS, group="heldout_task")


def convert_sms_spam(rows: Iterable[dict]) -> list[Record]:
    rows = ({"text": r["sms"].strip(), "label": r["label"]} for r in rows)
    return _records(rows, "sms_spam", SPAM_LABELS, group="heldout_task")


def convert_tweet_hate(rows: Iterable[dict]) -> list[Record]:
    return _records(rows, "tweet_hate", HATE_LABELS, group="heldout_task")


def convert_subjectivity(rows: Iterable[dict]) -> list[Record]:
    return _records(rows, "subjectivity", SUBJECTIVITY_LABELS, group="heldout_task")
