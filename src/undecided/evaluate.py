"""Evaluation on fixed record sets."""

from collections import defaultdict

import torch
from sklearn.metrics import f1_score

from undecided.data import Record
from undecided.encoding import collate, encode
from undecided.model import OptionScorer


@torch.no_grad()
def predict(scorer: OptionScorer, tokenizer, records: list[Record], max_length: int, batch_size: int,
            device: torch.device) -> list[list[float]]:
    """Probabilities per record, one per option, in the record's option order."""
    scorer.eval()
    probs = []
    for start in range(0, len(records), batch_size):
        chunk = records[start : start + batch_size]
        batch = collate([encode(tokenizer, r, max_length) for r in chunk], tokenizer.pad_token_id)
        logits = scorer(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        for record, row in zip(chunk, logits.softmax(-1).cpu()):
            probs.append(row[: len(record.options)].tolist())
    return probs


def score(records: list[Record], probs: list[list[float]]) -> dict:
    """Accuracy and macro-F1. Labels are compared by name, since option order differs per record.
    accuracy_by_position is keyed by the correct option's 1-based position, to expose position bias."""
    gold = [r.options[r.answer_idx] for r in records]
    predicted = [r.options[max(range(len(p)), key=p.__getitem__)] for r, p in zip(records, probs)]
    hits_by_position = defaultdict(list)
    for record, g, p in zip(records, gold, predicted):
        hits_by_position[record.answer_idx + 1].append(g == p)
    return {
        "n": len(records),
        "accuracy": sum(g == p for g, p in zip(gold, predicted)) / len(records),
        "macro_f1": f1_score(gold, predicted, average="macro", zero_division=0),
        "accuracy_by_position": {
            str(position): {"n": len(hits), "accuracy": sum(hits) / len(hits)}
            for position, hits in sorted(hits_by_position.items())
        },
    }


def evaluate(scorer: OptionScorer, tokenizer, records: list[Record], max_length: int, batch_size: int,
             device: torch.device) -> dict:
    return score(records, predict(scorer, tokenizer, records, max_length, batch_size, device))
