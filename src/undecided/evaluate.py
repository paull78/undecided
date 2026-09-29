"""Evaluation on fixed record sets."""

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
    """Accuracy and macro-F1. Labels are compared by name, since option order differs per record."""
    gold = [r.options[r.answer_idx] for r in records]
    predicted = [r.options[max(range(len(p)), key=p.__getitem__)] for r, p in zip(records, probs)]
    return {
        "n": len(records),
        "accuracy": sum(g == p for g, p in zip(gold, predicted)) / len(records),
        "macro_f1": f1_score(gold, predicted, average="macro", zero_division=0),
    }


def evaluate(scorer: OptionScorer, tokenizer, records: list[Record], max_length: int, batch_size: int,
             device: torch.device) -> dict:
    return score(records, predict(scorer, tokenizer, records, max_length, batch_size, device))
