"""Evaluation on fixed record sets."""

import torch

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


def evaluate(scorer: OptionScorer, tokenizer, records: list[Record], max_length: int, batch_size: int,
             device: torch.device) -> dict:
    probs = predict(scorer, tokenizer, records, max_length, batch_size, device)
    correct = sum(max(range(len(p)), key=p.__getitem__) == r.answer_idx for p, r in zip(probs, records))
    return {"n": len(records), "accuracy": correct / len(records)}
