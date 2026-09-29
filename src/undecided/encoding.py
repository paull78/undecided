"""Packed input format: [CLS] instruction [SEP] <opt> o1 <opt> o2 ... [SEP] text [SEP]."""

import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from undecided.data import Record

OPT_TOKEN = "<opt>"


def build_tokenizer(name: str) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(name)
    tokenizer.add_special_tokens({"additional_special_tokens": [OPT_TOKEN]})
    return tokenizer


def encode(tokenizer: PreTrainedTokenizerBase, record: Record, max_length: int) -> dict:
    sep = tokenizer.sep_token
    head = f"{record.instruction}{sep}" if record.instruction else ""
    head += "".join(f"{OPT_TOKEN} {option}" for option in record.options)

    head_ids = tokenizer(head, add_special_tokens=False)["input_ids"]
    text_ids = tokenizer(record.text, add_special_tokens=False)["input_ids"]
    text_budget = max_length - len(head_ids) - 3  # [CLS] and two [SEP]
    if text_budget < 1:
        raise ValueError(
            f"instruction and {len(record.options)} options take {len(head_ids)} tokens, "
            f"leaving no room for the text within max_length={max_length}"
        )
    text_ids = text_ids[:text_budget]

    input_ids = [
        tokenizer.cls_token_id,
        *head_ids,
        tokenizer.sep_token_id,
        *text_ids,
        tokenizer.sep_token_id,
    ]
    return {"input_ids": input_ids, "label": record.answer_idx}


def collate(examples: list[dict], pad_token_id: int) -> dict:
    """Pad a list of encoded examples to the longest one in the batch."""
    longest = max(len(e["input_ids"]) for e in examples)
    input_ids = torch.full((len(examples), longest), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(examples), longest), dtype=torch.long)
    for row, example in enumerate(examples):
        ids = example["input_ids"]
        input_ids[row, : len(ids)] = torch.tensor(ids)
        attention_mask[row, : len(ids)] = 1
    labels = torch.tensor([e["label"] for e in examples])
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
