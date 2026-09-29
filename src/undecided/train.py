"""Plain PyTorch training loop."""

import math
import random
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from undecided.augment import augment
from undecided.data import Record
from undecided.encoding import collate, encode
from undecided.model import OptionScorer


def build_optimizer(scorer: OptionScorer, head_lr: float, encoder_lr: float, weight_decay: float):
    """Two AdamW groups: the new head learns fast; the encoder and the <opt> vector learn slowly.

    <opt> feeds every attention layer, so moving it at the head's rate collapsed the encoder.
    """
    return torch.optim.AdamW(
        [
            {"params": list(scorer.head.parameters()), "lr": head_lr},
            {"params": [scorer.opt_embedding, *scorer.encoder.parameters()], "lr": encoder_lr},
        ],
        weight_decay=weight_decay,
    )


POOL_BATCHES = 50  # sort by length within pools of this many batches, so order stays random


def length_grouped_batches(records: list[Record], batch_size: int, rng: random.Random) -> list[list[Record]]:
    """Shuffle, sort by text length inside large pools, cut into batches, shuffle the batch order."""
    order = list(records)
    rng.shuffle(order)
    pool_size = batch_size * POOL_BATCHES
    batches = []
    for start in range(0, len(order), pool_size):
        pool = sorted(order[start : start + pool_size], key=lambda r: len(r.text))
        batches += [pool[i : i + batch_size] for i in range(0, len(pool), batch_size)]
    rng.shuffle(batches)
    return batches


@dataclass
class TrainConfig:
    epochs: int
    batch_size: int
    head_lr: float = 1e-3
    encoder_lr: float = 3e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    grad_clip: float = 1.0
    max_length: int = 512
    max_options: int = 16
    augment: bool = True
    seed: int = 0
    max_steps: int | None = None  # stop early, for short checks


def train(scorer: OptionScorer, tokenizer, records: list[Record], cfg: TrainConfig,
          device: torch.device, on_step: Callable[[int, float], None] | None = None) -> None:
    rng = random.Random(cfg.seed)
    torch.manual_seed(cfg.seed)
    scorer.to(device).train()

    steps_per_epoch = math.ceil(len(records) / cfg.batch_size)
    total_steps = steps_per_epoch * cfg.epochs
    if cfg.max_steps:
        total_steps = min(total_steps, cfg.max_steps)
    warmup_steps = max(1, int(total_steps * cfg.warmup_ratio))
    optimizer = build_optimizer(scorer, cfg.head_lr, cfg.encoder_lr, cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min((step + 1) / warmup_steps, max(0.0, (total_steps - step) / (total_steps - warmup_steps))),
    )

    step = 0
    for _ in range(cfg.epochs):
        for chunk in length_grouped_batches(records, cfg.batch_size, rng):
            if cfg.augment:
                chunk = [augment(r, rng, cfg.max_options) for r in chunk]
            batch = collate([encode(tokenizer, r, cfg.max_length) for r in chunk], tokenizer.pad_token_id)
            logits = scorer(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            loss = F.cross_entropy(logits, batch["labels"].to(device))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(scorer.parameters(), cfg.grad_clip)
            optimizer.step()
            scheduler.step()
            step += 1
            if on_step:
                on_step(step, loss.item())
            if step == total_steps:
                return
