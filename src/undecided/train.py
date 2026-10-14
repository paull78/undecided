"""Plain PyTorch training loop."""

import math
import os
import random
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from undecided.augment import augment
from undecided.data import Record
from undecided.encoding import collate, encode
from undecided.model import OptionScorer


def build_optimizer(scorer: OptionScorer, head_lr: float, encoder_lr: float, weight_decay: float,
                    lora_lr: float | None = None):
    """Two AdamW groups: the new head learns fast; the encoder and the <opt> vector learn slowly.

    <opt> feeds every attention layer, so moving it at the head's rate collapsed the encoder.
    Frozen parameters (requires_grad False) are left out. LoRA adapters, when present, get lora_lr.
    """
    lora = [p for n, p in scorer.encoder.named_parameters() if "lora_" in n and p.requires_grad]
    lora_ids = {id(p) for p in lora}
    groups = [
        {"params": [p for p in scorer.head.parameters() if p.requires_grad], "lr": head_lr},
        {"params": [p for p in [scorer.opt_embedding, *scorer.encoder.parameters()]
                    if p.requires_grad and id(p) not in lora_ids], "lr": encoder_lr},
        {"params": lora, "lr": lora_lr or encoder_lr},
    ]
    return torch.optim.AdamW([g for g in groups if g["params"]], weight_decay=weight_decay)


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
    recast_rate: float = 0.0  # share of examples asked as a yes/no check on one label
    freeze_encoder: bool = False  # train the head only; the encoder and the <opt> vector stay fixed
    lora_rank: int = 0  # > 0: freeze the encoder and train LoRA adapters on its attention projections
    lora_lr: float = 2e-4
    seed: int = 0
    max_steps: int | None = None  # stop early, for short checks


def loss_by_source(examples: list[tuple[str, float, int]]) -> dict[str, dict]:
    """Mean loss per source next to the loss of uniform guessing, ln(number of options).
    examples are (source, loss, number of options), as passed to on_step."""
    grouped = defaultdict(list)
    for source, loss, n_options in examples:
        grouped[source].append((loss, math.log(n_options)))
    return {
        source: {"n": len(pairs), "loss": sum(l for l, _ in pairs) / len(pairs),
                 "chance": sum(c for _, c in pairs) / len(pairs)}
        for source, pairs in grouped.items()
    }


def _save_state(path: Path, state: dict) -> None:
    """Write to a temporary file and rename, so an interruption mid-write never leaves a broken file."""
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def train(scorer: OptionScorer, tokenizer, records: list[Record], cfg: TrainConfig, device: torch.device,
          on_step: Callable[[int, float, list[tuple[str, float, int]]], None] | None = None,
          checkpoint: Path | None = None, checkpoint_every: int = 500, resume: bool = False,
          info: dict | None = None) -> list[float]:
    """Train and return the loss of every step.

    on_step gets the step, the batch loss and (source, loss, number of options) per example. With a
    checkpoint path, the full training state is saved there every checkpoint_every steps; resume=True
    continues from it with the same batches and augmentation as an uninterrupted run. info, if given,
    receives facts about the run (the number of trainable parameters).
    """
    rng = random.Random(cfg.seed)
    torch.manual_seed(cfg.seed)
    scorer.to(device).train()
    if cfg.freeze_encoder:
        for param in [scorer.opt_embedding, *scorer.encoder.parameters()]:
            param.requires_grad_(False)
    if cfg.lora_rank:
        from peft import LoraConfig, get_peft_model  # only needed for this ablation

        scorer.encoder = get_peft_model(scorer.encoder, LoraConfig(
            r=cfg.lora_rank, lora_alpha=2 * cfg.lora_rank, lora_dropout=0.0,
            target_modules=r".*\.attn\.(Wqkv|Wo)"))  # attention only; the MLP's output is also called Wo
        scorer.to(device)

    steps_per_epoch = math.ceil(len(records) / cfg.batch_size)
    total_steps = steps_per_epoch * cfg.epochs
    if cfg.max_steps:
        total_steps = min(total_steps, cfg.max_steps)
    warmup_steps = max(1, int(total_steps * cfg.warmup_ratio))
    optimizer = build_optimizer(scorer, cfg.head_lr, cfg.encoder_lr, cfg.weight_decay,
                                lora_lr=cfg.lora_lr if cfg.lora_rank else None)
    if info is not None:
        info["trainable_parameters"] = sum(p.numel() for g in optimizer.param_groups for p in g["params"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min((step + 1) / warmup_steps, max(0.0, (total_steps - step) / (total_steps - warmup_steps))),
    )

    step, start_epoch, skip_batches, losses = 0, 0, 0, []
    if resume and checkpoint and checkpoint.exists():
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        scorer.load_state_dict(state["scorer"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        step, start_epoch, skip_batches, losses = state["step"], state["epoch"], state["batch"], state["losses"]
        rng.setstate(state["epoch_rng"])
        torch.set_rng_state(state["torch_rng"])
        if device.type == "mps" and state.get("mps_rng") is not None:
            torch.mps.set_rng_state(state["mps_rng"])

    for epoch in range(start_epoch, cfg.epochs):
        epoch_rng = rng.getstate()  # batches and augmentation of this epoch all follow from it
        for batch_idx, chunk in enumerate(length_grouped_batches(records, cfg.batch_size, rng)):
            if cfg.augment:
                chunk = [augment(r, rng, cfg.max_options, cfg.recast_rate) for r in chunk]
            if epoch == start_epoch and batch_idx < skip_batches:
                continue  # already trained before the interruption; augmenting it kept rng in step
            batch = collate([encode(tokenizer, r, cfg.max_length) for r in chunk], tokenizer.pad_token_id)
            logits = scorer(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            example_losses = F.cross_entropy(logits, batch["labels"].to(device), reduction="none")
            loss = example_losses.mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(scorer.parameters(), cfg.grad_clip)
            optimizer.step()
            scheduler.step()
            step += 1
            losses.append(loss.item())
            if checkpoint and step % checkpoint_every == 0 and step < total_steps:
                _save_state(checkpoint, {
                    "scorer": scorer.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "step": step, "epoch": epoch, "batch": batch_idx + 1,
                    "losses": losses, "epoch_rng": epoch_rng, "torch_rng": torch.get_rng_state(),
                    "mps_rng": torch.mps.get_rng_state() if device.type == "mps" else None,
                })
            if on_step:
                examples = [(r.source, l, len(r.options)) for r, l in zip(chunk, example_losses.tolist())]
                on_step(step, losses[-1], examples)
            if step == total_steps:
                return _finish(scorer, losses)
    return _finish(scorer, losses)


def _finish(scorer: OptionScorer, losses: list[float]) -> list[float]:
    """Merge LoRA adapters into the encoder weights, so the result saves and loads like any model."""
    if hasattr(scorer.encoder, "merge_and_unload"):
        scorer.encoder = scorer.encoder.merge_and_unload()
    return losses
