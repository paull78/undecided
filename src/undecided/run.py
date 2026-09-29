"""Run one experiment from a YAML config.

    uv run python -m undecided.run configs/tracer.yaml

Checkpoints go to runs/<name>/ (gitignored); config, metrics and plots go to results/<name>/.
"""

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # must be set before torch is imported

import argparse
import json
import platform
import subprocess
import random
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib
import torch
import yaml
from transformers import AutoModel

from undecided.augment import shuffle_options
from undecided.device import pick_device
from undecided.encoding import OPT_TOKEN, build_tokenizer
from undecided.evaluate import evaluate
from undecided.model import OptionScorer, Undecided
from undecided.sources import load_records
from undecided.train import TrainConfig, train

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

def device_memory_gb(device: torch.device) -> float:
    """Memory the device driver holds for this process: live tensors plus the allocator cache."""
    if device.type == "mps":
        return torch.mps.driver_allocated_memory() / 1e9
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / 1e9
    return 0.0


def memory_report(device: torch.device) -> str:
    report = f"driver {device_memory_gb(device):5.1f} GB"
    if device.type == "mps":
        report += f"  tensors {torch.mps.current_allocated_memory() / 1e9:5.1f} GB"
    free = subprocess.run(["memory_pressure"], capture_output=True, text=True).stdout.strip().splitlines()
    if free:
        report += f"  system free {free[-1].rsplit(' ', 1)[-1]}"
    return report


def build_scorer(backbone: str, tokenizer) -> OptionScorer:
    return OptionScorer(AutoModel.from_pretrained(backbone), tokenizer.convert_tokens_to_ids(OPT_TOKEN))


def main(config_path: str) -> None:
    raw = yaml.safe_load(Path(config_path).read_text())
    name, seed = raw["name"], raw.get("seed", 0)
    train_cfg = TrainConfig(seed=seed, **raw["train"])
    device = pick_device()
    print(f"run {name} on {device}")

    tokenizer = build_tokenizer(raw["backbone"])
    scorer = build_scorer(raw["backbone"], tokenizer)

    train_records = [r for ds, n in raw["data"]["train"].items() for r in load_records(ds, "train", n, seed)]
    if raw["data"].get("eval_on_train"):
        eval_sets = {"train_set": train_records}
    else:
        rng = random.Random(seed)
        eval_sets = {
            ds: [shuffle_options(r, rng) for r in load_records(ds, "eval", n, seed)]
            for ds, n in raw["data"]["eval"].items()
        }

    losses, start, peak_gb = [], time.time(), 0.0

    def on_step(step: int, loss: float) -> None:
        nonlocal peak_gb
        losses.append(loss)
        if step % 20 == 0:
            peak_gb = max(peak_gb, device_memory_gb(device))
            recent = sum(losses[-20:]) / len(losses[-20:])
            rate = step * train_cfg.batch_size / (time.time() - start)
            print(f"step {step:5d}  loss {recent:.4f}  {rate:.1f} examples/s  {memory_report(device)}", flush=True)
        if step % 200 == 0 and device.type == "mps":
            torch.mps.empty_cache()  # batch shapes vary, so the allocator cache would keep growing

    train(scorer, tokenizer, train_records, train_cfg, device, on_step=on_step)
    seconds = time.time() - start

    metrics = {
        "run": name,
        "device": str(device),
        "machine": {"platform": platform.platform(), "processor": platform.processor(),
                    "torch": torch.__version__},
        "train": {
            "examples": len(train_records),
            "steps": len(losses),
            "seconds": round(seconds, 1),
            "examples_per_second": round(len(losses) * train_cfg.batch_size / seconds, 2),
            "final_loss": round(sum(losses[-20:]) / len(losses[-20:]), 4),
            "peak_memory_gb": round(peak_gb, 2),
        },
        "datasets": {
            ds: evaluate(scorer, tokenizer, records, train_cfg.max_length, 32, device)
            for ds, records in eval_sets.items()
        },
    }
    print(json.dumps(metrics["datasets"], indent=2))

    results = Path("results") / name
    (results / "plots").mkdir(parents=True, exist_ok=True)
    (results / "config.yaml").write_text(
        yaml.safe_dump({**raw, "train": asdict(train_cfg)}, sort_keys=False)
    )
    (results / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    plt.figure(figsize=(6, 3.5))
    plt.plot(losses, linewidth=0.8)
    plt.xlabel("step")
    plt.ylabel("training loss")
    plt.title(name)
    plt.tight_layout()
    plt.savefig(results / "plots" / "loss.png", dpi=120)

    Undecided(scorer, tokenizer, train_cfg.max_length, device).save_pretrained(Path("runs") / name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config")
    main(parser.parse_args().config)
