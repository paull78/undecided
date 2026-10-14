"""Run one experiment from a YAML config.

    uv run python -m undecided.run configs/tracer.yaml
    uv run python -m undecided.run configs/poc.yaml --eval-only   # re-evaluate runs/poc, no training
    uv run python -m undecided.run configs/poc.yaml --resume      # continue after an interruption

Checkpoints go to runs/<name>/ (gitignored); config, metrics and plots go to results/<name>/.
Training saves its full state to runs/<name>/train-state.pt every 500 steps, the model is saved
before evaluation starts, and each dataset's metrics go to results/<name>/metrics.partial.json as
soon as they exist, so an interrupted run loses at most a few minutes.
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
from undecided.results import group_summary
from undecided.sources import load_records
from undecided.train import TrainConfig, loss_by_source, train

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


MPS_MEMORY_GB = float(os.environ.get("UNDECIDED_MPS_GB", "24"))


def cap_device_memory(device: torch.device) -> None:
    """Cap MPS memory, so a runaway run fails on its own instead of pushing the machine into swap."""
    if device.type == "mps":
        fraction = min(1.0, MPS_MEMORY_GB * 1e9 / torch.mps.recommended_max_memory())
        torch.mps.set_per_process_memory_fraction(fraction)


def build_scorer(backbone: str, tokenizer) -> OptionScorer:
    return OptionScorer(AutoModel.from_pretrained(backbone), tokenizer.convert_tokens_to_ids(OPT_TOKEN))


def load_eval_sets(raw: dict, seed: int) -> dict[str, list]:
    """Fixed evaluation samples, options shuffled with the run's seed so the answer's position varies."""
    rng = random.Random(seed)
    return {
        ds: [shuffle_options(r, rng) for r in load_records(ds, "eval", n, seed)]
        for ds, n in raw["data"]["eval"].items()
    }


def evaluate_all(scorer, tokenizer, eval_sets: dict[str, list], max_length: int, device: torch.device,
                 partial: Path | None = None) -> dict:
    """Metrics per dataset and per group. With a partial file, datasets already scored on a sample of
    the same size are reused, and each new dataset is written to it as soon as it is scored."""
    done = json.loads(partial.read_text()) if partial and partial.exists() else {}
    datasets = {}
    for ds, records in eval_sets.items():
        if ds in done and done[ds].get("n") == len(records):
            datasets[ds] = done[ds]
            print(f"{ds}: accuracy {datasets[ds]['accuracy']:.3f}  (from {partial.name})", flush=True)
            continue
        start = time.time()
        datasets[ds] = {"group": records[0].group, "options": len(records[0].options),
                        **evaluate(scorer, tokenizer, records, max_length, 32, device)}
        print(f"{ds}: accuracy {datasets[ds]['accuracy']:.3f}  ({time.time() - start:.0f} s)", flush=True)
        if partial:
            done[ds] = datasets[ds]
            partial.parent.mkdir(parents=True, exist_ok=True)
            partial.write_text(json.dumps(done, indent=2) + "\n")
    return {"datasets": datasets, "groups": group_summary(datasets)}


def eval_only(raw: dict, seed: int) -> None:
    """Re-evaluate a saved checkpoint on the config's eval sets and update its metrics.json in place."""
    name = raw["name"]
    model = Undecided.from_pretrained(Path("runs") / name)
    cap_device_memory(model.device)
    print(f"evaluate {name} on {model.device}")
    results = Path("results") / name
    metrics = json.loads((results / "metrics.json").read_text())
    partial = results / "metrics.eval-only.partial.json"
    metrics.update(evaluate_all(model.scorer, model.tokenizer, load_eval_sets(raw, seed), model.max_length,
                                model.device, partial=partial))
    (results / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    partial.unlink(missing_ok=True)
    config = yaml.safe_load((results / "config.yaml").read_text())
    config["data"]["eval"] = raw["data"]["eval"]
    (results / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))


def main(config_path: str, only_eval: bool = False, resume: bool = False) -> None:
    raw = yaml.safe_load(Path(config_path).read_text())
    name, seed = raw["name"], raw.get("seed", 0)
    if only_eval:
        return eval_only(raw, seed)
    train_cfg = TrainConfig(seed=seed, **raw["train"])
    device = pick_device()
    cap_device_memory(device)
    print(f"run {name} on {device}")
    run_dir, results = Path("runs") / name, Path("results") / name
    finished_training = run_dir / "train.json"  # written with the final model, before evaluation

    if resume and finished_training.exists():
        model = Undecided.from_pretrained(run_dir, device)
        tokenizer, scorer = model.tokenizer, model.scorer
        print(f"training already finished, evaluating {run_dir}")
    elif raw.get("init_from"):  # continue training a saved checkpoint instead of the pretrained backbone
        start_from = Undecided.from_pretrained(raw["init_from"], device)
        tokenizer, scorer = start_from.tokenizer, start_from.scorer
        print(f"starting from {raw['init_from']}")
    else:
        tokenizer = build_tokenizer(raw["backbone"])
        scorer = build_scorer(raw["backbone"], tokenizer)

    train_records = [r for ds, n in raw["data"]["train"].items() for r in load_records(ds, "train", n, seed)]
    if raw["data"].get("eval_on_train"):
        eval_sets = {"train_set": train_records}
    else:
        eval_sets = load_eval_sets(raw, seed)

    window, start, peak_gb, steps_this_session = [], time.time(), 0.0, 0

    def on_step(step: int, loss: float, examples: list) -> None:
        nonlocal peak_gb, window, steps_this_session
        steps_this_session += 1
        window += examples
        if step % 20 == 0:
            peak_gb = max(peak_gb, device_memory_gb(device))
            recent = [l for _, l, _ in window[-20 * train_cfg.batch_size:]]
            rate = steps_this_session * train_cfg.batch_size / (time.time() - start)
            print(f"step {step:5d}  loss {sum(recent) / len(recent):.4f}  {rate:.1f} examples/s  "
                  f"{memory_report(device)}", flush=True)
        if step % 200 == 0:
            # Per-dataset loss against uniform guessing: the average can hide which tasks learn.
            per_source = loss_by_source(window)
            print("  by source (loss / chance): " + "  ".join(
                f"{s} {v['loss']:.2f}/{v['chance']:.2f}" for s, v in sorted(per_source.items())), flush=True)
            window = []
            if device.type == "mps":
                torch.mps.empty_cache()  # batch shapes vary, so the allocator cache would keep growing

    if resume and finished_training.exists():
        train_stats = json.loads(finished_training.read_text())
        losses = train_stats.pop("losses")
    else:
        state = run_dir / "train-state.pt"
        run_dir.mkdir(parents=True, exist_ok=True)
        if resume and state.exists():
            print(f"resuming from {state}")
        info = {}
        losses = train(scorer, tokenizer, train_records, train_cfg, device, on_step=on_step,
                       checkpoint=state, checkpoint_every=raw.get("checkpoint_every", 500), resume=resume,
                       info=info)
        seconds = time.time() - start
        train_stats = {
            "examples": len(train_records),
            "steps": len(losses),
            "seconds": round(seconds, 1),  # this session only, if resumed
            "examples_per_second": round(steps_this_session * train_cfg.batch_size / seconds, 2),
            "final_loss": round(sum(losses[-20:]) / len(losses[-20:]), 4),
            "peak_memory_gb": round(peak_gb, 2),
            "trainable_parameters": info.get("trainable_parameters"),
            "resumed": resume and steps_this_session < len(losses),
        }
        # Save the model before evaluating, so an interruption during evaluation loses nothing.
        Undecided(scorer, tokenizer, train_cfg.max_length, device).save_pretrained(run_dir)
        finished_training.write_text(json.dumps({**train_stats, "losses": losses}) + "\n")
        state.unlink(missing_ok=True)

    metrics = {
        "run": name,
        "device": str(device),
        "machine": {"platform": platform.platform(), "processor": platform.processor(),
                    "torch": torch.__version__},
        "train": train_stats,
        **evaluate_all(scorer, tokenizer, eval_sets, train_cfg.max_length, device,
                       partial=results / "metrics.partial.json"),
    }
    print(json.dumps(metrics["datasets"], indent=2))

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
    (results / "metrics.partial.json").unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config")
    parser.add_argument("--eval-only", action="store_true", help="evaluate runs/<name> without training")
    parser.add_argument("--resume", action="store_true", help="continue an interrupted run")
    args = parser.parse_args()
    main(args.config, args.eval_only, args.resume)
