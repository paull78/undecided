"""Baseline: an NLI zero-shot classifier on the same evaluation sets, one forward pass per option.

    uv run python -m undecided.baselines configs/poc-warm.yaml

MoritzLaurer/ModernBERT-base-zeroshot-v2.0 shares this project's backbone. Each option becomes a
hypothesis ("This example is billing.") paired with the text; the entailment logits of a record's
options go through a softmax, as in the Hugging Face zero-shot pipeline with one label per text.
The instruction is not used. MNLI is skipped, since NLI is what the baseline was trained on.

Writes results/zeroshot-v2/metrics.json in the same shape as a run, so the results command and the
chart read it unchanged, plus results/zeroshot-v2/latency.json.
"""

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import json
import math
import platform
import statistics
import time
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from undecided.data import Record
from undecided.device import pick_device
from undecided.evaluate import score
from undecided.latency import OPTION_COUNTS, TEXT
from undecided.results import group_summary
from undecided.run import cap_device_memory, load_eval_sets
from undecided.tasks import MASSIVE_NAMES

MODEL = "MoritzLaurer/ModernBERT-base-zeroshot-v2.0"
NAME = "zeroshot-v2"
TEMPLATE = "This example is {}."
ENTAILMENT = 0  # id2label {0: entailment, 1: not_entailment}
PAIR_BATCH = 64


def nli_pairs(record: Record, template: str) -> list[tuple[str, str]]:
    return [(record.text, template.format(option)) for option in record.options]


def options_softmax(logits: list[float]) -> list[float]:
    top = max(logits)
    e = [math.exp(v - top) for v in logits]
    return [v / sum(e) for v in e]


@torch.no_grad()
def entailment_logits(model, tokenizer, pairs: list[tuple[str, str]], device: torch.device) -> list[float]:
    out = []
    for start in range(0, len(pairs), PAIR_BATCH):
        chunk = pairs[start : start + PAIR_BATCH]
        batch = tokenizer([p for p, _ in chunk], [h for _, h in chunk], truncation="only_first", max_length=512,
                          padding=True, return_tensors="pt").to(device)
        out += model(**batch).logits[:, ENTAILMENT].float().cpu().tolist()
    return out


def predict_nli(model, tokenizer, records: list[Record], device: torch.device) -> list[list[float]]:
    pairs = [pair for r in records for pair in nli_pairs(r, TEMPLATE)]
    order = sorted(range(len(pairs)), key=lambda i: len(pairs[i][0]))  # length-sorted batches pad less
    logits = [0.0] * len(pairs)
    for i, value in zip(order, entailment_logits(model, tokenizer, [pairs[i] for i in order], device)):
        logits[i] = value
    probs, start = [], 0
    for r in records:
        probs.append(options_softmax(logits[start : start + len(r.options)]))
        start += len(r.options)
    return probs


def time_requests(model, tokenizer, device: torch.device, repeats: int = 100, warmup: int = 10) -> list[dict]:
    rows = []
    for n in OPTION_COUNTS:
        record = Record("", TEXT, list(MASSIVE_NAMES.values())[:n], 0, "latency", "inference")
        for _ in range(warmup):
            predict_nli(model, tokenizer, [record], device)
        times = []
        for _ in range(repeats):
            start = time.perf_counter()
            predict_nli(model, tokenizer, [record], device)  # returns floats, so the device has finished
            times.append((time.perf_counter() - start) * 1000)
        times.sort()
        rows.append({"options": n, "requests": repeats, "p50_ms": round(statistics.median(times), 1),
                     "p95_ms": round(times[int(0.95 * len(times)) - 1], 1)})
        print(f"{device.type} {n:3d} options  p50 {rows[-1]['p50_ms']:7.1f} ms", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", help="the run config whose evaluation sets to use")
    args = parser.parse_args()

    raw = yaml.safe_load(Path(args.config).read_text())
    device = pick_device()
    cap_device_memory(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    # The checkpoint is stored in bfloat16 and transformers keeps that by default; undecided runs in
    # float32, so the baseline does too (bfloat16 made it about 1.8 times slower on CPU).
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, dtype=torch.float32).to(device).eval()
    out = Path("results") / NAME
    out.mkdir(parents=True, exist_ok=True)
    partial = out / "metrics.partial.json"
    done = json.loads(partial.read_text()) if partial.exists() else {}

    eval_sets = {ds: rs for ds, rs in load_eval_sets(raw, raw.get("seed", 0)).items() if ds != "mnli"}
    for ds, records in eval_sets.items():
        if ds in done and done[ds].get("n") == len(records):
            continue
        start = time.time()
        done[ds] = {"group": records[0].group, "options": len(records[0].options),
                    **score(records, predict_nli(model, tokenizer, records, device))}
        partial.write_text(json.dumps(done, indent=2) + "\n")
        print(f"{ds}: accuracy {done[ds]['accuracy']:.3f}  ({time.time() - start:.0f} s)", flush=True)

    datasets = {ds: done[ds] for ds in eval_sets}
    metrics = {"run": NAME, "model": MODEL, "template": TEMPLATE, "device": str(device),
               "machine": {"platform": platform.platform(), "torch": torch.__version__},
               "datasets": datasets, "groups": group_summary(datasets)}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    partial.unlink(missing_ok=True)

    latency = {"model": MODEL, "machine": platform.platform(), "torch": torch.__version__,
               "text_words": len(TEXT.split()), "devices": {device.type: time_requests(model, tokenizer, device)}}
    if device.type != "cpu":
        cpu = torch.device("cpu")
        latency["devices"]["cpu"] = time_requests(model.to(cpu), tokenizer, cpu)
    (out / "latency.json").write_text(json.dumps(latency, indent=2) + "\n")


if __name__ == "__main__":
    main()
