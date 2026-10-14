"""Baseline: GLiClass, packed options like this project, on the same evaluation sets.

The gliclass package brings its own transformers version, so it runs in a throwaway environment:

    PYTHONPATH=src uv run --no-project --python 3.12 --with gliclass --with datasets --with pyyaml \\
        --with scikit-learn --with scipy --with matplotlib python -m undecided.baseline_gliclass configs/poc-warm.yaml

knowledgator/gliclass-modern-base-v2.0-init scores every option with its own sigmoid. The scores of
a record's options are normalised to sum to 1, which keeps accuracy and AUC and makes calibration
error comparable. The canonical instruction is passed as GLiClass's prompt. MNLI is skipped.
Writes results/gliclass/metrics.json and results/gliclass/latency.json.
"""

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import torch
import yaml
from gliclass import GLiClassModel, ZeroShotClassificationPipeline
from transformers import AutoTokenizer

from undecided.data import Record
from undecided.evaluate import score
from undecided.latency import OPTION_COUNTS, TEXT
from undecided.results import group_summary
from undecided.run import load_eval_sets
from undecided.tasks import INSTRUCTIONS, MASSIVE_NAMES

MODEL = "knowledgator/gliclass-modern-base-v2.0-init"
NAME = "gliclass"


def make_pipeline(device: str):
    return ZeroShotClassificationPipeline(GLiClassModel.from_pretrained(MODEL), AutoTokenizer.from_pretrained(MODEL),
                                          classification_type="multi-label", device=device, max_classes=100,
                                          progress_bar=False)


def predict(pipe, records: list[Record], instruction: str | None, batch_size: int = 16) -> list[list[float]]:
    """Every record of a dataset has the same label set, so all go through with one label list;
    scores are then mapped back to each record's own option order."""
    labels = sorted(records[0].options)
    results = pipe([r.text for r in records], labels, threshold=0.0, batch_size=batch_size,
                   prompt=instruction or None)
    probs = []
    for record, result in zip(records, results):
        by_name = {item["label"]: item["score"] for item in result}
        raw = [by_name.get(option, 0.0) for option in record.options]
        total = sum(raw) or 1.0
        probs.append([p / total for p in raw])
    return probs


def time_requests(pipe, repeats: int = 100, warmup: int = 10) -> list[dict]:
    rows = []
    for n in OPTION_COUNTS:
        options = list(MASSIVE_NAMES.values())[:n]
        record = Record("", TEXT, options, 0, "latency", "inference")
        for _ in range(warmup):
            predict(pipe, [record], None, batch_size=1)
        times = []
        for _ in range(repeats):
            start = time.perf_counter()
            predict(pipe, [record], None, batch_size=1)
            times.append((time.perf_counter() - start) * 1000)
        times.sort()
        rows.append({"options": n, "requests": repeats, "p50_ms": round(statistics.median(times), 1),
                     "p95_ms": round(times[int(0.95 * len(times)) - 1], 1)})
        print(f"{n:3d} options  p50 {rows[-1]['p50_ms']:7.1f} ms", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config")
    args = parser.parse_args()

    raw = yaml.safe_load(Path(args.config).read_text())
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    pipe = make_pipeline(device)
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
                    **score(records, predict(pipe, records, INSTRUCTIONS[records[0].source][0]))}
        partial.write_text(json.dumps(done, indent=2) + "\n")
        print(f"{ds}: accuracy {done[ds]['accuracy']:.3f}  ({time.time() - start:.0f} s)", flush=True)

    datasets = {ds: done[ds] for ds in eval_sets}
    metrics = {"run": NAME, "model": MODEL, "device": device,
               "machine": {"platform": platform.platform(), "torch": torch.__version__},
               "datasets": datasets, "groups": group_summary(datasets)}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    partial.unlink(missing_ok=True)

    latency = {"model": MODEL, "machine": platform.platform(), "torch": torch.__version__,
               "text_words": len(TEXT.split()), "devices": {device: time_requests(pipe)}}
    if device != "cpu":
        latency["devices"]["cpu"] = time_requests(make_pipeline("cpu"))
    (out / "latency.json").write_text(json.dumps(latency, indent=2) + "\n")


if __name__ == "__main__":
    main()
