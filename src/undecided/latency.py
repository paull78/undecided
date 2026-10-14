"""Time single requests through classify(): one text, n options, batch size 1.

    uv run python -m undecided.latency runs/poc            # writes results/poc/latency.json

Each option count is warmed up first, then timed over many requests; p50 and p95 are reported in
milliseconds. The options are MASSIVE intent names, so they look like real ones.
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

from undecided.model import Undecided
from undecided.tasks import MASSIVE_NAMES

TEXT = "Can you turn the lights in the kitchen down a bit and remind me to call mum at six tonight?"
OPTION_COUNTS = (2, 10, 50)


def _sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def time_requests(model: Undecided, n_options: int, repeats: int, warmup: int = 10) -> dict:
    options = list(MASSIVE_NAMES.values())[:n_options]
    for _ in range(warmup):
        model.classify(TEXT, options)
    _sync(model.device)
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        model.classify(TEXT, options)  # returns Python floats, so the device has finished
        times.append((time.perf_counter() - start) * 1000)
    times.sort()
    return {"options": n_options, "requests": repeats,
            "p50_ms": round(statistics.median(times), 1), "p95_ms": round(times[int(0.95 * len(times)) - 1], 1)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint", help="runs/<name>")
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--devices", nargs="+", default=None, help="default: the picked device, then cpu")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    devices = args.devices or list(dict.fromkeys([Undecided.from_pretrained(checkpoint).device.type, "cpu"]))
    report = {"checkpoint": str(checkpoint), "machine": platform.platform(), "torch": torch.__version__,
              "text_words": len(TEXT.split()), "devices": {}}
    for device in devices:
        model = Undecided.from_pretrained(checkpoint, torch.device(device))
        report["devices"][device] = [time_requests(model, n, args.repeats) for n in OPTION_COUNTS]
        for row in report["devices"][device]:
            print(f"{device:4s} {row['options']:3d} options  p50 {row['p50_ms']:7.1f} ms  p95 {row['p95_ms']:7.1f} ms")
        del model

    out = Path("results") / checkpoint.name / "latency.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
