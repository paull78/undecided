"""Inference-only probes of a trained checkpoint, on the evaluation sets of its config.

    uv run python -m undecided.probe instructions configs/poc-warm.yaml
    uv run python -m undecided.probe calibration configs/poc-warm.yaml
    uv run python -m undecided.probe orders configs/poc-warm.yaml

instructions: does the model read the instruction? Scores every dataset with its canonical
instruction, with none, and with another dataset's canonical instruction.

orders: option-order averaging. Scores every example under 4 option orders, then averages each
option's probability across them.

calibration: temperature scaling. A global T is fitted on fresh samples of the training datasets
(rows used neither for training nor for evaluation) and applied to every dataset; a per-dataset T
is fitted on 200 labelled examples of each held-out dataset. Writes reliability diagrams too.

Results go to results/<name>/probe-<probe>.json; each finished dataset is saved as it completes.
"""

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import json
import random
from dataclasses import replace
from pathlib import Path

import matplotlib
import yaml

from undecided.augment import FIXED_INSTRUCTION_SOURCES, shuffle_options
from undecided.calibration import apply_temperature, brier_score, fit_temperature, log_loss
from undecided.evaluate import expected_calibration_error, predict, score
from undecided.model import Undecided
from undecided.run import cap_device_memory, evaluate_all, load_eval_sets
from undecided.sources import SOURCES, load_records, sampled_indices
from undecided.tasks import INSTRUCTIONS

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ORDERS = 4
GLOBAL_FIT_N = 1000  # per training dataset
PER_DATASET_FIT_N = 200
FIT_SEED = 1

INSTRUCTION_VARIANTS = ("canonical", "none", "mismatched")


def with_instruction(eval_sets: dict[str, list], variant: str) -> dict[str, list]:
    """The evaluation sets with their instruction kept, removed, or swapped for the canonical
    instruction of the next dataset in the list (the last takes the first's). Tasks whose
    instruction carries a second text (MNLI and the other pair tasks) only take part unchanged."""
    if variant == "canonical":
        return eval_sets
    single = [ds for ds, records in eval_sets.items() if records[0].source not in FIXED_INSTRUCTION_SOURCES]
    if variant == "none":
        return {ds: [replace(r, instruction="") for r in eval_sets[ds]] for ds in single}
    if variant == "mismatched":
        swapped = {}
        for i, ds in enumerate(single):
            donor = eval_sets[single[(i + 1) % len(single)]][0].source
            swapped[ds] = [replace(r, instruction=INSTRUCTIONS[donor][0]) for r in eval_sets[ds]]
        return swapped
    raise ValueError(f"unknown instruction variant {variant!r}")


def probe_instructions(model: Undecided, raw: dict, eval_sets: dict[str, list], out_dir: Path) -> dict:
    report = {}
    for variant in INSTRUCTION_VARIANTS:
        print(f"--- instruction: {variant}", flush=True)
        partial = out_dir / f"probe-instructions-{variant}.partial.json"
        report[variant] = evaluate_all(model.scorer, model.tokenizer, with_instruction(eval_sets, variant),
                                       model.max_length, model.device, partial=partial)["datasets"]
    return report


def reorder(record, rng: random.Random):
    """The same example with its options in a random order."""
    return shuffle_options(record, rng)


def average_by_name(records: list, variants: list[list], variant_probs: list[list[list[float]]]) -> list[list[float]]:
    """Average each option's probability across differently ordered copies of the same examples,
    returned in the order of records."""
    averaged = []
    for i, record in enumerate(records):
        by_name = {option: 0.0 for option in record.options}
        for copies, probs in zip(variants, variant_probs):
            for option, p in zip(copies[i].options, probs[i]):
                by_name[option] += p / len(variants)
        averaged.append([by_name[option] for option in record.options])
    return averaged


def probe_orders(model: Undecided, raw: dict, eval_sets: dict[str, list], out_dir: Path) -> dict:
    rng = random.Random(FIT_SEED)
    variants = {ds: [records] + [[reorder(r, rng) for r in records] for _ in range(ORDERS - 1)]
                for ds, records in eval_sets.items()}
    probs = _cached_probs(model, {f"{ds}/{k}": copies for ds, vs in variants.items() for k, copies in enumerate(vs)},
                          out_dir / "probe-orders-probs.partial.json")
    report = {}
    for ds, records in eval_sets.items():
        singles = [score(variants[ds][k], probs[f"{ds}/{k}"]) for k in range(ORDERS)]
        averaged = score(records, average_by_name(records, variants[ds], [probs[f"{ds}/{k}"] for k in range(ORDERS)]))
        keys = [k for k in ("accuracy", "macro_f1", "ece", "auc") if k in averaged]
        report[ds] = {"group": records[0].group, "averaged": {k: averaged[k] for k in keys},
                      "single_orders": [{k: s[k] for k in keys} for s in singles]}
        accs = [s["accuracy"] for s in singles]
        print(f"{ds:14s} single orders {min(accs):.3f} to {max(accs):.3f}, averaged {averaged['accuracy']:.3f}",
              flush=True)
    return report


def fitting_sets(raw: dict, eval_sets: dict[str, list]) -> dict[str, list]:
    """Labelled examples for fitting temperatures, disjoint from training and evaluation rows:
    "global/<ds>" from each training dataset, "own/<ds>" from each held-out dataset."""
    seed = raw.get("seed", 0)
    rng = random.Random(FIT_SEED)
    sets = {}
    for ds, n_train in raw["data"]["train"].items():
        used = sampled_indices(ds, "train", n_train, seed)
        sets[f"global/{ds}"] = [shuffle_options(r, rng) for r in
                                load_records(ds, "train", GLOBAL_FIT_N, FIT_SEED, exclude=used)]
    for ds, records in eval_sets.items():
        if records[0].group == "train":
            continue
        source = SOURCES[ds]
        used = (sampled_indices(ds, "eval", raw["data"]["eval"][ds], seed)
                if source.train_split == source.eval_split else None)  # separate splits: nothing to avoid
        sets[f"own/{ds}"] = [shuffle_options(r, rng) for r in
                             load_records(ds, "train", PER_DATASET_FIT_N, FIT_SEED, exclude=used)]
    return sets


def _scores(records: list, probs: list[list[float]]) -> dict:
    hits = [r.options[r.answer_idx] == r.options[max(range(len(p)), key=p.__getitem__)] for r, p in zip(records, probs)]
    confidences = [max(p) for p in probs]
    return {"ece": expected_calibration_error(confidences, hits), "brier": brier_score(records, probs),
            "log_loss": log_loss(records, probs), "confidence": sum(confidences) / len(confidences),
            "accuracy": sum(hits) / len(hits)}


def _cached_probs(model: Undecided, sets: dict[str, list], cache: Path) -> dict[str, list]:
    """Probabilities per set, saved after each set so a rerun skips finished ones."""
    done = json.loads(cache.read_text()) if cache.exists() else {}
    for name, records in sets.items():
        if name in done and len(done[name]) == len(records):
            continue
        done[name] = predict(model.scorer, model.tokenizer, records, model.max_length, 32, model.device)
        cache.write_text(json.dumps(done))
        print(f"probabilities: {name} ({len(records)})", flush=True)
    return done


def probe_calibration(model: Undecided, raw: dict, eval_sets: dict[str, list], out_dir: Path) -> dict:
    fit = fitting_sets(raw, eval_sets)
    probs = _cached_probs(model, {**fit, **{f"eval/{ds}": r for ds, r in eval_sets.items()}},
                          out_dir / "probe-calibration-probs.partial.json")
    global_records = [r for name, rs in fit.items() if name.startswith("global/") for r in rs]
    global_probs = [p for name in fit if name.startswith("global/") for p in probs[name]]
    global_t = fit_temperature(global_records, global_probs)
    print(f"global T = {global_t:.3f}", flush=True)

    report = {"global_t": global_t, "global_fit_examples": len(global_records), "datasets": {}}
    for ds, records in eval_sets.items():
        raw_probs = probs[f"eval/{ds}"]
        entry = {"group": records[0].group, "raw": _scores(records, raw_probs),
                 "global_t": _scores(records, apply_temperature(raw_probs, global_t))}
        if f"own/{ds}" in fit:
            own_t = fit_temperature(fit[f"own/{ds}"], probs[f"own/{ds}"])
            entry["own_t_value"] = own_t
            entry["own_t"] = _scores(records, apply_temperature(raw_probs, own_t))
        report["datasets"][ds] = entry
        line = f"{ds:14s} ECE raw {entry['raw']['ece']:.3f}  global T {entry['global_t']['ece']:.3f}"
        if "own_t" in entry:
            line += f"  own T ({entry['own_t_value']:.2f}) {entry['own_t']['ece']:.3f}"
        print(line, flush=True)

    plot_reliability(eval_sets, probs, global_t, out_dir / "plots" / "reliability.png")
    return report


def _reliability_curve(records: list, probs: list[list[float]], bins: int = 10) -> tuple[list, list, list]:
    binned = [[] for _ in range(bins)]
    for r, p in zip(records, probs):
        best = max(range(len(p)), key=p.__getitem__)
        binned[min(int(p[best] * bins), bins - 1)].append((p[best], best == r.answer_idx))
    xs, ys, ns = [], [], []
    for items in binned:
        if len(items) >= 10:  # bins with fewer examples are too noisy to plot
            xs.append(sum(c for c, _ in items) / len(items))
            ys.append(sum(h for _, h in items) / len(items))
            ns.append(len(items))
    return xs, ys, ns


def plot_reliability(eval_sets: dict[str, list], probs: dict, global_t: float, path: Path) -> None:
    """One small panel per dataset: accuracy against stated confidence, raw and with global T."""
    names = list(eval_sets)
    cols = 4
    rows = -(-len(names) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(10, 2.6 * rows), facecolor="#fcfcfb", sharex=True, sharey=True)
    for ax, ds in zip(axes.flat, names):
        ax.set_facecolor("#fcfcfb")
        ax.plot([0, 1], [0, 1], color="#52514e", linewidth=0.8, linestyle=(0, (3, 3)))
        raw_probs = probs[f"eval/{ds}"]
        for label, p, color, marker in [("raw", raw_probs, "#2a78d6", "o"),
                                        (f"global T = {global_t:.2f}", apply_temperature(raw_probs, global_t),
                                         "#eb6834", "s")]:
            xs, ys, _ = _reliability_curve(eval_sets[ds], p)
            ax.plot(xs, ys, color=color, linewidth=2, marker=marker, markersize=4, label=label)
        ax.set_title(f"{ds} ({eval_sets[ds][0].group.replace('_', ' ')})", fontsize=8.5, color="#0b0b0b")
        ax.tick_params(labelsize=7, colors="#52514e", length=0)
        ax.grid(color="#e4e3df", linewidth=0.6)
        for side in ax.spines.values():
            side.set_visible(False)
    for ax in list(axes.flat)[len(names):]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=7, frameon=False, loc="upper left")
    fig.supxlabel("stated confidence", fontsize=9, color="#52514e")
    fig.supylabel("accuracy", fontsize=9, color="#52514e")
    fig.suptitle("Reliability: points on the diagonal are calibrated", fontsize=11, color="#0b0b0b", x=0.01,
                 ha="left")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, facecolor="#fcfcfb")


PROBES = {"instructions": probe_instructions, "calibration": probe_calibration, "orders": probe_orders}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("probe", choices=sorted(PROBES))
    parser.add_argument("config")
    args = parser.parse_args()

    raw = yaml.safe_load(Path(args.config).read_text())
    model = Undecided.from_pretrained(Path("runs") / raw["name"])
    cap_device_memory(model.device)
    out_dir = Path("results") / raw["name"]
    report = PROBES[args.probe](model, raw, load_eval_sets(raw, raw.get("seed", 0)), out_dir)
    (out_dir / f"probe-{args.probe}.json").write_text(json.dumps(report, indent=2) + "\n")
    for partial in out_dir.glob(f"probe-{args.probe}-*.partial.json"):
        partial.unlink()


if __name__ == "__main__":
    main()
