"""Inference-only probes of a trained checkpoint, on the evaluation sets of its config.

    uv run python -m undecided.probe instructions configs/poc-warm.yaml
    uv run python -m undecided.probe calibration configs/poc-warm.yaml
    uv run python -m undecided.probe orders configs/poc-warm.yaml
    uv run python -m undecided.probe examples configs/poc-warm.yaml
    uv run python -m undecided.probe yahoo|distractor|subjectivity|biencoder configs/poc-warm.yaml

instructions: does the model read the instruction? Scores every dataset with its canonical
instruction, with none, and with another dataset's canonical instruction.

orders: option-order averaging. Scores every example under 4 option orders, then averages each
option's probability across them.

examples: the five hand-written prompts shown in the README, with every option's probability.

yahoo: Yahoo Answers accuracy split into classes with a counterpart among the AG News training
labels (Sports, Business & Finance, Science & Mathematics, Computers & Internet) and the rest.

distractor: adds an irrelevant option ("recipe for pancakes") to every example and measures how
often it wins and how often the choice among the original options changes.

subjectivity: tests the explanation for SUBJ's inverted ranking: asks the same model for the
sentiment of each sentence and checks whether strong sentiment goes with "objective" sentences
and with the model's P(subjective).

biencoder: a training-free reference: frozen ModernBERT-base, mean-pooled embeddings of the text
and of each option, cosine similarity. Writes results/frozen-biencoder/metrics.json.

calibration: temperature scaling. A global T is fitted on fresh samples of the training datasets
(rows used neither for training nor for evaluation) and applied to every dataset; a per-dataset T
is fitted on 200 labelled examples of each held-out dataset. Writes reliability diagrams too.

Results go to results/<name>/probe-<probe>.json; each finished dataset is saved as it completes.
"""

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import json
import math
import random
from dataclasses import replace
from pathlib import Path

import matplotlib
import torch
import yaml

from undecided.augment import FIXED_INSTRUCTION_SOURCES, shuffle_options
from undecided.calibration import apply_temperature, brier_score, fit_temperature, log_loss
from undecided.evaluate import expected_calibration_error, predict, score
from undecided.results import group_summary
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


EXAMPLES = [
    ("NASA's rover found traces of water beneath the Martian surface.",
     ["space exploration", "cooking", "football", "stock market"], None),
    ("The striker scored twice in the second half.", ["space exploration", "cooking", "football", "stock market"], None),
    ("My card was charged twice.", ["billing", "bug report", "praise"], "Classify the support ticket."),
    ("I absolutely loved this film, the acting was superb.", ["positive", "negative"], None),
    ("The app crashes every time I open the settings page.", ["billing", "bug report", "praise"],
     "Classify the support ticket."),
]


def probe_examples(model: Undecided, raw: dict, eval_sets: dict[str, list], out_dir: Path) -> list:
    report = []
    for text, options, instruction in EXAMPLES:
        probs = model.classify(text, options, instruction)
        report.append({"text": text, "options": options, "instruction": instruction, "probabilities": probs})
        best = max(probs, key=probs.get)
        print(f"{text[:50]:50s} {best} {probs[best]:.1%}", flush=True)
    return report


YAHOO_OVERLAP = {"sports", "business & finance", "science & mathematics", "computers & internet"}
DISTRACTOR = "recipe for pancakes"
DISTRACTOR_SETS = ["ag_news", "emotion", "sst2", "yahoo_answers", "tweet_irony"]


def _argmax(row: list[float]) -> int:
    return max(range(len(row)), key=row.__getitem__)


def accuracy_by_overlap(records: list, probs: list[list[float]], overlapping: set[str]) -> dict:
    """Accuracy for examples whose gold label has a training counterpart, and for the rest."""
    split = {"overlapping": [], "new": []}
    for r, p in zip(records, probs):
        gold = r.options[r.answer_idx]
        split["overlapping" if gold in overlapping else "new"].append(r.options[_argmax(p)] == gold)
    return {k: {"n": len(v), "accuracy": sum(v) / len(v)} for k, v in split.items() if v}


def probe_yahoo(model: Undecided, raw: dict, eval_sets: dict[str, list], out_dir: Path) -> dict:
    records = eval_sets["yahoo_answers"]
    probs = predict(model.scorer, model.tokenizer, records, model.max_length, 32, model.device)
    report = accuracy_by_overlap(records, probs, YAHOO_OVERLAP)
    print(report, flush=True)
    return report


def with_distractor(record, distractor: str, rng: random.Random):
    """The same example with one irrelevant option inserted at a random position."""
    options = list(record.options)
    options.insert(rng.randint(0, len(options)), distractor)
    return replace(record, options=options, answer_idx=options.index(record.options[record.answer_idx]))


def probe_distractor(model: Undecided, raw: dict, eval_sets: dict[str, list], out_dir: Path) -> dict:
    rng = random.Random(FIT_SEED)
    report = {}
    for ds in DISTRACTOR_SETS:
        base = eval_sets[ds]
        extra = [with_distractor(r, DISTRACTOR, rng) for r in base]
        p_base = predict(model.scorer, model.tokenizer, base, model.max_length, 32, model.device)
        p_extra = predict(model.scorer, model.tokenizer, extra, model.max_length, 32, model.device)
        chosen = sum(r.options[_argmax(p)] == DISTRACTOR for r, p in zip(extra, p_extra))
        flips = 0
        for b, pb, e, pe in zip(base, p_base, extra, p_extra):
            among_originals = max((o for o in e.options if o != DISTRACTOR), key=lambda o: pe[e.options.index(o)])
            flips += among_originals != b.options[_argmax(pb)]
        report[ds] = {"n": len(base), "accuracy": score(base, p_base)["accuracy"],
                      "accuracy_with_distractor": score(extra, p_extra)["accuracy"],
                      "distractor_chosen": chosen / len(base), "choice_changed": flips / len(base),
                      "mean_distractor_probability": sum(p[e.options.index(DISTRACTOR)] for e, p in zip(extra, p_extra))
                      / len(base)}
        print(ds, {k: round(v, 3) for k, v in report[ds].items()}, flush=True)
    return report


def probe_subjectivity(model: Undecided, raw: dict, eval_sets: dict[str, list], out_dir: Path) -> dict:
    from sklearn.metrics import roc_auc_score

    records = eval_sets["subjectivity"]
    p_subj = predict(model.scorer, model.tokenizer, records, model.max_length, 32, model.device)
    subjective = [p[r.options.index("subjective")] for r, p in zip(records, p_subj)]
    sentiment = [replace(r, instruction=INSTRUCTIONS["sst2"][0], options=["positive", "negative"], answer_idx=0)
                 for r in records]
    p_sent = predict(model.scorer, model.tokenizer, sentiment, model.max_length, 32, model.device)
    intensity = [abs(p[0] - 0.5) * 2 for p in p_sent]  # 0 = neutral, 1 = strongly positive or negative
    objective = [r.options[r.answer_idx] == "objective" for r in records]
    n = len(records)
    mean = lambda xs: sum(xs) / len(xs)  # noqa: E731
    mi, ms = mean(intensity), mean(subjective)
    corr = (sum((a - mi) * (b - ms) for a, b in zip(intensity, subjective))
            / math.sqrt(sum((a - mi) ** 2 for a in intensity) * sum((b - ms) ** 2 for b in subjective)))
    report = {"n": n,
              "auc_intensity_for_objective": roc_auc_score(objective, intensity),
              "mean_intensity_objective": mean([i for i, o in zip(intensity, objective) if o]),
              "mean_intensity_subjective": mean([i for i, o in zip(intensity, objective) if not o]),
              "correlation_intensity_p_subjective": corr}
    print({k: round(v, 3) for k, v in report.items()}, flush=True)
    return report


@torch.no_grad()
def probe_biencoder(model: Undecided, raw: dict, eval_sets: dict[str, list], out_dir: Path) -> dict:
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(raw["backbone"])
    encoder = AutoModel.from_pretrained(raw["backbone"]).to(model.device).eval()

    def embed(texts: list[str]) -> torch.Tensor:
        out = []
        for start in range(0, len(texts), 32):
            batch = tokenizer(texts[start : start + 32], truncation=True, max_length=512, padding=True,
                              return_tensors="pt").to(model.device)
            hidden = encoder(**batch).last_hidden_state
            mask = batch["attention_mask"][..., None].float()
            out.append(torch.nn.functional.normalize((hidden * mask).sum(1) / mask.sum(1), dim=-1).cpu())
        return torch.cat(out)

    datasets = {}
    for ds, records in eval_sets.items():
        if records[0].source in FIXED_INSTRUCTION_SOURCES:
            continue  # pair tasks need the second text, which this reference ignores
        labels = sorted(records[0].options)
        label_vectors = dict(zip(labels, embed(labels)))
        text_vectors = embed([r.text for r in records])
        probs = []
        for r, t in zip(records, text_vectors):
            sims = torch.stack([t @ label_vectors[o] for o in r.options])
            probs.append(torch.softmax(sims * 20, dim=0).tolist())  # the scale only matters for calibration
        datasets[ds] = {"group": records[0].group, "options": len(records[0].options), **score(records, probs)}
        print(f"{ds}: accuracy {datasets[ds]['accuracy']:.3f}", flush=True)
    target = Path("results") / "frozen-biencoder"
    target.mkdir(parents=True, exist_ok=True)
    metrics = {"run": "frozen-biencoder", "model": raw["backbone"], "datasets": datasets,
               "groups": group_summary(datasets)}
    (target / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return {"written": str(target / "metrics.json")}


PROBES = {"instructions": probe_instructions, "calibration": probe_calibration, "orders": probe_orders,
          "examples": probe_examples, "yahoo": probe_yahoo, "distractor": probe_distractor,
          "subjectivity": probe_subjectivity, "biencoder": probe_biencoder}


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
