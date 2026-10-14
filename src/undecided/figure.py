"""Draw the README figures: latency against the number of options, and ranking AUC on the
two-option datasets.

    uv run python -m undecided.figure        # writes results/latency.png and results/auc.png
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RUNS = [("poc-warm", "undecided (poc-warm)", "#2a78d6", "o"), ("diverse-warm", "undecided (diverse-warm)", "#eb6834", "s"),
        ("zeroshot-v2", "NLI zero-shot baseline", "#1baf7a", "D"), ("gliclass", "GLiClass baseline", "#eda100", "^")]
TRAINED = [("sst2", "SST-2 sentiment")]
UNSEEN = [("sms_spam", "SMS spam"), ("tweet_hate", "hate speech"), ("tweet_irony", "irony"),
          ("subjectivity", "subjectivity")]
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


LATENCY = [("diverse-warm", "undecided (one pass)", "#2a78d6", "o", -11),
           ("zeroshot-v2", "NLI baseline (one pass per option)", "#1baf7a", "D", -3),
           ("gliclass", "GLiClass baseline (one pass)", "#eda100", "^", 5)]


def plot_latency() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for name, label, color, marker, label_dy in LATENCY:
        rows = json.loads((Path("results") / name / "latency.json").read_text())["devices"]["mps"]
        xs, ys = [r["options"] for r in rows], [r["p50_ms"] for r in rows]
        ax.plot(xs, ys, color=color, marker=marker, linewidth=2, markersize=6, label=label, zorder=3)
        ax.annotate(f"{ys[-1]:.0f} ms", (xs[-1], ys[-1]), textcoords="offset points", xytext=(6, label_dy),
                    color=INK, fontsize=8)
    ax.set_xticks([2, 10, 50])
    ax.set_xlim(0, 58)
    ax.set_ylim(0, None)
    ax.set_xlabel("options per request", color=MUTED, fontsize=9)
    ax.set_ylabel("p50 latency, ms (M4 Pro GPU)", color=MUTED, fontsize=9)
    ax.tick_params(colors=MUTED, length=0)
    ax.grid(color=GRID, linewidth=0.8, zorder=0)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.legend(loc="upper left", frameon=False, fontsize=8, labelcolor=INK)
    ax.set_title("One pass for all options: latency grows slowly with the number of options", color=INK,
                 fontsize=10.5, loc="left", pad=10)
    fig.tight_layout()
    fig.savefig(Path("results") / "latency.png", dpi=160, facecolor=SURFACE)


def main() -> None:
    plot_latency()
    metrics = {name: json.loads((Path("results") / name / "metrics.json").read_text())["datasets"]
               for name, _, _, _ in RUNS}
    rows = TRAINED + UNSEEN
    fig, ax = plt.subplots(figsize=(7.2, 4.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    offsets = {name: (i - 1.5) * 0.17 for i, (name, _, _, _) in enumerate(RUNS)}
    for name, label, color, marker in RUNS:
        ys, xs = [], []
        for i, (key, _) in enumerate(rows):
            auc = metrics[name].get(key, {}).get("auc")
            if auc is not None:
                ys.append(i + offsets[name])
                xs.append(auc)
        ax.scatter(xs, ys, s=46, color=color, marker=marker, label=label, zorder=3,
                   edgecolors=SURFACE, linewidths=1.5)

    ax.axvline(0.5, color=MUTED, linewidth=1, linestyle=(0, (3, 3)), zorder=1)
    ax.text(0.505, -0.55, "chance", color=MUTED, fontsize=8, va="center")
    ax.axhline(len(TRAINED) - 0.5, color=GRID, linewidth=1)
    ax.text(0.305, len(TRAINED) - 0.95, "a training set of undecided", color=MUTED, fontsize=8, fontweight="bold")
    ax.text(0.305, len(TRAINED) - 0.05, "never seen by undecided", color=MUTED, fontsize=8, fontweight="bold")

    ax.set_yticks(range(len(rows)), [label for _, label in rows], color=INK, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0.3, 1.02)
    ax.set_xlabel("ranking AUC (0.5 = no signal, 1.0 = perfect)", color=MUTED, fontsize=9)
    ax.tick_params(colors=MUTED, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False, fontsize=7.5,
              labelcolor=INK)
    ax.set_title("On unseen task types, undecided ranks near chance; the NLI baseline doesn't", color=INK,
                 fontsize=10.5, loc="left", pad=12)
    fig.text(0.01, 0.01, "Two-option datasets, options shuffled per example. The NLI baseline's model card lists spam\n"
             "and hate-speech datasets in its training results; irony and subjectivity are not listed.",
             color=MUTED, fontsize=7)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(Path("results") / "auc.png", dpi=160, facecolor=SURFACE)


if __name__ == "__main__":
    main()
