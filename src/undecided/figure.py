"""Draw the README figure: ranking AUC on every two-option dataset, per model.

    uv run python -m undecided.figure        # writes results/auc.png
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


def main() -> None:
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
