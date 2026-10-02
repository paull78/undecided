"""Print README-ready Markdown tables from one or more runs' metrics.

    uv run python -m undecided.results results/poc results/tracer
"""

import argparse
import json
from pathlib import Path

GROUPS = {"train": "In-distribution", "heldout_labels": "Held-out labels", "heldout_task": "Held-out task"}


def group_summary(datasets: dict[str, dict]) -> dict[str, dict]:
    """Mean accuracy and macro-F1 per group. Each dataset counts once, whatever its size."""
    summary = {}
    for group in GROUPS:
        names = [name for name, m in datasets.items() if m.get("group", "train") == group]
        if names:
            summary[group] = {"datasets": names}
            for metric in ("accuracy", "macro_f1"):  # runs from before Phase 3 have no macro-F1
                if all(metric in datasets[n] for n in names):
                    summary[group][metric] = round(sum(datasets[n][metric] for n in names) / len(names), 4)
    return summary


def _table(header: list[str], rows: list[list[str]]) -> str:
    def line(cells: list[str]) -> str:
        return "| " + " | ".join(cells) + " |"

    return "\n".join([line(header), "|" + "---|" * len(header), *map(line, rows)])


def markdown_tables(runs: list[dict]) -> tuple[str, str]:
    """A summary table (one row per run, one column per group) and a per-dataset table (one row per
    dataset, one column per run). Missing values are left blank."""
    summaries = [group_summary(run["datasets"]) for run in runs]
    summary = _table(
        ["Run", *(f"{label} acc." for label in GROUPS.values())],
        [[run["run"], *(f"{s[g]['accuracy']:.3f}" if g in s else "" for g in GROUPS)]
         for run, s in zip(runs, summaries)],
    )

    datasets = {}  # name -> first metrics seen, for group and option count
    for run in runs:
        for name, m in run["datasets"].items():
            datasets.setdefault(name, m)
    ordered = sorted(datasets, key=lambda n: list(GROUPS).index(datasets[n].get("group", "train")))
    rows = []
    for name in ordered:
        options = datasets[name].get("options")
        cells = [name, datasets[name].get("group", "train"), str(options or ""),
                 f"{1 / options:.3f}" if options else ""]
        for run in runs:
            m = run["datasets"].get(name)
            cells.append("" if not m else f"{m['accuracy']:.3f} / {m['macro_f1']:.3f}" if "macro_f1" in m
                         else f"{m['accuracy']:.3f}")
        rows.append(cells)
    per_dataset = _table(["Dataset", "Group", "Options", "Chance", *(f"{r['run']} acc. / F1" for r in runs)], rows)
    return summary, per_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results", nargs="+", help="results/<run> directories")
    runs = [json.loads((Path(d) / "metrics.json").read_text()) for d in parser.parse_args().results]
    summary, per_dataset = markdown_tables(runs)
    print(summary, per_dataset, sep="\n\n")


if __name__ == "__main__":
    main()
