"""Print augmented training examples and label distributions for a config, to check by eye.

    uv run python -m undecided.preview configs/poc.yaml
"""

import argparse
import random
import textwrap
from collections import Counter
from pathlib import Path

import yaml

from undecided.augment import augment
from undecided.sources import load_records
from undecided.tasks import PARAPHRASES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config")
    parser.add_argument("--examples", type=int, default=5)
    args = parser.parse_args()

    raw = yaml.safe_load(Path(args.config).read_text())
    seed, max_options = raw.get("seed", 0), raw["train"].get("max_options", 16)
    rng = random.Random(seed)

    for name, n in raw["data"]["train"].items():
        records = load_records(name, "train", n, seed)
        labels = records[0].options
        counts = Counter(r.options[r.answer_idx] for r in records)
        print(f"\n=== {name}: {len(records)} examples, {len(labels)} labels ===")
        top = ", ".join(f"{label} {count}" for label, count in counts.most_common(8))
        print(f"most common: {top}")
        if len(counts) > 8:
            print(f"least common: {', '.join(f'{lab} {c}' for lab, c in counts.most_common()[-3:])}")
        unknown = set(PARAPHRASES.get(name, {})) - set(labels)
        if unknown:
            print(f"WARNING: paraphrases for labels that don't exist: {sorted(unknown)}")

        for record in rng.sample(records, args.examples):
            out = augment(record, rng, max_options)
            options = "  ".join(f"[{o}]" if i == out.answer_idx else o for i, o in enumerate(out.options))
            print(f"\n  instruction: {out.instruction or '(none)'}")
            print(f"  options:     {options}")
            print(textwrap.indent(textwrap.shorten(out.text, 220), "  text:        "))


if __name__ == "__main__":
    main()
