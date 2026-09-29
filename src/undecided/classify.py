"""Classify a text with a trained checkpoint.

    uv run python -m undecided.classify runs/tracer "Rover finds water on Mars" \
        --options sports business science --instruction "What is this news about?"
"""

import argparse

from undecided import Undecided


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint", help="a run directory under runs/")
    parser.add_argument("text")
    parser.add_argument("--options", nargs="+", required=True)
    parser.add_argument("--instruction")
    args = parser.parse_args()

    probs = Undecided.from_pretrained(args.checkpoint).classify(args.text, args.options, args.instruction)
    width = max(len(o) for o in probs)
    for option, p in sorted(probs.items(), key=lambda kv: -kv[1]):
        print(f"{option:<{width}}  {p:6.1%}")


if __name__ == "__main__":
    main()
