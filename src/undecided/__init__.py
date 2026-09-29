"""undecided: a Jev-style zero-shot classifier, ModernBERT with a per-option linear head."""

import os

# Unsupported MPS operations fall back to the CPU instead of crashing. Set before torch loads.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from undecided.model import Undecided  # noqa: E402

__all__ = ["Undecided"]
