"""Calibration metrics and temperature scaling, working from predicted probabilities.

Dividing log-probabilities by T gives the same softmax as dividing the logits by T, because each
example's log-probabilities differ from its logits by one constant. So temperature scaling needs
only the probabilities that evaluation already produces.
"""

import math

from scipy.optimize import minimize_scalar

from undecided.data import Record

EPS = 1e-12


def apply_temperature(probs: list[list[float]], temperature: float) -> list[list[float]]:
    scaled = []
    for row in probs:
        z = [math.log(max(p, EPS)) / temperature for p in row]
        top = max(z)
        e = [math.exp(v - top) for v in z]
        scaled.append([v / sum(e) for v in e])
    return scaled


def log_loss(records: list[Record], probs: list[list[float]]) -> float:
    return -sum(math.log(max(p[r.answer_idx], EPS)) for r, p in zip(records, probs)) / len(records)


def brier_score(records: list[Record], probs: list[list[float]]) -> float:
    """Squared error between the probabilities and the one-hot answer, summed over options."""
    return sum(
        sum((p - (i == r.answer_idx)) ** 2 for i, p in enumerate(row)) for r, row in zip(records, probs)
    ) / len(records)


def fit_temperature(records: list[Record], probs: list[list[float]]) -> float:
    """The single T that minimises log loss on these examples, searched between 0.05 and 20."""
    result = minimize_scalar(lambda log_t: log_loss(records, apply_temperature(probs, math.exp(log_t))),
                             bounds=(math.log(0.05), math.log(20.0)), method="bounded")
    return math.exp(result.x)
