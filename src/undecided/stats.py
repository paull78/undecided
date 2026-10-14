"""95% confidence intervals from summary numbers, for results whose per-example predictions were
not kept. They treat each estimate as independent, which is conservative when two models are
compared on the same examples (a paired test would give tighter intervals)."""

import math

Z = 1.96


def proportion_half_width(p: float, n: int) -> float:
    """Normal approximation to the binomial: accuracy p measured on n examples."""
    return Z * math.sqrt(p * (1 - p) / n)


def auc_half_width(auc: float, n_pos: int, n_neg: int) -> float:
    """Hanley and McNeil (1982) standard error of a ROC AUC."""
    q1 = auc / (2 - auc)
    q2 = 2 * auc**2 / (1 + auc)
    variance = (auc * (1 - auc) + (n_pos - 1) * (q1 - auc**2) + (n_neg - 1) * (q2 - auc**2)) / (n_pos * n_neg)
    return Z * math.sqrt(variance)


def mean_half_width(half_widths: list[float]) -> float:
    """Half-width of the mean of independent estimates, from their own half-widths."""
    return math.sqrt(sum(h**2 for h in half_widths)) / len(half_widths)
