import math

from undecided.stats import auc_half_width, mean_half_width, proportion_half_width


def test_a_proportion_gets_the_normal_approximation_interval():
    # p = 0.5 on 100 examples: standard error 0.05, 95% half-width 1.96 * 0.05.
    assert math.isclose(proportion_half_width(0.5, 100), 0.098, rel_tol=1e-3)
    assert proportion_half_width(0.9, 1000) < proportion_half_width(0.5, 1000)


def test_auc_uses_hanley_mcneil_for_its_standard_error():
    # A = 0.5 with 100 positives and 100 negatives: Q1 = Q2 = 1/3,
    # variance = (0.25 + 99 * (1/3 - 0.25) * 2) / 10,000 = 0.001675.
    assert math.isclose(auc_half_width(0.5, 100, 100), 1.96 * math.sqrt(0.001675), rel_tol=1e-3)


def test_an_average_of_independent_estimates_combines_their_variances():
    # Two estimates with half-widths 0.06 and 0.08: the mean has half-width sqrt(0.06² + 0.08²) / 2.
    assert math.isclose(mean_half_width([0.06, 0.08]), 0.05, rel_tol=1e-9)
