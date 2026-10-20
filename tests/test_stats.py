"""Statistics helpers: CIs, permutation test, noise, kappa."""

import pytest

from elyaeval.reporting.stats import (
    bootstrap_ci,
    cohen_kappa,
    paired_permutation_pvalue,
    recommended_tolerance,
    within_golden_sd,
)


def test_bootstrap_ci_brackets_the_mean_and_is_deterministic():
    values = [0.2, 0.4, 0.5, 0.6, 0.9, 0.7, 0.3]
    lo, hi = bootstrap_ci(values)
    assert lo < sum(values) / len(values) < hi
    assert bootstrap_ci(values) == (lo, hi)  # seeded


def test_bootstrap_ci_needs_two_values():
    assert bootstrap_ci([0.5]) == (None, None)
    assert bootstrap_ci([]) == (None, None)


def test_permutation_pvalue_exact_small_n():
    # 8 identical negative deltas: only the all-negative and all-positive
    # sign patterns are as extreme -> 2 / 2**8
    assert paired_permutation_pvalue([-0.3] * 8) == pytest.approx(2 / 256)
    assert paired_permutation_pvalue([0.0, 0.0, 0.0]) == 1.0
    assert paired_permutation_pvalue([]) is None


def test_permutation_pvalue_noise_is_not_significant():
    assert paired_permutation_pvalue([0.1, -0.1, 0.05, -0.05, 0.1, -0.12]) > 0.5


def test_permutation_pvalue_monte_carlo_large_n():
    p = paired_permutation_pvalue([-0.2] * 30)
    assert p < 0.001
    assert p == paired_permutation_pvalue([-0.2] * 30)  # seeded


def test_within_golden_sd():
    assert within_golden_sd({"a": [0.5, 0.5], "b": [0.2, 0.2]}) == 0.0
    sd = within_golden_sd({"a": [0.4, 0.6], "b": [0.1, 0.3], "c": [0.9]})
    assert sd == pytest.approx(0.1414, abs=1e-3)
    assert within_golden_sd({"a": [0.5]}) is None


def test_recommended_tolerance_shrinks_with_more_data():
    small = recommended_tolerance(0.1, n_goldens=5, repeats=1)
    big = recommended_tolerance(0.1, n_goldens=50, repeats=3)
    assert small > big > 0


def test_cohen_kappa():
    assert cohen_kappa([True, False, True, False], [True, False, True, False]) == 1.0
    assert cohen_kappa([True, True, False, False], [True, False, True, False]) == 0.0
    assert cohen_kappa([], []) is None
    assert cohen_kappa([True, True], [True, True]) is None  # chance agreement is 1
