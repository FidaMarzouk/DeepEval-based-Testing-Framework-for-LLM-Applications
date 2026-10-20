"""
Small, dependency-free statistics for LLM-judge scores.

LLM-judge metrics are noisy (the judge is an LLM; the SUT usually is too),
and suites are small (tens of goldens, not thousands). So:

  - bootstrap_ci(): percentile bootstrap CI of a mean — no normality
    assumption, behaves sensibly for scores bounded in [0, 1].
  - paired_permutation_pvalue(): two-sided sign-flip permutation test on
    paired per-golden deltas — exact for small n, Monte Carlo above that.
    Asks "could this mean delta be judge/SUT noise?" without assuming a
    distribution.
  - within_golden_sd(): pooled standard deviation of repeated scores of
    the SAME golden — the run-to-run noise floor, used by `elyaeval
    calibrate` to recommend regression tolerances.

Everything is seeded so the same CSVs always produce the same verdict.
"""

from __future__ import annotations

import itertools
import math
import random
from collections.abc import Sequence
from typing import Optional

DEFAULT_SEED = 1234
_EXACT_PERMUTATION_MAX_N = 16  # 2**16 = 65,536 sign patterns


def mean(values: Sequence[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def bootstrap_ci(
    values: Sequence[float],
    confidence: float = 0.95,
    n_resamples: int = 2000,
    seed: int = DEFAULT_SEED,
) -> tuple[Optional[float], Optional[float]]:
    """Percentile-bootstrap CI of the mean. (None, None) for < 2 values —
    one observation says nothing about spread."""
    n = len(values)
    if n < 2:
        return None, None
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(values, k=n)) / n for _ in range(n_resamples))
    alpha = (1 - confidence) / 2
    lo = means[max(0, math.floor(alpha * n_resamples))]
    hi = means[min(n_resamples - 1, math.ceil((1 - alpha) * n_resamples) - 1)]
    return lo, hi


def paired_permutation_pvalue(
    deltas: Sequence[float],
    n_resamples: int = 10000,
    seed: int = DEFAULT_SEED,
) -> Optional[float]:
    """Two-sided p-value for H0 'mean paired delta is 0', by randomly
    flipping the sign of each delta. None for an empty input; 1.0 when every
    delta is 0."""
    n = len(deltas)
    if n == 0:
        return None
    observed = abs(sum(deltas))
    if observed == 0:
        return 1.0
    eps = 1e-12
    if n <= _EXACT_PERMUTATION_MAX_N:
        extreme = sum(
            1
            for signs in itertools.product((1, -1), repeat=n)
            if abs(sum(s * d for s, d in zip(signs, deltas, strict=True))) >= observed - eps
        )
        return extreme / 2**n
    rng = random.Random(seed)
    extreme = sum(
        1
        for _ in range(n_resamples)
        if abs(sum(d if rng.random() < 0.5 else -d for d in deltas)) >= observed - eps
    )
    return (extreme + 1) / (n_resamples + 1)


def within_golden_sd(scores_by_golden: dict[str, list[float]]) -> Optional[float]:
    """Pooled SD of repeated scores per golden (goldens with >= 2 repeats).
    None if no golden was scored more than once."""
    num, dof = 0.0, 0
    for scores in scores_by_golden.values():
        k = len(scores)
        if k < 2:
            continue
        m = sum(scores) / k
        num += sum((s - m) ** 2 for s in scores)
        dof += k - 1
    return math.sqrt(num / dof) if dof else None


def recommended_tolerance(noise_sd: float, n_goldens: int, repeats: int, z: float = 1.96) -> float:
    """Smallest mean-delta between two runs that judge/SUT noise alone would
    rarely produce: z * sd * sqrt(2 / (n * k)) for two independent runs of
    n goldens x k repeats."""
    return z * noise_sd * math.sqrt(2.0 / (max(n_goldens, 1) * max(repeats, 1)))


def cohen_kappa(a: Sequence[bool], b: Sequence[bool]) -> Optional[float]:
    """Agreement between two binary raters beyond chance. None if undefined
    (empty, or chance agreement is 1)."""
    n = len(a)
    if n == 0 or n != len(b):
        return None
    observed = sum(1 for x, y in zip(a, b, strict=True) if x == y) / n
    pa, pb = sum(a) / n, sum(b) / n
    expected = pa * pb + (1 - pa) * (1 - pb)
    if expected >= 1:
        return None
    return (observed - expected) / (1 - expected)
