"""One-sample Wilcoxon signed rank test as in R 4.4 (``stats::wilcox.test``).

R 4.4.x (the reference environment, R 4.4.3) drops zeros, ranks
``abs(x)`` without rounding (``digits.rank = Inf``) and uses the exact
null distribution when ``n < 50`` with no ties and no zeros; otherwise the
normal approximation with tie correction and continuity correction.
R >= 4.6 changed the zero handling; this follows 4.4.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import stats

__all__ = ["wilcox_signed_rank_pvalue", "psignrank"]


def _csignrank_counts(n: int) -> np.ndarray:
    """Number of subsets of {1..n} with each sum 0..n(n+1)/2."""
    u = n * (n + 1) // 2
    c = np.zeros(u + 1, dtype=object)
    c[0] = 1
    for k in range(1, n + 1):
        c[k:] = c[k:] + c[:-k] if k <= u else c[k:]
    return c


def psignrank(q: float, n: int, lower_tail: bool = True) -> float:
    """``stats::psignrank`` (exact)."""
    q = math.floor(q + 1e-7)
    u = n * (n + 1) // 2
    if q < 0:
        return 0.0 if lower_tail else 1.0
    if q >= u:
        return 1.0 if lower_tail else 0.0
    c = _csignrank_counts(n)
    total = 2**n
    low = sum(c[: q + 1])
    p = low / total
    return float(p) if lower_tail else float((total - low) / total)


def wilcox_signed_rank_pvalue(x, mu: float = 0.0, correct: bool = True, exact=None) -> float:
    """``wilcox.test(x, mu = mu)$p.value``, two-sided, R 4.4 semantics."""
    x = np.asarray(x, dtype=np.float64)
    x = x[~np.isnan(x)]
    if len(x) < 1:
        raise ValueError("not enough (non-missing) 'x' observations")
    x = x - mu
    zeroes = bool(np.any(x == 0))
    if zeroes:
        x = x[x != 0]
    n = float(len(x))
    if exact is None:
        exact = n < 50
    r = stats.rankdata(np.abs(x), method="average")
    statistic = float(r[x > 0].sum())
    ties = len(r) != len(np.unique(r))
    if exact and not ties and not zeroes:
        ni = int(n)
        if statistic > n * (n + 1) / 4:
            p = psignrank(statistic - 1, ni, lower_tail=False)
        else:
            p = psignrank(statistic, ni)
        return min(2 * p, 1.0)
    _, nties = np.unique(r, return_counts=True)
    z = statistic - n * (n + 1) / 4
    sigma = math.sqrt(max(0.0, n * (n + 1) * (2 * n + 1) / 24 - float(np.sum(nties.astype(float) ** 3 - nties)) / 48))
    correction = math.copysign(0.5, z) if (correct and z != 0) else 0.0
    # IEEE arithmetic as in R: sigma = 0 (e.g. all values were zeros and were
    # dropped) gives z = NaN (0/0) or +-Inf, and the p-value NaN or 0.
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.float64(z - correction) / np.float64(sigma)
    if np.isnan(z):
        return float("nan")
    return float(2 * min(stats.norm.cdf(z), stats.norm.sf(z)))
