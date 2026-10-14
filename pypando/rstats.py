"""The pieces of R's ``stats`` that Pando's model fits go through.

* :func:`dqrdc2` / :func:`dqrls` — R's modified LINPACK QR with limited
  column pivoting (``src/appl/dqrdc2.f``, ``dqrls.f``), the decomposition
  behind ``lm`` and ``glm``. Columns whose remaining norm falls below
  ``tol`` times their original norm are moved to the end (aliased).
* :func:`glm_gaussian` — ``glm(family = gaussian)`` (``glm.fit``: IRLS with
  identity link, which converges at the second iteration) plus the
  coefficient table of ``summary.glm``.
* :func:`p_adjust` — ``p.adjust`` (BH / fdr and the other methods).
* :func:`t_test_one`, :func:`wilcox_test_one` — one-sample tests used by
  Pando's bagging ridge p-values.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import special, stats

__all__ = [
    "dqrdc2",
    "dqrls",
    "glm_gaussian",
    "p_adjust",
    "pt_two_sided",
    "t_test_one",
    "wilcox_test_one",
    "r_rank",
]


def dqrdc2(x: np.ndarray, tol: float):
    """R's ``dqrdc2``. Returns ``(qr, rank, qraux, pivot)`` (pivot 0-based).

    ``x`` is copied; the returned ``qr`` holds R in its upper triangle and the
    Householder vectors below, exactly like R's ``qr()$qr``.
    """
    x = np.array(x, dtype=np.float64, order="F", copy=True)
    n, p = x.shape
    jpvt = np.arange(p)
    qraux = np.zeros(p)
    work = np.zeros((p, 2))
    if n > 0:
        for j in range(p):
            qraux[j] = np.linalg.norm(x[:, j])
            work[j, 0] = qraux[j]
            work[j, 1] = qraux[j] if qraux[j] != 0.0 else 1.0
    lup = min(n, p)
    k = p + 1  # Fortran 1-based k
    for l in range(lup):  # Fortran l = l + 1
        # cycle negligible columns to the end
        while not ((l + 1) >= k or qraux[l] >= work[l, 1] * tol):
            x[:, l:] = np.concatenate([x[:, l + 1 :], x[:, l : l + 1]], axis=1)
            i = jpvt[l]
            t, tt, ttt = qraux[l], work[l, 0], work[l, 1]
            jpvt[l:-1] = jpvt[l + 1 :]
            qraux[l:-1] = qraux[l + 1 :]
            work[l:-1, :] = work[l + 1 :, :]
            jpvt[-1], qraux[-1], work[-1, 0], work[-1, 1] = i, t, tt, ttt
            k -= 1
        if (l + 1) != n:
            nrmxl = np.linalg.norm(x[l:, l])
            if nrmxl != 0.0:
                if x[l, l] != 0.0:
                    nrmxl = math.copysign(nrmxl, x[l, l])
                x[l:, l] *= 1.0 / nrmxl
                x[l, l] = 1.0 + x[l, l]
                for j in range(l + 1, p):
                    t = -np.dot(x[l:, l], x[l:, j]) / x[l, l]
                    x[l:, j] += t * x[l:, l]
                    if qraux[j] != 0.0:
                        tt = 1.0 - (abs(x[l, j]) / qraux[j]) ** 2
                        tt = max(tt, 0.0)
                        t = tt
                        if abs(t) >= 1e-6:
                            qraux[j] = qraux[j] * math.sqrt(t)
                        else:
                            qraux[j] = np.linalg.norm(x[l + 1 :, j])
                            work[j, 0] = qraux[j]
                qraux[l] = x[l, l]
                x[l, l] = -nrmxl
    rank = min(k - 1, n)
    return x, rank, qraux, jpvt


def _qty(qr: np.ndarray, qraux: np.ndarray, k: int, y: np.ndarray) -> np.ndarray:
    """Q'y from ``dqrsl`` (job digit b)."""
    n = qr.shape[0]
    qty = np.array(y, dtype=np.float64, copy=True)
    ju = min(k, n - 1)
    for j in range(ju):
        if qraux[j] != 0.0:
            v = qr[j:, j].copy()
            v[0] = qraux[j]
            t = -np.dot(v, qty[j:]) / v[0]
            qty[j:] += t * v
    return qty


def dqrls(x: np.ndarray, y: np.ndarray, tol: float):
    """R's ``dqrls`` (the work-horse of ``C_Cdqrls``) for a single response.

    Returns ``coefficients`` (in pivoted order; entries past the rank are 0,
    as Cdqrls returns them), ``rank``, ``pivot``, ``qr`` and ``qraux``.
    """
    qr, k, qraux, pivot = dqrdc2(x, tol)
    p = qr.shape[1]
    b = np.zeros(p)
    if k > 0:
        qty = _qty(qr, qraux, k, y)
        bb = qty[:k].copy()
        for jj in range(k):  # back substitution, dqrsl job digit c
            j = k - jj - 1
            if qr[j, j] == 0.0:
                break
            bb[j] = bb[j] / qr[j, j]
            if j > 0:
                bb[:j] += -bb[j] * qr[:j, j]
        b[:k] = bb
    return {"coefficients": b, "rank": k, "pivot": pivot, "qr": qr, "qraux": qraux}


def _r_sum(v: np.ndarray) -> float:
    """R's ``sum()``: sequential accumulation in long double (``rsum`` in summary.c)."""
    v = np.asarray(v, dtype=np.float64)
    if len(v) == 0:
        return 0.0
    return float(np.cumsum(v, dtype=np.longdouble)[-1])


def _matvec(X: np.ndarray, b: np.ndarray) -> np.ndarray:
    """R's ``X %*% b`` for a vector ``b``: BLAS ``dgemv`` (``matprod`` in array.c)."""
    from scipy.linalg.blas import dgemv

    if np.isnan(X).any() or np.isinf(X).any() or np.isnan(b).any() or np.isinf(b).any():
        return X @ b  # R's simple_matprod path
    return dgemv(1.0, np.asfortranarray(X), np.ascontiguousarray(b))


def _chol2inv(R: np.ndarray) -> np.ndarray:
    """``chol2inv(R)`` via LAPACK ``dpotri`` (upper), symmetrised."""
    from scipy.linalg.lapack import dpotri

    inv, info = dpotri(np.asfortranarray(np.triu(R)), lower=0)
    if info != 0:
        raise np.linalg.LinAlgError(f"dpotri failed ({info})")
    return np.triu(inv) + np.triu(inv, 1).T


def _qrls(X, z, tol, cache=None):
    """``C_Cdqrls(X, z, tol)``.

    acceleration (E, memoisation): glm.fit decomposes ``x * w`` at every IRLS
    iteration; for the Gaussian family with unit prior weights ``w`` is 1, so
    ``x * w`` is bit-identical to ``x`` and so is its QR. ``cache`` keeps the
    decomposition from the first iteration and only the solve with the new
    working response is redone (identical arithmetic to recomputing).
    """
    from . import _blas

    if not _blas.available:
        return dqrls(X, z, tol)
    if cache is not None and "qr" in cache:
        qr, k, qraux, pivot = cache["qr"]
        b = np.zeros(qr.shape[1])
        if k > 0:
            ib = np.zeros(4, dtype=np.int32)
            db = np.zeros(2)
            _blas._dqrsl_b(qr, k, qraux, np.ascontiguousarray(z, dtype=np.float64), b, ib, db)
        return {"coefficients": b, "rank": k, "pivot": pivot, "qr": qr, "qraux": qraux}
    out = _blas.dqrls_blas(X, z, tol)
    if cache is not None:
        cache["qr"] = (out["qr"], out["rank"], out["qraux"], out["pivot"])
    return out


def pt_two_sided(t: np.ndarray, df: float) -> np.ndarray:
    """``2 * pt(-abs(t), df)``."""
    return 2.0 * special.stdtr(df, -np.abs(t))


def glm_gaussian(X: np.ndarray, y: np.ndarray, colnames, epsilon: float = 1e-8, maxit: int = 25):
    """``summary(glm(y ~ X, family = gaussian))`` for a design matrix ``X``.

    ``X`` already contains the intercept column (as ``model.matrix`` builds
    it). Returns ``rsq = 1 - deviance / null.deviance`` and the coefficient
    table (``term, estimate, std_err, statistic, pval``) in R's order: the
    pivoted order of the non-aliased coefficients.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n, p = X.shape
    tol = min(1e-7, epsilon / 1000)
    eta = y.copy()  # gaussian initialize: mustart <- y; identity link
    mu = eta.copy()
    devold = _r_sum((y - mu) ** 2)
    coef = None
    fit = None
    qr_cache = {}
    for _ in range(maxit):
        z = eta + (y - mu) / 1.0  # mu.eta = 1, weights = 1
        fit = _qrls(X, z, tol, cache=qr_cache)
        start = np.zeros(p)
        start[fit["pivot"]] = fit["coefficients"]
        eta = _matvec(X, start)
        mu = eta + 0.0
        dev = _r_sum((y - mu) ** 2)
        if abs(dev - devold) / (0.1 + abs(dev)) < epsilon:
            coef = start
            break
        devold = dev
        coef = start
    rank = fit["rank"]
    pivot = fit["pivot"]
    coef = coef.copy()
    if rank < p:
        coef[pivot[rank:]] = np.nan
    resid = y - mu
    wtdmu = _r_sum(y) / n
    nulldev = _r_sum((y - wtdmu) ** 2)
    rsq = 1.0 - dev / nulldev
    df_r = n - rank
    dispersion = _r_sum(resid**2) / df_r if df_r > 0 else np.nan
    names = np.asarray(colnames, dtype=object)
    if rank > 0:
        cov_unscaled = _chol2inv(fit["qr"][:rank, :rank])
        coef_p = coef[pivot[:rank]]
        if df_r > 0:
            se = np.sqrt(np.diag(cov_unscaled) * dispersion)
            tval = coef_p / se
            pval = pt_two_sided(tval, df_r)
        else:
            se = tval = pval = np.full(rank, np.nan)
        table = {
            "term": names[pivot[:rank]],
            "estimate": coef_p,
            "std_err": se,
            "statistic": tval,
            "pval": pval,
        }
    else:
        table = {k: np.array([]) for k in ("term", "estimate", "std_err", "statistic", "pval")}
    return {"rsq": rsq, "coefs": table, "rank": rank, "df_residual": df_r}


def _r_order_decreasing(p: np.ndarray) -> np.ndarray:
    # R's order(p, decreasing = TRUE) with the default radix method is stable
    return np.argsort(-p, kind="stable")


def p_adjust(p, method: str = "BH") -> np.ndarray:
    """``stats::p.adjust``; NA values are kept and excluded from ``n``."""
    method = {"fdr": "BH"}.get(method, method)
    p = np.asarray(p, dtype=np.float64)
    out = p.copy()
    nna = ~np.isnan(p)
    q = p[nna]
    lp = len(q)
    n = lp
    if n <= 1:
        return out
    if method == "BH":
        i = np.arange(lp, 0, -1, dtype=np.float64)
        o = _r_order_decreasing(q)
        ro = np.argsort(o, kind="stable")
        adj = np.minimum(1.0, np.minimum.accumulate(n / i * q[o]))[ro]
    elif method == "bonferroni":
        adj = np.minimum(1.0, n * q)
    elif method == "holm":
        i = np.arange(1, lp + 1, dtype=np.float64)
        o = np.argsort(q, kind="stable")
        ro = np.argsort(o, kind="stable")
        adj = np.minimum(1.0, np.maximum.accumulate((n + 1 - i) * q[o]))[ro]
    elif method == "hochberg":
        i = np.arange(lp, 0, -1, dtype=np.float64)
        o = _r_order_decreasing(q)
        ro = np.argsort(o, kind="stable")
        adj = np.minimum(1.0, np.minimum.accumulate((n + 1 - i) * q[o]))[ro]
    elif method == "BY":
        i = np.arange(lp, 0, -1, dtype=np.float64)
        o = _r_order_decreasing(q)
        ro = np.argsort(o, kind="stable")
        c = np.sum(1.0 / np.arange(1, n + 1))
        adj = np.minimum(1.0, np.minimum.accumulate(c * n / i * q[o]))[ro]
    elif method == "none":
        adj = q
    else:
        raise ValueError(f"unsupported p.adjust method {method!r}")
    out[nna] = adj
    return out


def r_rank(x: np.ndarray) -> np.ndarray:
    """``rank(x)`` with ``ties.method = "average"``."""
    return stats.rankdata(x, method="average")


def t_test_one(x: np.ndarray, mu: float = 0.0) -> float:
    """``t.test(x)$p.value`` (one sample, two-sided)."""
    x = np.asarray(x, dtype=np.float64)
    x = x[~np.isnan(x)]
    nx = len(x)
    if nx < 2:
        raise ValueError("not enough 'x' observations")
    mx = _r_sum(x) / nx
    vx = _r_sum((x - mx) ** 2) / (nx - 1)
    stderr = math.sqrt(vx / nx)
    if stderr < 10 * np.finfo(float).eps * abs(mx):
        raise ValueError("data are essentially constant")
    tstat = (mx - mu) / stderr
    return float(2.0 * special.stdtr(nx - 1, -abs(tstat)))


def wilcox_test_one(x: np.ndarray, mu: float = 0.0, correct: bool = True, exact=None) -> float:
    """``wilcox.test(x)$p.value`` (one-sample signed rank, two-sided).

    Implemented for the R version of the reference environment; see
    :mod:`pypando._wilcox` (selected at import).
    """
    from ._wilcox import wilcox_signed_rank_pvalue

    return wilcox_signed_rank_pvalue(x, mu=mu, correct=correct, exact=exact)
