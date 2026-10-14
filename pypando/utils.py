"""Small numerical helpers from Pando ``utils.R``."""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

__all__ = ["sparse_cor", "aggregate_matrix", "r_logical_rows"]


def _as_csc(x):
    if sp.issparse(x):
        return sp.csc_matrix(x, dtype=np.float64)
    return sp.csc_matrix(np.asarray(x, dtype=np.float64))


def sparse_cor(x, y=None, remove_na: bool = True, remove_inf: bool = True, allow_neg: bool = True) -> np.ndarray:
    """Pando ``sparse_cor`` (Pearson via ``sparse_covcor``).

    ``x``: cells x k, ``y``: cells x m (or ``None`` for ``cor(x)``). Uses
    Pando's moment formula, so zero-variance columns give ``NaN`` and are set
    to 0 (``remove_na``); infinities become 1 (``remove_inf``).
    """
    X = _as_csc(x)
    n = X.shape[0]
    muX = np.asarray(X.mean(axis=0)).ravel()
    with np.errstate(invalid="ignore", divide="ignore"):
        if y is None:
            cross = (X.T @ X).toarray()
            cov = (cross - n * np.outer(muX, muX)) / (n - 1)
            sd = np.sqrt(np.diag(cov))
            cor = cov / np.outer(sd, sd)
        else:
            Y = _as_csc(y)
            muY = np.asarray(Y.mean(axis=0)).ravel()
            cross = (X.T @ Y).toarray()
            cov = (cross - n * np.outer(muX, muY)) / (n - 1)
            sdx = np.sqrt((np.asarray(X.multiply(X).sum(axis=0)).ravel() - n * muX**2) / (n - 1))
            sdy = np.sqrt((np.asarray(Y.multiply(Y).sum(axis=0)).ravel() - n * muY**2) / (n - 1))
            cor = cov / np.outer(sdx, sdy)
    if remove_na:
        cor[np.isnan(cor)] = 0.0
    if remove_inf:
        cor[np.isinf(cor)] = 1.0
    if not allow_neg:
        cor[cor < 0] = 0.0
    return cor


_SUMMARY = {
    "mean": lambda a: a.mean(axis=0),
    "median": lambda a: np.median(a, axis=0),
    "max": lambda a: a.max(axis=0),
    "min": lambda a: a.min(axis=0),
    "count": lambda a: (a != 0).sum(axis=0),
    "any": lambda a: (a != 0).any(axis=0),
    "all": lambda a: (a != 0).all(axis=0),
    "sd": lambda a: a.std(axis=0, ddof=1),
}


def aggregate_matrix(x, groups, fun: str = "mean") -> pd.DataFrame:
    """Pando ``aggregate_matrix``: summarise rows of ``x`` by ``groups``.

    Returns a DataFrame (groups x columns); groups are sorted as R's
    ``factor`` levels are (C-locale string order).
    """
    groups = np.asarray(groups).astype(str)
    levels = np.array(sorted(set(groups.tolist())))
    dense = x.toarray() if sp.issparse(x) else np.asarray(x)
    out = np.vstack([_SUMMARY[fun](dense[groups == g]) if fun != "sum" else dense[groups == g].sum(axis=0) for g in levels])
    return pd.DataFrame(out, index=levels)


def r_logical_rows(mask: np.ndarray, nrow: int) -> np.ndarray:
    """Row indices selected by R's ``m[mask, ]`` for a logical ``mask``.

    R recycles a shorter logical index over the rows; a longer one that
    selects past the last row yields ``NA`` rows, which Pando cannot use, so
    that case raises.
    """
    mask = np.asarray(mask, dtype=bool)
    L = len(mask)
    if L == nrow:
        return np.flatnonzero(mask)
    if L < nrow:
        if L == 0:
            return np.zeros(0, dtype=np.int64)
        return np.flatnonzero(np.resize(mask, nrow))
    if mask[nrow:].any():
        raise IndexError(
            "motif-to-TF table has fewer rows than motifs scanned: R would index NA rows here; "
            "align `motif_tfs` with `pfm` (same motifs, same order)"
        )
    return np.flatnonzero(mask[:nrow])
