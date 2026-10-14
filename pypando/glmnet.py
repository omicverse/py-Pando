"""Gaussian elastic net with glmnet's path rules (``glmnet`` / ``cv.glmnet``).

Pando fits ``glmnetUtils::glmnet(formula, data, family = "gaussian", alpha)``
and ``glmnetUtils::cv.glmnet(...)``. This module follows glmnet's
covariance-update coordinate descent (``elnet1`` in glmnet's Fortran, mirrored
by the C++ ``glmnetpp`` in glmnet >= 4.1) and its defaults:

* weights ``1/n``; ``x`` centred and scaled to unit (population) variance,
  ``y`` centred and scaled (``ys``) — lambdas are reported on ``y``'s scale;
* ``nlambda = 100``, ``lambda.min.ratio = 1e-4`` if ``n >= p`` else ``0.01``;
  ``lambda_1 = +inf`` (all-zero fit), ``lambda_2 = alf * lambda_max``,
  geometric after that; reported ``lambda_1`` is extrapolated as R's
  ``fix.lam`` does;
* convergence ``thresh = 1e-7`` on ``max_j xv_j * delta_j^2``;
* path stops when ``dev.ratio`` gains less than ``fdev = 1e-5`` (relative),
  exceeds ``devmax = 0.999``, after at least ``mnlam = 5`` lambdas.

``dev.ratio`` is accumulated from coordinate updates exactly as glmnet does.
"""
from __future__ import annotations

import math

import numpy as np

from .rrng import RRandom

__all__ = ["glmnet_gaussian", "cv_glmnet_gaussian", "lambda_interp", "predict_s"]

_BIG = 9.9e35
_FDEV = 1e-5
_DEVMAX = 0.999
_MNLAM = 5
_EPS = 1e-6
_THRESH = 1e-7
_MAXIT = 100000


def _standardize(x: np.ndarray, y: np.ndarray):
    n = x.shape[0]
    w = np.full(n, 1.0 / n)
    v = np.sqrt(w)
    ju = np.array([np.any(x[:, j] != x[0, j]) for j in range(x.shape[1])], dtype=bool)
    xm = w @ x
    xs = np.ones(x.shape[1])
    xz = (x - xm) * v[:, None]
    xv = np.einsum("ij,ij->j", xz, xz)
    xs = np.sqrt(xv)
    xs[~ju] = 1.0
    xz = xz / xs
    ym = float(w @ y)
    yz = v * (y - ym)
    ys = math.sqrt(float(yz @ yz))
    yz = yz / ys
    return xz, yz, xm, xs, ym, ys, ju


def _elnet_path(xz, yz, ju, alpha, ulam=None, nlam=100, flmin=1e-4):
    """Coordinate descent over the lambda path on standardised data.

    Returns (a_path [p x L], rsq [L], alm [L] on the standardised scale).
    """
    n, p = xz.shape
    bta = alpha
    omb = 1.0 - bta
    vp = np.ones(p)
    xv = np.ones(p)
    g = xz.T @ yz
    if ulam is not None:
        nlam = len(ulam)
        alf = 1.0
    else:
        eqs = max(_EPS, flmin)
        alf = eqs ** (1.0 / (nlam - 1))
    a = np.zeros(p)
    mm = np.full(p, -1, dtype=np.int64)  # position in the active set
    ia = []
    c = np.zeros((p, 0))
    nin = 0
    rsq = 0.0
    nlp = 0
    iz = 0
    mnl = min(_MNLAM, nlam)
    ao, rsqo, almo = [], [], []
    alm = 0.0
    for m in range(1, nlam + 1):
        if ulam is not None:
            alm = ulam[m - 1]
        elif m > 2:
            alm = alm * alf
        elif m == 1:
            alm = _BIG
        else:
            alm0 = 0.0
            for j in range(p):
                if ju[j] and vp[j] > 0.0:
                    alm0 = max(alm0, abs(g[j]) / vp[j])
            alm = alf * alm0 / max(bta, 1e-3)
        dem = alm * omb
        ab = alm * bta
        rsq0 = rsq
        jz = 1
        while True:
            if not (iz * jz != 0):
                nlp += 1
                dlx = 0.0
                for k in range(p):
                    if not ju[k]:
                        continue
                    ak = a[k]
                    u = g[k] + ak * xv[k]
                    vv = abs(u) - vp[k] * ab
                    a[k] = 0.0
                    if vv > 0.0:
                        a[k] = math.copysign(vv, u) / (xv[k] + vp[k] * dem)
                    if a[k] == ak:
                        continue
                    if mm[k] < 0:
                        nin += 1
                        col = np.empty(p)
                        for j in range(p):
                            if not ju[j]:
                                col[j] = 0.0
                                continue
                            if mm[j] >= 0:
                                col[j] = c[k, mm[j]]
                                continue
                            if j == k:
                                col[j] = xv[j]
                                continue
                            col[j] = float(xz[:, j] @ xz[:, k])
                        c = np.concatenate([c, col[:, None]], axis=1)
                        mm[k] = nin - 1
                        ia.append(k)
                    dl = a[k] - ak
                    rsq += dl * (2.0 * g[k] - dl * xv[k])
                    dlx = max(xv[k] * dl * dl, dlx)
                    jj = ju
                    g[jj] -= c[jj, mm[k]] * dl
                if dlx < _THRESH:
                    break
                if nlp > _MAXIT:
                    raise RuntimeError("glmnet: maximum iterations reached")
            # inner loop on the active set
            iz = 1
            ia_arr = np.asarray(ia, dtype=np.int64)
            da = a[ia_arr].copy()
            while True:
                nlp += 1
                dlx = 0.0
                for l in range(nin):
                    k = ia_arr[l]
                    ak = a[k]
                    u = g[k] + ak * xv[k]
                    vv = abs(u) - vp[k] * ab
                    a[k] = 0.0
                    if vv > 0.0:
                        a[k] = math.copysign(vv, u) / (xv[k] + vp[k] * dem)
                    if a[k] == ak:
                        continue
                    dl = a[k] - ak
                    rsq += dl * (2.0 * g[k] - dl * xv[k])
                    dlx = max(xv[k] * dl * dl, dlx)
                    g[ia_arr] -= c[ia_arr, mm[k]] * dl
                if dlx < _THRESH:
                    break
                if nlp > _MAXIT:
                    raise RuntimeError("glmnet: maximum iterations reached")
            da = a[ia_arr] - da
            for j in range(p):
                if mm[j] >= 0 or not ju[j]:
                    continue
                g[j] -= float(da @ c[j, :nin])
            jz = 0
        ao.append(a.copy())
        rsqo.append(rsq)
        almo.append(alm)
        if m < mnl:
            continue
        if ulam is not None:
            continue
        if rsq - rsq0 < _FDEV * rsq:
            break
        if rsq > _DEVMAX:
            break
    return np.array(ao).T, np.array(rsqo), np.array(almo)


try:
    from numba import njit as _njit
except ImportError:  # pragma: no cover
    _njit = None


def _elnet_path_nb_impl(xz, yz, ju, alpha, ulam, has_ulam, nlam, flmin, big, fdev, devmax, mnlam, thresh, maxit):
    n, p = xz.shape
    bta = alpha
    omb = 1.0 - bta
    g = np.zeros(p)
    for j in range(p):
        s = 0.0
        for i in range(n):
            s += xz[i, j] * yz[i]
        g[j] = s
    if has_ulam:
        nlam = ulam.shape[0]
        alf = 1.0
    else:
        eqs = max(1e-6, flmin)
        alf = eqs ** (1.0 / (nlam - 1))
    a = np.zeros(p)
    mm = np.full(p, -1, dtype=np.int64)
    ia = np.zeros(p, dtype=np.int64)
    c = np.zeros((p, p))
    nin = 0
    rsq = 0.0
    nlp = 0
    iz = 0
    mnl = min(mnlam, nlam)
    ao = np.zeros((p, nlam))
    rsqo = np.zeros(nlam)
    almo = np.zeros(nlam)
    lmu = 0
    alm = 0.0
    da = np.zeros(p)
    for m in range(1, nlam + 1):
        if has_ulam:
            alm = ulam[m - 1]
        elif m > 2:
            alm = alm * alf
        elif m == 1:
            alm = big
        else:
            alm0 = 0.0
            for j in range(p):
                if ju[j]:
                    v = abs(g[j])
                    if v > alm0:
                        alm0 = v
            alm = alf * alm0 / max(bta, 1e-3)
        dem = alm * omb
        ab = alm * bta
        rsq0 = rsq
        jz = 1
        while True:
            if not (iz * jz != 0):
                nlp += 1
                dlx = 0.0
                for k in range(p):
                    if not ju[k]:
                        continue
                    ak = a[k]
                    u = g[k] + ak
                    vv = abs(u) - ab
                    a[k] = 0.0
                    if vv > 0.0:
                        a[k] = (vv if u >= 0 else -vv) / (1.0 + dem)
                    if a[k] == ak:
                        continue
                    if mm[k] < 0:
                        for j in range(p):
                            if not ju[j]:
                                c[j, nin] = 0.0
                            elif mm[j] >= 0:
                                c[j, nin] = c[k, mm[j]]
                            elif j == k:
                                c[j, nin] = 1.0
                            else:
                                s = 0.0
                                for i in range(n):
                                    s += xz[i, j] * xz[i, k]
                                c[j, nin] = s
                        mm[k] = nin
                        ia[nin] = k
                        nin += 1
                    dl = a[k] - ak
                    rsq += dl * (2.0 * g[k] - dl)
                    if dl * dl > dlx:
                        dlx = dl * dl
                    col = mm[k]
                    for j in range(p):
                        if ju[j]:
                            g[j] -= c[j, col] * dl
                if dlx < thresh:
                    break
                if nlp > maxit:
                    return ao, rsqo, almo, -1
            iz = 1
            for l in range(nin):
                da[l] = a[ia[l]]
            while True:
                nlp += 1
                dlx = 0.0
                for l in range(nin):
                    k = ia[l]
                    ak = a[k]
                    u = g[k] + ak
                    vv = abs(u) - ab
                    a[k] = 0.0
                    if vv > 0.0:
                        a[k] = (vv if u >= 0 else -vv) / (1.0 + dem)
                    if a[k] == ak:
                        continue
                    dl = a[k] - ak
                    rsq += dl * (2.0 * g[k] - dl)
                    if dl * dl > dlx:
                        dlx = dl * dl
                    col = mm[k]
                    for jj in range(nin):
                        j = ia[jj]
                        g[j] -= c[j, col] * dl
                if dlx < thresh:
                    break
                if nlp > maxit:
                    return ao, rsqo, almo, -1
            for l in range(nin):
                da[l] = a[ia[l]] - da[l]
            for j in range(p):
                if mm[j] >= 0 or not ju[j]:
                    continue
                s = 0.0
                for l in range(nin):
                    s += da[l] * c[j, l]
                g[j] -= s
            jz = 0
        for j in range(p):
            ao[j, m - 1] = a[j]
        rsqo[m - 1] = rsq
        almo[m - 1] = alm
        lmu = m
        if m < mnl:
            continue
        if has_ulam:
            continue
        if rsq - rsq0 < fdev * rsq:
            break
        if rsq > devmax:
            break
    return ao, rsqo, almo, lmu


_elnet_path_nb = _njit(cache=True)(_elnet_path_nb_impl) if _njit is not None else None


def _elnet_path_fast(xz, yz, ju, alpha, ulam=None, nlam=100, flmin=1e-4):
    """Compiled version of :func:`_elnet_path` (same updates, same order)."""
    has = ulam is not None
    ul = np.asarray(ulam, dtype=np.float64) if has else np.zeros(1)
    ao, rsqo, almo, lmu = _elnet_path_nb(np.asfortranarray(xz), np.ascontiguousarray(yz), ju.astype(np.bool_),
                                         float(alpha), ul, has, int(nlam), float(flmin), _BIG, _FDEV,
                                         _DEVMAX, _MNLAM, _THRESH, _MAXIT)
    if lmu < 0:
        raise RuntimeError("glmnet: maximum iterations reached")
    return ao[:, :lmu], rsqo[:lmu], almo[:lmu]


def glmnet_gaussian(x: np.ndarray, y: np.ndarray, alpha: float = 1.0, lambda_=None):
    """``glmnet(x, y, family = "gaussian", alpha = alpha)``.

    Returns a dict with ``lambda``, ``dev_ratio``, ``a0`` and ``beta``
    (``p x L``, original scale), like the R object.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n, p = x.shape
    xz, yz, xm, xs, ym, ys, ju = _standardize(x, y)
    flmin = 0.01 if n < p else 1e-4
    ulam = None if lambda_ is None else np.asarray(lambda_, dtype=np.float64) / ys
    path = _elnet_path_fast if _elnet_path_nb is not None else _elnet_path
    a, rsq, alm = path(xz, yz, ju, alpha, ulam=ulam, flmin=flmin)
    beta = a * ys / xs[:, None]
    beta[~ju, :] = 0.0
    a0 = ym - xm @ beta
    lam = alm * ys
    if lambda_ is None and len(lam) >= 3:
        lam = lam.copy()
        lam[0] = math.exp(2.0 * math.log(lam[1]) - math.log(lam[2]))
    return {"lambda": lam, "dev_ratio": rsq, "a0": a0, "beta": beta}


def _predict(fit, x):
    return fit["a0"][None, :] + x @ fit["beta"]


def lambda_interp(lam: np.ndarray, s: np.ndarray):
    """glmnet's ``lambda.interp``: left/right path indices (0-based) and weights."""
    lam = np.asarray(lam, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    if len(lam) == 1:
        z = np.zeros(len(s), dtype=np.int64)
        return z, z, np.ones(len(s))
    k = len(lam)
    sfrac = (lam[0] - s) / (lam[0] - lam[k - 1])
    lnorm = (lam[0] - lam) / (lam[0] - lam[k - 1])
    sfrac = np.clip(sfrac, lnorm.min(), lnorm.max())
    coord = np.interp(sfrac, lnorm, np.arange(1, k + 1, dtype=np.float64))  # approx(x, seq(x), xout)
    left = np.floor(coord).astype(np.int64) - 1
    right = np.ceil(coord).astype(np.int64) - 1
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = (sfrac - lnorm[right]) / (lnorm[left] - lnorm[right])
    frac[left == right] = 1.0
    frac[np.abs(lnorm[left] - lnorm[right]) < np.finfo(float).eps] = 1.0
    return left, right, frac


def predict_s(fit, x, s):
    """``predict(glmnet_fit, x, s = s)``: coefficients interpolated along the path."""
    left, right, frac = lambda_interp(fit["lambda"], s)
    a0 = fit["a0"][left] * frac + fit["a0"][right] * (1 - frac)
    beta = fit["beta"][:, left] * frac + fit["beta"][:, right] * (1 - frac)
    return a0[None, :] + x @ beta


def draw_folds(rng: RRandom, n: int, nfolds: int = 10) -> np.ndarray:
    """``sample(rep(seq(nfolds), length = n))`` from R's stream."""
    return rng.sample(np.resize(np.arange(1, nfolds + 1), n))


def cv_glmnet_gaussian(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float = 1.0,
    nfolds: int = 10,
    rng: RRandom | None = None,
    foldid=None,
    fold_lambda: str = "own",
):
    """``cv.glmnet(x, y, family = "gaussian", alpha)`` with ``type.measure = "mse"``.

    Folds are ``sample(rep(seq(nfolds), length = N))`` drawn from ``rng`` (an
    :class:`~pypando.rrng.RRandom`, i.e. R's stream), unless ``foldid`` is given.
    Returns the full fit plus ``cvm``, ``cvsd``, ``lambda_min``, ``lambda_1se``
    and ``index`` (0-based ``min`` / ``1se``).

    ``fold_lambda="own"`` follows glmnet 5.0: every fold is fitted on its own
    lambda path and its predictions are interpolated onto the full-data
    lambdas (``predict(fit, s = lambda)``). ``"full"`` follows glmnet 4.x:
    folds are refitted on the full-data lambda sequence.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = x.shape[0]
    # cv.glmnet draws the folds before its first glmnet fit (which may then fail)
    if foldid is None:
        if rng is None:
            raise ValueError("cv_glmnet_gaussian needs `rng` or `foldid`")
        foldid = draw_folds(rng, n, nfolds)
    foldid = np.asarray(foldid)
    full = glmnet_gaussian(x, y, alpha=alpha)
    lam = full["lambda"]
    nf = int(foldid.max())
    L = len(lam)
    predmat = np.full((n, L), np.nan)
    for i in range(1, nf + 1):
        test = foldid == i
        if fold_lambda == "own":
            fit = glmnet_gaussian(x[~test], y[~test], alpha=alpha)
            preds = predict_s(fit, x[test], lam)
        else:
            fit = glmnet_gaussian(x[~test], y[~test], alpha=alpha, lambda_=lam)
            preds = _predict(fit, x[test])
        nlami = min(preds.shape[1], L)
        predmat[test, :nlami] = preds
        if nlami < L:
            predmat[test, nlami - 1 :] = preds[:, [nlami - 1]]
    cvraw = (y[:, None] - predmat) ** 2
    outmat = np.full((nf, L), np.nan)
    wisum = np.zeros(nf)
    for i in range(1, nf + 1):
        sel = foldid == i
        outmat[i - 1] = cvraw[sel].mean(axis=0)
        wisum[i - 1] = sel.sum() / n * n  # weights are 1 per observation
    N = np.full(L, nf)
    cvm = (outmat * wisum[:, None]).sum(axis=0) / wisum.sum()
    cvsd = np.sqrt(((outmat - cvm) ** 2 * wisum[:, None]).sum(axis=0) / wisum.sum() / (N - 1))
    cvmin = np.nanmin(cvm)
    idmin_mask = cvm <= cvmin
    lambda_min = np.max(lam[idmin_mask])
    idmin = int(np.flatnonzero(lam == lambda_min)[0])
    semin = (cvm + cvsd)[idmin]
    id1se_mask = cvm <= semin
    lambda_1se = np.max(lam[id1se_mask])
    id1se = int(np.flatnonzero(lam == lambda_1se)[0])
    return {
        "glmnet_fit": full,
        "lambda": lam,
        "cvm": cvm,
        "cvsd": cvsd,
        "lambda_min": lambda_min,
        "lambda_1se": lambda_1se,
        "index": {"min": idmin, "1se": id1se},
        "foldid": foldid,
    }
