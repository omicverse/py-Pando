"""R's QR (``dqrdc2`` / ``dqrls``) compiled with numba, calling the BLAS.

R's Fortran routines call the BLAS level-1 kernels ``dnrm2``, ``ddot``,
``daxpy`` and ``dscal``. Floating-point results of those kernels depend on
their summation order, so for decisions that sit on a knife edge (a column
is "negligible" if its residual norm falls below ``1e-11`` times its
original norm) the only way to reproduce R is to make the same calls in the
same order. This module does that through SciPy's BLAS (the library NumPy
and SciPy are linked to); with the same BLAS build as R's, the QR is
bit-identical to R's.
"""
from __future__ import annotations

import ctypes

import numpy as np

__all__ = ["dqrdc2_blas", "dqrls_blas", "available"]

try:
    import numba
    from numba import njit
    import scipy.linalg.cython_blas as _cb

    def _fn(name, restype, *argtypes):
        cap = _cb.__pyx_capi__[name]
        ctypes.pythonapi.PyCapsule_GetName.restype = ctypes.c_char_p
        ctypes.pythonapi.PyCapsule_GetName.argtypes = [ctypes.py_object]
        ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.c_void_p
        ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
        ptr = ctypes.pythonapi.PyCapsule_GetPointer(cap, ctypes.pythonapi.PyCapsule_GetName(cap))
        return ctypes.CFUNCTYPE(restype, *argtypes)(ptr)

    _P = ctypes.c_void_p
    _dnrm2 = _fn("dnrm2", ctypes.c_double, _P, _P, _P)
    _ddot = _fn("ddot", ctypes.c_double, _P, _P, _P, _P, _P)
    _daxpy = _fn("daxpy", None, _P, _P, _P, _P, _P, _P)
    _dscal = _fn("dscal", None, _P, _P, _P, _P)
    available = True
except Exception:  # pragma: no cover - numba or the capsules missing
    available = False


if available:

    @njit(cache=False)
    def _dqrdc2(x, tol, jpvt, qraux, work, ib, db):
        # x: Fortran-ordered (n, p). ib: int32 scratch[4], db: float64 scratch[2]
        n, p = x.shape
        xp = x.ctypes.data
        ld = n
        step = 8  # sizeof(double)
        one = ib.ctypes.data  # ib[0] = 1 (incx)
        nn = ib.ctypes.data + 4  # ib[1] = length
        ib[0] = 1
        if n > 0:
            for j in range(p):
                ib[1] = n
                v = _dnrm2(nn, xp + (j * ld) * step, one)
                qraux[j] = v
                work[j, 0] = v
                work[j, 1] = v if v != 0.0 else 1.0
        lup = min(n, p)
        k = p + 1
        for l in range(lup):
            while not ((l + 1) >= k or qraux[l] >= work[l, 1] * tol):
                for i in range(n):
                    t = x[i, l]
                    for j in range(l + 1, p):
                        x[i, j - 1] = x[i, j]
                    x[i, p - 1] = t
                i0 = jpvt[l]
                t = qraux[l]
                tt = work[l, 0]
                ttt = work[l, 1]
                for j in range(l + 1, p):
                    jpvt[j - 1] = jpvt[j]
                    qraux[j - 1] = qraux[j]
                    work[j - 1, 0] = work[j, 0]
                    work[j - 1, 1] = work[j, 1]
                jpvt[p - 1] = i0
                qraux[p - 1] = t
                work[p - 1, 0] = tt
                work[p - 1, 1] = ttt
                k -= 1
            if (l + 1) != n:
                m = n - l
                ib[1] = m
                col_l = xp + (l * ld + l) * step
                nrmxl = _dnrm2(nn, col_l, one)
                if nrmxl != 0.0:
                    if x[l, l] != 0.0:
                        nrmxl = abs(nrmxl) if x[l, l] >= 0 else -abs(nrmxl)
                    db[0] = 1.0 / nrmxl
                    _dscal(nn, db.ctypes.data, col_l, one)
                    x[l, l] = 1.0 + x[l, l]
                    for j in range(l + 1, p):
                        col_j = xp + (j * ld + l) * step
                        t = -_ddot(nn, col_l, one, col_j, one) / x[l, l]
                        db[1] = t
                        _daxpy(nn, db.ctypes.data + 8, col_l, one, col_j, one)
                        if qraux[j] != 0.0:
                            r = abs(x[l, j]) / qraux[j]
                            tt = 1.0 - r * r
                            if tt < 0.0:
                                tt = 0.0
                            t = tt
                            if abs(t) >= 1e-6:
                                qraux[j] = qraux[j] * np.sqrt(t)
                            else:
                                ib[2] = n - l - 1
                                qraux[j] = _dnrm2(ib.ctypes.data + 8, xp + (j * ld + l + 1) * step, one)
                                work[j, 0] = qraux[j]
                    qraux[l] = x[l, l]
                    x[l, l] = -nrmxl
        return min(k - 1, n)

    @njit(cache=False)
    def _dqrsl_b(x, k, qraux, y, b, ib, db):
        # qty (job digit b) then b (job digit c), as dqrsl with job 1110
        n = x.shape[0]
        xp = x.ctypes.data
        step = 8
        ib[0] = 1
        one = ib.ctypes.data
        nn = ib.ctypes.data + 4
        qty = y.copy()
        qp = qty.ctypes.data
        ju = min(k, n - 1)
        for j in range(ju):
            if qraux[j] != 0.0:
                temp = x[j, j]
                x[j, j] = qraux[j]
                ib[1] = n - j
                col = xp + (j * n + j) * step
                t = -_ddot(nn, col, one, qp + j * step, one) / x[j, j]
                db[1] = t
                _daxpy(nn, db.ctypes.data + 8, col, one, qp + j * step, one)
                x[j, j] = temp
        for i in range(k):
            b[i] = qty[i]
        bp = b.ctypes.data
        for jj in range(k):
            j = k - jj - 1
            if x[j, j] == 0.0:
                return j + 1  # info
            b[j] = b[j] / x[j, j]
            if j > 0:
                db[1] = -b[j]
                ib[1] = j
                _daxpy(nn, db.ctypes.data + 8, xp + (j * n) * step, one, bp, one)
        return 0


def dqrdc2_blas(x: np.ndarray, tol: float):
    """Same contract as :func:`pypando.rstats.dqrdc2`, BLAS-exact."""
    X = np.array(x, dtype=np.float64, order="F", copy=True)
    p = X.shape[1]
    jpvt = np.arange(p, dtype=np.int64)
    qraux = np.zeros(p)
    work = np.zeros((p, 2))
    ib = np.zeros(4, dtype=np.int32)
    db = np.zeros(2)
    k = _dqrdc2(X, float(tol), jpvt, qraux, work, ib, db)
    return X, int(k), qraux, jpvt


def dqrls_blas(x: np.ndarray, y: np.ndarray, tol: float):
    """Same contract as :func:`pypando.rstats.dqrls`, BLAS-exact."""
    qr, k, qraux, pivot = dqrdc2_blas(x, tol)
    p = qr.shape[1]
    b = np.zeros(p)
    if k > 0:
        ib = np.zeros(4, dtype=np.int32)
        db = np.zeros(2)
        _dqrsl_b(qr, k, qraux, np.ascontiguousarray(y, dtype=np.float64), b, ib, db)
    return {"coefficients": b, "rank": k, "pivot": pivot, "qr": qr, "qraux": qraux}
