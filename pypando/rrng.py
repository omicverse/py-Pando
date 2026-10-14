"""R's default random number stream, for the steps where Pando draws numbers in R.

Only ``cv.glmnet`` draws random numbers on the R side (the fold assignment
``sample(rep(seq(nfolds), length = N))``). To reproduce R's folds exactly,
this module emulates

* ``set.seed(seed)`` with the default ``RNGkind`` ("Mersenne-Twister",
  "Inversion", "Rejection"), following ``RNG_Init`` / ``FixupSeeds`` in
  R's ``src/main/RNG.c``;
* ``unif_rand()`` (MT19937 with R's ``fixup`` to the open interval);
* ``R_unif_index()`` with rejection sampling (R >= 3.6.0);
* ``sample(x)`` without replacement (``do_sample`` in ``src/main/random.c``).
"""
from __future__ import annotations

import math

import numpy as np

__all__ = ["RRandom"]

_N, _M = 624, 397
_MATRIX_A = 0x9908B0DF
_UPPER = 0x80000000
_LOWER = 0x7FFFFFFF
_I2_32M1 = 2.328306437080797e-10  # 1 / (2^32 - 1)
_MASK32 = 0xFFFFFFFF


class RRandom:
    """A stream equal to R's after ``set.seed(seed)``."""

    def __init__(self, seed: int):
        s = seed & _MASK32
        for _ in range(50):  # initial scrambling
            s = (69069 * s + 1) & _MASK32
        dummy = []
        for _ in range(_N + 1):  # i_seed[0] is mti
            s = (69069 * s + 1) & _MASK32
            dummy.append(s)
        self._mt = dummy[1:]
        self._mti = _N  # FixupSeeds(initial=TRUE): dummy[0] = 624

    def _genrand(self) -> float:
        mt = self._mt
        if self._mti >= _N:
            for kk in range(_N - _M):
                y = (mt[kk] & _UPPER) | (mt[kk + 1] & _LOWER)
                mt[kk] = mt[kk + _M] ^ (y >> 1) ^ (_MATRIX_A if y & 1 else 0)
            for kk in range(_N - _M, _N - 1):
                y = (mt[kk] & _UPPER) | (mt[kk + 1] & _LOWER)
                mt[kk] = mt[kk + (_M - _N)] ^ (y >> 1) ^ (_MATRIX_A if y & 1 else 0)
            y = (mt[_N - 1] & _UPPER) | (mt[0] & _LOWER)
            mt[_N - 1] = mt[_M - 1] ^ (y >> 1) ^ (_MATRIX_A if y & 1 else 0)
            self._mti = 0
        y = mt[self._mti]
        self._mti += 1
        y ^= y >> 11
        y ^= (y << 7) & 0x9D2C5680
        y ^= (y << 15) & 0xEFC60000
        y ^= y >> 18
        return (y & _MASK32) * 2.3283064365386963e-10

    def unif_rand(self) -> float:
        x = self._genrand()
        if x <= 0.0:
            return 0.5 * _I2_32M1
        if 1.0 - x <= 0.0:
            return 1.0 - 0.5 * _I2_32M1
        return x

    def _rbits(self, bits: int) -> float:
        v = 0
        n = 0
        while n <= bits:
            v1 = int(math.floor(self.unif_rand() * 65536))
            v = (65536 * v + v1) & 0xFFFFFFFFFFFFFFFF
            n += 16
        return float(v & ((1 << bits) - 1))

    def unif_index(self, dn: float) -> float:
        """``R_unif_index(dn)`` with ``sample.kind = "Rejection"``."""
        if dn <= 0:
            return 0.0
        bits = int(math.ceil(math.log2(dn)))
        while True:
            dv = self._rbits(bits)
            if dn > dv:
                return dv

    def sample_int(self, n: int, size: int | None = None) -> np.ndarray:
        """``sample.int(n, size)`` without replacement, 1-based like R."""
        k = n if size is None else size
        if k < 2:
            return np.array([int(self.unif_index(n)) + 1 for _ in range(k)], dtype=np.int64)
        x = list(range(n))
        out = np.empty(k, dtype=np.int64)
        m = n
        for i in range(k):
            j = int(self.unif_index(m))
            out[i] = x[j] + 1
            m -= 1
            x[j] = x[m]
        return out

    def sample(self, x) -> np.ndarray:
        """``sample(x)`` for a vector of length > 1 (a permutation of ``x``)."""
        x = np.asarray(x)
        return x[self.sample_int(len(x)) - 1]
