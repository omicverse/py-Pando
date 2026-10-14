"""Benchmark ``infer_grn`` on the canonical fixture (omicverse-rebuildr EVALUATION.md).

    python tests/bench_infer_grn.py <ref_dir> <hg38.fa> <method> <variant> <out.json>

One warm-up run is discarded, then three timed runs (``time.perf_counter``);
BLAS is pinned to one thread. ``variant`` selects the implementation state
being measured, so each iteration of ITERATION_LOG.md can be re-timed:

* glm: ``numpy_qr`` (iteration 0, pure-NumPy QR), ``blas_qr`` (iteration 1,
  BLAS-exact QR recomputed every IRLS step), ``blas_qr_cached`` (iteration 2);
* glmnet / cv.glmnet: ``python`` (interpreted coordinate descent) or ``numba``.
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def _prepared(ref, fasta):
    cache = Path(ref) / "bench_grn.pkl"
    if cache.exists():
        return pickle.loads(cache.read_bytes())
    import _run_candidate as rc
    import pypando as pp

    rna, atac, annot, vf, targets = rc.load_inputs(Path(ref))
    grn = pp.find_motifs(pp.initiate_grn(rna, atac, annot, variable_features=vf), genome=fasta, verbose=False)
    cache.write_bytes(pickle.dumps((grn, targets)))
    return grn, targets


def _apply_variant(method, variant):
    from pypando import glmnet as G
    from pypando import rstats

    if method == "glm":
        if variant == "numpy_qr":
            rstats._qrls = lambda X, z, tol, cache=None: rstats.dqrls(X, z, tol)
        elif variant == "blas_qr":
            orig = rstats._qrls
            rstats._qrls = lambda X, z, tol, cache=None: orig(X, z, tol, None)
        elif variant != "blas_qr_cached":
            raise ValueError(variant)
    else:
        if variant == "python":
            G._elnet_path_nb = None
        elif variant != "numba":
            raise ValueError(variant)


def main(ref, fasta, method, variant, out):
    import pypando as pp

    grn, targets = _prepared(ref, fasta)
    _apply_variant(method, variant)
    times = []
    for i in range(4):
        t0 = time.perf_counter()
        pp.infer_grn(grn, genes=targets, method=method, seed=42, verbose=False)
        times.append(time.perf_counter() - t0)
        print(f"{method}/{variant} run {i}: {times[-1]:.1f}s", flush=True)
    runs = times[1:]
    res = {"method": method, "variant": variant, "warmup_run_s": times[0], "wall_clock_runs_s": runs,
           "wall_clock_mean_s": float(np.mean(runs)), "wall_clock_stddev_s": float(np.std(runs, ddof=1))}
    Path(out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res))


if __name__ == "__main__":
    main(*sys.argv[1:6])
