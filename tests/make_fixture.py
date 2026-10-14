"""Build the raw part of the canonical fixture (step 1 of 2).

Reads the public 10x Multiome PBMC 10k filtered matrix, draws 2,000 cells with
``numpy.random.default_rng(42)`` and writes raw counts as Matrix Market files.
Everything after this (normalisation, variable genes, peak window, TF-IDF) is
done once in R by ``tests/r_reference_driver.R`` and the results are shared
with the Python side, so both start from identical numbers.

Usage::

    python tests/make_fixture.py <filtered_feature_bc_matrix.h5> <out_dir>
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import h5py
import numpy as np
import scipy.io
import scipy.sparse as sp

SHA256 = "f6824171378787baab244f559b8b438f79db2eb39f78d17b2196f7ecd2c03549"
N_CELLS = 2000
SEED = 42


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(h5: str, out: str) -> None:
    h5, out = Path(h5), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    digest = _sha256(h5)
    if digest != SHA256:
        raise SystemExit(f"sha256 mismatch for {h5}: {digest}")

    with h5py.File(h5, "r") as f:
        m = f["matrix"]
        X = sp.csc_matrix((m["data"][:], m["indices"][:], m["indptr"][:]), shape=tuple(m["shape"][:]))
        barcodes = np.array([b.decode() for b in m["barcodes"][:]])
        names = np.array([b.decode() for b in m["features/name"][:]])
        types = np.array([b.decode() for b in m["features/feature_type"][:]])
        ids = np.array([b.decode() for b in m["features/id"][:]])

    rng = np.random.default_rng(SEED)
    cells = np.sort(rng.choice(X.shape[1], size=N_CELLS, replace=False))
    X = X[:, cells]
    barcodes = barcodes[cells]

    gex = types == "Gene Expression"
    peaks = types == "Peaks"
    # peak names in the 10x file are "chr1:10109-10357" -> Signac style "chr1-10109-10357"
    peak_names = np.array([p.replace(":", "-") for p in names[peaks]])

    scipy.io.mmwrite(out / "rna_counts.mtx", sp.csc_matrix(X[gex, :]).astype(np.int32))
    scipy.io.mmwrite(out / "peak_counts.mtx", sp.csc_matrix(X[peaks, :]).astype(np.int32))
    (out / "barcodes.txt").write_text("\n".join(barcodes) + "\n")
    (out / "genes.tsv").write_text("".join(f"{i}\t{n}\n" for i, n in zip(ids[gex], names[gex])))
    (out / "peaks.txt").write_text("\n".join(peak_names) + "\n")
    print(f"cells {X.shape[1]}, genes {gex.sum()}, peaks {peaks.sum()} -> {out}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
