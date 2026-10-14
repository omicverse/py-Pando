"""Python candidate run on the shared fixture written by ``tests/r_reference_driver.R``.

    python tests/_run_candidate.py <ref_dir> <out_dir> <hg38.fa> [methods] [phastcons.tsv.gz]

Reads ``<ref_dir>/shared`` (the exact numbers R used), runs py-Pando and writes
outputs in the same layout the R driver uses, plus ``run.json`` with timings.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp

import pypando as pp
from pypando import ranges as R

SEED = 42


def _load_sparse(shared: Path, stem: str) -> sp.csc_matrix:
    m = scipy.io.mmread(shared / f"{stem}.mtx").tocoo()
    x = np.fromfile(shared / f"{stem}_x.f64", dtype="<f8")
    # writeMM emits entries in column-major (dgCMatrix @x) order
    order = np.lexsort((m.row, m.col))
    rows, cols = m.row[order], m.col[order]
    if len(x) != len(rows):
        raise ValueError("binary dump does not match the Matrix Market entries")
    return sp.csc_matrix((x, (rows, cols)), shape=m.shape)


def load_inputs(ref: Path):
    shared = ref / "shared"
    cells = (shared / "cells.txt").read_text().split()
    genes = (shared / "rna_genes.txt").read_text().split("\n")[:-1]
    peaks = (shared / "peaks.txt").read_text().split()
    rna = ad.AnnData(X=_load_sparse(shared, "rna_data").T.tocsr())
    rna.obs_names = cells
    rna.var_names = genes
    atac = ad.AnnData(X=_load_sparse(shared, "peak_data").T.tocsr())
    atac.obs_names = cells
    atac.var_names = peaks
    annot = pd.read_csv(shared / "annotation.tsv", sep="\t", keep_default_na=False)
    vf = (shared / "variable_features.txt").read_text().split()
    targets = (shared / "targets.txt").read_text().split()
    return rna, atac, annot, vf, targets


def main(ref, out, fasta, methods=None, phastcons=None):
    ref, out = Path(ref), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    methods = (methods or "glm,glmnet,cv.glmnet,xgb,bagging_ridge,bayesian_ridge").split(",")
    rna, atac, annot, vf, targets = load_inputs(ref)
    timing = {}

    t0 = time.perf_counter()
    grn = pp.initiate_grn(rna, atac, annot, variable_features=vf)
    timing["initiate_grn"] = time.perf_counter() - t0
    reg = grn.grn.regions
    pd.DataFrame({"region": R.ranges_to_string(reg.ranges), "peak": reg.peaks + 1}).to_csv(
        out / "candidate_regions.tsv", sep="\t", index=False)

    if phastcons:
        pc = pd.read_csv(phastcons, sep="\t", keep_default_na=False)
        t0 = time.perf_counter()
        g2 = pp.initiate_grn(rna, atac, annot, regions=pc[["seqnames", "start", "end", "strand"]])
        timing["initiate_grn_phastcons"] = time.perf_counter() - t0
        r2 = g2.grn.regions
        pd.DataFrame({"region": R.ranges_to_string(r2.ranges), "peak": r2.peaks + 1}).to_csv(
            out / "candidate_regions_phastcons.tsv", sep="\t", index=False)

    t0 = time.perf_counter()
    grn = pp.find_motifs(grn, genome=fasta, verbose=False)
    timing["find_motifs"] = time.perf_counter() - t0
    reg = grn.grn.regions
    coo = reg.motifs.data.tocoo()
    pd.DataFrame({"i": coo.row + 1, "j": coo.col + 1}).to_csv(out / "motif_matrix_ij.tsv", sep="\t", index=False)
    (out / "motif_matrix_rows.txt").write_text("\n".join(reg.motifs.regions) + "\n")
    (out / "motif_matrix_cols.txt").write_text("\n".join(reg.motifs.motifs) + "\n")
    (out / "tfs_use.txt").write_text("\n".join(reg.tfs.columns) + "\n")

    ga = annot[(annot["type"] != "gap") & annot["gene_name"].isin(set(targets))].reset_index(drop=True)
    for m in ("Signac", "GREAT"):
        t0 = time.perf_counter()
        hits, _, cols = pp.find_peaks_near_genes(reg.ranges, ga, method=m, upstream=100000, downstream=0)
        timing[f"find_peaks_near_genes_{m}"] = time.perf_counter() - t0
        c = hits.tocoo()
        pd.DataFrame({"i": c.row + 1, "j": c.col + 1, "x": c.data}).to_csv(
            out / f"peaks_near_genes_{m}_ij.tsv", sep="\t", index=False)
        (out / f"peaks_near_genes_{m}_cols.txt").write_text("\n".join(cols) + "\n")

    for m in methods:
        print("infer_grn:", m, flush=True)
        t0 = time.perf_counter()
        g = pp.infer_grn(grn, genes=targets, method=m, seed=SEED, verbose=False)
        timing[f"infer_grn_{m}"] = time.perf_counter() - t0
        pp.coef(g).to_csv(out / f"coefs_{m}.tsv", sep="\t", index=False, na_rep="NA", float_format="%.17g")
        pp.gof(g).to_csv(out / f"gof_{m}.tsv", sep="\t", index=False, na_rep="NA", float_format="%.17g")
        if m == "glm":
            t0 = time.perf_counter()
            pp.find_modules(g, verbose=False)
            timing["find_modules"] = time.perf_counter() - t0
            mods = pp.NetworkModules(g)
            mods.meta.to_csv(out / "modules_meta.tsv", sep="\t", index=False, na_rep="NA", float_format="%.17g")
            (out / "modules_features.json").write_text(json.dumps(mods.features))
    (out / "run.json").write_text(json.dumps({"timing": timing, "pypando": pp.__version__}, indent=1))
    print("done")


if __name__ == "__main__":
    main(*sys.argv[1:6])
