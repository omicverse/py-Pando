"""Pando's bundled data sets.

Shipped with the wheel: the motif collection (:func:`pypando.motifs.load_pando_motifs`),
the motif-to-TF table (:func:`pypando.motifs.load_motif2tf`) and the Ensembl v93
GRCh38 annotation used by ``peak_to_gene_method="GREAT"`` (:func:`ensdb_v93_hg38`).

Downloaded on first use (pooch, checksum-pinned), because they are too large for a
wheel: the phastCons 20-mammal conserved elements and the SCREEN cCREs that Pando
offers as ``regions`` for :func:`pypando.initiate_grn`.
"""
from __future__ import annotations

from importlib import resources

import pandas as pd

__all__ = ["ensdb_v93_hg38", "phastcons_hg38", "screen_ccre_hg38"]

_ASSET_BASE = "https://github.com/omicverse/py-Pando/releases/download/data-v1/"
_ASSETS = {
    "phastcons20_hg38.tsv.gz": "sha256:4e7e30bb0d2727a3ff5409fa8e8e1948d4543a3a635b0925ce785c3d8ceb5950",
    "screen_ccre_hg38.tsv.gz": "sha256:13815dd04916f6a31702b943cc49adf047647cef5451c9268191adff059f466d",
}


def _read(name: str) -> pd.DataFrame:
    with resources.files(__name__).joinpath(name).open("rb") as fh:
        return pd.read_csv(fh, sep="\t", compression="gzip", keep_default_na=False, low_memory=False)


def ensdb_v93_hg38() -> pd.DataFrame:
    """``EnsDb.Hsapiens.v93.annot.UCSC.hg38`` (one row per transcript, gene ranges)."""
    df = _read("ensdb_v93_hg38.tsv.gz")
    for c in ("start", "end"):
        df[c] = df[c].astype("int64")
    return df


def _fetch(name: str) -> str:
    import pooch

    return pooch.retrieve(_ASSET_BASE + name, known_hash=_ASSETS[name], progressbar=False)


def phastcons_hg38() -> pd.DataFrame:
    """``phastConsElements20Mammals.UCSC.hg38`` (2,244,053 conserved elements)."""
    return pd.read_csv(_fetch("phastcons20_hg38.tsv.gz"), sep="\t", keep_default_na=False)


def screen_ccre_hg38() -> pd.DataFrame:
    """``SCREEN.ccRE.UCSC.hg38`` (926,535 candidate cis-regulatory elements)."""
    return pd.read_csv(_fetch("screen_ccre_hg38.tsv.gz"), sep="\t", keep_default_na=False)
