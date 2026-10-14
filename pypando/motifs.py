"""Motif scanning as Pando does it (motifmatchr + Pando's patched background).

``find_motifs`` in Pando converts each position frequency matrix with
``TFBSTools::toPWM`` (pseudocount 0.8, log2 probability ratio against the
matrix's own background), re-expresses it against the scanning background
(``convert_pwm_fixed`` in ``motifmatchr_patched.R``), and scans every
candidate region with MOODS (``get_motif_ix`` in motifmatchr): a threshold
from ``MOODS::tools::threshold_from_p`` per motif, the same threshold for
the reverse complement, window ``w = 7``. A region matches a motif if
either strand has a hit.

The same MOODS C++ library is used here through MOODS-python.
"""
from __future__ import annotations

import gzip
import json
from importlib import resources
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp

__all__ = [
    "PFM",
    "to_pwm",
    "convert_pwm",
    "nucleotide_frequencies",
    "scan_motifs",
    "load_pando_motifs",
    "load_motif2tf",
]

BASES = ("A", "C", "G", "T")


class PFM:
    """A position frequency matrix (``TFBSTools::PFMatrix``): counts 4 x width."""

    def __init__(self, ID: str, counts, name: str | None = None, bg=None):
        self.ID = ID
        self.name = name if name is not None else ID
        self.counts = np.asarray(counts, dtype=np.int64)
        if self.counts.shape[0] != 4:
            raise ValueError("counts must be 4 x width (rows A, C, G, T)")
        self.bg = np.asarray(bg if bg is not None else [0.25, 0.25, 0.25, 0.25], dtype=np.float64)

    def __repr__(self):
        return f"PFM({self.ID!r}, width={self.counts.shape[1]})"


def to_pwm(pfm: PFM, pseudocounts: float = 0.8) -> np.ndarray:
    """``TFBSTools::toPWM(pfm)`` (type ``log2probratio``, bg = ``bg(pfm)``)."""
    x = pfm.counts.astype(np.float64)
    bg = pfm.bg
    nseq = x.sum(axis=0)
    prior_n = bg.sum()
    pc = np.zeros(x.shape[1]) + pseudocounts
    p = (x + np.outer(bg, pc)) / (nseq + prior_n * pc)[None, :]
    prior_probs = bg / prior_n
    return np.log2(p / prior_probs[:, None])


def convert_pwm(pwm: np.ndarray, pwm_bg: np.ndarray, bg_freqs: np.ndarray) -> np.ndarray:
    """Pando's ``convert_pwm_fixed`` for a log2 PWM: ``mat + log2(B) - log2(bg)``."""
    return pwm + np.log2(pwm_bg)[:, None] - np.log2(bg_freqs)[:, None]


def nucleotide_frequencies(seqs: Sequence[str]) -> np.ndarray:
    """``Biostrings::letterFrequency(seqs, "ACGT")`` summed and normalised."""
    counts = np.zeros(4)
    for s in seqs:
        for i, b in enumerate(BASES):
            counts[i] += s.count(b)
    return counts / counts.sum()


def scan_motifs(
    pfms: Sequence[PFM],
    seqs: Sequence[str],
    bg: str | Sequence[float] = "subject",
    p_cutoff: float = 5e-05,
    w: int = 7,
    genome_freqs=None,
) -> sp.csr_matrix:
    """Region x motif match matrix (bool), as motifmatchr's ``out = "matches"``.

    ``seqs`` must be upper-case (``getSeq`` on a BSgenome returns upper case).
    ``bg``: ``"subject"`` (composition of ``seqs``), ``"even"``, ``"genome"``
    (pass ``genome_freqs``) or four frequencies A, C, G, T.
    """
    import MOODS.scan
    import MOODS.tools

    if isinstance(bg, str):
        if bg == "subject":
            bg_freqs = nucleotide_frequencies(seqs)
        elif bg == "even":
            bg_freqs = np.array([0.25, 0.25, 0.25, 0.25])
        elif bg == "genome":
            if genome_freqs is None:
                raise ValueError("bg='genome' needs genome_freqs")
            bg_freqs = np.asarray(genome_freqs, dtype=np.float64)
        else:
            raise ValueError("bg must be 'subject', 'genome', 'even' or 4 frequencies")
    else:
        bg_freqs = np.asarray(bg, dtype=np.float64)
        if len(bg_freqs) != 4 or not np.isclose(bg_freqs.sum(), 1) or bg_freqs.min() <= 0:
            raise ValueError("Invalid background frequencies")
    bgl = [float(v) for v in bg_freqs]
    mats, rcs, thr = [], [], []
    for pfm in pfms:
        m = convert_pwm(to_pwm(pfm), pfm.bg, bg_freqs)
        mat = [list(map(float, row)) for row in m]
        mats.append(mat)
        rcs.append(MOODS.tools.reverse_complement(mat))
        thr.append(MOODS.tools.threshold_from_p(mat, bgl, p_cutoff))
    n = len(mats)
    scanner = MOODS.scan.Scanner(w)
    scanner.set_motifs(mats + rcs, bgl, thr + thr)
    rows, cols = [], []
    for i, s in enumerate(seqs):
        res = scanner.scan(s)
        for j in range(n):
            if len(res[j]) > 0 or len(res[j + n]) > 0:
                rows.append(i)
                cols.append(j)
    data = np.ones(len(rows), dtype=bool)
    return sp.csr_matrix((data, (rows, cols)), shape=(len(seqs), n), dtype=bool)


def _data_path(name: str):
    return resources.files("pypando").joinpath("data").joinpath(name)


def load_pando_motifs() -> list:
    """Pando's bundled ``motifs`` (a ``PFMatrixList``), as :class:`PFM` objects."""
    with gzip.open(_data_path("motifs.json.gz"), "rt") as fh:
        raw = json.load(fh)
    return [PFM(m["ID"], m["counts"], name=m.get("name"), bg=m.get("bg")) for m in raw]


def load_motif2tf() -> pd.DataFrame:
    """Pando's bundled ``motif2tf`` table (motif, tf, origin, gene_id, family, name, symbol, motif_tf)."""
    return pd.read_csv(_data_path("motif2tf.tsv.gz"), sep="\t", keep_default_na=False)
