"""The GenomicRanges / IRanges / Signac range operations Pando uses.

Ranges are pandas DataFrames with columns ``seqnames`` (str), ``start`` and
``end`` (int, 1-based, closed, as in GRanges) and ``strand`` (``"+"``, ``"-"``
or ``"*"``). Extra columns (``gene_name``, ``type``, ...) are carried along.

Only what Pando calls is implemented, with Bioconductor's semantics:

========================  ==============================================
R                         here
========================  ==============================================
Signac::StringToGRanges   :func:`string_to_ranges`
Signac::GRangesToString   :func:`ranges_to_string`
Signac::Extend            :func:`extend`
IRanges::resize(fix=st.)  :func:`resize_start`
IRanges::findOverlaps     :func:`find_overlaps` (type="any", select="all")
IRanges::pintersect       :func:`pintersect`
GenomicRanges::reduce     :func:`reduce`   (via ``intersect(x, x)``)
GenomicRanges::subtract   :func:`subtract` (``ignore.strand=TRUE``)
========================  ==============================================
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "make_ranges",
    "string_to_ranges",
    "ranges_to_string",
    "extend",
    "resize_start",
    "find_overlaps",
    "pintersect",
    "reduce",
    "subtract",
]


def make_ranges(seqnames, start, end, strand="*", **extra) -> pd.DataFrame:
    """Build a ranges table; ``strand`` may be a scalar or a vector."""
    n = len(seqnames)
    df = pd.DataFrame(
        {
            "seqnames": np.asarray(seqnames, dtype=object).astype(str),
            "start": np.asarray(start, dtype=np.int64),
            "end": np.asarray(end, dtype=np.int64),
            "strand": np.broadcast_to(np.asarray(strand, dtype=object), (n,)).astype(str),
        }
    )
    for k, v in extra.items():
        df[k] = v
    return df


def string_to_ranges(regions: Iterable[str], sep: Sequence[str] = ("-", "-")) -> pd.DataFrame:
    """``Signac::StringToGRanges``: ``"chr1-100-200"`` -> chr1:100-200, strand ``*``.

    Signac splits on every occurrence of either separator (``tidyr::separate``
    with ``sep = "-|-"``) and keeps the first three pieces.
    """
    regions = list(regions)
    import re

    pattern = re.escape(sep[0]) + "|" + re.escape(sep[1])
    parts = [re.split(pattern, r) for r in regions]
    chrom = [p[0] for p in parts]
    start = [int(p[1]) for p in parts]
    end = [int(p[2]) for p in parts]
    return make_ranges(chrom, start, end, "*")


def ranges_to_string(ranges: pd.DataFrame, sep: Sequence[str] = ("-", "-")) -> np.ndarray:
    """``Signac::GRangesToString``."""
    return (
        ranges["seqnames"].astype(str)
        + sep[0]
        + ranges["start"].astype(np.int64).astype(str)
        + sep[1]
        + ranges["end"].astype(np.int64).astype(str)
    ).to_numpy(dtype=object)


def _on_plus(ranges: pd.DataFrame) -> np.ndarray:
    s = ranges["strand"].to_numpy()
    return (s == "+") | (s == "*")


def extend(ranges: pd.DataFrame, upstream: int = 0, downstream: int = 0) -> pd.DataFrame:
    """``Signac::Extend`` (``from.midpoint = FALSE``); ``*`` is treated as ``+``.

    Signac then calls ``trim()``. Trimming only clips ranges at the sequence
    ends, which never changes which peaks overlap, so it is not reproduced
    (the clipped coordinates are not used anywhere downstream in Pando).
    """
    out = ranges.copy()
    plus = _on_plus(ranges)
    start = ranges["start"].to_numpy(np.int64)
    end = ranges["end"].to_numpy(np.int64)
    out["start"] = start - np.where(plus, upstream, downstream)
    out["end"] = end + np.where(plus, downstream, upstream)
    return out


def resize_start(ranges: pd.DataFrame, width: int = 1) -> pd.DataFrame:
    """``IRanges::resize(x, width, fix = "start")`` (strand-aware; ``*`` as ``+``)."""
    out = ranges.copy()
    plus = _on_plus(ranges)
    start = ranges["start"].to_numpy(np.int64)
    end = ranges["end"].to_numpy(np.int64)
    out["start"] = np.where(plus, start, end - width + 1)
    out["end"] = np.where(plus, start + width - 1, end)
    return out


def _strand_compatible(sq: np.ndarray, ss: np.ndarray) -> np.ndarray:
    return (sq == "*") | (ss == "*") | (sq == ss)


def find_overlaps(
    query: pd.DataFrame,
    subject: pd.DataFrame,
    minoverlap: int = 1,
    ignore_strand: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """``findOverlaps(query, subject, type = "any", select = "all")``.

    Returns ``(query_hits, subject_hits)`` (0-based), sorted by query then
    subject, as Bioconductor's ``Hits`` object is. Two closed ranges overlap
    when they share at least ``minoverlap`` positions, are on the same sequence
    and have compatible strands (``*`` matches anything).
    """
    qh_all, sh_all = [], []
    qchr = query["seqnames"].to_numpy()
    schr = subject["seqnames"].to_numpy()
    qs, qe = query["start"].to_numpy(np.int64), query["end"].to_numpy(np.int64)
    ss, se = subject["start"].to_numpy(np.int64), subject["end"].to_numpy(np.int64)
    qstr, sstr = query["strand"].to_numpy(), subject["strand"].to_numpy()
    for chrom in np.unique(qchr):
        qi = np.flatnonzero(qchr == chrom)
        si = np.flatnonzero(schr == chrom)
        if len(qi) == 0 or len(si) == 0:
            continue
        # sort subject by start; a subject can overlap query q only if
        # s_start <= q_end - minoverlap + 1
        order = si[np.argsort(ss[si], kind="stable")]
        s_start_sorted = ss[order]
        s_end_sorted = se[order]
        max_width = int((se[si] - ss[si]).max()) + 1
        for q in qi:
            hi = np.searchsorted(s_start_sorted, qe[q] - minoverlap + 1, side="right")
            # subjects starting before q_start - max_width cannot reach q
            lo = np.searchsorted(s_start_sorted, qs[q] - max_width, side="left")
            cand = order[lo:hi]
            if len(cand) == 0:
                continue
            ov = np.minimum(qe[q], se[cand]) - np.maximum(qs[q], ss[cand]) + 1
            keep = ov >= minoverlap
            if not ignore_strand:
                keep &= _strand_compatible(np.full(len(cand), qstr[q]), sstr[cand])
            hit = np.sort(cand[keep])
            if len(hit):
                qh_all.append(np.full(len(hit), q, dtype=np.int64))
                sh_all.append(hit.astype(np.int64))
    if not qh_all:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    qh = np.concatenate(qh_all)
    sh = np.concatenate(sh_all)
    o = np.lexsort((sh, qh))
    return qh[o], sh[o]


def pintersect(x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    """``pintersect(x, y)`` for parallel, overlapping ranges (strand from ``x``)."""
    out = x.reset_index(drop=True).copy()
    out["start"] = np.maximum(x["start"].to_numpy(np.int64), y["start"].to_numpy(np.int64))
    out["end"] = np.minimum(x["end"].to_numpy(np.int64), y["end"].to_numpy(np.int64))
    return out


def reduce(ranges: pd.DataFrame, ignore_strand: bool = False) -> pd.DataFrame:
    """``GenomicRanges::reduce``: merge overlapping *and adjacent* ranges.

    Output is sorted by sequence (order of first appearance in ``seqlevels``,
    i.e. the input order of sequences), strand (``+``, ``-``, ``*``) and start.
    Extra columns are dropped.
    """
    if len(ranges) == 0:
        return make_ranges([], [], [], "*")
    seq_levels = list(dict.fromkeys(ranges["seqnames"].tolist()))
    strand_levels = ["+", "-", "*"]
    rows = []
    strands = ranges["strand"].to_numpy() if not ignore_strand else np.full(len(ranges), "*", dtype=object)
    df = pd.DataFrame(
        {
            "seqnames": ranges["seqnames"].to_numpy(),
            "strand": strands,
            "start": ranges["start"].to_numpy(np.int64),
            "end": ranges["end"].to_numpy(np.int64),
        }
    )
    for chrom in seq_levels:
        for st in strand_levels:
            sub = df[(df["seqnames"] == chrom) & (df["strand"] == st)]
            if len(sub) == 0:
                continue
            o = np.lexsort((sub["end"].to_numpy(), sub["start"].to_numpy()))
            s = sub["start"].to_numpy()[o]
            e = sub["end"].to_numpy()[o]
            cs, ce = s[0], e[0]
            for a, b in zip(s[1:], e[1:]):
                if a <= ce + 1:  # overlapping or adjacent
                    ce = max(ce, b)
                else:
                    rows.append((chrom, cs, ce, st))
                    cs, ce = a, b
            rows.append((chrom, cs, ce, st))
    out = pd.DataFrame(rows, columns=["seqnames", "start", "end", "strand"])
    return out[["seqnames", "start", "end", "strand"]]


def subtract(x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    """``unlist(GenomicRanges::subtract(x, y, ignore.strand = TRUE))``.

    Each range of ``x`` minus the union of ``y``; a range fully covered
    disappears, a range cut in the middle yields several pieces (in genomic
    order). Pieces keep ``x``'s strand; extra columns are dropped.
    """
    yr = reduce(y, ignore_strand=True)
    rows = []
    by_chr = {c: g for c, g in yr.groupby("seqnames", sort=False)}
    for chrom, s, e, st in zip(x["seqnames"], x["start"], x["end"], x["strand"]):
        g = by_chr.get(chrom)
        if g is None:
            rows.append((chrom, int(s), int(e), st))
            continue
        ys, ye = g["start"].to_numpy(), g["end"].to_numpy()
        m = (ys <= e) & (ye >= s)
        cur = int(s)
        for a, b in zip(ys[m], ye[m]):  # yr is sorted and non-overlapping
            if a > cur:
                rows.append((chrom, cur, int(a) - 1, st))
            cur = max(cur, int(b) + 1)
        if cur <= e:
            rows.append((chrom, cur, int(e), st))
    return pd.DataFrame(rows, columns=["seqnames", "start", "end", "strand"])
