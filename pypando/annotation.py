"""Gene annotation in the shape Pando reads (Signac's ``Annotation(object)``).

Pando uses a gene annotation for two things: a gene's extent (the union of
its exon / CDS / UTR rows; Signac's intronic ``gap`` rows are dropped) and
the exons it removes from the candidate regions. Signac builds this table
with ``GetGRangesFromEnsDb``; :func:`annotation_from_gtf` builds the same
table from a GENCODE or Ensembl GTF/GFF3 file.
"""
from __future__ import annotations

import gzip
import re
from typing import Sequence

import pandas as pd

__all__ = ["annotation_from_gtf"]

# Signac::GetGRangesFromEnsDb default biotypes (GENCODE >= v32 calls lincRNA "lncRNA")
DEFAULT_BIOTYPES = ("protein_coding", "lncRNA", "lincRNA", "rRNA", "processed_transcript")
_TYPES = {"exon": "exon", "CDS": "cds", "five_prime_UTR": "utr", "three_prime_UTR": "utr", "UTR": "utr"}
_ATTR = re.compile(r'(\w+)[ =]"?([^";]+)"?')


def annotation_from_gtf(path, biotypes: Sequence[str] | None = DEFAULT_BIOTYPES, chr_prefix: str | None = None) -> pd.DataFrame:
    """Signac-style annotation from a GTF / GFF3 (optionally gzipped).

    Returns columns ``seqnames, start, end, strand, tx_id, gene_name, gene_id,
    gene_biotype, type`` with ``type`` in ``exon``, ``cds``, ``utr``, for genes
    whose biotype is in ``biotypes`` (``None`` keeps all). ``chr_prefix``
    (e.g. ``"chr"``) is prepended to sequence names that lack it, for
    Ensembl files used with UCSC-style peaks.
    """
    opener = gzip.open if str(path).endswith(".gz") else open
    keep = None if biotypes is None else set(biotypes)
    rows = []
    with opener(path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9 or f[2] not in _TYPES:
                continue
            attrs = dict(_ATTR.findall(f[8]))
            biotype = attrs.get("gene_type") or attrs.get("gene_biotype") or ""
            if keep is not None and biotype not in keep:
                continue
            chrom = f[0]
            if chr_prefix and not chrom.startswith(chr_prefix):
                chrom = chr_prefix + chrom
            rows.append((chrom, int(f[3]), int(f[4]), f[6],
                         (attrs.get("transcript_id") or "").split(".")[0],
                         attrs.get("gene_name", ""), (attrs.get("gene_id") or "").split(".")[0],
                         biotype, _TYPES[f[2]]))
    return pd.DataFrame(rows, columns=["seqnames", "start", "end", "strand", "tx_id", "gene_name",
                                       "gene_id", "gene_biotype", "type"])
