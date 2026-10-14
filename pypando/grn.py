"""GRN inference: Pando ``regions.R`` and ``grn.R``.

The pipeline is the one in Pando's README::

    grn = initiate_grn(rna, atac, annotation)          # candidate regions
    grn = find_motifs(grn, pfm=motifs, genome=fasta)   # TF binding sites
    grn = infer_grn(grn, genes=targets)                # per-gene models
    grn = find_modules(grn)                            # TF modules
"""
from __future__ import annotations

import logging
import re
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp

from . import models as _models
from . import ranges as R
from .formula import PandoFormula
from .motifs import PFM, load_motif2tf, load_pando_motifs, scan_motifs
from .objects import (
    GRNData,
    Modules,
    MotifMatches,
    Network,
    Regions,
    RegulatoryNetwork,
    GetNetwork,
    DefaultNetwork,
)
from .rrng import RRandom
from .rstats import p_adjust
from .utils import r_logical_rows, sparse_cor

__all__ = [
    "initiate_grn",
    "find_motifs",
    "find_peaks_near_genes",
    "infer_grn",
    "fit_grn_models",
    "format_coefs",
    "find_modules",
    "region_background",
    "subset_regions",
    "regions_near_genes",
]

log = logging.getLogger("pypando")


def _msg(verbose, *parts):
    if verbose:
        print("".join(str(p) for p in parts))


# ---------------------------------------------------------------------------
# initiate_grn
# ---------------------------------------------------------------------------
def initiate_grn(
    rna,
    atac,
    annotation: pd.DataFrame,
    regions: Optional[pd.DataFrame] = None,
    exclude_exons: bool = True,
    variable_features: Optional[Sequence[str]] = None,
    rna_layer: Optional[str] = None,
    peak_layer: Optional[str] = None,
) -> GRNData:
    """Pando ``initiate_grn``: candidate regions = peaks (optionally intersected
    with ``regions``, e.g. conserved elements) minus exons.

    ``regions`` is a ranges table (``seqnames, start, end``; see
    :mod:`pypando.ranges`); Pando's bundled phastCons and SCREEN cCRE sets are
    available from :func:`pypando.data.phastcons_hg38` and
    :func:`pypando.data.screen_ccre_hg38`.
    """
    if annotation is None or len(annotation) == 0:
        raise ValueError("Please provide a gene annotation.")
    peak_ranges = R.string_to_ranges(atac.var_names)
    if regions is not None:
        reg = regions.copy()
        if "strand" not in reg:
            reg["strand"] = "*"
        qh, sh = R.find_overlaps(reg, peak_ranges)
        cand = R.pintersect(peak_ranges.iloc[sh].reset_index(drop=True), reg.iloc[qh].reset_index(drop=True))
        cand = cand[["seqnames", "start", "end", "strand"]]
    else:
        cand = peak_ranges.copy()
    if exclude_exons:
        exons = annotation[annotation["type"] == "exon"][["seqnames", "start", "end", "strand"]]
        cand = R.subtract(cand, exons)
    cand = cand.reset_index(drop=True)
    _, peak_matches = R.find_overlaps(cand, peak_ranges)
    regions_obj = Regions(ranges=cand, peaks=peak_matches)
    params = {"peak_assay": "peaks", "rna_assay": "RNA", "exclude_exons": exclude_exons}
    grn = RegulatoryNetwork(regions=regions_obj, params=params)
    return GRNData(rna, atac, annotation, grn, variable_features=variable_features,
                   rna_layer=rna_layer, peak_layer=peak_layer)


# ---------------------------------------------------------------------------
# find_motifs
# ---------------------------------------------------------------------------
def _motif2tf_matrix(motif_tfs: pd.DataFrame) -> pd.DataFrame:
    """``select(motif=1, tf=2) %>% distinct() %>% pivot_wider(...)``."""
    df = motif_tfs.iloc[:, :2].copy()
    df.columns = ["motif", "tf"]
    df = df.drop_duplicates()
    motifs = list(dict.fromkeys(df["motif"].tolist()))
    tfs = list(dict.fromkeys(df["tf"].tolist()))
    mi = {m: i for i, m in enumerate(motifs)}
    ti = {t: i for i, t in enumerate(tfs)}
    mat = np.zeros((len(motifs), len(tfs)), dtype=np.int8)
    mat[[mi[m] for m in df["motif"]], [ti[t] for t in df["tf"]]] = 1
    return pd.DataFrame(mat, index=motifs, columns=tfs)


def _get_seqs(genome, ranges: pd.DataFrame) -> tuple[list, np.ndarray]:
    """``getSeq(genome, ranges)`` (upper case, + strand); ranges on sequences
    missing from the genome are flagged and get no matches."""
    import pyfaidx

    fa = genome if isinstance(genome, pyfaidx.Fasta) else pyfaidx.Fasta(str(genome), sequence_always_upper=True)
    keys = set(fa.keys())
    keep = ranges["seqnames"].isin(keys).to_numpy()
    seqs = []
    for chrom, s, e, k in zip(ranges["seqnames"], ranges["start"], ranges["end"], keep):
        if k:
            seqs.append(str(fa[chrom][int(s) - 1 : int(e)]).upper())
    return seqs, keep


def find_motifs(
    object: GRNData,
    pfm: Optional[Sequence[PFM]] = None,
    genome=None,
    motif_tfs: Optional[pd.DataFrame] = None,
    bg="subject",
    p_cutoff: float = 5e-05,
    verbose: bool = True,
) -> GRNData:
    """Pando ``find_motifs``: scan candidate regions and attach the motif-to-TF map.

    Parameters
    ----------
    pfm
        Position frequency matrices (:class:`~pypando.motifs.PFM`); default
        Pando's bundled ``motifs``.
    genome
        Path to a FASTA file (or a ``pyfaidx.Fasta``) of the same assembly as
        the peaks, e.g. UCSC ``hg38.fa`` for ``BSgenome.Hsapiens.UCSC.hg38``.
    motif_tfs
        Motif-to-TF table (first column motif, second TF); default Pando's
        ``motif2tf``. As in Pando, its rows are used *positionally* against
        ``pfm`` when TFs are looked up per peak, so keep both aligned.
    """
    if genome is None:
        raise ValueError("`genome` (a FASTA path) is required")
    if pfm is None:
        pfm = load_pando_motifs()
    _msg(verbose, "Adding TF info")
    m2t = load_motif2tf() if motif_tfs is None else motif_tfs
    m2t = _motif2tf_matrix(m2t)
    rna_genes = list(dict.fromkeys(np.asarray(object.rna.var_names).tolist()))
    cols = set(m2t.columns)
    tfs_use = [g for g in rna_genes if g in cols]
    if not tfs_use:
        raise ValueError(
            "None of the provided TFs were found in the dataset. "
            "Consider providing a custom motif-to-TF map as `motif_tfs`"
        )
    object.grn.regions.tfs = m2t[tfs_use]

    cand = object.grn.regions.ranges
    if not cand["seqnames"].isin(_fasta_keys(genome)).all():
        log.warning("Not all seqlevels present in supplied genome")
    _msg(verbose, "Building motif matrix")
    seqs, keep = _get_seqs(genome, cand)
    hits = scan_motifs(pfm, seqs, bg=bg, p_cutoff=p_cutoff)
    if not keep.all():
        coo = hits.tocoo()
        rows = np.flatnonzero(keep)[coo.row]
        hits = sp.csr_matrix((coo.data, (rows, coo.col)), shape=(len(cand), hits.shape[1]), dtype=bool)
    object.grn.regions.motifs = MotifMatches(
        data=hits,
        regions=R.ranges_to_string(cand),
        motifs=np.array([m.ID for m in pfm], dtype="O"),
    )
    return object


def region_background(object: GRNData, genome) -> np.ndarray:
    """A, C, G, T frequencies of all current candidate regions (new in Python).

    This is what ``find_motifs(bg = "subject")`` uses. Computing it once and
    passing it as ``bg`` lets you scan a subset of regions (see
    :func:`subset_regions`) and still get exactly the hits a full scan gives,
    because MOODS scans each region independently given the background.
    """
    from .motifs import nucleotide_frequencies

    seqs, _ = _get_seqs(genome, object.grn.regions.ranges)
    return nucleotide_frequencies(seqs)


def subset_regions(object: GRNData, keep) -> GRNData:
    """Keep only some candidate regions (new in Python).

    ``keep`` is a boolean mask over ``object.grn.regions.ranges``, e.g. the
    regions within reach of the target genes. Regions are assigned to genes
    one at a time (Signac windows and GREAT domains alike), so dropping
    regions that no target gene can reach does not change ``infer_grn``.
    Call before :func:`find_motifs`, with ``bg`` set to
    :func:`region_background` of the full set to keep motif calls identical.
    """
    keep = np.asarray(keep, dtype=bool)
    reg = object.grn.regions
    if len(reg.peaks) != len(reg.ranges):
        raise ValueError("candidate regions and peak indices are not one-to-one")
    reg.ranges = reg.ranges[keep].reset_index(drop=True)
    reg.peaks = reg.peaks[keep]
    if reg.motifs is not None:
        reg.motifs = MotifMatches(data=reg.motifs.data[np.flatnonzero(keep)],
                                  regions=reg.motifs.regions[keep], motifs=reg.motifs.motifs)
    return object


def regions_near_genes(object: GRNData, genes, upstream: int = 100000, downstream: int = 0,
                       method: str = "Signac", extend: int = 1000000) -> np.ndarray:
    """Mask of candidate regions a target gene can reach (new in Python).

    A superset of what ``infer_grn`` can use: every region overlapping a
    target gene's annotation extended by ``max(upstream, downstream)`` on both
    sides (``Signac``), or by ``upstream + extend`` (``GREAT``).
    """
    ann = object.annotation
    if "type" in ann.columns:
        ann = ann[ann["type"] != "gap"]
    ann = ann[ann["gene_name"].isin(set(genes))]
    if method == "GREAT":
        from .data import ensdb_v93_hg38

        great = ensdb_v93_hg38()
        ann = pd.concat([ann[["seqnames", "start", "end", "strand"]],
                         great[great["gene_name"].isin(set(genes))][["seqnames", "start", "end", "strand"]]])
        pad = upstream + downstream + extend
    else:
        pad = max(upstream, downstream)
    win = ann[["seqnames", "start", "end"]].copy()
    win["start"] = win["start"] - pad
    win["end"] = win["end"] + pad
    win["strand"] = "*"
    qh, _ = R.find_overlaps(object.grn.regions.ranges, win)
    mask = np.zeros(len(object.grn.regions.ranges), dtype=bool)
    mask[qh] = True
    return mask


def _fasta_keys(genome):
    import pyfaidx

    fa = genome if isinstance(genome, pyfaidx.Fasta) else pyfaidx.Fasta(str(genome))
    return list(fa.keys())


# ---------------------------------------------------------------------------
# find_peaks_near_genes
# ---------------------------------------------------------------------------
def find_peaks_near_genes(
    peaks: pd.DataFrame,
    genes: pd.DataFrame,
    sep=("-", "-"),
    method: str = "Signac",
    upstream: int = 100000,
    downstream: int = 0,
    extend: int = 1000000,
    only_tss: bool = False,
    great_annotation: Optional[pd.DataFrame] = None,
):
    """Pando ``find_peaks_near_genes``.

    Returns ``(hits, peak_names, gene_names)``: a sparse peak x gene-row hit
    matrix (one column per annotation row, named by ``gene_name``).

    ``method="GREAT"`` uses Pando's bundled Ensembl v93 (GRCh38) annotation
    (``great_annotation`` overrides it) and assigns each peak to the basal
    domain, else the gene body, else the extended domain.
    """
    if method == "Signac":
        g = R.resize_start(genes) if only_tss else genes
        ext = R.extend(g, upstream=upstream, downstream=downstream)
        qh, sh = R.find_overlaps(peaks, ext)
        hits = sp.csr_matrix(
            (np.ones(len(qh)), (qh, sh)), shape=(len(peaks), len(ext))
        )
        return hits, R.ranges_to_string(peaks, sep), ext["gene_name"].to_numpy(dtype="O")
    if method != "GREAT":
        raise ValueError("method must be 'Signac' or 'GREAT'")
    if great_annotation is None:
        from .data import ensdb_v93_hg38

        great_annotation = ensdb_v93_hg38()
    use = great_annotation[great_annotation["gene_name"].isin(set(genes["gene_name"]))].reset_index(drop=True)
    tss = use[["seqnames", "start", "end", "strand", "gene_name"]]
    basal = R.extend(tss, upstream=upstream, downstream=downstream)
    peak_all = R.ranges_to_string(peaks, sep)
    bq, bs = R.find_overlaps(peaks, basal, minoverlap=2)
    n = len(peaks)
    unmapped = np.setdiff1d(np.arange(n), bq)  # sorted, as R's setdiff on seq()
    un_r = R.string_to_ranges(peak_all[unmapped], sep)
    gq, gs = R.find_overlaps(un_r, tss, minoverlap=2)
    body_idx = unmapped[gq]
    mapped = np.concatenate([bq, body_idx])
    unmapped2 = np.array([i for i in range(n) if i not in set(mapped.tolist())], dtype=np.int64)
    un2_r = R.string_to_ranges(peak_all[unmapped2], sep)
    ext_reg = R.extend(basal, upstream=extend, downstream=extend)
    eq, es = R.find_overlaps(un2_r, ext_reg, minoverlap=2)
    ext_idx = unmapped2[eq]
    rows = np.concatenate([bq, body_idx, ext_idx])
    cols = np.concatenate([bs, gs, es])
    hits = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, len(basal)))
    return hits, peak_all, basal["gene_name"].to_numpy(dtype="O")


def _peaks2gene(hits: sp.csr_matrix, gene_names: np.ndarray) -> dict:
    """``aggregate_matrix(t(hits), groups = colnames, fun = "sum")`` as gene -> region mask."""
    hits = sp.csc_matrix(hits)
    out = {}
    for g in dict.fromkeys(gene_names.tolist()):
        cols = np.flatnonzero(gene_names == g)
        out[g] = np.asarray(hits[:, cols].sum(axis=1)).ravel() > 0
    return out


# ---------------------------------------------------------------------------
# infer_grn / fit_grn_models
# ---------------------------------------------------------------------------
def _col(X, j):
    v = X[:, j]
    return np.asarray(v.toarray()).ravel() if sp.issparse(v) else np.asarray(v).ravel()


def _sanitise(name: str) -> str:
    return name.replace("-", "_")


def _blas_limit(n):
    if n is None:
        return None
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:  # pragma: no cover
        return None
    return threadpool_limits(limits=n, user_api="blas")


def _r_scale(M: np.ndarray) -> np.ndarray:
    mu = M.mean(axis=0)
    sd = M.std(axis=0, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (M - mu) / sd


def infer_grn(
    object: GRNData,
    genes: Optional[Sequence[str]] = None,
    network_name: Optional[str] = None,
    peak_to_gene_method: str = "Signac",
    upstream: int = 100000,
    downstream: int = 0,
    extend: int = 1000000,
    only_tss: bool = False,
    peak_to_gene_domains: Optional[pd.DataFrame] = None,
    parallel: bool = False,
    tf_cor: float = 0.1,
    peak_cor: float = 0.0,
    aggregate_rna_col: Optional[str] = None,
    aggregate_peaks_col: Optional[str] = None,
    method: str = "glm",
    alpha: float = 0.5,
    family: str = "gaussian",
    interaction_term: str = ":",
    adjust_method: str = "fdr",
    scale: bool = False,
    seed: Optional[int] = None,
    blas_threads: Optional[int] = 1,
    verbose=True,
    **kwargs,
) -> GRNData:
    """Pando ``infer_grn``: fit one model per target gene.

    For each target gene: candidate regions near the gene (``Signac`` window
    or ``GREAT`` domains) with a motif and with ``|cor(peak, gene)| >
    peak_cor``; TFs whose motif is in those peaks and with ``|cor(TF, gene)|
    > tf_cor``; then ``gene ~ sum(peak <interaction_term> TF)``.

    ``blas_threads`` (new in Python, default 1): BLAS threads used while
    fitting. Multi-threaded level-3 BLAS changes summation order, so results
    are only reproducible (and equal to R's run with one BLAS thread) when
    this is fixed; ``None`` leaves the BLAS as configured.

    ``seed`` (new in Python): for ``method="cv.glmnet"`` the folds are drawn
    from an emulation of R's stream after ``set.seed(seed)``, so the same
    seed reproduces R's folds; for ``method="bagging_ridge"`` it seeds NumPy's
    global RNG, as Pando's scikit-learn call draws from it.
    """
    if method not in _models.METHODS:
        raise ValueError(f"method must be one of {_models.METHODS}")
    if network_name is None:
        network_name = f"{method}_network"
    return fit_grn_models(
        object, genes=genes, network_name=network_name, peak_to_gene_method=peak_to_gene_method,
        upstream=upstream, downstream=downstream, extend=extend, only_tss=only_tss,
        peak_to_gene_domains=peak_to_gene_domains, parallel=parallel, tf_cor=tf_cor,
        peak_cor=peak_cor, aggregate_rna_col=aggregate_rna_col,
        aggregate_peaks_col=aggregate_peaks_col, method=method, alpha=alpha, family=family,
        interaction_term=interaction_term, adjust_method=adjust_method, scale=scale, seed=seed,
        blas_threads=blas_threads, verbose=verbose, **kwargs,
    )


def fit_grn_models(
    object: GRNData,
    genes=None,
    network_name=None,
    peak_to_gene_method="Signac",
    upstream=100000,
    downstream=0,
    extend=1000000,
    only_tss=False,
    peak_to_gene_domains=None,
    parallel=False,
    tf_cor=0.1,
    peak_cor=0.0,
    aggregate_rna_col=None,
    aggregate_peaks_col=None,
    method="glm",
    alpha=0.5,
    family="gaussian",
    interaction_term=":",
    adjust_method="fdr",
    scale=False,
    seed=None,
    blas_threads=1,
    verbose=True,
    **kwargs,
) -> GRNData:
    """Pando ``fit_grn_models`` (called by :func:`infer_grn`)."""
    if network_name is None:
        network_name = f"{method}_network"
    regions = object.grn.regions
    motif2tf = regions.tfs
    if motif2tf is None:
        raise ValueError("Motif matches have not been found. Please run `find_motifs()` first.")
    gene_annot = object.annotation
    if "type" in gene_annot.columns:
        gene_annot = gene_annot[gene_annot["type"] != "gap"]
    if genes is None:
        genes = object.variable_features
        if genes is None:
            raise ValueError("Please provide a set of features or set variable features")
    genes = list(genes)

    rna_names = np.asarray(object.rna.var_names).astype(str)
    rna_index = {g: i for i, g in enumerate(rna_names)}
    gene_data = object.rna_data()
    gene_data = sp.csc_matrix(gene_data) if sp.issparse(gene_data) else np.asarray(gene_data)
    if aggregate_rna_col is not None:
        raise NotImplementedError(
            "aggregate_rna_col: Pando reads object@meta.data, which a GRNData does not have "
            "(it errors in R). Aggregate the AnnData beforehand instead."
        )
    if aggregate_peaks_col is not None:
        raise NotImplementedError("aggregate_peaks_col: aggregate the AnnData beforehand instead.")
    peak_data = object.peak_data()
    peak_data = sp.csc_matrix(peak_data) if sp.issparse(peak_data) else np.asarray(peak_data)

    genes_set = set(genes)
    annot_names = list(dict.fromkeys(gene_annot["gene_name"].tolist()))
    features = [g for g in annot_names if g in genes_set and g in rna_index]
    feat_set = set(features)
    gene_annot = gene_annot[gene_annot["gene_name"].isin(feat_set)].reset_index(drop=True)

    region_names = regions.motifs.regions
    peak_data = peak_data[:, regions.peaks]
    peaks2motif = regions.motifs.data.tocsr()

    if peak_to_gene_domains is None:
        _msg(verbose, "Selecting candidate regulatory regions near genes")
        hits, _, hit_genes = find_peaks_near_genes(
            regions.ranges, gene_annot, method=peak_to_gene_method, upstream=upstream,
            downstream=downstream, extend=extend, only_tss=only_tss,
        )
    else:
        _msg(verbose, "Selecting candidate regulatory regions in provided domains")
        hits, _, hit_genes = find_peaks_near_genes(
            regions.ranges, peak_to_gene_domains, method="Signac", upstream=0, downstream=0, only_tss=False
        )
    p2g = _peaks2gene(hits, np.asarray(hit_genes, dtype="O"))

    peaks_at_gene = np.zeros(len(region_names), dtype=bool)
    for m in p2g.values():
        peaks_at_gene |= m
    peaks_with_motif = np.asarray(peaks2motif.sum(axis=1)).ravel() > 0
    peaks_use = peaks_at_gene & peaks_with_motif
    keep_idx = np.flatnonzero(peaks_use)
    p2g = {g: m[keep_idx] for g, m in p2g.items()}
    peaks2motif = peaks2motif[keep_idx]
    peak_data = peak_data[:, keep_idx]
    region_names = region_names[keep_idx]

    _msg(verbose, "Preparing model input")
    tf_names = np.asarray(motif2tf.columns, dtype="O")
    m2t = motif2tf.to_numpy()
    n_m2t = m2t.shape[0]

    rng = None
    if method == "cv.glmnet":
        rng = RRandom(seed if seed is not None else int(np.random.randint(0, 2**31 - 1)))
    if method == "bagging_ridge" and seed is not None:
        np.random.seed(seed)

    _msg(verbose, "Fitting models for ", len(features), " target genes")
    fits = {}
    limiter = _blas_limit(blas_threads)
    for g in features:
        if g not in p2g:
            continue
        gene_peaks = p2g[g]
        if gene_peaks.sum() == 0:
            continue
        gp_idx = np.flatnonzero(gene_peaks)
        g_x = _col(gene_data, rna_index[g])
        peak_x = peak_data[:, gp_idx]
        pg_cor = sparse_cor(peak_x, g_x[:, None])[:, 0]
        sel = np.abs(pg_cor) > peak_cor
        if not sel.any():
            continue
        p_idx = gp_idx[sel]  # indices into region arrays, in peak order
        peak_x = peak_data[:, p_idx]
        peak_motifs = peaks2motif[p_idx]
        gene_peak_tfs = []
        for r in range(len(p_idx)):
            x = np.asarray(peak_motifs[r].toarray()).ravel()
            rows = r_logical_rows(x, n_m2t)
            if len(rows):
                present = m2t[rows].max(axis=0) > 0
            else:
                present = np.zeros(len(tf_names), dtype=bool)
            tfs = [t for t in tf_names[present] if t != g]
            gene_peak_tfs.append(list(dict.fromkeys(tfs)))
        gene_tfs = list(dict.fromkeys([t for tl in gene_peak_tfs for t in tl]))
        if gene_tfs:
            tf_x = np.column_stack([_col(gene_data, rna_index[t]) for t in gene_tfs])
            tg_cor = sparse_cor(tf_x, g_x[:, None])[:, 0]
        else:
            tg_cor = np.zeros(0)
        tfs_use = [t for t, c in zip(gene_tfs, tg_cor) if abs(c) > tf_cor]
        if not tfs_use:
            continue
        tf_cor_map = dict(zip(gene_tfs, tg_cor))
        corr_df = pd.DataFrame({"tf": list(dict.fromkeys(tfs_use)), "corr": [tf_cor_map[t] for t in dict.fromkeys(tfs_use)]})
        use_set = set(tfs_use)
        pairs = []
        per_peak_tfs = []
        for r, tl in enumerate(gene_peak_tfs):
            tl2 = [t for t in tl if t in use_set]
            if not tl2:
                continue
            pname = _sanitise(region_names[p_idx[r]])
            per_peak_tfs.append(tl2)
            pairs.extend((pname, _sanitise(t)) for t in tl2)
        if not pairs:
            continue
        frm = PandoFormula(_sanitise(g), pairs, interaction_term)
        frm.check_parse()
        nfeats = sum(len(t) for t in per_peak_tfs)
        gene_tfs2 = list(dict.fromkeys([t for tl in per_peak_tfs for t in tl]))
        cols = [g] + [t for t in gene_tfs2 if t != g]
        data_cols = {c: _col(gene_data, rna_index[c]) for c in cols}
        for j, r in enumerate(p_idx):
            data_cols[region_names[r]] = _col(peak_x, j)
        data = {}
        for c, v in data_cols.items():
            data[_sanitise(c)] = v
        if scale:
            names = list(data)
            M = _r_scale(np.column_stack([data[k] for k in names]))
            data = {k: M[:, i] for i, k in enumerate(names)}
        # na.omit over the variables in the model (relevant only with scale=TRUE)
        used = frm.variables()
        okrow = np.ones(len(g_x), dtype=bool)
        for v in used:
            okrow &= ~np.isnan(data[v])
        if not okrow.all():
            data = {k: v[okrow] for k, v in data.items()}
        _msg(verbose == 2, "Fitting model with ", nfeats, " variables for ", g)
        try:
            res = _models.fit_model(frm, data, method=method, family=family, alpha=alpha, rng=rng, **kwargs) \
                if method == "cv.glmnet" else \
                _models.fit_model(frm, data, method=method, family=family, alpha=alpha, **kwargs)
        except Exception as err:  # Pando: try(fit_model(...)) -> skip the gene
            _msg(verbose, "Warning: Fitting model failed for ", g)
            _msg(verbose == 2, repr(err))
            continue
        res["gof"]["nvariables"] = nfeats
        res["corr"] = corr_df
        fits[g] = res

    if limiter is not None:
        limiter.restore_original_limits()
    if not fits:
        _msg(verbose, "Warning: Fitting model failed for all genes.")
    coefs = pd.concat(
        [r["coefs"].assign(target=g)[["target"] + list(r["coefs"].columns)] for g, r in fits.items()],
        ignore_index=True,
    ) if fits else pd.DataFrame()
    coefs = format_coefs(coefs, term=interaction_term, adjust_method=adjust_method)
    if len(coefs):
        corrs = pd.concat([r["corr"].assign(target=g)[["target", "tf", "corr"]] for g, r in fits.items()],
                          ignore_index=True)
        coefs = coefs.merge(corrs, on=["target", "tf"], how="left", sort=False)
    gof = pd.DataFrame([{"target": g, **r["gof"]} for g, r in fits.items()])
    params = {
        "method": method,
        "family": family,
        "dist": {"upstream": upstream, "downstream": downstream},
        "only_tss": only_tss,
        "interaction": interaction_term,
        "tf_cor": tf_cor,
        "peak_cor": peak_cor,
    }
    net = Network(features=np.array(features, dtype="O"), coefs=coefs, fit=gof, params=params)
    object.grn.networks[network_name] = net
    object.grn.active_network = network_name
    return object


# ---------------------------------------------------------------------------
# format_coefs
# ---------------------------------------------------------------------------
_REGION_PATTERN = re.compile(r"[\d\w]+_\d+_\d+")


def format_coefs(coefs: pd.DataFrame, term: str = ":", adjust_method: str = "fdr") -> pd.DataFrame:
    """Pando ``format_coefs``: split terms into TF and region, adjust p-values."""
    if coefs is None or len(coefs) == 0:
        return coefs if coefs is not None else pd.DataFrame()
    coefs = coefs.copy()
    if "pval" in coefs.columns:
        coefs["padj"] = p_adjust(coefs["pval"].to_numpy(dtype=np.float64), method=adjust_method)
    term_pat = re.compile("(.+)" + re.escape(term) + "(.+)")
    non_int = ~coefs["term"].isin(["(Intercept)", "Intercept"])
    terms_have_sep = coefs.loc[non_int, "term"].str.contains(re.escape(term), regex=True).any()
    use = coefs[non_int].copy()
    t = use["term"].astype(str)
    if terms_have_sep:
        tf_ = t.map(lambda s: term_pat.sub(r"\1", s, count=1) if term_pat.search(s) else s)
        region_ = t.map(lambda s: term_pat.sub(r"\2", s, count=1) if term_pat.search(s) else s)
    else:
        is_reg = t.map(lambda s: bool(_REGION_PATTERN.search(s)))
        tf_ = t.where(~is_reg, None)
        region_ = t.where(is_reg, None)
    tf_reg = tf_.map(lambda s: s is not None and bool(_REGION_PATTERN.search(s)))
    tf_notna = tf_.notna()
    tf = np.where(tf_notna & tf_reg, region_, tf_)
    region = np.where(tf_notna & ~tf_reg, region_, tf_)
    use["tf"] = pd.Series(tf, index=use.index).map(lambda s: s.replace("_", "-") if isinstance(s, str) else s)
    use["region"] = pd.Series(region, index=use.index).map(lambda s: s.replace("_", "-") if isinstance(s, str) else s)
    use["target"] = use["target"].astype(str).str.replace("_", "-", regex=False)
    rest = [c for c in use.columns if c not in ("tf", "target", "region", "term")]
    return use[["tf", "target", "region", "term"] + rest].reset_index(drop=True)


# ---------------------------------------------------------------------------
# find_modules
# ---------------------------------------------------------------------------
def _find_modules_network(net: Network, p_thresh=0.05, rsq_thresh=0.1, nvar_thresh=10,
                          min_genes_per_module=5, xgb_method="tf", xgb_top=50, verbose=True) -> Modules:
    fit_method = net.params["method"]
    if fit_method not in ("glm", "cv.glmnet", "glmnet", "brms", "xgb"):
        raise NotImplementedError(f'find_modules() is not yet implemented for "{fit_method}" models')
    g = net.fit
    models_use = g.loc[(g["rsq"] > rsq_thresh) & (g["nvariables"] > nvar_thresh), "target"].unique()
    mods = net.coefs[net.coefs["target"].isin(set(models_use))].copy()
    if fit_method in ("cv.glmnet", "glmnet"):
        mods = mods[mods["estimate"] != 0]
    elif fit_method == "xgb":
        # group_by(xgb_method) %>% top_n(xgb_top, gain): keeps ties at the boundary
        keep = []
        for _, grp in mods.groupby(xgb_method, sort=False):
            cut = np.sort(grp["gain"].to_numpy())[::-1][min(xgb_top, len(grp)) - 1]
            keep.append(grp[grp["gain"] >= cut])
        mods = pd.concat(keep) if keep else mods.iloc[0:0]
        mods = mods.assign(estimate=np.sign(mods["corr"]) * mods["gain"])
    else:
        mods = mods[mods["padj"].isna() | (mods["padj"] < p_thresh)]

    if len(mods):
        tf_per_gene = mods.groupby("target")["tf"].transform("nunique")
        peak_per_gene = mods.groupby("target")["region"].transform("nunique")
        gene_per_tf = mods.groupby("tf")["target"].transform("nunique")
        mods = mods.assign(tf_per_gene=tf_per_gene, peak_per_gene=peak_per_gene, gene_per_tf=gene_per_tf)
        agg = {
            "estimate": ("estimate", "sum"),
            "n_regions": ("peak_per_gene", "first"),
            "n_genes": ("gene_per_tf", "first"),
            "n_tfs": ("tf_per_gene", "first"),
            "regions": ("region", lambda s: ";".join(s)),
        }
        if fit_method not in ("cv.glmnet", "glmnet", "xgb"):
            agg["pval"] = ("pval", "min")
            agg["padj"] = ("padj", "min")
        meta = mods.groupby(["target", "tf"], sort=True).agg(**agg).reset_index()
        meta = meta.sort_values("tf", kind="stable").reset_index(drop=True)
        meta = meta[["tf", "target"] + [c for c in meta.columns if c not in ("tf", "target")]]
    else:
        meta = pd.DataFrame(columns=["tf", "target", "estimate", "n_regions", "n_genes", "n_tfs", "regions"])

    def split(sign):
        sub = meta[meta["estimate"] > 0] if sign > 0 else meta[meta["estimate"] < 0]
        genes, regs = {}, {}
        for tf, grp in sub.groupby("tf", sort=True):
            if len(grp) > min_genes_per_module:
                genes[tf] = grp["target"].tolist()
                regs[tf] = [r for s in grp["regions"] for r in s.split(";")]
        return genes, regs

    gp, rp = split(1)
    gn, rn = split(-1)
    _msg(verbose, f"Found {meta['tf'].nunique()} TF modules")
    return Modules(
        meta=meta,
        features={"genes_pos": gp, "genes_neg": gn, "regions_pos": rp, "regions_neg": rn},
        params={"p_thresh": p_thresh, "rsq_thresh": rsq_thresh, "nvar_thresh": nvar_thresh,
                "min_genes_per_module": min_genes_per_module},
    )


def find_modules(object, network: Optional[str] = None, p_thresh: float = 0.05, rsq_thresh: float = 0.1,
                 nvar_thresh: int = 10, min_genes_per_module: int = 5, verbose: bool = True, **kwargs):
    """Pando ``find_modules``: TF modules from significant model terms.

    On a :class:`~pypando.objects.GRNData` the module regions are also
    mapped back to peak names (``peaks_pos`` / ``peaks_neg``).
    """
    if isinstance(object, Network):
        return _find_modules_network(object, p_thresh, rsq_thresh, nvar_thresh, min_genes_per_module,
                                     verbose=verbose, **kwargs)
    network = network or DefaultNetwork(object)
    net = GetNetwork(object, network)
    mods = _find_modules_network(net, p_thresh, rsq_thresh, nvar_thresh, min_genes_per_module, verbose=verbose)
    regions = object.grn.regions
    peak_names = np.asarray(object.atac.var_names).astype(str)[regions.peaks]
    reg2peak = dict(zip(R.ranges_to_string(regions.ranges), peak_names))
    mods.features["peaks_pos"] = {tf: list(dict.fromkeys(reg2peak.get(r) for r in rs)) for tf, rs in mods.features["regions_pos"].items()}
    mods.features["peaks_neg"] = {tf: list(dict.fromkeys(reg2peak.get(r) for r in rs)) for tf, rs in mods.features["regions_neg"].items()}
    net.modules = mods
    return object
