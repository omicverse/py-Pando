"""py-Pando — pure-Python port of Pando (Fleck et al., Nature 2023).

Pando infers gene regulatory networks from paired single-cell RNA and ATAC
data: for every target gene it regresses expression on TF expression x
accessibility of TF-binding peaks near the gene.

Functional API (mirrors the R package)::

    import pypando as pp
    grn = pp.initiate_grn(rna, atac, annotation)
    grn = pp.find_motifs(grn, genome="hg38.fa")
    grn = pp.infer_grn(grn, genes=targets)
    grn = pp.find_modules(grn)
    pp.coef(grn); pp.gof(grn); pp.NetworkModules(grn)
"""
from . import data
from .annotation import annotation_from_gtf
from .core import Pando
from .grn import (
    find_modules,
    find_motifs,
    find_peaks_near_genes,
    fit_grn_models,
    format_coefs,
    infer_grn,
    initiate_grn,
    region_background,
    regions_near_genes,
    subset_regions,
)
from .graph import get_network_graph
from .models import (
    fit_bagging_ridge,
    fit_bayesian_ridge,
    fit_brms,
    fit_cvglmnet,
    fit_glm,
    fit_glmnet,
    fit_model,
    fit_xgb,
)
from .motifs import PFM, load_motif2tf, load_pando_motifs
from .objects import (
    GRNData,
    Modules,
    Network,
    Regions,
    RegulatoryNetwork,
    DefaultNetwork,
    GetGRN,
    GetNetwork,
    NetworkFeatures,
    NetworkGraph,
    NetworkModules,
    NetworkParams,
    NetworkRegions,
    NetworkTFs,
    Params,
    coef,
    gof,
)
from .objects import GetAssaySummary, aggregate_assay, create_GRNData
from .plots import plot_gof, plot_module_metrics, plot_network_graph
from .utils import aggregate_matrix, sparse_cor

__all__ = [
    "Pando",
    "annotation_from_gtf",
    "initiate_grn",
    "find_motifs",
    "find_peaks_near_genes",
    "infer_grn",
    "fit_grn_models",
    "format_coefs",
    "find_modules",
    "region_background",
    "regions_near_genes",
    "subset_regions",
    "get_network_graph",
    "fit_model",
    "fit_glm",
    "fit_glmnet",
    "fit_cvglmnet",
    "fit_xgb",
    "fit_bagging_ridge",
    "fit_bayesian_ridge",
    "fit_brms",
    "PFM",
    "load_pando_motifs",
    "load_motif2tf",
    "GRNData",
    "RegulatoryNetwork",
    "Regions",
    "Network",
    "Modules",
    "GetGRN",
    "GetNetwork",
    "DefaultNetwork",
    "NetworkFeatures",
    "NetworkTFs",
    "NetworkRegions",
    "NetworkModules",
    "NetworkParams",
    "NetworkGraph",
    "Params",
    "coef",
    "gof",
    "aggregate_matrix",
    "aggregate_assay",
    "GetAssaySummary",
    "create_GRNData",
    "plot_gof",
    "plot_module_metrics",
    "plot_network_graph",
    "sparse_cor",
]
__version__ = "0.1.0"
