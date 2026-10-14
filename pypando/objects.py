"""Pando's S4 classes as dataclasses, plus its accessor functions.

=====================  =====================================
R (Pando)              py-Pando
=====================  =====================================
``GRNData``            :class:`GRNData` (RNA + ATAC AnnData)
``RegulatoryNetwork``  :class:`RegulatoryNetwork`
``Regions``            :class:`Regions`
``Network``            :class:`Network`
``Modules``            :class:`Modules`
=====================  =====================================
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp

__all__ = [
    "Regions",
    "Modules",
    "Network",
    "RegulatoryNetwork",
    "GRNData",
    "GetNetwork",
    "GetGRN",
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
    "create_GRNData",
    "aggregate_assay",
    "GetAssaySummary",
]


@dataclass
class MotifMatches:
    """Region x motif matches (``Signac`` Motif object's ``@data``)."""

    data: sp.csr_matrix
    regions: np.ndarray
    motifs: np.ndarray

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.data.toarray(), index=self.regions, columns=self.motifs)


@dataclass
class Regions:
    ranges: pd.DataFrame
    peaks: np.ndarray  # 0-based peak index per candidate region
    motifs: Optional[MotifMatches] = None
    tfs: Optional[pd.DataFrame] = None  # motif x TF (0/1)

    def __repr__(self):
        return (
            f"A Regions object\nwith {len(self.ranges)} candidate genomic regions "
            f"in {len(np.unique(self.peaks))} peaks"
        )


@dataclass
class Modules:
    meta: pd.DataFrame = field(default_factory=pd.DataFrame)
    features: dict = field(default_factory=dict)
    params: dict = field(default_factory=dict)

    def __repr__(self):
        return f"An Modules object with {len(self.features.get('genes_pos', {}))} TF modules"


@dataclass
class Network:
    features: np.ndarray
    fit: pd.DataFrame
    coefs: pd.DataFrame
    modules: Modules = field(default_factory=Modules)
    graphs: dict = field(default_factory=dict)
    params: dict = field(default_factory=dict)

    def __repr__(self):
        src = self.modules.meta if len(self.modules.meta) else self.coefs
        n_genes = src["target"].nunique() if len(src) else 0
        n_tfs = src["tf"].nunique() if len(src) else 0
        return f"A Network object\nwith {n_tfs} TFs and {n_genes} target genes"


@dataclass
class RegulatoryNetwork:
    regions: Regions
    params: dict
    networks: dict = field(default_factory=dict)
    active_network: str = ""

    def __repr__(self):
        tfs = self.regions.tfs
        tf_string = (
            "\nCandidate regions have not been scanned for motifs"
            if tfs is None
            else f"based on {tfs.shape[1]} transcription factors\n"
        )
        names = list(self.networks)
        if not names:
            conn = "\nNo network has been inferred\n"
        elif len(names) == 1:
            conn = f"1 inferred network: {names[0]}\n"
        else:
            conn = f"{len(names)} inferred networks: {', '.join(names)}\n"
        return f"A RegulatoryNetwork object {tf_string}\n{conn}"


class GRNData:
    """Paired RNA + ATAC data with a :class:`RegulatoryNetwork`.

    Parameters
    ----------
    rna
        AnnData, cells x genes. ``X`` (or ``rna_layer``) holds the
        normalised expression Pando reads from the RNA assay's ``data`` layer.
    atac
        AnnData, cells x peaks, same cells in the same order. ``var_names``
        are peak strings ``"chr-start-end"``; ``X`` (or ``peak_layer``) holds
        the ``data`` layer of the peak assay.
    annotation
        Gene annotation as Signac's ``Annotation()``: columns ``seqnames``,
        ``start``, ``end``, ``strand``, ``gene_name``, ``type`` (exon, cds,
        utr, gap, ...).
    variable_features
        Default target genes (``VariableFeatures`` of the RNA assay).
    """

    def __init__(self, rna, atac, annotation: pd.DataFrame, grn: RegulatoryNetwork,
                 variable_features=None, rna_layer=None, peak_layer=None):
        if rna.n_obs != atac.n_obs or not np.array_equal(np.asarray(rna.obs_names), np.asarray(atac.obs_names)):
            raise ValueError("rna and atac must contain the same cells in the same order")
        self.rna = rna
        self.atac = atac
        self.annotation = annotation
        self.grn = grn
        self.variable_features = None if variable_features is None else list(variable_features)
        self.rna_layer = rna_layer
        self.peak_layer = peak_layer

    def rna_data(self):
        return self.rna.layers[self.rna_layer] if self.rna_layer else self.rna.X

    def peak_data(self):
        return self.atac.layers[self.peak_layer] if self.peak_layer else self.atac.X

    def __repr__(self):
        return repr(self.grn)


def _grn(obj) -> RegulatoryNetwork:
    return obj.grn if isinstance(obj, GRNData) else obj


def DefaultNetwork(obj) -> str:
    return _grn(obj).active_network


def GetGRN(obj) -> RegulatoryNetwork:
    return obj.grn


def GetNetwork(obj, network: str | None = None) -> Network:
    g = _grn(obj)
    network = network or g.active_network
    if not network:
        raise KeyError("Network not found, try running `infer_grn()` first.")
    if network not in g.networks:
        raise KeyError(f'The requested network "{network}" does not exist.')
    return g.networks[network]


def NetworkFeatures(obj, network=None):
    return GetNetwork(obj, network).features


def NetworkTFs(obj) -> pd.DataFrame:
    return _grn(obj).regions.tfs


def NetworkRegions(obj) -> Regions:
    return _grn(obj).regions


def NetworkModules(obj, network=None) -> Modules:
    if isinstance(obj, Network):
        return obj.modules
    return GetNetwork(obj, network).modules


def NetworkParams(obj, network=None) -> dict:
    if isinstance(obj, Network):
        return obj.params
    return GetNetwork(obj, network).params


def NetworkGraph(obj, network=None, graph: str = "module_graph"):
    net = obj if isinstance(obj, Network) else GetNetwork(obj, network)
    if graph not in net.graphs:
        raise KeyError(f'The requested graph "{graph}" does not exist. Try (re-)running `get_network_graph()`.')
    return net.graphs[graph]


def Params(obj) -> dict:
    return _grn(obj).params


def coef(obj, network=None) -> pd.DataFrame:
    """Network coefficients (``coef(object)`` in Pando)."""
    if isinstance(obj, Network):
        return obj.coefs
    return GetNetwork(obj, network).coefs


def gof(obj, network=None) -> pd.DataFrame:
    """Goodness of fit per target gene (``gof(object)`` in Pando)."""
    if isinstance(obj, Network):
        return obj.fit
    return GetNetwork(obj, network).fit


def create_GRNData(rna, atac, annotation: pd.DataFrame, grn_object: RegulatoryNetwork, **kwargs) -> GRNData:
    """Pando ``create_GRNData``: bundle data with an existing RegulatoryNetwork."""
    return GRNData(rna, atac, annotation, grn_object, **kwargs)


def aggregate_assay(object: GRNData, group_name: str, fun: str = "mean", assay: str = "RNA") -> GRNData:
    """Pando ``aggregate_assay``: summarise the RNA (or peak) data by an ``obs`` column.

    The summary (groups x features) is stored in ``.uns["pando_summary"][group_name]``
    of the RNA or ATAC AnnData, as Pando stores it in the assay's ``misc``.
    """
    from .utils import aggregate_matrix

    ad_ = object.rna if assay == "RNA" else object.atac
    X = object.rna_data() if assay == "RNA" else object.peak_data()
    groups = ad_.obs[group_name].astype(str).to_numpy()
    agg = aggregate_matrix(X, groups, fun=fun)
    agg.columns = np.asarray(ad_.var_names)
    ad_.uns.setdefault("pando_summary", {})[group_name] = agg
    return object


def GetAssaySummary(object: GRNData, group_name: str, assay: str = "RNA", verbose: bool = True) -> pd.DataFrame:
    """Pando ``GetAssaySummary``: the stored summary, computed on first use."""
    ad_ = object.rna if assay == "RNA" else object.atac
    smry = ad_.uns.get("pando_summary", {}).get(group_name)
    if smry is None:
        if verbose:
            print(f'Summary of "{group_name}" does not yet exist.\nSummarizing.')
        aggregate_assay(object, group_name, assay=assay)
        smry = ad_.uns["pando_summary"][group_name]
    return smry
