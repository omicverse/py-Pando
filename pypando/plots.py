"""Plots from Pando ``plots.R`` (matplotlib versions of the ggplot2 figures)."""
from __future__ import annotations

import numpy as np

from .objects import NetworkGraph, NetworkModules, gof

__all__ = ["plot_gof", "plot_module_metrics", "plot_network_graph"]


def plot_gof(object, network=None, point_size: float = 4, ax=None):
    """``plot_gof``: model R^2 against the number of variables per target gene."""
    import matplotlib.pyplot as plt

    g = gof(object, network)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5)) if ax is None else (None, ax)
    axes[0].scatter(g["rsq"], g["nvariables"], s=point_size, c="k")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("R^2")
    axes[0].set_ylabel("number of variables")
    axes[1].hist(g["rsq"], bins=30, color="grey")
    axes[1].set_xlabel("R^2")
    axes[1].set_ylabel("target genes")
    return fig


def plot_module_metrics(object, network=None):
    """``plot_module_metrics``: genes per TF, TFs per gene, regions per gene."""
    import matplotlib.pyplot as plt

    meta = NetworkModules(object, network).meta
    genes_per_tf = meta.groupby("tf")["target"].nunique()
    tfs_per_gene = meta.groupby("target")["tf"].nunique()
    regions_per_gene = meta.groupby("target")["n_regions"].first()
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    for ax, v, lab in zip(axes, (genes_per_tf, tfs_per_gene, regions_per_gene),
                          ("genes per TF", "TFs per gene", "regions per gene")):
        ax.hist(v, bins=30, color="grey")
        ax.set_xlabel(lab)
    axes[0].set_ylabel("count")
    fig.tight_layout()
    return fig


def plot_network_graph(object, network=None, graph: str = "module_graph", label_nodes: bool = True,
                       node_size_scale: float = 2000, ax=None):
    """``plot_network_graph``: nodes at their UMAP coordinates, sized by PageRank,
    edges coloured by sign (needs ``get_network_graph`` with a UMAP layout)."""
    import matplotlib.pyplot as plt

    g = NetworkGraph(object, network, graph)
    nodes, edges = g["nodes"], g["edges"]
    if "UMAP_1" not in nodes or nodes["UMAP_1"].isna().all():
        raise ValueError("graph has no layout; rerun get_network_graph with a UMAP method and umap-learn installed")
    pos = dict(zip(nodes["name"], zip(nodes["UMAP_1"], nodes["UMAP_2"])))
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    for a, b, d in zip(edges["from_node"], edges["to_node"], edges["dir"]):
        if a in pos and b in pos:
            ax.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]], lw=0.3, alpha=0.4,
                    color="#d73027" if d > 0 else "#4575b4")
    ax.scatter(nodes["UMAP_1"], nodes["UMAP_2"], s=nodes["centrality"] * node_size_scale, c="k", zorder=3)
    if label_nodes:
        top = nodes.nlargest(15, "centrality")
        for _, r in top.iterrows():
            ax.annotate(r["name"], (r["UMAP_1"], r["UMAP_2"]), fontsize=7)
    ax.set_axis_off()
    return ax
