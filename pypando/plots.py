"""Plots from Pando ``plots.R`` (matplotlib versions of the ggplot2 figures)."""
from __future__ import annotations

import numpy as np

from .objects import NetworkGraph, NetworkModules, NetworkTFs, gof

__all__ = ["plot_gof", "plot_module_metrics", "plot_network_graph", "plot_tf_network"]


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


def _tree_layout(root: str, edges, circular: bool):
    """Dendrogram-style tree layout (ggraph ``layout = 'tree'``): depth = BFS distance from the root,
    leaves spaced evenly in depth-first order, parents centred over their children."""
    children: dict = {}
    for a, b in edges:
        children.setdefault(a, [])
        if b not in children[a]:
            children[a].append(b)
    depth, parent, order = {root: 0}, {}, [root]
    for v in order:                       # BFS: each node hangs under its first shortest-path parent
        for w in children.get(v, []):
            if w not in depth:
                depth[w], parent[w] = depth[v] + 1, v
                order.append(w)
    kids: dict = {}
    for w, v in parent.items():
        kids.setdefault(v, []).append(w)
    x, nxt = {}, [0.0]

    def place(v):
        ch = kids.get(v, [])
        if not ch:
            x[v] = nxt[0]; nxt[0] += 1.0
        else:
            for c in ch:
                place(c)
            x[v] = (x[ch[0]] + x[ch[-1]]) / 2

    place(root)
    n = max(nxt[0], 1.0)
    pos = {}
    for v in depth:
        if circular:
            a = 2 * np.pi * x[v] / n
            pos[v] = (depth[v] * np.cos(a), depth[v] * np.sin(a))
        else:
            pos[v] = (x[v], -float(depth[v]))
    return pos


def plot_tf_network(
    object,
    tf: str,
    network=None,
    graph: str = "module_graph",
    circular: bool = True,
    edge_width: float = 0.6,
    edge_color: dict | None = None,
    node_size: float = 30,
    text_size: float = 8,
    label_nodes: str = "tfs",
    color_edges: bool = True,
    ax=None,
):
    """``plot_tf_network``: the ``get_tf_network`` sub-network as a (circular) tree.

    Edges are drawn as diagonals coloured by their sign (``dir``); ``label_nodes`` is ``"tfs"``
    (the network's TFs), ``"all"`` or ``"none"`` (the root only). Layout coordinates are not
    ggraph's; the drawn graph is.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path

    if label_nodes not in ("tfs", "all", "none"):
        raise ValueError("label_nodes must be 'tfs', 'all' or 'none'")
    edge_color = edge_color or {-1: "darkgrey", 1: "orange"}
    tg = NetworkGraph(object, network, "tf_graphs")
    if tf not in tg:
        raise KeyError(f"No TF network for {tf}; run get_tf_network(object, tf={tf!r}) first.")
    e = tg[tf]["edges"]
    pairs = list(dict.fromkeys(zip(e["from_node"], e["to_node"], e["dir"])))
    pos = _tree_layout(tf, [(a, b) for a, b, _ in pairs], circular)
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6) if circular else (8, 4))
    else:
        fig = ax.figure
    for a, b, d in pairs:
        if a not in pos or b not in pos:
            continue
        (x0, y0), (x1, y1) = pos[a], pos[b]
        if circular:      # diagonal: bend through the radial midpoint
            r0, r1 = np.hypot(x0, y0), np.hypot(x1, y1)
            rm, a1 = (r0 + r1) / 2, np.arctan2(y1, x1)
            a0 = np.arctan2(y0, x0) if r0 > 0 else a1
            c1, c2 = (rm * np.cos(a0), rm * np.sin(a0)), (rm * np.cos(a1), rm * np.sin(a1))
        else:
            ym = (y0 + y1) / 2
            c1, c2 = (x0, ym), (x1, ym)
        col = edge_color.get(int(d), "darkgrey") if color_edges else list(edge_color.values())[0]
        ax.add_patch(PathPatch(Path([(x0, y0), c1, c2, (x1, y1)], [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]),
                               fc="none", ec=col, lw=edge_width, zorder=1))
    names = list(pos)
    xy = np.array([pos[v] for v in names])
    ax.scatter(xy[:, 0], xy[:, 1], s=node_size, c="lightgrey", edgecolors="darkgrey", linewidths=0.5, zorder=2)
    if label_nodes == "tfs":
        tfs = set(NetworkTFs(object).columns)
        lab = [v for v in names if v in tfs or v == tf]
    elif label_nodes == "all":
        lab = names
    else:
        lab = [tf]
    for v in lab:
        ax.annotate(v, pos[v], fontsize=text_size, ha="center", va="center", zorder=3,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="grey", lw=0.4))
    ax.set_aspect("equal" if circular else "auto")
    ax.autoscale_view()
    ax.axis("off")
    return fig
