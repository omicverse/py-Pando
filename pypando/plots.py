"""Plots from Pando ``plots.R``: matplotlib versions styled after the ggplot2 / ggraph figures."""
from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd

from .objects import NetworkGraph, NetworkModules, NetworkTFs, gof

__all__ = ["plot_gof", "plot_module_metrics", "plot_network_graph", "plot_tf_network"]

_PT = 72.27 / 25.4                       # ggplot2 .pt: points per mm
_EDGE = {-1: "darkgrey", 1: "orange"}     # Pando's edge_color
_LN10 = math.log(10.0)


def _plog(x):
    """scales::pseudo_log_trans(sigma = 1, base = 10)."""
    return np.arcsinh(np.asarray(x, dtype=float) / 2) / _LN10


def _plog_ticks(ax, axis: str, breaks=(0, 1, 10, 100, 1000, 10000, 100000)):
    """Pseudo-log breaks inside the current axis limits (ggplot2 drops the others)."""
    lo, hi = ax.get_ylim() if axis == "y" else ax.get_xlim()
    b = [v for v in breaks if lo <= _plog(v) <= hi]
    (ax.set_yticks if axis == "y" else ax.set_xticks)(_plog(b), [str(v) for v in b])


def _theme_bw(ax, minor_x=None, minor_y=None):
    """ggplot2 theme_bw: white panel, grey92 grid, grey20 border, grey30 tick labels."""
    ax.set_facecolor("white")
    ax.grid(True, color="#EBEBEB", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_color("#333333"); sp.set_linewidth(0.8)
    ax.tick_params(colors="#333333", labelcolor="#4D4D4D", labelsize=8.8, length=2.5, width=0.6)
    ax.xaxis.label.set_size(11); ax.yaxis.label.set_size(11)


def _density_neighbors(x, y, adjust: float = 1.0):
    """ggpointdensity's "neighbors" density: points within an ellipse scaled to the axis ranges."""
    xr, yr = (np.ptp(x) or 1.0) * adjust, (np.ptp(y) or 1.0) * adjust
    r2, xy = (xr + yr) / 70, xr / yr
    dx = (x[:, None] - x[None, :]) ** 2 / xy
    dy = (y[:, None] - y[None, :]) ** 2 * xy
    return ((dx + dy) < r2).sum(axis=1).astype(float)


def _r_reds():
    """rev(pals::brewer.reds(100)): dark red (sparse) to light (dense)."""
    from matplotlib.colors import LinearSegmentedColormap

    reds = ["#FFF5F0", "#FEE0D2", "#FCBBA1", "#FC9272", "#FB6A4A", "#EF3B2C", "#CB181D", "#A50F15", "#67000D"]
    return LinearSegmentedColormap.from_list("rev_brewer_reds", reds[::-1])


def plot_gof(object, network=None, point_size: float = 0.5):
    """``plot_gof``: explained variance against model size, with marginal histograms.

    As in Pando: points coloured by local density, models failing ``find_modules``' thresholds
    (``rsq_thresh``, ``nvar_thresh``) faded, dashed threshold lines, pseudo-log y axis, 20 bins.
    """
    import matplotlib.pyplot as plt

    params = NetworkModules(object, network).params
    if not params:
        raise ValueError("No modules found, please run `find_modules()` first.")
    g = gof(object, network)
    g = g[(g["rsq"] <= 1) & (g["rsq"] >= 0)]
    nice = ((g["rsq"] > params["rsq_thresh"]) & (g["nvariables"] > params["nvar_thresh"])).to_numpy()
    x, y = g["rsq"].to_numpy(float), _plog(g["nvariables"].to_numpy(float))
    dens = _density_neighbors(x, y)

    fig = plt.figure(figsize=(6, 5))
    gs = fig.add_gridspec(3, 5, hspace=0, wspace=0, left=0.13, right=0.98, bottom=0.11, top=0.98)
    ax, top, right = fig.add_subplot(gs[1:, :4]), fig.add_subplot(gs[0, :4]), fig.add_subplot(gs[1:, 4])
    _theme_bw(ax)
    cmap, norm = _r_reds(), plt.Normalize(dens.min(), dens.max())
    for m, a in ((~nice, 0.5), (nice, 1.0)):
        ax.scatter(x[m], y[m], c=dens[m], cmap=cmap, norm=norm, s=(1.2 * point_size * _PT) ** 2, alpha=a,
                   linewidths=0, zorder=3)
    ax.axhline(_plog(params["nvar_thresh"]), color="darkgrey", ls="--", lw=0.5 * _PT * 0.75, zorder=2)
    ax.axvline(params["rsq_thresh"], color="darkgrey", ls="--", lw=0.5 * _PT * 0.75, zorder=2)
    xl = (min(x.min(), 0) - 0.05 * np.ptp(x), x.max() + 0.05 * np.ptp(x))
    yl = (y.min() - 0.05 * np.ptp(y), y.max() + 0.05 * np.ptp(y))
    ax.set_xlim(xl); ax.set_ylim(yl)
    ax.set_xticks([v for v in np.round(np.arange(0, 1.01, 0.2), 1) if xl[0] <= v <= xl[1]])
    _plog_ticks(ax, "y")
    ax.set_xlabel("Explained variance ($R^2$)"); ax.set_ylabel("# variables in model")
    hist = dict(edgecolor="black", facecolor="darkgrey", linewidth=0.2 * _PT)
    top.hist(x, bins=np.linspace(x.min(), x.max(), 21), **hist); top.set_xlim(xl); top.axis("off")
    right.hist(y, bins=np.linspace(y.min(), y.max(), 21), orientation="horizontal", **hist)
    right.set_ylim(yl); right.axis("off")
    return fig


def _violin(ax, values, log: bool = False):
    """geom_violin (trimmed, nrd0 bandwidth, darkgrey) + geom_boxplot(width = 0.2, no outliers)."""
    v = _plog(values) if log else np.asarray(values, dtype=float)
    sd = v.std(ddof=1) if len(v) > 1 else 0.0
    iqr = np.subtract(*np.percentile(v, [75, 25])) if len(v) > 1 else 0.0
    lo = min(sd, iqr / 1.34) or sd or 1.0
    bw = 0.9 * lo * len(v) ** -0.2                      # stats::bw.nrd0
    if len(np.unique(v)) > 1:
        parts = ax.violinplot(v, positions=[1], widths=0.9, showextrema=False, bw_method=bw / (sd or 1.0))
        for b in parts["bodies"]:
            b.set_facecolor("darkgrey"); b.set_edgecolor("black"); b.set_linewidth(0.2 * _PT * 0.75); b.set_alpha(1)
    ax.boxplot(v, positions=[1], widths=0.2, showfliers=False, patch_artist=True,
               boxprops=dict(facecolor="white", edgecolor="black", lw=0.4), medianprops=dict(color="black", lw=0.6),
               whiskerprops=dict(lw=0.4), capprops=dict(lw=0))
    _theme_bw(ax)
    ax.set_xticks([]); ax.set_xlim(0.4, 1.6)
    lo, hi = float(v.min()), float(v.max())
    ax.set_ylim(lo - 0.05 * (hi - lo or 1), hi + 0.05 * (hi - lo or 1))
    if log:
        _plog_ticks(ax, "y", breaks=(0, 1, 2, 3, 5, 10, 20, 30, 40, 50, 100, 200, 500, 1000, 10000))


def plot_module_metrics(object, network=None):
    """``plot_module_metrics``: violins of target genes per TF (pseudo-log), regions per target gene
    and TFs per target gene, in Pando's panel order."""
    import matplotlib.pyplot as plt

    meta = NetworkModules(object, network).meta
    if len(meta) == 0:
        raise ValueError("No modules found, please run `find_modules()` first.")
    fig, axes = plt.subplots(1, 3, figsize=(7, 4))
    panels = ((meta.drop_duplicates(["tf", "n_genes"])["n_genes"], True, "# genes", "# target genes\nper TF"),
              (meta.drop_duplicates(["target", "n_regions"])["n_regions"], False, "# peaks", "# regions\nper target gene"),
              (meta.drop_duplicates(["target", "n_tfs"])["n_tfs"], False, "# TFs", "# TFs per\ntarget gene"))
    for ax, (v, log, ylab, title) in zip(axes, panels):
        _violin(ax, v.to_numpy(float), log=log)
        ax.set_ylabel(ylab); ax.set_title(title, loc="left", fontsize=13)
    fig.tight_layout(w_pad=1.5)
    return fig


def _diagonal(p0, p1, n: int = 40):
    """ggraph geom_edge_diagonal (vertical): cubic Bezier with control points at the mid height."""
    (x0, y0), (x1, y1) = p0, p1
    t = np.linspace(0, 1, n)[:, None]
    ym = (y0 + y1) / 2
    pts = np.array([[x0, y0], [x0, ym], [x1, ym], [x1, y1]])
    return ((1 - t) ** 3) * pts[0] + 3 * ((1 - t) ** 2) * t * pts[1] + 3 * (1 - t) * t ** 2 * pts[2] + t ** 3 * pts[3]


def _label_nodes(ax, xs, ys, names, size, box: bool):
    texts = []
    for x, y, nm in zip(xs, ys, names):
        kw = dict(bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="#333333", lw=0.5)) if box else {}
        texts.append(ax.text(x, y, nm, fontsize=size, ha="center", va="center", zorder=5, **kw))
    try:
        from adjustText import adjust_text
    except ImportError:
        warnings.warn("install adjustText (pip install 'pypando[plot]') to repel overlapping labels")
        return texts
    adjust_text(texts, x=np.asarray(xs), y=np.asarray(ys), ax=ax, expand=(1.15, 1.3),
                arrowprops=dict(arrowstyle="-", color="black", lw=0.6))
    return texts


def plot_network_graph(object, network=None, graph: str = "module_graph", edge_width: float = 0.2,
                       edge_color: dict | None = None, node_color: str = "magma", node_size=(1, 5),
                       text_size: float = 10, color_nodes: bool = True, label_nodes: bool = True,
                       color_edges: bool = True, ax=None):
    """``plot_network_graph``: the module graph at its UMAP layout (``get_network_graph``).

    As ggraph draws it in Pando: diagonal edges coloured by sign (orange activating, dark grey
    repressing), nodes filled with magma and sized 1-5 mm by PageRank centrality with a grey
    outline, every gene labelled with repelled text (adjustText) and leader lines.
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    edge_color = edge_color or _EDGE
    g = NetworkGraph(object, network, graph)
    nodes, edges = g["nodes"], g["edges"]
    if "UMAP_1" not in nodes or nodes["UMAP_1"].isna().all():
        raise ValueError("No UMAP coordinates found, please run `get_network_graph()` first.")
    pos = dict(zip(nodes["name"], zip(nodes["UMAP_1"], nodes["UMAP_2"])))
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))
    segs, cols = [], []
    for a, b, d in zip(edges["from_node"], edges["to_node"], edges["dir"]):
        if a in pos and b in pos:
            segs.append(_diagonal(pos[a], pos[b]))
            cols.append(edge_color.get(int(d), "darkgrey") if color_edges else list(edge_color.values())[0])
    ax.add_collection(LineCollection(segs, colors=cols, linewidths=edge_width * _PT, zorder=1))
    c = nodes["centrality"].to_numpy(float)
    rng = np.ptp(c) or 1.0
    size_mm = node_size[0] + (c - c.min()) / rng * (node_size[1] - node_size[0]) if color_nodes else np.full(len(c), node_size[0])
    ax.scatter(nodes["UMAP_1"], nodes["UMAP_2"], s=(0.75 * size_mm * _PT) ** 2,
               c=c if color_nodes else "lightgrey", cmap=node_color if color_nodes else None,
               edgecolors="darkgrey", linewidths=0.5, zorder=3)
    ax.set_aspect("auto"); ax.autoscale_view(); ax.axis("off")
    if label_nodes:
        _label_nodes(ax, nodes["UMAP_1"].to_numpy(), nodes["UMAP_2"].to_numpy(), nodes["name"], text_size, box=False)
    return ax


def _tree_layout(root: str, edges, node_order, circular: bool):
    """igraph ``layout_as_tree`` (Reingold-Tilford), as ggraph's ``layout = 'tree'`` calls it.

    BFS from the root over out-edges (children in vertex order) gives the tree; each subtree is packed
    against its left siblings so that contours at every depth stay one unit apart, and a parent sits
    midway between its first and last child. ``circular`` applies ggraph's radial transform: radius =
    depth / max depth, angle = pi/2 - 2 pi (x - min + 0.5) / (range + 1), clockwise from the top.
    On the fixture this reproduces ggraph's coordinates for PAX5 exactly.
    """
    rank = {v: i for i, v in enumerate(node_order)}
    out: dict = {}
    for a, b in edges:
        out.setdefault(a, set()).add(b)
    depth, kids, queue = {root: 0}, {}, [root]
    for v in queue:
        for w in sorted(out.get(v, ()), key=lambda u: rank.get(u, len(rank))):
            if w not in depth:
                depth[w] = depth[v] + 1
                kids.setdefault(v, []).append(w)
                queue.append(w)
    rel: dict = {}

    def place(v):                       # returns contour {depth offset: (left, right)} relative to v
        ch = kids.get(v, [])
        if not ch:
            return {0: (0.0, 0.0)}
        offs, acc = [], {}
        for c in ch:
            cc = place(c)
            shift = 0.0 if not acc else max(acc[d][1] - cc[d][0] + 1.0 for d in cc if d in acc)
            offs.append(shift)
            for d, (l, r) in cc.items():
                acc[d] = (min(acc[d][0], l + shift), max(acc[d][1], r + shift)) if d in acc else (l + shift, r + shift)
        mid = (offs[0] + offs[-1]) / 2
        for c, o in zip(ch, offs):
            rel[c] = o - mid
        cont = {0: (0.0, 0.0)}
        for d, (l, r) in acc.items():
            cont[d + 1] = (l - mid, r - mid)
        return cont

    place(root)
    x = {root: 0.0}
    for v in queue[1:]:
        parent = next(p for p, cs in kids.items() if v in cs)
        x[v] = x[parent] + rel[v]
    if not circular:
        return {v: (x[v], -float(depth[v])) for v in x}, depth
    lo, hi = min(x.values()) - 0.5, max(x.values()) + 0.5
    maxd = max(depth.values()) or 1
    out_pos = {}
    for v in x:
        a = np.pi / 2 - 2 * np.pi * (x[v] - lo) / (hi - lo)
        r = depth[v] / maxd
        out_pos[v] = (r * np.cos(a), r * np.sin(a))
    return out_pos, depth


def _radial_diagonal(p0, p1, n: int = 50):
    """geom_edge_diagonal on a circular layout: control points at the mid radius, on the rays of
    the two end points (a cubic Bezier in Cartesian space, as ggraph draws it)."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    r0, r1 = np.hypot(*p0), np.hypot(*p1)
    rm = (r0 + r1) / 2
    u1 = p1 / (r1 or 1.0)
    u0 = p0 / r0 if r0 > 0 else u1
    pts = np.array([p0, u0 * rm, u1 * rm, p1])
    t = np.linspace(0, 1, n)[:, None]
    return ((1 - t) ** 3) * pts[0] + 3 * ((1 - t) ** 2) * t * pts[1] + 3 * (1 - t) * t ** 2 * pts[2] + t ** 3 * pts[3]


def plot_tf_network(
    object,
    tf: str,
    network=None,
    graph: str = "module_graph",
    circular: bool = True,
    edge_width: float = 0.2,
    edge_color: dict | None = None,
    node_size: float = 3,
    text_size: float = 10,
    label_nodes: str = "tfs",
    color_edges: bool = True,
    ax=None,
):
    """``plot_tf_network``: the ``get_tf_network`` sub-network as a (circular) tree.

    Layout as ggraph ``layout = 'tree'`` (igraph Reingold-Tilford; ``circular`` puts the root at the
    centre and one ring per step), diagonal edges coloured by sign, grey points of ``node_size`` mm,
    boxed labels on the network's TFs (``"tfs"``), every node (``"all"``) or the root (``"none"``).
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    if label_nodes not in ("tfs", "all", "none"):
        raise ValueError("label_nodes must be 'tfs', 'all' or 'none'")
    edge_color = edge_color or _EDGE
    tg = NetworkGraph(object, network, "tf_graphs")
    if tf not in tg:
        raise KeyError(f"No TF network for {tf}; run get_tf_network(object, tf={tf!r}) first.")
    e = tg[tf]["edges"]
    order = list(tg[tf]["nodes"]["name"]) if "nodes" in tg[tf] else None
    order = order or list(dict.fromkeys(list(e["from_node"]) + list(e["to_node"])))
    pairs = list(dict.fromkeys(zip(e["from_node"], e["to_node"], e["dir"])))
    pos, _ = _tree_layout(tf, [(a, b) for a, b, _ in pairs], order, circular)
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7) if circular else (9, 5))
    else:
        fig = ax.figure
    segs, cols = [], []
    for a, b, d in pairs:
        if a not in pos or b not in pos:
            continue
        segs.append(_radial_diagonal(pos[a], pos[b]) if circular else _diagonal(pos[a], pos[b]))
        cols.append(edge_color.get(int(d), "darkgrey") if color_edges else list(edge_color.values())[0])
    ax.add_collection(LineCollection(segs, colors=cols, linewidths=edge_width * _PT, zorder=1))
    names = list(pos)
    xy = np.array([pos[v] for v in names])
    ax.scatter(xy[:, 0], xy[:, 1], s=(0.75 * node_size * _PT) ** 2, c="lightgrey", edgecolors="darkgrey",
               linewidths=0.5, zorder=2)
    if label_nodes == "tfs":
        try:
            tfs = set(NetworkTFs(object).columns)
        except (AttributeError, TypeError):     # a bare Network: its relaying nodes are TFs
            tfs = set(e["from_node"])
        lab = [v for v in names if v in tfs or v == tf]
    elif label_nodes == "all":
        lab = names
    else:
        lab = [tf]
    for v in lab:
        ax.annotate(v, pos[v], fontsize=text_size, ha="center", va="center", zorder=3,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="#333333", lw=0.5))
    ax.set_aspect("equal" if circular else "auto")
    ax.autoscale_view()
    ax.axis("off")
    return fig
