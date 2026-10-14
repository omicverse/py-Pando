"""Module graph (Pando ``get_network_graph``) without the plotting.

The graph has one edge per TF -> target pair in the module table (with the
module columns as edge attributes and ``dir = sign(estimate)``) and node
PageRank centrality (``tidygraph::centrality_pagerank``: directed, damping
0.85, unweighted). For ``umap_method != "none"`` Pando also lays genes out
with ``uwot::umap`` on a co-expression matrix; that layout is reproduced with
umap-learn when it is installed, but R's and Python's UMAP streams differ,
so coordinates are not expected to match.
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np
import pandas as pd

from .objects import GRNData, GetNetwork, DefaultNetwork, NetworkFeatures, NetworkGraph, NetworkModules
from .utils import sparse_cor

__all__ = ["get_network_graph", "get_tf_network", "tf_network_from_edges", "pagerank"]


def pagerank(nodes: list, edges: pd.DataFrame, damping: float = 0.85) -> np.ndarray:
    """Exact PageRank of a directed unweighted graph (dangling nodes jump uniformly)."""
    n = len(nodes)
    idx = {v: i for i, v in enumerate(nodes)}
    A = np.zeros((n, n))
    for a, b in zip(edges["from_node"], edges["to_node"]):
        A[idx[a], idx[b]] += 1.0
    out = A.sum(axis=1)
    P = np.zeros((n, n))
    nz = out > 0
    P[nz] = A[nz] / out[nz, None]
    P[~nz] = 1.0 / n
    M = np.eye(n) - damping * P.T
    pr = np.linalg.solve(M, np.full(n, (1 - damping) / n))
    return pr / pr.sum()


def get_network_graph(
    object: GRNData,
    network: str | None = None,
    graph_name: str = "module_graph",
    umap_method: str = "weighted",
    features=None,
    random_seed: int = 111,
    n_pcs: int = 30,
    verbose: bool = True,
    **umap_kwargs,
) -> GRNData:
    network = network or DefaultNetwork(object)
    modules = NetworkModules(object, network)
    if not modules.params:
        raise ValueError("No modules found, please run `find_modules()` first.")
    if features is None:
        features = list(NetworkFeatures(object, network))
    meta = modules.meta
    rna_names = np.asarray(object.rna.var_names).astype(str)
    X = object.rna_data()
    coex = None
    if umap_method in ("weighted", "corr"):
        feats = [f for f in features if f in set(rna_names)]
        cols = [int(np.flatnonzero(rna_names == f)[0]) for f in feats]
        gene_cor = sparse_cor(X[:, cols])
        cor_df = pd.DataFrame(gene_cor, index=feats, columns=feats)
        use = meta[meta["target"].isin(feats) & meta["tf"].isin(feats)].copy()
        use["corr"] = [cor_df.at[t, g] if (t in cor_df.index and g in cor_df.columns) else 0.0
                       for t, g in zip(use["tf"], use["target"])]
        if umap_method == "weighted":
            reg = use.pivot_table(index="target", columns="tf", values="estimate", fill_value=0.0, sort=False)
            coex = cor_df.loc[reg.index, reg.columns].to_numpy() * np.sqrt(np.abs(reg.to_numpy()) + 1)
            coex = pd.DataFrame(coex, index=reg.index, columns=reg.columns)
        else:
            coex = cor_df
        gene_net = use
    else:
        gene_net = meta[meta["target"].isin(features) & meta["tf"].isin(features)].copy()
        if umap_method == "coef":
            coex = gene_net.pivot_table(index="target", columns="tf", values="estimate", fill_value=0.0, sort=False)
    edges = gene_net.rename(columns={"tf": "from_node", "target": "to_node"}).copy()
    edges["dir"] = np.sign(edges["estimate"])
    nodes = list(dict.fromkeys(edges["from_node"].tolist() + edges["to_node"].tolist()))
    node_df = pd.DataFrame({"name": nodes, "centrality": pagerank(nodes, edges)})
    if coex is not None:
        # tidygraph's inner_join on the nodes drops genes without a layout row (TFs that are never a
        # target here) together with their edges; centrality is computed before, as in R
        node_df = node_df.merge(_umap(coex, random_seed, n_pcs, **umap_kwargs), left_on="name", right_on="gene", how="inner").drop(columns="gene")
        kept = set(node_df["name"])
        edges = edges[edges["from_node"].isin(kept) & edges["to_node"].isin(kept)]
    net = GetNetwork(object, network)
    net.graphs[graph_name] = {"nodes": node_df, "edges": edges.reset_index(drop=True)}
    return object


def _umap(coex: pd.DataFrame, seed: int, n_pcs: int, **kw) -> pd.DataFrame:
    try:
        import umap
    except ImportError:
        return pd.DataFrame({"gene": coex.index, "UMAP_1": np.nan, "UMAP_2": np.nan})
    M = coex.to_numpy(dtype=np.float64)
    if M.shape[1] > 100:
        from sklearn.decomposition import TruncatedSVD

        M = TruncatedSVD(n_components=n_pcs, random_state=seed).fit_transform(M - M.mean(axis=0))
    emb = umap.UMAP(random_state=seed, **kw).fit_transform(M)
    return pd.DataFrame({"gene": coex.index, "UMAP_1": emb[:, 0], "UMAP_2": emb[:, 1]})


# --------------------------------------------------------------------------- get_tf_network

def _r_mean(x) -> float:
    """R's ``mean`` of doubles: long-double sum, then one refinement pass (summary.c ``rmean``)."""
    n = len(x)
    if n == 0:
        return math.nan
    x = np.asarray(x, dtype=np.longdouble)
    s = x.sum() / n
    if np.isfinite(s):
        s = s + (x - s).sum() / n
    return float(s)


def _all_shortest_paths(out_edges: dict, source: str, targets) -> list[list[str]]:
    """``igraph::all_shortest_paths(g, source, targets, mode = "out")$res`` on an unweighted graph."""
    dist, preds, q = {source: 0}, {source: []}, deque([source])
    while q:
        v = q.popleft()
        for w in out_edges.get(v, ()):
            if w not in dist:
                dist[w], preds[w] = dist[v] + 1, [v]
                q.append(w)
            elif dist[w] == dist[v] + 1:
                preds[w].append(v)
    paths = []

    def back(v, tail):
        if v == source:
            paths.append([source] + tail)
            return
        for u in preds[v]:
            back(u, [v] + tail)

    for t in targets:
        if t in dist:
            back(t, [])
    return paths


def tf_network_from_edges(
    edges: pd.DataFrame,
    tf: str,
    features,
    order: float = 3,
    keep_all_edges: bool = False,
) -> dict:
    """Pando's ``get_tf_network`` on a module-graph edge table (``from_node``, ``to_node``, attributes).

    Follows R's algorithm step by step (see DISCOVERY.md), with the two fixes described there:
    ``keep_all_edges=False`` prunes on the per-path mean ``-log10(padj)`` when the edges have ``padj``,
    and only paths with at most ``order`` edges are kept (R ignores ``order``; ``order=np.inf``
    reproduces the installed R).
    """
    edges = edges.reset_index(drop=True)
    nodes = set(edges["from_node"]) | set(edges["to_node"])
    feats = [f for f in dict.fromkeys(features) if f in nodes]
    out_edges: dict = {}
    for a, b in zip(edges["from_node"], edges["to_node"]):
        out_edges.setdefault(a, []).append(b)
    eidx = {(a, b): i for i, (a, b) in enumerate(zip(edges["from_node"], edges["to_node"]))}
    spaths = _all_shortest_paths(out_edges, tf, feats)
    if len(spaths) < 3:
        raise ValueError("Selected TF has fewer than 3 targets.")
    has_padj = "padj" in edges.columns
    est = edges["estimate"].to_numpy(dtype=float)
    padj = edges["padj"].to_numpy(dtype=float) if has_padj else None
    rows, path_edges = [], []
    for p in spaths:
        # edges of the path in the module graph's edge order (as tidygraph's subgraph keeps them)
        ix = sorted(eidx[(a, b)] for a, b in zip(p[:-1], p[1:]))
        e = est[ix]
        row = {"end_node": p[-1], "dir": float(np.prod(np.sign(e))) if len(e) else 1.0,
               "path": ";".join(p), "order": len(p) - 1, "mean_estimate": _r_mean(e)}
        if has_padj:
            row["mean_log_padj"] = _r_mean([-math.log10(v) if v > 0 else math.inf for v in padj[ix]])
        rows.append(row)
        path_edges.append(ix)
    paths = pd.DataFrame(rows)
    keep = (paths["order"] <= order).to_numpy()
    if not keep_all_edges:
        col = "mean_log_padj" if "mean_log_padj" in paths.columns else "mean_estimate"
        best = np.zeros(len(paths), dtype=bool)
        for _, idx in paths[keep].groupby("end_node", sort=False).groups.items():
            v = paths.loc[idx, col].to_numpy()
            if not np.isnan(v).any():          # R: max() is NaN and every comparison NA otherwise
                best[np.asarray(idx)[v == v.max()]] = True
        keep &= (paths["order"].to_numpy() == 1) | best
    seen, sel = set(), []
    for k in np.flatnonzero(keep):
        for i in path_edges[k]:
            key = (i, paths.at[k, "end_node"], paths.at[k, "dir"])
            if key not in seen:
                seen.add(key)
                sel.append(key)
    # inner_join(module edges, kept path edges): module-edge order, then path order; arrange(comb_dir)
    by_edge: dict = {}
    for i, end, d in sel:
        by_edge.setdefault(i, []).append((end, d))
    out = []
    for i in sorted(by_edge):
        for end, d in by_edge[i]:
            out.append((i, end, d))
    res = edges.loc[[i for i, _, _ in out]].reset_index(drop=True)
    res["end_node"] = [e for _, e, _ in out]
    res["comb_dir"] = [d for _, _, d in out]
    res = res.sort_values("comb_dir", kind="stable").reset_index(drop=True)
    node_names = list(dict.fromkeys(res["from_node"].tolist() + res["to_node"].tolist()))
    return {"nodes": pd.DataFrame({"name": node_names}), "edges": res, "paths": paths.assign(kept=keep)}


def get_tf_network(
    object: GRNData,
    tf: str,
    network: str | None = None,
    graph: str = "module_graph",
    features=None,
    order: float = 3,
    keep_all_edges: bool = False,
    verbose: bool = True,
    parallel: bool = False,
) -> GRNData:
    """Pando ``get_tf_network``: sub-network rooted at ``tf``, stored in ``graphs["tf_graphs"][tf]``.

    Deviation from Pando 1.1.1 (a bug fix, see DISCOVERY.md): with ``keep_all_edges=False`` each leaf
    keeps its direct edge and its most significant path (largest mean ``-log10(padj)``); R's check for
    this looks up a column that never exists and silently keeps the path with the largest mean
    estimate. ``order`` limits the path length; R ignores it (pass ``order=np.inf`` for R's output with
    ``keep_all_edges=True``). ``parallel`` is accepted for API compatibility.
    """
    network = network or DefaultNetwork(object)
    g = NetworkGraph(object, network, graph)
    if features is None:
        features = list(NetworkFeatures(object, network))
    res = tf_network_from_edges(g["edges"], tf, features, order=order, keep_all_edges=keep_all_edges)
    net = GetNetwork(object, network)
    net.graphs.setdefault("tf_graphs", {})[tf] = {"nodes": res["nodes"], "edges": res["edges"]}
    return object
