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

import numpy as np
import pandas as pd

from .objects import GRNData, GetNetwork, DefaultNetwork, NetworkFeatures, NetworkModules
from .utils import sparse_cor

__all__ = ["get_network_graph", "pagerank"]


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
        node_df = node_df.merge(_umap(coex, random_seed, n_pcs, **umap_kwargs), left_on="name", right_on="gene", how="inner").drop(columns="gene")
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
