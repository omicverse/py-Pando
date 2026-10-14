"""Fast tests without R: building blocks against values computed once in R 4.4.3."""
import numpy as np
import pandas as pd
import pytest

import pypando as pp
from pypando import ranges as R
from pypando import rstats
from pypando._wilcox import psignrank, wilcox_signed_rank_pvalue
from pypando.formula import PandoFormula
from pypando.glmnet import glmnet_gaussian, lambda_interp
from pypando.rrng import RRandom


def test_import_and_bundled_data():
    assert pp.__version__ == "0.1.2"
    motifs = pp.load_pando_motifs()
    m2t = pp.load_motif2tf()
    assert len(motifs) == 1590
    # Pando's lookup is positional: motif2tf's distinct motifs are in the same order as the motifs
    assert list(dict.fromkeys(m2t["motif"])) == [m.ID for m in motifs]


def test_r_rng_matches_r():
    # R 4.4.3: set.seed(42); sample(rep(seq(10), length = 37))
    r = RRandom(42)
    got = r.sample(np.resize(np.arange(1, 11), 37))
    assert got.tolist() == [7, 1, 5, 10, 8, 6, 7, 5, 4, 7, 4, 5, 5, 4, 10, 3, 10, 3, 9, 7,
                            2, 1, 3, 1, 2, 2, 8, 3, 6, 2, 6, 4, 1, 6, 9, 8, 9]
    # set.seed(7); sample(5); runif(2)
    r = RRandom(7)
    assert r.sample_int(5).tolist() == [2, 3, 4, 5, 1]
    assert r.unif_rand() == pytest.approx(0.7920104258228093, abs=1e-15)


def test_ranges_ops():
    peaks = R.string_to_ranges(["chr1-100-200", "chr1-300-400", "chr2-50-60"])
    exons = R.make_ranges(["chr1", "chr1"], [150, 390], [160, 500], "+")
    sub = R.subtract(peaks, exons)
    assert R.ranges_to_string(sub).tolist() == ["chr1-100-149", "chr1-161-200", "chr1-300-389", "chr2-50-60"]
    genes = R.make_ranges(["chr1", "chr1"], [1000, 1000], [2000, 2000], ["+", "-"])
    ext = R.extend(genes, upstream=100, downstream=10)
    assert ext["start"].tolist() == [900, 990] and ext["end"].tolist() == [2010, 2100]
    q, s = R.find_overlaps(R.string_to_ranges(["chr1-890-905", "chr1-2095-2200"]), ext)
    assert list(zip(q, s)) == [(0, 0), (1, 1)]
    assert R.reduce(R.make_ranges(["c", "c", "c"], [1, 11, 30], [10, 20, 40])).shape[0] == 2  # adjacent merge


def test_glm_matches_r_with_aliased_column():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(60, 3))
    y = X @ [1.0, 0.5, -0.3] + rng.normal(size=60)
    D = np.column_stack([np.ones(60), X[:, 0], 2 * X[:, 1], X[:, 1], X[:, 2]])
    res = rstats.glm_gaussian(D, y, ["(Intercept)", "a", "A", "b", "c"])
    assert res["rank"] == 4
    assert list(res["coefs"]["term"]) == ["(Intercept)", "a", "A", "c"]  # 'b' aliased, as in R


def test_p_adjust_bh_with_na():
    p = np.array([0.01, np.nan, 0.04, 0.03, 0.2])
    out = rstats.p_adjust(p, "fdr")
    assert np.isnan(out[1])
    np.testing.assert_allclose(out[[0, 2, 3, 4]], [0.04, 0.0533333333333, 0.0533333333333, 0.2], rtol=1e-10)


def test_wilcox_exact_and_asymptotic():
    assert psignrank(3, 5) == pytest.approx(5 / 32)
    x = np.array([1.2, -0.3, 2.2, 0.8, -1.1, 0.4])
    # R 4.4.3: wilcox.test(c(1.2,-0.3,2.2,0.8,-1.1,0.4))$p.value == 0.3125
    assert wilcox_signed_rank_pvalue(x) == pytest.approx(0.3125)


def test_formula_labels_follow_r():
    f = PandoFormula("g", [("p1", "tfA"), ("p1", "tfB"), ("p2", "tfA")], ":")
    data = {k: np.arange(4.0) + i for i, k in enumerate(["g", "p1", "tfA", "tfB", "p2"])}
    _, names = f.model_matrix(data)
    assert names == ["(Intercept)", "p1:tfA", "p1:tfB", "tfA:p2"]
    _, gnames = f.glmnet_matrix(data)
    assert gnames == ["p1:tfA", "p1:tfB", "p2:tfA"]
    with pytest.raises(ValueError):
        PandoFormula("1bad", [("p1", "tfA")]).check_parse()


def test_glmnet_lambda_interp_identity_at_knots():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 4)); y = X[:, 0] + rng.normal(size=50)
    fit = glmnet_gaussian(X, y, alpha=0.5)
    left, right, frac = lambda_interp(fit["lambda"], fit["lambda"][[3, 10]])
    assert left.tolist() == [3, 10] and np.allclose(frac, 1.0)


def test_format_coefs_swaps_tf_region():
    coefs = pd.DataFrame({"target": ["G", "G", "G"], "term": ["(Intercept)", "chr1_10_20:TFA", "TFB:chr2_5_9"],
                          "estimate": [1.0, 2.0, 3.0], "pval": [0.5, 0.01, 0.02]})
    out = pp.format_coefs(coefs)
    assert out["tf"].tolist() == ["TFA", "TFB"]
    assert out["region"].tolist() == ["chr1-10-20", "chr2-5-9"]
    assert list(out.columns[:4]) == ["tf", "target", "region", "term"]


def _toy_module_graph():
    # A -> B -> D (activating, weak) and A -> C -> D (repressive, strong); A -> E; D -> F
    e = pd.DataFrame({
        "from_node": ["A", "A", "B", "C", "A", "D"],
        "to_node": ["B", "C", "D", "D", "E", "F"],
        "estimate": [0.9, -0.5, 0.9, -0.5, 0.2, 0.3],
        "padj": [0.04, 1e-10, 0.04, 1e-10, 0.01, 0.02],
    })
    e["dir"] = np.sign(e["estimate"])
    return e


def test_tf_network_paths_and_pruning():
    from pypando.graph import tf_network_from_edges

    e = _toy_module_graph()
    feats = ["A", "B", "C", "D", "E", "F"]
    full = tf_network_from_edges(e, "A", feats, order=np.inf, keep_all_edges=True)
    p = full["paths"].set_index("path")
    # all shortest paths, including R's zero-length path to the root itself
    assert set(p.index) == {"A", "A;B", "A;C", "A;E", "A;B;D", "A;C;D", "A;B;D;F", "A;C;D;F"}
    assert p.loc["A;C;D", "dir"] == 1.0 and p.loc["A;C;D", "order"] == 2
    assert np.isclose(p.loc["A;C;D", "mean_log_padj"], 10.0)
    # default: leaf D keeps its most significant path (A;C;D), not the largest mean estimate (A;B;D)
    d = tf_network_from_edges(e, "A", feats)["edges"]
    to_d = set(zip(d.loc[d.end_node == "D", "from_node"], d.loc[d.end_node == "D", "to_node"]))
    assert to_d == {("A", "C"), ("C", "D")}
    # order is honoured (R ignores it): with order=1 only direct edges remain
    o1 = tf_network_from_edges(e, "A", feats, order=1, keep_all_edges=True)["edges"]
    assert set(o1.end_node) == {"B", "C", "E"}
    # one row per (edge, end node, combined sign), sorted by comb_dir as R's arrange()
    assert len(full["edges"]) == 12   # D -> F reached through B and through C, same sign: kept once
    assert full["edges"]["comb_dir"].is_monotonic_increasing
    with pytest.raises(ValueError, match="fewer than 3"):
        tf_network_from_edges(e, "D", ["D", "F"])


def test_tree_layout_matches_ggraph():
    # ggraph 2.2.2 / igraph 2.3.4: create_layout(as_tbl_graph(edges), "tree", circular = TRUE) on the toy graph
    from pypando.graph import tf_network_from_edges
    from pypando.plots import _tree_layout

    want = {True: {"A": (0, 0), "B": (0.288675, 0.166667), "C": (0, -0.333333), "D": (0.57735, 0.333333),
                   "E": (-0.288675, 0.166667), "F": (0.866025, 0.5)},
            False: {"A": (0, 0), "C": (0.288675, 0.166667), "D": (0.57735, 0.333333), "B": (0, -0.333333),
                    "E": (-0.288675, 0.166667), "F": (0.866025, 0.5)}}
    for keep, w in want.items():
        r = tf_network_from_edges(_toy_module_graph(), "A", list("ABCDEF"), keep_all_edges=keep, order=np.inf)
        e = r["edges"]
        pos, _ = _tree_layout("A", list(zip(e.from_node, e.to_node)), list(r["nodes"]["name"]), True)
        for v, xy in w.items():
            assert np.allclose(pos[v], xy, atol=1e-6), (keep, v, pos[v], xy)


def _toy_network():
    from pypando.graph import tf_network_from_edges
    from pypando.objects import Modules, Network

    e = _toy_module_graph()
    fit = pd.DataFrame({"target": list("BCDEFGH"), "rsq": [0.05, 0.3, 0.6, 0.2, 0.8, 0.0, 0.45],
                        "nvariables": [3, 20, 150, 8, 1200, 0, 40]})
    meta = e.rename(columns={"from_node": "tf", "to_node": "target"}).assign(
        n_regions=[2, 5, 3, 1, 8, 4], n_tfs=[1, 1, 2, 2, 1, 1], n_genes=[3, 3, 2, 2, 3, 1])
    net = Network(features=np.array(list("ABCDEF")), fit=fit, coefs=pd.DataFrame(),
                  modules=Modules(meta=meta, params={"p_thresh": 0.05, "rsq_thresh": 0.1, "nvar_thresh": 10}))
    nodes = pd.DataFrame({"name": list("ABCDEF"), "centrality": np.linspace(0.1, 0.3, 6),
                          "UMAP_1": np.arange(6.0), "UMAP_2": np.arange(6.0) ** 0.5})
    net.graphs["module_graph"] = {"nodes": nodes, "edges": e}
    res = tf_network_from_edges(e, "A", list("ABCDEF"))
    net.graphs["tf_graphs"] = {"A": {"nodes": res["nodes"], "edges": res["edges"]}}
    return net, res


def test_plots_return_figures():
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.collections import LineCollection

    net, res = _toy_network()
    fig = pp.plot_gof(net)
    assert len(fig.axes) == 3                          # scatter + two marginal histograms
    fig = pp.plot_module_metrics(net)
    assert [a.get_title(loc="left") for a in fig.axes] == ["# target genes\nper TF", "# regions\nper target gene",
                                                           "# TFs per\ntarget gene"]
    ax = pp.plot_network_graph(net)
    lc = [c for c in ax.collections if isinstance(c, LineCollection)][0]
    assert len(lc.get_segments()) == len(net.graphs["module_graph"]["edges"])
    assert len(ax.texts) == 6                          # every gene labelled
    n_edges = len(set(zip(res["edges"].from_node, res["edges"].to_node, res["edges"].dir)))
    for circ in (True, False):
        fig = pp.plot_tf_network(net, "A", circular=circ, label_nodes="all")
        lc = [c for c in fig.axes[0].collections if isinstance(c, LineCollection)][0]
        assert len(lc.get_segments()) == n_edges
