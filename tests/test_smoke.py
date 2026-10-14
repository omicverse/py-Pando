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
    assert pp.__version__ == "0.1.0"
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
