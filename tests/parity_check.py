"""Apply the pre-registered gates of ``data/manifest.yaml`` to an R and a Python run.

    python tests/parity_check.py <ref_dir> <cand_dir> [out.json]

Prints one line per gate and writes all measured values to ``out.json``.
Thresholds are read from the manifest, never from this file.
"""
from __future__ import annotations

import json
from collections import Counter
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

HERE = Path(__file__).resolve().parent
MANIFEST = yaml.safe_load((HERE.parent / "data" / "manifest.yaml").read_text())
GATES = {o["name"]: o for o in MANIFEST["outputs"]}


def _read_ref_coefs(ref: Path, m: str) -> pd.DataFrame:
    df = pd.read_csv(ref / f"coefs_{m}.tsv", sep="\t", keep_default_na=True, na_values=["NA"])
    for col in df.columns:
        f = ref / f"coefs_{m}_{col}.f64"
        if f.exists():
            df[col] = np.fromfile(f, dtype="<f8")
    return df


def _read_ref_gof(ref: Path, m: str) -> pd.DataFrame:
    df = pd.read_csv(ref / f"gof_{m}.tsv", sep="\t", na_values=["NA"])
    for col in df.columns:
        f = ref / f"gof_{m}_{col}.f64"
        if f.exists():
            df[col] = np.fromfile(f, dtype="<f8")
    return df


def _norm_rel(a, b, rtol, atol):
    a, b = np.asarray(a, float), np.asarray(b, float)
    both_nan = np.isnan(a) & np.isnan(b)
    with np.errstate(invalid="ignore"):
        err = np.abs(a - b) / (rtol * np.abs(a) + atol)
    err[both_nan] = 0.0
    err[np.isnan(err)] = np.inf
    return float(err.max()) if len(err) else 0.0


def _neglog10(p):
    p = np.asarray(p, float)
    with np.errstate(divide="ignore"):
        return -np.log10(np.clip(p, 1e-323, None))


def _set_compare(ref_rows, cand_rows):
    r, c = set(ref_rows), set(cand_rows)
    return len(r & c) / max(1, len(r | c))


def gate_regions(ref, cand, name="candidate_regions"):
    r = pd.read_csv(ref / f"{name}.tsv", sep="\t")
    c = pd.read_csv(cand / f"{name}.tsv", sep="\t")
    same = len(r) == len(c) and (r["region"].to_numpy() == c["region"].to_numpy()).all() \
        and (r["peak"].to_numpy() == c["peak"].to_numpy()).all()
    return {"identical": bool(same), "n_ref": len(r), "n_cand": len(c),
            "pass": bool(same) and GATES["candidate_regions"]["threshold"] == 1.0}


def gate_peaks_near(ref, cand, m):
    def load(d):
        ij = pd.read_csv(d / f"peaks_near_genes_{m}_ij.tsv", sep="\t")
        cols = (d / f"peaks_near_genes_{m}_cols.txt").read_text().split("\n")[:-1]
        return set(zip(ij["i"], ij["j"], ij["x"])), cols
    (rs, rc), (cs, cc) = load(ref), load(cand)
    same = rs == cs and rc == cc
    return {"identical": bool(same), "n_ref": len(rs), "n_cand": len(cs), "pass": bool(same)}


def gate_motifs(ref, cand):
    rr = (ref / "motif_matrix_rows.txt").read_text().split()
    cr = (cand / "motif_matrix_rows.txt").read_text().split()
    rc = (ref / "motif_matrix_cols.txt").read_text().split()
    cc = (cand / "motif_matrix_cols.txt").read_text().split()
    if rr != cr or rc != cc:
        return {"pass": False, "reason": "row/column names differ"}
    r = pd.read_csv(ref / "motif_matrix_ij.tsv", sep="\t")
    c = pd.read_csv(cand / "motif_matrix_ij.tsv", sep="\t")
    rs, cs = set(zip(r["i"], r["j"])), set(zip(c["i"], c["j"]))
    total = len(rr) * len(rc)
    diff = len(rs ^ cs)
    agreement = 1 - diff / total
    jacc = len(rs & cs) / max(1, len(rs | cs))
    th = GATES["motif_matrix"]["threshold"]
    tfs_same = (ref / "tfs_use.txt").read_text().split() == (cand / "tfs_use.txt").read_text().split()
    return {"agreement": agreement, "jaccard": jacc, "n_diff": diff, "n_ref": len(rs), "n_cand": len(cs),
            "tfs_use_identical": tfs_same,
            "pass": agreement >= th["agreement"] and jacc >= th["jaccard"] and tfs_same}


KEY = ["target", "tf", "region"]


def _merge(ref, cand, cols, key=KEY):
    r = ref.copy(); c = cand.copy()
    for d in (r, c):
        d["region"] = d["region"].fillna("NA_")
    return r[key + cols].merge(c[key + cols], on=key, how="outer", suffixes=("_r", "_c"), indicator=True)


def gate_glm(ref, cand):
    r, c = _read_ref_coefs(ref, "glm"), pd.read_csv(cand / "coefs_glm.tsv", sep="\t", na_values=["NA"])
    rg, cg = _read_ref_gof(ref, "glm"), pd.read_csv(cand / "gof_glm.tsv", sep="\t")
    keys_same = set(map(tuple, r[KEY].astype(str).values)) == set(map(tuple, c[KEY].astype(str).values)) \
        and set(rg["target"]) == set(cg["target"]) and len(r) == len(c)
    m = _merge(r, c, ["estimate", "std_err", "statistic", "pval", "padj", "corr"])
    m = m[m["_merge"] == "both"]
    g = GATES["glm_coefficients"]
    rtol, atol = g["parity_rtol"], g["parity_atol"]
    errs = {col: _norm_rel(m[f"{col}_r"], m[f"{col}_c"], rtol, atol) for col in ("estimate", "std_err", "statistic", "corr")}
    gm = rg.merge(cg, on="target", suffixes=("_r", "_c"))
    errs["gof_rsq"] = _norm_rel(gm["rsq_r"], gm["rsq_c"], rtol, atol)
    errs["gof_nvariables"] = float(np.max(np.abs(gm["nvariables_r"] - gm["nvariables_c"]))) if len(gm) else 0.0
    coef_pass = max(v for k, v in errs.items() if k != "gof_nvariables") <= g["threshold"] and errs["gof_nvariables"] == 0
    lp_r, lp_c = _neglog10(m["pval_r"]), _neglog10(m["pval_c"])
    la_r, la_c = _neglog10(m["padj_r"]), _neglog10(m["padj_c"])
    ok = ~(np.isnan(lp_r) | np.isnan(lp_c))
    dlog = float(np.max(np.abs(lp_r[ok] - lp_c[ok]))) if ok.any() else 0.0
    dlog_adj = float(np.max(np.abs(la_r[ok] - la_c[ok]))) if ok.any() else 0.0
    nan_same = bool((np.isnan(lp_r) == np.isnan(lp_c)).all())
    sp = float(stats.spearmanr(lp_r[ok], lp_c[ok]).statistic) if ok.sum() > 2 else 1.0
    pth = GATES["glm_pvalues"]["threshold"]
    return {
        "glm_keys": {"identical": bool(keys_same), "n_ref": len(r), "n_cand": len(c), "pass": bool(keys_same)},
        "glm_coefficients": {"max_normalised_error": errs, "pass": bool(coef_pass)},
        "glm_pvalues": {"max_abs_dlog10p": dlog, "max_abs_dlog10padj": dlog_adj, "spearman": sp,
                        "nan_pattern_identical": nan_same,
                        "pass": dlog <= pth["max_abs_dlog10p"] and dlog_adj <= pth["max_abs_dlog10p"]
                        and sp >= pth["spearman"] and nan_same},
    }


def gate_glmnet(ref, cand, m, gname):
    r, c = _read_ref_coefs(ref, m), pd.read_csv(cand / f"coefs_{m}.tsv", sep="\t", na_values=["NA"])
    rg, cg = _read_ref_gof(ref, m), pd.read_csv(cand / f"gof_{m}.tsv", sep="\t")
    mm = _merge(r, c, ["estimate"])
    er = mm["estimate_r"].fillna(0.0).to_numpy(); ec = mm["estimate_c"].fillna(0.0).to_numpy()
    pear = float(np.corrcoef(er, ec)[0, 1]) if len(er) > 2 else 1.0
    sr = set(map(tuple, r.loc[r["estimate"] != 0, KEY].fillna("NA_").astype(str).values))
    sc = set(map(tuple, c.loc[c["estimate"] != 0, KEY].fillna("NA_").astype(str).values))
    jac = len(sr & sc) / max(1, len(sr | sc))
    gm = rg.merge(cg, on="target", suffixes=("_r", "_c"))
    lam_rel = float(np.max(np.abs(gm["lambda_r"] - gm["lambda_c"]) / np.abs(gm["lambda_r"]))) if len(gm) else 0.0
    same_lambda_frac = float(np.mean(np.abs(gm["lambda_r"] - gm["lambda_c"]) / np.abs(gm["lambda_r"]) <= 1e-4)) if len(gm) else 1.0
    th = GATES[gname]["threshold"]
    return {"pearson": pear, "support_jaccard": jac, "lambda_rel_max": lam_rel,
            "frac_targets_same_lambda": same_lambda_frac, "n_targets_ref": len(rg), "n_targets_cand": len(cg),
            "max_abs_estimate_diff": float(np.max(np.abs(er - ec))) if len(er) else 0.0,
            "rsq_max_abs_diff": float(np.max(np.abs(gm["rsq_r"] - gm["rsq_c"]))) if len(gm) else 0.0,
            "pass": pear >= th["pearson"] and jac >= th["support_jaccard"] and lam_rel <= th["lambda_rel"]}


def gate_xgb(ref, cand):
    r, c = _read_ref_coefs(ref, "xgb"), pd.read_csv(cand / "coefs_xgb.tsv", sep="\t", na_values=["NA"])
    sps, jacs = [], []
    for t, rr in r.groupby("target"):
        cc = c[c["target"] == t]
        mm = rr[KEY + ["gain"]].fillna("NA_").merge(cc[KEY + ["gain"]].fillna("NA_"), on=KEY, how="outer").fillna(0.0)
        if len(mm) > 2:
            sps.append(stats.spearmanr(mm["gain_x"], mm["gain_y"]).statistic)
        top_r = set(map(tuple, rr.nlargest(10, "gain")[KEY].fillna("NA_").values))
        top_c = set(map(tuple, cc.nlargest(10, "gain")[KEY].fillna("NA_").values))
        jacs.append(len(top_r & top_c) / max(1, len(top_r | top_c)))
    th = GATES["xgb_gain"]["threshold"]
    msp, mj = float(np.nanmedian(sps)), float(np.median(jacs))
    return {"median_spearman": msp, "median_top10_jaccard": mj, "n_targets": len(jacs),
            "pass": msp >= th["spearman"] and mj >= th["top10_jaccard"]}


def gate_ridge(ref, cand, m):
    r, c = _read_ref_coefs(ref, m), pd.read_csv(cand / f"coefs_{m}.tsv", sep="\t", na_values=["NA"])
    g = GATES[m]
    cols = ["estimate", "pval"] + (["est_variance"] if m == "bayesian_ridge" else [])
    mm = _merge(r, c, cols)
    keys_same = bool((mm["_merge"] == "both").all())
    mm = mm[mm["_merge"] == "both"]
    err = {col: _norm_rel(mm[f"{col}_r"], mm[f"{col}_c"], g["parity_rtol"], g["parity_atol"]) for col in cols if col != "pval"}
    lp_r, lp_c = _neglog10(mm["pval_r"]), _neglog10(mm["pval_c"])
    ok = ~(np.isnan(lp_r) | np.isnan(lp_c))
    dlog = float(np.max(np.abs(lp_r[ok] - lp_c[ok]))) if ok.any() else 0.0
    th = g["threshold"]
    return {"keys_identical": keys_same, "max_normalised_error": err, "max_abs_dlog10p": dlog,
            "pass": keys_same and max(err.values()) <= th["coef"] and dlog <= th["max_abs_dlog10p"]}


def gate_modules(ref, cand):
    rf = json.loads((ref / "modules_features.json").read_text())
    cf = json.loads((cand / "modules_features.json").read_text())
    jac, tf_sets = [], []
    for k in ("genes_pos", "genes_neg", "regions_pos", "regions_neg"):
        rk, ck = rf.get(k) or {}, cf.get(k) or {}
        if isinstance(rk, list):  # empty named list -> []
            rk = {}
        tf_sets.append(_set_compare(rk.keys(), ck.keys()) if (rk or ck) else 1.0)
        for tf in set(rk) | set(ck):
            jac.append(_set_compare(rk.get(tf, []), ck.get(tf, [])))
    th = GATES["modules"]["threshold"]
    mj = float(np.mean(jac)) if jac else 1.0
    return {"mean_jaccard": mj, "tf_set_jaccard": min(tf_sets), "n_modules": len(jac),
            "pass": mj >= th["mean_jaccard"] and min(tf_sets) >= th["tf_set_jaccard"]}


def main(ref, cand, out=None):
    ref, cand = Path(ref), Path(cand)
    res = {}
    res["candidate_regions"] = gate_regions(ref, cand)
    if (ref / "candidate_regions_phastcons.tsv").exists() and (cand / "candidate_regions_phastcons.tsv").exists():
        res["candidate_regions_phastcons"] = gate_regions(ref, cand, "candidate_regions_phastcons")
    res["peaks_near_genes_signac"] = gate_peaks_near(ref, cand, "Signac")
    res["peaks_near_genes_great"] = gate_peaks_near(ref, cand, "GREAT")
    res["motif_matrix"] = gate_motifs(ref, cand)
    if (ref / "coefs_glm.tsv").exists():
        res.update(gate_glm(ref, cand))
        res["modules"] = gate_modules(ref, cand)
    for m, gname in (("glmnet", "glmnet_coefficients"), ("cv.glmnet", "cvglmnet_coefficients")):
        if (ref / f"coefs_{m}.tsv").exists() and (cand / f"coefs_{m}.tsv").exists():
            res[gname] = gate_glmnet(ref, cand, m, gname)
    if (ref / "coefs_xgb.tsv").exists() and (cand / "coefs_xgb.tsv").exists():
        res["xgb_gain"] = gate_xgb(ref, cand)
    for m in ("bagging_ridge", "bayesian_ridge"):
        if (ref / f"coefs_{m}.tsv").exists() and (cand / f"coefs_{m}.tsv").exists():
            res[m] = gate_ridge(ref, cand, m)
    for k, v in res.items():
        print(f"{'PASS' if v['pass'] else 'FAIL'}  {k}: " + json.dumps({a: b for a, b in v.items() if a != 'pass'}, default=float))
    if out:
        Path(out).write_text(json.dumps(res, indent=1, default=float))
    return res


# --- get_tf_network (manifest amendment 2026-10-14) -------------------------------------
TF_NUMERIC = ("estimate", "n_regions", "n_genes", "n_tfs", "pval", "padj", "corr", "dir", "comb_dir")
TF_VARIANTS = {"tf_network_all_edges": [("R_all", "R_all")],
               "tf_network_default_intent": [("intent_default", "intent_default"), ("intent_all", "intent_all"),
                                             ("intent_default_o2", "intent_default_o2")]}


def _tf_key(df):
    return list(zip(df["from_name"], df["to_name"], df["end_node"], df["comb_dir"].astype(float)))


def gate_tf_network(ref, cand, gname):
    g = GATES[gname]
    rt = pd.read_csv(ref / "tf_networks.tsv", sep="\t", na_values=["NA"], keep_default_na=False)
    ct = pd.read_csv(cand / "tf_networks.tsv", sep="\t", na_values=["NA"], keep_default_na=False)
    if (ref / "tf_networks_estimate.f64").exists():  # full-precision copy, same row order
        rt["estimate"] = np.fromfile(ref / "tf_networks_estimate.f64", dtype="<f8")
    rs = json.loads((ref / "tf_networks_status.json").read_text())
    cs = json.loads((cand / "tf_networks_status.json").read_text())
    tfs_r = (ref / "tf_network_tfs.txt").read_text().split()
    tfs_c = (cand / "tf_network_tfs.txt").read_text().split()
    same_nodes = same_edges = tfs_r == tfs_c
    worst, n_edges, per_tf = 0.0, 0, {}
    for vr, vc in TF_VARIANTS[gname]:
        for tf in tfs_r:
            er = rs.get(f"{vr}/{tf}", "").startswith("error")
            ec = cs.get(f"{vc}/{tf}", "").startswith("error")
            if er or ec:
                ok = er and ec
                same_nodes &= ok; same_edges &= ok
                per_tf[f"{vr}/{tf}"] = "both error" if ok else "error on one side"
                continue
            r = rt[(rt.variant == vr) & (rt.root == tf)].copy()
            c = ct[(ct.variant == vc) & (ct.root == tf)].copy()
            nr = set(r.from_name) | set(r.to_name); nc = set(c.from_name) | set(c.to_name)
            kr, kc = Counter(_tf_key(r)), Counter(_tf_key(c))
            same_nodes &= nr == nc; same_edges &= kr == kc
            n_edges += len(r)
            r["k"], c["k"] = _tf_key(r), _tf_key(c)
            m = r.drop_duplicates("k").merge(c.drop_duplicates("k"), on="k", suffixes=("_r", "_c"))
            for col in TF_NUMERIC:
                if f"{col}_r" in m and f"{col}_c" in m:
                    worst = max(worst, _norm_rel(m[f"{col}_r"], m[f"{col}_c"], g["parity_rtol"], g["parity_atol"]))
            per_tf[f"{vr}/{tf}"] = f"{len(r)} vs {len(c)} edges" + ("" if kr == kc else " DIFFER")
    th = g["threshold"]
    return {"identical_nodes": bool(same_nodes), "identical_edges": bool(same_edges), "attr": worst,
            "n_edges_ref": n_edges, "tfs": tfs_r, "per_tf": per_tf,
            "pass": bool(same_nodes) == th["identical_nodes"] and bool(same_edges) == th["identical_edges"] and worst <= th["attr"]}


def tf_main(ref, cand, out=None):
    ref, cand = Path(ref), Path(cand)
    res = {k: gate_tf_network(ref, cand, k) for k in TF_VARIANTS}
    for k, v in res.items():
        print(f"{'PASS' if v['pass'] else 'FAIL'}  {k}: " + json.dumps({a: b for a, b in v.items() if a not in ('pass', 'per_tf')}, default=float))
    if out:
        Path(out).write_text(json.dumps(res, indent=1, default=float))
    return res


if __name__ == "__main__":
    if sys.argv[1] == "--tf":
        tf_main(*sys.argv[2:5])
    else:
        main(*sys.argv[1:4])
