"""Per-gene model fits (Pando ``models.R``).

Each ``fit_*`` takes a :class:`~pypando.formula.PandoFormula` and a column
dictionary (sanitised variable name -> values) and returns
``{"gof": {...}, "coefs": DataFrame}`` with the columns Pando returns.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import rstats
from .formula import PandoFormula
from .glmnet import cv_glmnet_gaussian, glmnet_gaussian

__all__ = [
    "fit_model",
    "fit_glm",
    "fit_glmnet",
    "fit_cvglmnet",
    "fit_xgb",
    "fit_bagging_ridge",
    "fit_bayesian_ridge",
    "fit_brms",
    "r2",
]

METHODS = ("glm", "glmnet", "cv.glmnet", "brms", "xgb", "bagging_ridge", "bayesian_ridge")


def _sse(y_true, y_pred):
    return float(np.sum((np.asarray(y_true) - np.asarray(y_pred)) ** 2))


def r2(y_true, y_pred) -> float:
    """Pando's ``r2``: ``1 - SSE / SST``."""
    y_true = np.asarray(y_true, dtype=np.float64)
    return 1.0 - _sse(y_true, y_pred) / _sse(y_true, np.mean(y_true))


def fit_glm(formula: PandoFormula, data: dict, family="gaussian", **kwargs):
    """``glm(formula, data, family = gaussian)`` + ``summary`` (Gaussian only)."""
    fam = family if isinstance(family, str) else getattr(family, "__name__", str(family))
    if fam != "gaussian":
        raise NotImplementedError("py-Pando's glm supports family = 'gaussian' (Pando's default)")
    X, names = formula.model_matrix(data, intercept=True)
    y = np.asarray(data[formula.target], dtype=np.float64)
    res = rstats.glm_gaussian(X, y, names)
    coefs = pd.DataFrame(res["coefs"])[["term", "estimate", "std_err", "statistic", "pval"]]
    return {"gof": {"rsq": res["rsq"]}, "coefs": coefs}


def _glmnet_check(X):
    # glmnet: "x should be a matrix with 2 or more columns" -> Pando skips the gene
    if X.shape[1] < 2:
        raise ValueError("x should be a matrix with 2 or more columns")


def _coef_table(names, intercept, beta):
    return pd.DataFrame({"term": ["(Intercept)"] + list(names), "estimate": np.r_[intercept, beta]})


def fit_glmnet(formula: PandoFormula, data: dict, family="gaussian", alpha: float = 0.5, **kwargs):
    """``glmnetUtils::glmnet``; lambda = first with dev.ratio > 0.95 * max."""
    X, names = formula.glmnet_matrix(data)
    _glmnet_check(X)
    y = np.asarray(data[formula.target], dtype=np.float64)
    fit = glmnet_gaussian(X, y, alpha=alpha)
    dev = fit["dev_ratio"]
    which_max = int(np.flatnonzero(dev > np.max(dev) * 0.95)[0])
    gof = {"lambda": fit["lambda"][which_max], "rsq": dev[which_max], "alpha": alpha}
    coefs = _coef_table(names, fit["a0"][which_max], fit["beta"][:, which_max])
    return {"gof": gof, "coefs": coefs}


def fit_cvglmnet(formula: PandoFormula, data: dict, family="gaussian", alpha: float = 0.5, rng=None, **kwargs):
    """``glmnetUtils::cv.glmnet``; coefficients and rsq at ``lambda.1se``."""
    X, names = formula.glmnet_matrix(data)
    if X.shape[1] < 2 and rng is not None and kwargs.get("foldid") is None:
        # R's cv.glmnet draws the folds before glmnet rejects the matrix: keep the stream in step
        from .glmnet import draw_folds

        draw_folds(rng, X.shape[0], kwargs.get("nfolds", 10))
    _glmnet_check(X)
    y = np.asarray(data[formula.target], dtype=np.float64)
    fit = cv_glmnet_gaussian(X, y, alpha=alpha, rng=rng, **kwargs)
    i = fit["index"]["1se"]
    full = fit["glmnet_fit"]
    gof = {"lambda": fit["lambda_1se"], "rsq": full["dev_ratio"][i], "alpha": alpha}
    coefs = _coef_table(names, full["a0"][i], full["beta"][:, i])
    return {"gof": gof, "coefs": coefs}


def fit_xgb(
    formula: PandoFormula,
    data: dict,
    params: dict | None = None,
    nrounds: int = 1000,
    nthread: int = 1,
    **kwargs,
):
    """``xgboost`` on ``model.matrix(formula)``; gain / cover / frequency importance."""
    import xgboost as xgb

    if params is None:
        params = {"max_depth": 3, "eta": 0.01, "objective": "reg:squarederror"}
    X, names = formula.model_matrix(data, intercept=True)
    y = np.asarray(data[formula.target], dtype=np.float64)
    dtrain = xgb.DMatrix(X, label=y, feature_names=list(names), nthread=nthread)
    booster = xgb.train({**params, "nthread": nthread}, dtrain, num_boost_round=nrounds, verbose_eval=False)
    pred = booster.predict(dtrain)
    gain = booster.get_score(importance_type="total_gain")
    cover = booster.get_score(importance_type="total_cover")
    freq = booster.get_score(importance_type="weight")
    feats = list(gain)
    g = np.array([gain[f] for f in feats])
    c = np.array([cover[f] for f in feats])
    fq = np.array([freq[f] for f in feats])
    coefs = pd.DataFrame(
        {"term": feats, "gain": g / g.sum(), "cover": c / c.sum(), "frequency": fq / fq.sum()}
    )
    coefs = coefs.sort_values("gain", ascending=False, kind="stable").reset_index(drop=True)
    return {"gof": {"rsq": r2(y, pred)}, "coefs": coefs}


def fit_bagging_ridge(
    formula: PandoFormula,
    data: dict,
    alpha: float = 1.0,
    solver: str = "auto",
    bagging_number: int = 200,
    n_jobs: int = 1,
    p_method: str = "wilcox",
    **kwargs,
):
    """``BaggingRegressor(Ridge)`` exactly as Pando configures it from R.

    ``BaggingRegressor`` has no ``random_state`` in Pando, so it draws from
    NumPy's global RNG: seed it with ``numpy.random.seed`` to reproduce runs.
    """
    from sklearn.ensemble import BaggingRegressor
    from sklearn.linear_model import Ridge

    X, names = formula.model_matrix(data, intercept=True)
    X, names = X[:, 1:], names[1:]
    if X.shape[1] < 2:
        raise ValueError("The bagging ridge model requires at least two variables.")
    y = np.asarray(data[formula.target], dtype=np.float64)
    model = BaggingRegressor(
        estimator=Ridge(alpha=alpha, solver=solver, random_state=123, **kwargs),
        n_estimators=int(bagging_number),
        bootstrap=True,
        max_features=0.8,
        n_jobs=int(n_jobs),
        verbose=False,
    )
    model.fit(np.asfortranarray(X), y)
    p = X.shape[1]
    coefs = np.full((bagging_number, p), np.nan)
    for i, (feat, est) in enumerate(zip(model.estimators_features_, model.estimators_)):
        coefs[i, feat] = np.ravel(est.coef_)
    if p_method == "t":
        pv = np.array([rstats.t_test_one(col[~np.isnan(col)]) for col in coefs.T])
    else:
        pv = np.array([rstats.wilcox_test_one(col[~np.isnan(col)]) for col in coefs.T])
    est = np.nanmean(coefs, axis=0)
    table = pd.DataFrame(
        {
            "term": names,
            "estimate": est,
            "pval": pv,
            "neglog10p": -np.log10(np.where(np.isnan(pv), 1.0, pv)),
        }
    )
    y_pred = X @ est
    return {"gof": {"rsq": r2(y, y_pred)}, "coefs": table}


def fit_bayesian_ridge(formula: PandoFormula, data: dict, **kwargs):
    """``BayesianRidge`` with Pando's normal-approximation p-values."""
    from scipy.stats import norm
    from sklearn.linear_model import BayesianRidge

    X, names = formula.model_matrix(data, intercept=True)
    X, names = X[:, 1:], names[1:]
    if X.shape[1] < 2:
        raise ValueError("The bayesian ridge model requires at least two variables.")
    y = np.asarray(data[formula.target], dtype=np.float64)
    model = BayesianRidge(**kwargs).fit(np.asfortranarray(X), y)
    coef = model.coef_
    var = np.diag(model.sigma_)
    p = norm.cdf(0.0, loc=np.abs(coef), scale=np.sqrt(var)) * 2
    table = pd.DataFrame(
        {
            "term": names,
            "estimate": coef,
            "est_variance": var,
            "pval": p,
            "neglog10p": -np.log10(np.where(np.isnan(p), 1.0, p)),
        }
    )
    return {"gof": {"rsq": model.score(np.asfortranarray(X), y)}, "coefs": table}


def fit_brms(*args, **kwargs):
    """Not ported: Pando's brms models need Stan. Use another ``method``."""
    raise NotImplementedError("method='brms' (Stan via brms) is not ported to Python")


def fit_model(formula: PandoFormula, data: dict, method: str = "glm", family="gaussian", alpha: float = 1.0, **kwargs):
    """Pando ``fit_model``: dispatch to the model function for ``method``."""
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    if method == "glm":
        return fit_glm(formula, data, family=family, **kwargs)
    if method == "glmnet":
        return fit_glmnet(formula, data, family=family, alpha=alpha, **kwargs)
    if method == "cv.glmnet":
        return fit_cvglmnet(formula, data, family=family, alpha=alpha, **kwargs)
    if method == "brms":
        return fit_brms(formula, data, family=family, **kwargs)
    if method == "xgb":
        return fit_xgb(formula, data, **kwargs)
    if method == "bagging_ridge":
        return fit_bagging_ridge(formula, data, alpha=alpha, **kwargs)
    return fit_bayesian_ridge(formula, data, **kwargs)
