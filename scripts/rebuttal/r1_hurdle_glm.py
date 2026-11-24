"""Experiment 2: hurdle Poisson GLM (logistic regression + zero-truncated Poisson GLM).

  P(Y = 0) = 1 - p,  logit p = a0 + a' x
  P(Y = k) = p * Poisson(k | lam) / (1 - exp(-lam)),  log lam = b0 + b' x,  k >= 1

The features x are those of the paper's Poisson GLM (scripts/14_train_window_poisson_glm.py with
config/14_window_poisson_glm_salinity.yaml): coordinates, day of year, the covariates of the target
day (with the counts at t-1 and t-7) and a window of the W previous counts of the cell, standardized
on the training days. The logistic part is fitted on all training cell-days, the zero-truncated part
on the active ones, each with an L2 penalty on the same scale as sklearn's PoissonRegressor
(mean loss + alpha/2 ||w||^2; intercepts unpenalized).

W, alpha_logit and alpha_ztp are selected on the four validation weeks with the paper's criterion
(App. B.2): min-max normalized mean log-likelihood and RMSE of the daily totals, averaged over the
weeks. The selected model is then refitted before each validation and test week.

Outputs (reports/rebuttal/reviewer1/):
  hparam/hurdle_glm_grid.csv, hparam/hurdle_glm_scores.csv, hparam/hurdle_glm_selected.yaml
  predictions/hurdle_glm.parquet   (model, seed, split, week, month, date, ci, cj, y, p, lam)
"""

from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from r1_common import ALL_WEEKS, OUT, PRED_DIR, ROOT, VAL_WEEKS, cell_key, hurdle_mean, hurdle_scores, load_module
from src.models.metrics_lgcp import evaluate_metrics

BASE_CONFIG = "config/14_window_poisson_glm_salinity.yaml"
W_GRID = [1, 3, 7, 14]
ALPHA_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
HP_DIR = OUT / "hparam"

_glm = load_module("r1_glm_feats", "scripts/14_train_window_poisson_glm.py")
_hp = load_module("r1_hp", "scripts/51c_baseline_hparam_rerun_2weeks_before_salinity.py")


def fit_logit(X, b, alpha):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return LogisticRegression(C=1.0 / (alpha * len(b)), max_iter=5000).fit(X, b)


def fit_ztp(X, y, alpha):
    """Zero-truncated Poisson GLM: minimize mean NLL + alpha/2 ||w||^2 with L-BFGS."""
    n, d = X.shape

    def f(w):
        eta = np.clip(w[0] + X @ w[1:], -30, 30)
        lam = np.exp(eta)
        one_m = -np.expm1(-lam)
        g = lam / one_m - y                       # d NLL / d eta
        obj = np.mean(lam + np.log(one_m) - y * eta) + 0.5 * alpha * w[1:] @ w[1:]
        return obj, np.concatenate([[g.mean()], X.T @ g / n + alpha * w[1:]])

    w0 = np.zeros(d + 1)
    w0[0] = np.log(max(y.mean() - 1.0, 0.1))
    res = minimize(f, w0, jac=True, method="L-BFGS-B", options={"maxiter": 10000})
    if not res.success:
        print(f"    ZTP fit (alpha={alpha}) did not converge: {res.message}")
    return res.x


def predict_ztp(w, X):
    return np.exp(np.clip(w[0] + X @ w[1:], -30, 30))


class Features:
    """Window features for a given W, with the train/test split of a week."""

    def __init__(self):
        self.cfg = _glm.load_config(str(ROOT / BASE_CONFIG))
        self.raw = _glm.load_parquet_years(str(ROOT / self.cfg["parquet_path"]), self.cfg.get("years"))
        self.raw["date"] = pd.to_datetime(self.raw["date"])
        self.cache = {}

    def get(self, W):
        if W not in self.cache:
            cfg = dict(self.cfg, window_size=W)
            win, feats = _glm.build_window_dataframe(self.raw, cfg)
            self.cache[W] = (win.reset_index(drop=True), feats)
        return self.cache[W]

    def split(self, W, start, end):
        win, feats = self.get(W)
        tr, te = _glm.make_split_masks(dates=win["date"], split_strategy="fixed_test_window",
                                       train_fraction=self.cfg["train_fraction"], random_seed=self.cfg["data_seed"],
                                       test_start_date=start, test_end_date=end)
        X = win[feats].values.astype(float)
        y = win[self.cfg["target_col"]].values.astype(float)
        sc = StandardScaler().fit(X[tr])
        return sc.transform(X[tr]), y[tr], sc.transform(X[te]), y[te], win.loc[te]


def grid_search(F: Features) -> pd.DataFrame:
    rows = []
    for name, start, end, month in VAL_WEEKS:
        for W in W_GRID:
            Xtr, ytr, Xte, yte, wte = F.split(W, start, end)
            dates = wte["date"].values
            ps = {a: fit_logit(Xtr, (ytr > 0).astype(int), a).predict_proba(Xte)[:, 1] for a in ALPHA_GRID}
            pos = ytr > 0
            lams = {a: predict_ztp(fit_ztp(Xtr[pos], ytr[pos], a), Xte) for a in ALPHA_GRID}
            for a_l, a_z in itertools.product(ALPHA_GRID, ALPHA_GRID):
                ll, _ = hurdle_scores(yte, ps[a_l], lams[a_z])
                m = evaluate_metrics(yte, hurdle_mean(ps[a_l], lams[a_z]), dates)
                rows.append({"week": name, "window_size": W, "alpha_logit": a_l, "alpha_ztp": a_z,
                             "mean_ll_obs": float(ll.mean()), "rmse_daily": m["rmse_daily"], "mae_obs": m["mae_obs"]})
            print(f"  [{name}] W={W} done", flush=True)
    return pd.DataFrame(rows)


def main():
    HP_DIR.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    F = Features()

    print("Grid search on the validation weeks …", flush=True)
    grid = grid_search(F)
    grid.to_csv(HP_DIR / "hurdle_glm_grid.csv", index=False)
    scores = _hp.compute_relative_scores(grid, week_col="week", group_cols=["window_size", "alpha_logit", "alpha_ztp"],
                                         metrics_directions=_hp.METRIC_DIRECTIONS)
    scores.to_csv(HP_DIR / "hurdle_glm_scores.csv", index=False)
    best = scores.iloc[0]
    sel = {"window_size": int(best.window_size), "alpha_logit": float(best.alpha_logit), "alpha_ztp": float(best.alpha_ztp)}
    (HP_DIR / "hurdle_glm_selected.yaml").write_text(yaml.safe_dump(
        {"base_config": BASE_CONFIG, "criterion": "App. B.2 relative score (mean_ll_obs, rmse_daily)",
         "grid": {"window_size": list(W_GRID), "alpha_logit": list(ALPHA_GRID), "alpha_ztp": list(ALPHA_GRID)},
         "selected": sel, "score": float(best.combined_relative_score)}, sort_keys=False))
    print(f"Selected: {sel}", flush=True)

    out = []
    for split, name, start, end, month in ALL_WEEKS:
        Xtr, ytr, Xte, yte, wte = F.split(sel["window_size"], start, end)
        pos = ytr > 0
        p = fit_logit(Xtr, pos.astype(int), sel["alpha_logit"]).predict_proba(Xte)[:, 1]
        lam = predict_ztp(fit_ztp(Xtr[pos], ytr[pos], sel["alpha_ztp"]), Xte)
        ci, cj = cell_key(wte.longitude, wte.latitude)
        out.append(pd.DataFrame({"model": "Hurdle GLM", "seed": 1, "split": split, "week": name, "month": month,
                                 "date": pd.to_datetime(wte["date"].values).normalize(), "ci": ci, "cj": cj,
                                 "y": yte, "p": p, "lam": lam}))
    pred = pd.concat(out, ignore_index=True)
    pred.to_parquet(PRED_DIR / "hurdle_glm.parquet", index=False)
    print(f"Saved {len(pred):,} rows → {PRED_DIR / 'hurdle_glm.parquet'}")


if __name__ == "__main__":
    main()
