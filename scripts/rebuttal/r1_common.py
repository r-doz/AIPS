"""Shared helpers for the reviewer-1 rebuttal experiments.

Windows, data loading, scoring and statistical tests used by
  r1_poisson_baselines.py  (experiment 1: validation-tuned activity threshold for the Poisson baselines)
  r1_hurdle_glm.py         (experiment 2: hurdle Poisson GLM)
  r1_neural.py             (experiment 3: GNN / ConvLSTM with lag t-7 and a hurdle head)
  r1_evaluate.py           (threshold tuning, test metrics, paired tests, tables)

Nothing here changes the definitive code: models and metrics are imported from scripts/ and src/.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import gammaln
from scipy.stats import poisson, wilcoxon

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.hurdle_lgcp import apply_modulated_gate  # noqa: E402
from src.models.metrics_lgcp import evaluate_activity_metrics, evaluate_metrics  # noqa: E402

OUT = ROOT / "reports/rebuttal/reviewer1"
PRED_DIR = OUT / "predictions"
EPS = 1e-8

# The validation weeks of the paper (App. B.2): the two 7-day blocks before each reporting month.
VAL_WEEKS = [
    ("val_may_w1", "2025-04-21", "2025-04-27", "may"),
    ("val_may_w2", "2025-04-28", "2025-05-04", "may"),
    ("val_nov_w1", "2025-10-20", "2025-10-26", "november"),
    ("val_nov_w2", "2025-10-27", "2025-11-02", "november"),
]
# The 8 test weeks of the paper, named as in the definitive results.
TEST_WEEKS = [
    (f"2025_{m}_{d:02d}_{d + 6:02d}", f"2025-{mn:02d}-{d:02d}", f"2025-{mn:02d}-{d + 6:02d}", m)
    for m, mn in (("may", 5), ("november", 11)) for d in (1, 8, 15, 22)
]
ALL_WEEKS = [("val",) + w for w in VAL_WEEKS] + [("test",) + w for w in TEST_WEEKS]
SEEDS = list(range(1, 11))
TAU_GRID = np.round(np.arange(0.01, 1.0, 0.01), 2)


def load_module(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def cell_key(lon, lat):
    """Integer cell key (the grid step is 1/24 deg, so 48 * coordinate is integer-spaced)."""
    return np.round(np.asarray(lon, float) * 48).astype(int), np.round(np.asarray(lat, float) * 48).astype(int)


def load_panel() -> pd.DataFrame:
    df = pd.concat([pd.read_parquet(ROOT / f"data/processed/cpr_gfw_salinity_{y}.parquet") for y in (2024, 2025)])
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["ci"], df["cj"] = cell_key(df.longitude, df.latitude)
    return df.sort_values(["ci", "cj", "date"]).reset_index(drop=True)


# ---------------------------------------------------------------- scores
def hurdle_mean(p, lam):
    """E[Y] of a hurdle with P(Y>0) = p and zero-truncated Poisson(lam) positives."""
    lam = np.clip(np.asarray(lam, float), EPS, None)
    return np.asarray(p, float) * lam / -np.expm1(-lam)


def poisson_scores(y, lam):
    """Per-observation Poisson log-likelihood and discrete CRPS (same as reports/probabilistic_scores)."""
    y = np.asarray(y, float)
    lam = np.clip(np.asarray(lam, float), EPS, None)
    ll = y * np.log(lam) - lam - gammaln(y + 1)
    K = int(max(y.max(), (lam + 10 * np.sqrt(lam)).max()) + 30)
    cdf = poisson.cdf(np.arange(K + 1)[None, :], lam[:, None])
    ind = (np.arange(K + 1)[None, :] >= y[:, None]).astype(float)
    return ll, ((cdf - ind) ** 2).sum(1)


def hurdle_scores(y, p, lam):
    """Per-observation log-likelihood and discrete CRPS of the hurdle: P(0) = 1-p, P(k) = p ZTP(k | lam)."""
    y = np.asarray(y, int)
    p = np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12)
    lam = np.clip(np.asarray(lam, float), EPS, None)
    K = int(max(y.max(), (lam + 10 * np.sqrt(lam)).max()) + 30)
    k = np.arange(1, K + 1)
    log_ztp = k[None, :] * np.log(lam)[:, None] - lam[:, None] - gammaln(k + 1)[None, :] - np.log(-np.expm1(-lam))[:, None]
    pmf = np.concatenate([(1 - p)[:, None], p[:, None] * np.exp(log_ztp)], axis=1)
    ll = np.log(np.clip(pmf[np.arange(len(y)), y], 1e-300, None))
    cdf = np.cumsum(pmf, axis=1)
    ind = (np.arange(K + 1)[None, :] >= y[:, None]).astype(float)
    return ll, ((cdf - ind) ** 2).sum(1)


def activity(y, active) -> dict:
    """Accuracy, precision, recall and F1 of a binary activity prediction (paper's definitions)."""
    return evaluate_activity_metrics(y, np.asarray(active, float))


def point_metrics(y, pred, dates) -> dict:
    m = evaluate_metrics(np.asarray(y, float), np.asarray(pred, float), np.asarray(dates))
    return {k: m[k] for k in ("mae_obs", "rmse_obs", "wasserstein", "accuracy", "precision", "recall", "f1", "rmse_daily")}


def gate(mu, p, tau, dates, redistribute: bool = True):
    """Red-LGCP's gate (App. B.2, rho = gamma = 1) with a constant threshold tau (sigma = 0)."""
    mu, p = np.asarray(mu, float), np.asarray(p, float)
    if redistribute:
        return apply_modulated_gate(mu, p, np.full_like(p, tau), dates, gamma=1.0, scale=1.0)
    return np.where(p >= tau, mu, 0.0)


# ---------------------------------------------------------------- statistics
def holm(pvals: pd.Series) -> pd.Series:
    order = pvals.sort_values()
    adj, run_max = {}, 0.0
    for rank, (i, p) in enumerate(order.items()):
        run_max = max(run_max, (len(order) - rank) * p)
        adj[i] = min(1.0, run_max)
    return pd.Series(adj).reindex(pvals.index)


def paired_test(ours: pd.Series, theirs: pd.Series, higher_is_better: bool) -> dict:
    """Weeks won by `ours` and the two-sided exact Wilcoxon p over paired weekly values."""
    theirs = theirs.loc[ours.index]
    d = (ours - theirs) if higher_is_better else (theirs - ours)
    if np.allclose(d, 0):
        return {"weeks_won": 0, "p": np.nan}
    return {"weeks_won": int((d > 0).sum()), "p": float(wilcoxon(d.to_numpy(), method="exact").pvalue)}
