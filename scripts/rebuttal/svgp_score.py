"""Score one Red-LGCP run with and without the SVGP residual variance in the predictive distribution.

For a run folder (model.pt, params.yaml, hurdle_predictions.csv) writes two rows, residual = no / yes:
  - log-likelihood and CRPS of the hurdle predictive distribution, computed as in
    reports/probabilistic_scores/compute_scores.py (300 latent samples, torch seed 1000 + seed, 100 dropout
    passes); without the residual the computation is identical to that script;
  - point metrics of the paper's gate: the run's saved P(active) and threshold tau + sigma, rho = gamma = 1,
    applied to the expected count p E[lambda / (1 - exp(-lambda))];
  - GP diagnostics: prior variance k0 = k(x, x), mean residual variance and mean projected variance a'Sa on
    the training and test inputs, variance of the posterior-mean field A m on the training inputs, kernel
    variances, period and intercept.

Usage: python svgp_score.py --run <run dir> --label <model label> --out <csv>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import gammaln

from svgp_residual import ROOT, prior_variance, residual_variance
from src.models.data_pp_lgcp import prepare_data
from src.models.hurdle_lgcp import HurdleSparseLGCP, _conditional_mean, apply_modulated_gate
from src.models.lgcp import build_Kfu, build_Kuu
from src.models.metrics_lgcp import evaluate_metrics


def pmf_scores(y, pmf):
    y = np.asarray(y, int)
    ll = np.log(np.clip(pmf[np.arange(len(y)), y], 1e-300, None))
    cdf = np.cumsum(pmf, axis=1)
    ind = (np.arange(pmf.shape[1])[None, :] >= y[:, None]).astype(float)
    return ll, ((cdf - ind) ** 2).sum(1)


def hurdle_pmf(lam: np.ndarray, p_bar: np.ndarray, K: int) -> np.ndarray:
    """P(0) = 1 - p, P(k) = p E_q[ZTP(k | lambda)], k = 1..K, from latent samples lam [S, n] (as compute_scores.py)."""
    k = np.arange(1, K + 1)
    log1mexp = np.where(lam < 1e-6, np.log(np.clip(lam, 1e-300, None)), np.log(-np.expm1(-lam)))
    log_tp = (k[None, None, :] * np.log(np.clip(lam, 1e-300, None))[:, :, None] - lam[:, :, None]
              - gammaln(k + 1)[None, None, :] - log1mexp[:, :, None])
    return np.concatenate([(1 - p_bar)[:, None], p_bar[:, None] * np.exp(log_tp).mean(0)], axis=1)


def hurdle_scores(y: np.ndarray, lam: np.ndarray, p_bar: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Log-likelihood and CRPS per test row, and the number of rows whose support had to be extended.

    The support 0..K follows compute_scores.py (K = min(400, ...)); rows whose pmf then sums to < 0.9999
    (heavy-tailed lambda samples, e.g. the residual added to an SoR model at prediction) are recomputed on
    0..K_i with K_i = max(y_i, max_s lam_si + 10 sqrt(lam_si)) + 30, which holds all their mass. compute_scores.py
    asserts that no row of the paper's model needs this, so without the residual the scores are unchanged.
    """
    y = np.asarray(y, int)
    K = int(min(400, max(y.max() + 10, np.quantile(lam, 0.999) * 3 + 30)))
    pmf = hurdle_pmf(lam, p_bar, K)
    ll, cr = pmf_scores(y, pmf)
    short = np.flatnonzero(pmf.sum(1) < 0.9999)
    for i in short:
        Ki = int(max(y[i], (lam[:, i] + 10 * np.sqrt(lam[:, i])).max()) + 30)
        assert Ki < 2_000_000, Ki
        pi = hurdle_pmf(lam[:, i:i + 1], p_bar[i:i + 1], Ki)
        assert pi.sum() > 0.9999, (i, Ki, pi.sum())
        (ll[i],), (cr[i],) = pmf_scores(y[i:i + 1], pi)
    return ll, cr, len(short)


def score(run: Path, label: str) -> list[dict]:
    torch.set_default_dtype(torch.float32)
    torch.set_num_threads(1)
    ck = torch.load(run / "model.pt", map_location="cpu", weights_only=False)
    cfg, h = ck["config"], ck["config"].get("hurdle", {})
    seed, window = int(cfg["np_seed"]), run.parent.name
    trc, trv, try_, tec, tev, tey, _, _ = prepare_data(
        str(ROOT / cfg["parquet_path"]), train_fraction=cfg["train_fraction"], random_seed=cfg["data_seed"],
        split_strategy=cfg["split_strategy"], covariate_cols=cfg["covariate_cols"],
        test_start_date=cfg["test_start_date"], test_end_date=cfg["test_end_date"],
        lag_features=cfg["lag_features"], years=cfg["years"])
    model = HurdleSparseLGCP(trc, trv, try_, kernel_config=cfg["kernel_config"], M_inducing=cfg["M_inducing"],
                             device="cpu", np_seed=cfg["np_seed"], clf_hidden=int(h.get("clf_hidden", 32)),
                             clf_dropout=float(h.get("clf_dropout", 0.1)), clf_alpha=float(h.get("clf_alpha", 1e-4)))
    model.load_state_dict(ck["model_state"])
    pr = pd.read_csv(run / "hurdle_predictions.csv")
    y = tey.astype(float)
    assert np.array_equal(pr.observed.to_numpy(float), y), "test rows out of order"
    dates = pd.to_datetime(pr.date).to_numpy()

    torch.manual_seed(1000 + seed)
    with torch.no_grad():
        Xn, Cn = torch.tensor(tec, dtype=torch.float32), torch.tensor(tev, dtype=torch.float32)
        kern = model.kernel_params()
        eye = torch.eye(model.M)
        chol = torch.linalg.cholesky(build_Kuu(model.Z, kern, 1e-6) + 1e-8 * eye)
        Kfu = build_Kfu(Xn, model.Z, kern)
        A = torch.cholesky_solve(Kfu.T, chol).T
        S = model.get_S()
        Lq = torch.linalg.cholesky(S + 1e-8 * eye)
        U = model.m.unsqueeze(0) + torch.randn(300, model.M) @ Lq.T
        eta = model.alpha + (Cn @ model.beta).unsqueeze(0) + U @ A.T
        lam0 = torch.exp(eta).numpy().astype(float)
        model.classifier.train()
        feats = torch.cat([Xn, Cn], dim=1)
        p_bar = torch.stack([torch.sigmoid(model.classifier(feats).squeeze(-1)) for _ in range(100)]).mean(0).numpy().astype(float)
        r_te = residual_variance(kern, Xn, Kfu, A)
        lam1 = torch.exp(eta + torch.sqrt(r_te).unsqueeze(0) * torch.randn(300, Xn.shape[0])).numpy().astype(float)
        # GP diagnostics
        k0 = float(prior_variance(kern, Xn))
        Kfu_tr = build_Kfu(model.X, model.Z, kern)
        A_tr = torch.cholesky_solve(Kfu_tr.T, chol).T
        r_tr = residual_variance(kern, model.X, Kfu_tr, A_tr)
        diag = {"k0": k0, "resid_var_train": float(r_tr.mean()), "resid_var_test": float(r_te.mean()),
                "proj_var_train": float(((A_tr @ S) * A_tr).sum(1).mean()), "proj_var_test": float(((A @ S) * A).sum(1).mean()),
                "var_fbar_train": float((A_tr @ model.m).var()), "alpha": float(model.alpha),
                "var_s_broad": float(kern["spatial"][0]["params"]["variance"]),
                "var_s_loc": float(kern["spatial"][1]["params"]["variance"]),
                "var_t_rbf": float(kern["temporal"][0]["params"]["variance"]),
                "var_t_per": float(kern["temporal"][1]["params"]["variance"])}

    rows = []
    for residual, lam in (("no", lam0), ("yes", lam1)):
        ll, cr, n_ext = hurdle_scores(y, lam, p_bar)
        mu = pr.p_active_mean.to_numpy() * _conditional_mean(torch.tensor(lam)).mean(0).numpy()
        point = apply_modulated_gate(mu, pr.p_active_mean.to_numpy(), pr.tau_higher.to_numpy(), dates, gamma=1.0, scale=1.0)
        m = evaluate_metrics(y, point, dates)
        rows.append({"model": label, "residual_in_prediction": residual, "seed": seed, "week": window,
                     "month": window.split("_")[1], "log_lik": float(ll.mean()), "crps": float(cr.mean()),
                     **{k: m[k] for k in ("mae_obs", "rmse_obs", "wasserstein", "accuracy", "f1")},
                     "rows_extended_support": n_ext, "max_lambda_sample": float(lam.max()), **diag,
                     "run": str(run.relative_to(ROOT))})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(score(Path(args.run).resolve(), args.label)).to_csv(out, index=False)


if __name__ == "__main__":
    main()
