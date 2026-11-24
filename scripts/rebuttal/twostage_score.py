"""Score the two-stage ablation ladder (rebuttal) for every seed and test week.

Step 0 is the paper's two-stage model: the Poisson LGCP (trained on all cell-days, zeros included) with a
post-hoc scikit-learn gate classifier (no dropout, fixed tau = 0.3), whose log-likelihood and CRPS were
computed from the Poisson predictive alone (reports/probabilistic_scores/ablation_scores.py). Each step adds
one change:

  S1  + hurdle scoring: the same model scored with the hurdle predictive P(0) = 1 - p,
      P(k) = p E_q[ZTP(k | lambda)], with p from the gate classifier (retrained with the run's settings
      and seed) and lambda from the Poisson LGCP. Point forecasts unchanged.
  S2  + MC-dropout classifier: the classifier of the two-stage hurdle run (Red-LGCP's classifier, trained
      alone); p = mean of 100 dropout passes and gate tau + sigma. The count part is still the Poisson LGCP;
      the point forecast is the paper's gate (tau + sigma, rho = gamma = 1) applied to E_q[lambda], the
      quantity the two-stage model gates.
  S3  + zero-truncated count model: the two-stage hurdle run itself (twostage_hurdle.py), i.e. Red-LGCP
      with its two parts trained separately; scored as compute_scores.py scores the joint model, point
      metrics from its metrics_higher.yaml.

Checks written with every row: the Poisson-predictive log-likelihood and CRPS of step 0 recomputed as in
ablation_scores.py (torch seed 2000 + seed), and step 0's point forecasts recomputed from the retrained
gate classifier and the exact posterior mean E_q[lambda] = exp(alpha + C beta + a'm + a'Sa / 2) (the run
used a 300-sample Monte Carlo mean).

Output: reports/rebuttal/twostage/scores/all_scores.csv
Usage: python twostage_score.py [--workers N]
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.special import gammaln

from svgp_residual import ROOT
from svgp_score import hurdle_scores, pmf_scores
from src.models.data_pp_lgcp import prepare_data
from src.models.hurdle_lgcp import HurdleSparseLGCP, apply_modulated_gate
from src.models.lgcp import SparseLGCP, build_Kfu, build_Kuu
from src.models.metrics_lgcp import evaluate_metrics
from src.models.zero_gate import train_zero_gate

BASE = Path(os.environ.get("TWOSTAGE_BASE", ROOT / "reports/rebuttal/twostage"))  # override only for tests
TWOSTAGE = ROOT / "reports/seeded_monthly_windows_salinity/11_train_lgcp"
WINDOWS = [f"2025_{m}_{d:02d}_{d + 6:02d}" for m in ("may", "november") for d in (1, 8, 15, 22)]
POINT = ["mae_obs", "rmse_obs", "wasserstein", "accuracy", "f1"]
DATA_KEYS = ["parquet_path", "train_fraction", "data_seed", "split_strategy", "covariate_cols", "test_start_date",
             "test_end_date", "lag_features", "years", "kernel_config", "M_inducing", "np_seed"]


def definitive_run(root: Path, seed: int, window: str) -> Path:
    mm = yaml.safe_load(open(root / f"seed_{seed}/monthly_metrics.yaml"))
    p = [w["metrics_path"] for w in mm["per_window"] if w["name"] == window][0]
    return (ROOT / p.split("projects/AIPS/")[-1]).parent


def posterior_tools(model, Xn):
    kern = model.kernel_params()
    eye = torch.eye(model.M)
    chol = torch.linalg.cholesky(build_Kuu(model.Z, kern, 1e-6) + 1e-8 * eye)
    A = torch.cholesky_solve(build_Kfu(Xn, model.Z, kern).T, chol).T
    S = model.get_S()
    return A, S, torch.linalg.cholesky(S + 1e-8 * eye)


def lambda_samples(model, Xn, Cn, n=300):
    A, _, Lq = posterior_tools(model, Xn)
    U = model.m.unsqueeze(0) + torch.randn(n, model.M) @ Lq.T
    return torch.exp(model.alpha + (Cn @ model.beta).unsqueeze(0) + U @ A.T).numpy().astype(float)


def poisson_scores(y, lam):
    """Poisson predictive E_q[Poisson(y | lambda)], as lgcp_scores in ablation_scores.py."""
    K = int(min(400, max(y.max() + 10, np.quantile(lam, 0.999) * 3 + 30)))
    k = np.arange(K + 1)
    logp = k[None, None, :] * np.log(np.clip(lam, 1e-300, None))[:, :, None] - lam[:, :, None] - gammaln(k + 1)[None, None, :]
    pmf = np.exp(logp).mean(0)
    assert pmf.sum(1).min() > 0.9999
    return pmf_scores(y, pmf)


def score(job) -> list[dict]:
    seed, window = job
    torch.set_default_dtype(torch.float32)
    torch.set_num_threads(1)
    month = window.split("_")[1]
    ts_run = definitive_run(TWOSTAGE, seed, window)
    (h_run,) = [p.parent for p in (BASE / f"runs/seed_{seed}/runs/{window}").glob("*/model.pt")]
    ck_p = torch.load(ts_run / "model.pt", map_location="cpu", weights_only=False)
    ck_h = torch.load(h_run / "model.pt", map_location="cpu", weights_only=False)
    cfg, hcfg = ck_p["config"], ck_h["config"]
    for k in DATA_KEYS:
        assert cfg.get(k) == hcfg.get(k), k
    trc, trv, try_, tec, tev, tey, _, _ = prepare_data(
        str(ROOT / cfg["parquet_path"]), train_fraction=cfg["train_fraction"], random_seed=cfg["data_seed"],
        split_strategy=cfg["split_strategy"], covariate_cols=cfg["covariate_cols"],
        test_start_date=cfg["test_start_date"], test_end_date=cfg["test_end_date"],
        lag_features=cfg["lag_features"], years=cfg["years"])
    y = tey.astype(float)
    sp = pd.read_csv(ts_run / "spatial_predictions.csv")
    hp = pd.read_csv(h_run / "hurdle_predictions.csv")
    assert np.array_equal(sp.observed.to_numpy(float), y) and np.array_equal(hp.observed.to_numpy(float), y)
    dates = pd.to_datetime(sp.date).to_numpy()
    Xn, Cn = torch.tensor(tec, dtype=torch.float32), torch.tensor(tev, dtype=torch.float32)

    # Poisson LGCP of the two-stage run
    pois = SparseLGCP(trc, trv, try_, kernel_config=cfg["kernel_config"], M_inducing=cfg["M_inducing"],
                      device="cpu", np_seed=cfg["np_seed"])
    pois.load_state_dict(ck_p["model_state"])
    with torch.no_grad():
        torch.manual_seed(2000 + seed)
        lam_p = lambda_samples(pois, Xn, Cn)
        A, S, _ = posterior_tools(pois, Xn)
        mean_lam_p = torch.exp(pois.alpha + Cn @ pois.beta + A @ pois.m + 0.5 * ((A @ S) * A).sum(1)).numpy().astype(float)
    ll0, cr0 = poisson_scores(tey, lam_p)

    # scikit-learn gate classifier of the two-stage run, retrained
    gcfg = cfg["zero_gate"]
    clf = train_zero_gate(trc, trv, try_, gcfg, seed=cfg.get("np_seed", 0))
    p_sk = clf.predict_proba(np.hstack([tec, tev]))[:, list(clf.classes_).index(1)]
    tau = float(gcfg["threshold"])
    pred0 = apply_modulated_gate(mean_lam_p, p_sk, tau, dates, gamma=float(gcfg["redistribution_gamma"]),
                                 scale=float(gcfg["redistribution_scale"]))
    m0 = evaluate_metrics(tey, pred0, dates)
    saved0 = yaml.safe_load(open(ts_run / "metrics.yaml"))
    checks = {"check_s0_log_lik": float(ll0.mean()), "check_s0_crps": float(cr0.mean()),
              "check_s0_mae_recomputed": m0["mae_obs"], "check_s0_mae_saved": saved0["mae_obs"],
              "check_s0_zero_pattern_agreement": float(np.mean((pred0 == 0) == (sp.predicted.to_numpy() == 0)))}

    # two-stage hurdle run (classifier and zero-truncated LGCP trained separately)
    hur = HurdleSparseLGCP(trc, trv, try_, kernel_config=hcfg["kernel_config"], M_inducing=hcfg["M_inducing"],
                           device="cpu", np_seed=hcfg["np_seed"], clf_hidden=int(hcfg["hurdle"]["clf_hidden"]),
                           clf_dropout=float(hcfg["hurdle"]["clf_dropout"]), clf_alpha=float(hcfg["hurdle"]["clf_alpha"]))
    hur.load_state_dict(ck_h["model_state"])
    with torch.no_grad():
        torch.manual_seed(1000 + seed)  # as compute_scores.py: latent samples, then dropout passes
        lam_h = lambda_samples(hur, Xn, Cn)
        hur.classifier.train()
        feats = torch.cat([Xn, Cn], dim=1)
        p_bar = torch.stack([torch.sigmoid(hur.classifier(feats).squeeze(-1)) for _ in range(100)]).mean(0).numpy().astype(float)

    pred2 = apply_modulated_gate(mean_lam_p, hp.p_active_mean.to_numpy(), hp.tau_higher.to_numpy(), dates,
                                 gamma=float(gcfg["redistribution_gamma"]), scale=float(gcfg["redistribution_scale"]))
    steps = {
        "S1": (hurdle_scores(tey, lam_p, p_sk), {k: saved0[k] for k in POINT}),
        "S2": (hurdle_scores(tey, lam_p, p_bar), evaluate_metrics(tey, pred2, dates)),
        "S3": (hurdle_scores(tey, lam_h, p_bar), yaml.safe_load(open(h_run / "metrics_higher.yaml"))),
    }
    rows = []
    for step, ((ll, cr, n_ext), m) in steps.items():
        rows.append({"step": step, "seed": seed, "week": window, "month": month, "log_lik": float(ll.mean()),
                     "crps": float(cr.mean()), **{k: m[k] for k in POINT}, "rows_extended_support": n_ext,
                     **checks, "twostage_run": os.path.relpath(ts_run, ROOT), "hurdle_run": os.path.relpath(h_run, ROOT)})
    return rows


def _one(job):
    try:
        return score(job)
    except Exception as exc:  # report and continue
        print(f"FAILED {job}: {exc!r}", flush=True)
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    ap.add_argument("--only", nargs=2, metavar=("SEED", "WINDOW"), help="score one (seed, window) and print it")
    args = ap.parse_args()
    if args.only:
        print(pd.DataFrame(score((int(args.only[0]), args.only[1]))).T.to_string())
        return
    jobs = [(s, w) for s in range(1, 11) for w in WINDOWS]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        rows = [r for part in ex.map(_one, jobs, chunksize=1) for r in part]
    out = BASE / "scores"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "all_scores.csv", index=False)
    print(f"Saved {len(rows)} rows (expected {3 * len(jobs)}) → {out / 'all_scores.csv'}")


if __name__ == "__main__":
    main()
