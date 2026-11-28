"""Score every run of the inducing-point and kernel ablation (rebuttal), and the paper's runs.

Per run:
  - log-likelihood and CRPS of the hurdle predictive, as reports/probabilistic_scores/compute_scores.py
    (torch seed 1000 + seed, 300 latent samples, then 100 dropout passes);
  - point metrics of the paper's gate (tau + sigma) from the run's metrics_higher.yaml;
  - GP diagnostics: prior variance k(x, x), share of it captured by the inducing points on the training and
    test inputs (1 - mean residual / k(x, x), residual = diag(K_ff - K_fu K_uu^-1 K_uf)), learned kernel
    variances, intercept, training wall time (model.pt time minus the run folder's start stamp);
  - the posterior-mean latent field A m, its variance a'Sa and the posterior-mean log-intensity
    alpha + C beta + A m at the test cell-days, saved to
    scores/fields/<variant>/seed_<s>_<window>.npz to compare fields across M.

Output: reports/rebuttal/inducing_kernels/scores/all_scores.csv
Usage: python ik_score.py [--workers N]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from svgp_residual import ROOT, prior_variance, residual_variance
from svgp_score import hurdle_scores
from src.models.data_pp_lgcp import prepare_data
from src.models.hurdle_lgcp import HurdleSparseLGCP
from src.models.lgcp import build_Kfu, build_Kuu

BASE = Path(os.environ.get("IK_BASE", ROOT / "reports/rebuttal/inducing_kernels"))  # overrides only for tests
# copy of reports/seeded_monthly_windows_hurdle (Demetra's lacks seed_1)
DEFINITIVE = Path(os.environ.get("IK_DEFINITIVE", ROOT / "reports/rebuttal/svgp_residual/definitive"))
POINT = ["mae_obs", "rmse_obs", "wasserstein", "accuracy", "f1"]
DATA_KEYS = ("parquet_path", "train_fraction", "data_seed", "split_strategy", "covariate_cols", "test_start_date",
             "test_end_date", "lag_features", "years")


_DATA = {}


def data(cfg: dict):
    """prepare_data for a run's data settings, cached per worker process (one entry per test week)."""
    key = json.dumps({k: cfg[k] for k in DATA_KEYS}, sort_keys=True)
    if key not in _DATA:
        _DATA[key] = prepare_data(
            str(ROOT / cfg["parquet_path"]), train_fraction=cfg["train_fraction"], random_seed=cfg["data_seed"],
            split_strategy=cfg["split_strategy"], covariate_cols=cfg["covariate_cols"],
            test_start_date=cfg["test_start_date"], test_end_date=cfg["test_end_date"],
            lag_features=cfg["lag_features"], years=cfg["years"])
    return _DATA[key]


def score(job) -> dict:
    variant, run = job
    torch.set_num_threads(1)
    ck = torch.load(run / "model.pt", map_location="cpu", weights_only=False)
    cfg, h = ck["config"], ck["config"]["hurdle"]
    dtype = torch.float64 if cfg.get("ik_dtype", "float32") == "float64" else torch.float32  # as in training
    torch.set_default_dtype(dtype)
    seed, window = int(cfg["np_seed"]), run.parent.name
    trc, trv, try_, tec, tev, tey, _, _ = data(cfg)
    model = HurdleSparseLGCP(trc, trv, try_, kernel_config=cfg["kernel_config"], M_inducing=cfg["M_inducing"],
                             device="cpu", np_seed=cfg["np_seed"], clf_hidden=int(h["clf_hidden"]),
                             clf_dropout=float(h["clf_dropout"]), clf_alpha=float(h["clf_alpha"]))
    model.load_state_dict(ck["model_state"])
    y = tey.astype(float)
    pr = pd.read_csv(run / "hurdle_predictions.csv")
    assert np.array_equal(pr.observed.to_numpy(float), y), "test rows out of order"

    torch.manual_seed(1000 + seed)
    with torch.no_grad():
        Xn, Cn = torch.tensor(tec, dtype=dtype), torch.tensor(tev, dtype=dtype)
        kern = model.kernel_params()
        eye = torch.eye(model.M)
        chol = torch.linalg.cholesky(build_Kuu(model.Z, kern, 1e-6) + 1e-8 * eye)
        Kfu = build_Kfu(Xn, model.Z, kern)
        A = torch.cholesky_solve(Kfu.T, chol).T
        S = model.get_S()
        U = model.m.unsqueeze(0) + torch.randn(300, model.M) @ torch.linalg.cholesky(S + 1e-8 * eye).T
        lam = torch.exp(model.alpha + (Cn @ model.beta).unsqueeze(0) + U @ A.T).numpy().astype(float)
        model.classifier.train()
        feats = torch.cat([Xn, Cn], dim=1)
        p_bar = torch.stack([torch.sigmoid(model.classifier(feats).squeeze(-1)) for _ in range(100)]).mean(0).numpy().astype(float)
        k0 = float(prior_variance(kern, Xn))
        r_te = residual_variance(kern, Xn, Kfu, A)
        Kfu_tr = build_Kfu(model.X, model.Z, kern)
        r_tr = residual_variance(kern, model.X, Kfu_tr, torch.cholesky_solve(Kfu_tr.T, chol).T)
        fbar, fvar = (A @ model.m).numpy(), ((A @ S) * A).sum(1).numpy()
        eta = (model.alpha + Cn @ model.beta).numpy() + fbar  # posterior-mean log-intensity of the count part
    ll, cr, n_ext = hurdle_scores(y, lam, p_bar)

    fields = BASE / "scores/fields" / variant
    fields.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(fields / f"seed_{seed}_{window}.npz", fbar=fbar, fvar=fvar, eta=eta)
    m = yaml.safe_load(open(run / "metrics_higher.yaml"))
    start = dt.datetime.strptime(run.name.rsplit("_", 1)[1], "%Y%m%d-%H%M%S")
    train_min = (dt.datetime.fromtimestamp(os.path.getmtime(run / "model.pt")) - start).total_seconds() / 60
    variances = {f"var_{g}_{i}_{c['type'].lower()}": float(c["params"]["variance"])
                 for g in ("spatial", "temporal") for i, c in enumerate(kern[g])}
    return {"variant": variant, "seed": seed, "week": window, "month": window.split("_")[1], "M": int(cfg["M_inducing"]),
            "spacetime": kern.get("spacetime_composition", "product"), "dtype": str(dtype).split(".")[-1], "log_lik": float(ll.mean()), "crps": float(cr.mean()),
            **{k: m[k] for k in POINT}, "rows_extended_support": n_ext, "k0": k0,
            "captured_share_train": 1 - float(r_tr.mean()) / k0, "captured_share_test": 1 - float(r_te.mean()) / k0,
            "alpha": float(model.alpha.detach()), "train_minutes": train_min, **variances, "run": os.path.relpath(run, ROOT)}


def _one(job):
    try:
        return [score(job)]
    except Exception as exc:  # report and continue
        print(f"FAILED {job}: {exc!r}", flush=True)
        return []


def main():
    from concurrent.futures import ProcessPoolExecutor
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    args = ap.parse_args()
    jobs = [("paper", p.parent) for p in sorted(DEFINITIVE.glob("seed_*/runs/*/*/model.pt"))]
    jobs += [(p.parts[-6], p.parent) for p in sorted((BASE / "runs").glob("*/seed_*/runs/*/*/model.pt"))]
    print(pd.Series([j[0] for j in jobs]).value_counts().to_dict(), flush=True)
    jobs.sort(key=lambda j: -yaml.safe_load(open(j[1] / "params.yaml"))["M_inducing"])
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        rows = [r for part in ex.map(_one, jobs, chunksize=1) for r in part]
    out = BASE / "scores"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "all_scores.csv", index=False)
    print(f"Saved {len(rows)} rows (expected {len(jobs)}) → {out / 'all_scores.csv'}")


if __name__ == "__main__":
    main()
