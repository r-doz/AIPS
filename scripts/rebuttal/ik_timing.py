"""Cost of Red-LGCP against the number of inducing points M (rebuttal, inducing-point ablation).

For one window (data and settings of a definitive per-run config), builds the model for each M and times,
on one thread, (i) training steps of the joint ELBO (Adam, minibatch, num_mc and clipping of the config)
after a warm-up and (ii) the test-set prediction (num_pred_samples latent samples, 100 dropout passes).

Usage: python ik_timing.py --config <per-run config> --M 50 100 300 --steps 30 --out <csv>
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import torch

from twostage_hurdle import load_train_module
from src.models.data_pp_lgcp import prepare_data
from src.models.hurdle_lgcp import HurdleSparseLGCP


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--M", type=int, nargs="+", required=True)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float32)
    mod = load_train_module()
    cfg = mod.base.load_config(args.config)
    h = cfg["hurdle"]
    trc, trv, try_, tec, tev, tey, scalers, df = prepare_data(
        cfg["parquet_path"], train_fraction=cfg["train_fraction"], random_seed=cfg["data_seed"],
        split_strategy=cfg["split_strategy"], covariate_cols=cfg["covariate_cols"],
        test_start_date=cfg["test_start_date"], test_end_date=cfg["test_end_date"],
        lag_features=cfg["lag_features"], years=cfg["years"])
    kc = mod.base.resolve_period_days(cfg["kernel_config"], scalers, (df["date"].max() - df["date"].min()).days)
    rows = []
    for M in args.M:
        mod.base.set_seeds(cfg["np_seed"])
        model = HurdleSparseLGCP(trc, trv, try_, kernel_config=kc, M_inducing=M, device="cpu", np_seed=cfg["np_seed"],
                                 clf_hidden=int(h["clf_hidden"]), clf_dropout=float(h["clf_dropout"]), clf_alpha=float(h["clf_alpha"]))
        mod.base.configure_log_noise(model, cfg)
        opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
        times = []
        for i in range(args.warmup + args.steps):
            t0 = time.perf_counter()
            opt.zero_grad()
            elbo, _ = model.elbo_mc(num_mc=cfg["num_mc"], minibatch_size=min(cfg["minibatch"], model.N))
            (-elbo).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()
            if i >= args.warmup:
                times.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        model.predict_hurdle(tec, tev, num_samples=int(cfg["num_pred_samples"]), num_dropout=int(h["num_dropout_samples"]))
        pred = time.perf_counter() - t0
        sps = float(np.median(times))
        rows.append({"M": M, "sec_per_step": sps, "train_min_estimate": sps * cfg["num_steps"] / 60,
                     "predict_sec": pred, "n_train": model.N, "n_test": len(tey), "minibatch": cfg["minibatch"],
                     "num_mc": cfg["num_mc"], "num_steps": cfg["num_steps"]})
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
