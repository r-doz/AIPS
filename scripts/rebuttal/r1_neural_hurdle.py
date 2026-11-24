"""Experiment 3: neural baselines with Red-LGCP's lag access (t-1, t-7) and a hurdle head.

Models (training code in r1_neural.py):
  GNN-hurdle           config/rebuttal/r1_gnn_hurdle.yaml   the definitive GNN inputs (they already include
                                                            the counts at t-1 and t-7) with a hurdle head
  ConvLSTM-lag         config/rebuttal/r1_convlstm_lag.yaml the definitive ConvLSTM plus the t-1 and t-7
                                                            count channels, Poisson head (isolates the lags)
  ConvLSTM-lag-hurdle  config/rebuttal/r1_convlstm_lag.yaml the same with a hurdle head

Hyperparameters are re-selected on the four validation weeks with the paper's protocol for each family
(App. B.2): for the GNN, the grid hidden_dim x num_layers x lr scored by the relative criterion
(mean log-likelihood and RMSE of daily totals); for the ConvLSTM, the grid seq_length x lr scored by the
mean log-likelihood; seed 0 as in the original searches. For hurdle models the log-likelihood is that
of the hurdle and the daily totals use the hurdle mean. The selected models are then trained with
seeds 1-10 on every validation and test week.

Outputs (reports/rebuttal/reviewer1/):
  hparam/<model>_grid.csv, hparam/<model>_selected.yaml
  predictions/neural_hurdle.parquet  (model, seed, split, week, month, date, ci, cj, y, and p, lam or rate)

Usage: python r1_neural_hurdle.py [--stage grid|final|all]   (final reuses the saved selections)
"""

from __future__ import annotations

import argparse
import itertools
import time

import numpy as np
import pandas as pd
import yaml

from r1_common import ALL_WEEKS, OUT, PRED_DIR, SEEDS, VAL_WEEKS, hurdle_mean, hurdle_scores, load_module, poisson_scores
from r1_neural import load_cfg, make_tasks, run_tasks
from src.models.metrics_lgcp import evaluate_metrics

HP_DIR = OUT / "hparam"
VAL = [("val",) + w for w in VAL_WEEKS]
_hp = load_module("r1_hp3", "scripts/51c_baseline_hparam_rerun_2weeks_before_salinity.py")

MODELS = {
    "GNN-hurdle": dict(kind="gnn", head="hurdle", config="config/rebuttal/r1_gnn_hurdle.yaml",
                       grid={"hidden_dim": [16, 32, 64], "num_layers": [1, 2, 3], "lr": [1e-4, 1e-3, 1e-2, 1e-1]},
                       criterion="relative"),
    "ConvLSTM-lag": dict(kind="convlstm", head="poisson", config="config/rebuttal/r1_convlstm_lag.yaml",
                         grid={"seq_length": [3, 7, 14], "lr": [1e-3, 1e-2]}, criterion="mean_ll"),
    "ConvLSTM-lag-hurdle": dict(kind="convlstm", head="hurdle", config="config/rebuttal/r1_convlstm_lag.yaml",
                                grid={"seq_length": [3, 7, 14], "lr": [1e-3, 1e-2]}, criterion="mean_ll"),
}


def week_scores(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Mean log-likelihood and RMSE of daily totals per (config, week)."""
    rows = []
    for k, g in df.groupby(keys + ["week"]):
        if "p" in g and g["p"].notna().all():
            ll, _ = hurdle_scores(g.y, g.p, g.lam)
            mean = hurdle_mean(g.p, g.lam)
        else:
            ll, _ = poisson_scores(g.y, g.rate)
            mean = g.rate.to_numpy()
        m = evaluate_metrics(g.y.to_numpy(), mean, g.date.to_numpy())
        rows.append(dict(zip(keys + ["week"], k), mean_ll_obs=float(ll.mean()), rmse_daily=m["rmse_daily"], mae_obs=m["mae_obs"]))
    return pd.DataFrame(rows)


def grid_stage(name: str, spec: dict):
    keys = list(spec["grid"])
    tasks = []
    for values in itertools.product(*spec["grid"].values()):
        hp = dict(zip(keys, values))
        cfg = load_cfg(spec["kind"], spec["config"], hp)
        tasks += make_tasks(name, spec["kind"], spec["head"], cfg, VAL, [0], tags=hp)
    t0 = time.time()
    print(f"[{name}] grid: {len(tasks)} runs …", flush=True)
    res = week_scores(run_tasks(tasks), keys)
    res.to_csv(HP_DIR / f"{name}_grid.csv", index=False)
    if spec["criterion"] == "relative":
        scores = _hp.compute_relative_scores(res, week_col="week", group_cols=keys, metrics_directions=_hp.METRIC_DIRECTIONS)
        best, score = scores.iloc[0], float(scores.iloc[0].combined_relative_score)
    else:
        scores = res.groupby(keys).mean_ll_obs.mean().sort_values(ascending=False).reset_index()
        best, score = scores.iloc[0], float(scores.iloc[0].mean_ll_obs)
    scores.to_csv(HP_DIR / f"{name}_scores.csv", index=False)
    sel = {k: (int(best[k]) if k in ("hidden_dim", "num_layers", "seq_length") else float(best[k])) for k in keys}
    (HP_DIR / f"{name}_selected.yaml").write_text(yaml.safe_dump(
        {"config": spec["config"], "head": spec["head"], "grid": spec["grid"], "seed": 0,
         "criterion": "App. B.2 relative score (mean_ll_obs, rmse_daily)" if spec["criterion"] == "relative"
         else "mean log-likelihood over the 4 validation weeks",
         "selected": sel, "score": score, "minutes": round((time.time() - t0) / 60, 1)}, sort_keys=False))
    print(f"[{name}] selected {sel} (score {score:.4f}, {(time.time() - t0) / 60:.1f} min)", flush=True)


def final_stage(names: list[str]):
    tasks = []
    for name in names:
        spec = MODELS[name]
        sel = yaml.safe_load((HP_DIR / f"{name}_selected.yaml").read_text())["selected"]
        tasks += make_tasks(name, spec["kind"], spec["head"], load_cfg(spec["kind"], spec["config"], sel), ALL_WEEKS, SEEDS)
    t0 = time.time()
    print(f"final: {len(tasks)} runs …", flush=True)
    pred = run_tasks(tasks)
    pred["date"] = pd.to_datetime(pred["date"]).dt.normalize()
    path = PRED_DIR / "neural_hurdle.parquet"
    if path.exists():  # keep models trained in an earlier call
        old = pd.read_parquet(path)
        pred = pd.concat([old[~old.model.isin(names)], pred], ignore_index=True)
    pred.to_parquet(path, index=False)
    print(f"Saved {len(pred):,} rows → {path} ({(time.time() - t0) / 60:.1f} min)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["grid", "final", "all"], default="all")
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    args = ap.parse_args()
    HP_DIR.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    if args.stage in ("grid", "all"):
        for name in args.models:
            grid_stage(name, MODELS[name])
    if args.stage in ("final", "all"):
        final_stage(args.models)


if __name__ == "__main__":
    main()
