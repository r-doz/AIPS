"""Experiment 1: predicted rates of the five Poisson baselines on the validation and test weeks.

The test-week rates are those of the definitive runs (Table 1), checked against their saved MAE:
  LAP, GCMP  rebuilt from the data (deterministic)
  GLM        deterministic refit with the definitive config
  GNN        read from the definitive runs (spatial_predictions.csv)
  ConvLSTM   retrained with the definitive config and seeds (reproduces the definitive runs exactly)
The validation-week rates come from the same models, trained on all days before each validation
week with the same configs and seeds 1-10. r1_evaluate.py tunes the activity threshold on them.

Also retrains the GNN on the test weeks ("GNN (rerun)"): single GNN runs are not bitwise reproducible
across software environments, and this checks that they reproduce in the one used (Demetra).

Output: reports/rebuttal/reviewer1/predictions/poisson_baselines.parquet
        (model, seed, split, week, month, date, ci, cj, y, rate)
"""

from __future__ import annotations

import glob
import time

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler

from r1_common import ALL_WEEKS, PRED_DIR, ROOT, SEEDS, TEST_WEEKS, VAL_WEEKS, cell_key, load_module, load_panel
from r1_neural import load_cfg, make_tasks, run_tasks

GNN_ROOT = ROOT / "reports/seeded_monthly_windows_salinity_gnn_cellpoisson/15_train_gnn"
CONV_ROOT = ROOT / "reports/seeded_monthly_windows_salinity/18_train_convlstm"
REF = {
    "LAP": ROOT / "reports/seeded_monthly_windows/13_evaluate_last_available_poisson/per_window_per_seed_metrics.csv",
    "GCMP": ROOT / "reports/seeded_monthly_windows/12b_global_cell_mean/per_window_per_seed_metrics.csv",
    "GLM": ROOT / "reports/seeded_monthly_windows_salinity/14_train_window_poisson_glm/per_window_per_seed_metrics.csv",
    "GNN": GNN_ROOT / "per_window_per_seed_metrics.csv",
    "ConvLSTM": CONV_ROOT / "per_window_per_seed_metrics.csv",
}
VAL = [("val",) + w for w in VAL_WEEKS]
TEST = [("test",) + w for w in TEST_WEEKS]
COLS = ["model", "seed", "split", "week", "month", "date", "ci", "cj", "y", "rate"]


def check_mae(pred: pd.DataFrame, model: str, tol: float = 2e-4):
    """Assert that the test-week rates reproduce the definitive per-window MAE."""
    ref = pd.read_csv(REF[model])
    t = pred[pred.split == "test"].assign(ae=lambda d: (d.y - d.rate).abs())
    got = t.groupby(["seed", "week"]).ae.mean()
    worst = 0.0
    for (seed, week), mae in got.items():
        r = ref[(ref.name == week) & (ref.seed == seed if "seed" in ref else True)].mae_obs
        worst = max(worst, abs(mae - float(r.iloc[0])))
    assert worst < tol, f"{model}: test rates do not reproduce the definitive MAE (max gap {worst:.2e})"
    print(f"  {model}: {len(got)} test runs match the definitive MAE (max gap {worst:.1e})", flush=True)


def lap_gcmp(panel: pd.DataFrame) -> list[pd.DataFrame]:
    p = panel.copy()
    g = p.groupby(["ci", "cj"]).ais_vessels_count
    p["LAP"] = g.shift(1)
    p["GCMP"] = (g.cumsum() - p.ais_vessels_count) / p.groupby(["ci", "cj"]).cumcount()
    out = []
    for model in ("LAP", "GCMP"):
        for split, name, start, end, month in ALL_WEEKS:
            q = p[(p.date >= start) & (p.date <= end)]
            out.append(pd.DataFrame({"model": model, "seed": 1, "split": split, "week": name, "month": month,
                                     "date": q.date.values, "ci": q.ci.values, "cj": q.cj.values,
                                     "y": q.ais_vessels_count.astype(float).values, "rate": q[model].values}))
    return out


def glm() -> pd.DataFrame:
    mod = load_module("r1_glm", "scripts/14_train_window_poisson_glm.py")
    cfg = mod.load_config(str(ROOT / "config/14_window_poisson_glm_salinity.yaml"))
    raw = mod.load_parquet_years(str(ROOT / cfg["parquet_path"]), cfg.get("years"))
    raw["date"] = pd.to_datetime(raw["date"])
    win, feats = mod.build_window_dataframe(raw, cfg)
    ci, cj = cell_key(win.longitude, win.latitude)
    out = []
    for split, name, start, end, month in ALL_WEEKS:
        tr, te = mod.make_split_masks(dates=win["date"], split_strategy="fixed_test_window",
                                      train_fraction=cfg["train_fraction"], random_seed=cfg["data_seed"],
                                      test_start_date=start, test_end_date=end)
        X = win[feats].values.astype(np.float32)
        y = win[cfg["target_col"]].values.astype(np.float32)
        sc = StandardScaler().fit(X[tr])
        m = PoissonRegressor(alpha=float(cfg["poisson_alpha"]), max_iter=int(cfg["max_iter"])).fit(sc.transform(X[tr]), y[tr])
        out.append(pd.DataFrame({"model": "GLM", "seed": 1, "split": split, "week": name, "month": month,
                                 "date": pd.to_datetime(win["date"].values[te]).normalize(), "ci": ci[te], "cj": cj[te],
                                 "y": y[te].astype(float), "rate": np.clip(m.predict(sc.transform(X[te])), 0, None)}))
    return pd.concat(out, ignore_index=True)


def gnn_definitive_test() -> pd.DataFrame:
    out = []
    for seed in SEEDS:
        for split, name, start, end, month in TEST:
            (run,) = glob.glob(str(GNN_ROOT / f"seed_{seed}/runs/{name}/*/spatial_predictions.csv"))
            sp = pd.read_csv(run)
            ci, cj = cell_key(sp.x * sp.lon_std + sp.lon_mean, sp.y * sp.lat_std + sp.lat_mean)
            out.append(pd.DataFrame({"model": "GNN", "seed": seed, "split": split, "week": name, "month": month,
                                     "date": pd.to_datetime(sp.date).dt.normalize(), "ci": ci, "cj": cj,
                                     "y": sp.observed.astype(float), "rate": sp.predicted}))
    return pd.concat(out, ignore_index=True)


def main():
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    parts = lap_gcmp(load_panel())
    parts.append(glm())
    pred = pd.concat(parts, ignore_index=True)
    for model in ("LAP", "GCMP", "GLM"):
        check_mae(pred[pred.model == model], model)

    gnn_cfg = load_cfg("gnn", "config/15_gnn_salinity_cellpoisson.yaml")
    conv_cfg = load_cfg("convlstm", "config/18_convlstm_salinity.yaml")
    tasks = (make_tasks("GNN", "gnn", "poisson", gnn_cfg, VAL, SEEDS)
             + make_tasks("GNN (rerun)", "gnn", "poisson", gnn_cfg, TEST, SEEDS)
             + make_tasks("ConvLSTM", "convlstm", "poisson", conv_cfg, VAL + TEST, SEEDS))
    print(f"Training {len(tasks)} neural models …", flush=True)
    neural = run_tasks(tasks)
    gnn_test = gnn_definitive_test()
    check_mae(gnn_test, "GNN")
    check_mae(neural[neural.model == "ConvLSTM"], "ConvLSTM", tol=1e-5)

    pred = pd.concat([pred, neural[COLS], gnn_test[COLS]], ignore_index=True)
    pred["date"] = pd.to_datetime(pred["date"]).dt.normalize()
    pred.to_parquet(PRED_DIR / "poisson_baselines.parquet", index=False)
    meta = {"models": sorted(pred.model.unique()), "rows": len(pred), "minutes": round((time.time() - t0) / 60, 1),
            "configs": {"GLM": "config/14_window_poisson_glm_salinity.yaml",
                        "GNN": "config/15_gnn_salinity_cellpoisson.yaml",
                        "ConvLSTM": "config/18_convlstm_salinity.yaml"}}
    (PRED_DIR / "poisson_baselines.yaml").write_text(yaml.safe_dump(meta, sort_keys=False))
    print(f"Saved {len(pred):,} rows → {PRED_DIR / 'poisson_baselines.parquet'} ({meta['minutes']} min)")


if __name__ == "__main__":
    main()
