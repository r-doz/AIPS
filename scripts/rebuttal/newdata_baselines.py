"""Cheap baselines on the new datasets (rebuttal): LAP, GCMP, Poisson GLM and hurdle GLM.

Same models and settings as in the paper and in the reviewer-1 rebuttal, with only the data changed
(datasets as in newdata_make_configs.py: venice, trieste_4y, trieste_2y):
  LAP, GCMP   r1_poisson_baselines.lap_gcmp (last available count; cell mean of all previous days)
  GLM         the paper's window Poisson GLM (config/14_window_poisson_glm_salinity.yaml), refitted before each week
  Hurdle GLM  r1_hurdle_glm with the settings selected for reviewer 1 on the paper's validation weeks
              (reports/rebuttal/reviewer1/hparam/hurdle_glm_selected.yaml), not re-tuned
Predictions for the paper's 4 validation weeks and 8 test weeks of 2025. r1_evaluate.evaluate_model then tunes
the activity threshold on the validation weeks and scores the test weeks, as in the reviewer-1 tables.

Output: reports/rebuttal/new_data/baselines/<dataset>_predictions.parquet and <dataset>_per_window.csv
Usage: python newdata_baselines.py [--datasets venice trieste_4y trieste_2y]
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler

import r1_hurdle_glm as hg
from newdata_make_configs import EXPERIMENTS
from r1_common import ALL_WEEKS, ROOT, cell_key, load_module
from r1_evaluate import evaluate_model
from r1_poisson_baselines import lap_gcmp

OUT = ROOT / "reports/rebuttal/new_data/baselines"
_glm = load_module("nd_glm", "scripts/14_train_window_poisson_glm.py")
GLM_CONFIG = "config/14_window_poisson_glm_salinity.yaml"


def glm_config(ds: str) -> dict:
    cfg = _glm.load_config(str(ROOT / GLM_CONFIG))
    cfg["parquet_path"], cfg["years"] = EXPERIMENTS[ds]["parquet_path"], EXPERIMENTS[ds]["years"]
    return cfg


def panel(ds: str) -> pd.DataFrame:
    base = EXPERIMENTS[ds]["parquet_path"].replace(".parquet", "")
    df = pd.concat([pd.read_parquet(ROOT / f"{base}_{y}.parquet") for y in EXPERIMENTS[ds]["years"]])
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["ci"], df["cj"] = cell_key(df.longitude, df.latitude)
    return df.sort_values(["ci", "cj", "date"]).reset_index(drop=True)


def glm(ds: str) -> pd.DataFrame:
    cfg = glm_config(ds)
    raw = _glm.load_parquet_years(str(ROOT / cfg["parquet_path"]), cfg.get("years"))
    raw["date"] = pd.to_datetime(raw["date"])
    win, feats = _glm.build_window_dataframe(raw, cfg)
    ci, cj = cell_key(win.longitude, win.latitude)
    out = []
    for split, name, start, end, month in ALL_WEEKS:
        tr, te = _glm.make_split_masks(dates=win["date"], split_strategy="fixed_test_window", train_fraction=cfg["train_fraction"],
                                       random_seed=cfg["data_seed"], test_start_date=start, test_end_date=end)
        X, y = win[feats].values.astype(np.float32), win[cfg["target_col"]].values.astype(np.float32)
        sc = StandardScaler().fit(X[tr])
        m = PoissonRegressor(alpha=float(cfg["poisson_alpha"]), max_iter=int(cfg["max_iter"])).fit(sc.transform(X[tr]), y[tr])
        out.append(pd.DataFrame({"model": "GLM", "seed": 1, "split": split, "week": name, "month": month,
                                 "date": pd.to_datetime(win["date"].values[te]).normalize(), "ci": ci[te], "cj": cj[te],
                                 "y": y[te].astype(float), "rate": np.clip(m.predict(sc.transform(X[te])), 0, None)}))
    return pd.concat(out, ignore_index=True)


class Features(hg.Features):
    """r1_hurdle_glm.Features on another dataset."""

    def __init__(self, ds: str):
        self.cfg = glm_config(ds)
        self.raw = _glm.load_parquet_years(str(ROOT / self.cfg["parquet_path"]), self.cfg.get("years"))
        self.raw["date"] = pd.to_datetime(self.raw["date"])
        self.cache = {}


def hurdle_glm(ds: str) -> pd.DataFrame:
    sel = yaml.safe_load(open(ROOT / "reports/rebuttal/reviewer1/hparam/hurdle_glm_selected.yaml"))["selected"]
    F = Features(ds)
    out = []
    for split, name, start, end, month in ALL_WEEKS:
        Xtr, ytr, Xte, yte, wte = F.split(sel["window_size"], start, end)
        pos = ytr > 0
        p = hg.fit_logit(Xtr, pos.astype(int), sel["alpha_logit"]).predict_proba(Xte)[:, 1]
        lam = hg.predict_ztp(hg.fit_ztp(Xtr[pos], ytr[pos], sel["alpha_ztp"]), Xte)
        ci, cj = cell_key(wte.longitude, wte.latitude)
        out.append(pd.DataFrame({"model": "Hurdle GLM", "seed": 1, "split": split, "week": name, "month": month,
                                 "date": pd.to_datetime(wte["date"].values).normalize(), "ci": ci, "cj": cj,
                                 "y": yte, "p": p, "lam": lam}))
    return pd.concat(out, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=list(EXPERIMENTS))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for ds in args.datasets:
        preds = lap_gcmp(panel(ds)) + [glm(ds), hurdle_glm(ds)]
        pred = pd.concat(preds, ignore_index=True)
        pred["date"] = pd.to_datetime(pred["date"]).dt.normalize()
        pred.to_parquet(OUT / f"{ds}_predictions.parquet", index=False)
        taus, per = [], []
        for name, g in pred.groupby("model", sort=False):
            tau, rows, _ = evaluate_model(name, g)
            taus.append(tau)
            per.append(rows)
        pd.concat(per, ignore_index=True).assign(dataset=ds).to_csv(OUT / f"{ds}_per_window.csv", index=False)
        pd.DataFrame(taus).to_csv(OUT / f"{ds}_thresholds.csv", index=False)
        print(f"[{ds}] {len(pred):,} prediction rows; thresholds {pd.DataFrame(taus).set_index('model')[['tau_acc', 'tau_f1']].to_dict('index')}", flush=True)


if __name__ == "__main__":
    main()
