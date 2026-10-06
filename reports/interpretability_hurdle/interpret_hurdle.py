"""Interpretable parameters of the joint hurdle Red-LGCP (80 runs: 10 seeds x 8 weekly windows).

In the hurdle model, alpha, beta and the latent field f describe the intensity lambda of the
zero-truncated Poisson, i.e. the counts of active cells; activity itself is modeled by the
classifier, which has no coefficient-level interpretation.

1. Parameters: alpha, beta, kernel variances and period of every run.
2. Effects: multiplicative effect exp(beta / sd) on lambda per physical unit (per training SD for
   chlorophyll-a, on vs off for the binary indicators), where sd comes from the covariate scaler
   fitted on the training data of each window, rebuilt with the same prepare_data call as in training.
3. Latent field: the posterior mean f = K_xZ K_ZZ^{-1} m on the 49 cells x training days is split
   into a time-constant map, a periodic component and a short-term component, splitting the
   periodic kernel into its time-constant Fourier term and an oscillating remainder.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.special import ive
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.models.data_pp_lgcp import prepare_data  # noqa: E402
from src.models.lgcp import build_Kuu, periodic_kernel, rbf_kernel  # noqa: E402

torch.set_default_dtype(torch.float64)
RUNS = ROOT / "reports/seeded_monthly_windows_hurdle"
OUT = ROOT / "reports/interpretability_hurdle"
COVS = ["chl", "thetao", "fishing_block", "is_holiday", "is_weekend", "cell_lag_1", "cell_lag_7", "salinity"]
T0, SPAN_DAYS = pd.Timestamp("2024-01-01"), 730
_scalers = {}


def window_scalers(run, window):
    """Coordinate and covariate scalers fitted on the training data of a window, as in training."""
    if window not in _scalers:
        cfg = yaml.safe_load(open(run / "params.yaml"))
        *_, scalers, _ = prepare_data(
            str(ROOT / cfg["parquet_path"]), train_fraction=cfg["train_fraction"], random_seed=cfg["data_seed"],
            split_strategy=cfg.get("split_strategy", "random_day"), covariate_cols=cfg.get("covariate_cols"),
            test_start_date=cfg.get("test_start_date"), test_end_date=cfg.get("test_end_date"),
            lag_features=cfg.get("lag_features"), years=cfg.get("years"))
        _scalers[window] = scalers
    return _scalers[window]


data = pd.concat([pd.read_parquet(ROOT / f"data/processed/cpr_gfw_salinity_{y}.parquet",
                                  columns=["date", "longitude", "latitude", "ais_vessels_count"]) for y in (2024, 2025)])
data["date"] = pd.to_datetime(data["date"])
cells = (data[["longitude", "latitude"]].drop_duplicates().sort_values(["latitude", "longitude"])
         .to_numpy().astype(np.float64))
assert len(cells) == 49

rows, maps, profiles = [], [], []
files = sorted(RUNS.glob("seed_*/runs/*/*/model.pt"))
assert len(files) == 80, len(files)
for f in files:
    run = f.parent
    seed, window = int(f.parts[-5].split("_")[1]), f.parts[-3]
    month, test_start = window.split("_")[1], yaml.safe_load(open(run / "params.yaml"))["test_start_date"]
    assert yaml.safe_load(open(run / "params.yaml"))["covariate_cols"] == COVS
    st = {k: v.double() for k, v in torch.load(f, map_location="cpu", weights_only=False)["model_state"].items()}
    td = yaml.safe_load(open(run / "time_diagnostics.yaml"))
    sc = window_scalers(run, window)
    cov_sd = dict(zip(COVS, sc["cov_scaler"].scale_))
    cells_std = (cells - sc["coord_scaler"].mean_) / sc["coord_scaler"].scale_
    ex = lambda k: torch.exp(st[k])  # noqa: E731
    vs0, vs1, vt0, vt1 = (ex(f"log_{k}_variance") for k in ("spatial_0", "spatial_1", "temporal_0", "temporal_1"))
    ls0, ls1, lt0, lt1 = (ex(f"log_{k}_lengthscale") for k in ("spatial_0", "spatial_1", "temporal_0", "temporal_1"))
    per = ex("log_temporal_1_period")
    rec = {"seed": seed, "window": window, "month": month, "alpha": float(st["alpha"]),
           "var_s_broad": float(vs0), "var_s_loc": float(vs1), "var_t_rbf": float(vt0), "var_t_per": float(vt1),
           "period_days": float(per) * td["days_per_standardized_unit"],
           "ls_t_rbf_days": float(lt0) * td["days_per_standardized_unit"]}
    for c, b in zip(COVS, st["beta"].tolist()):
        sd = cov_sd[c]
        rec[f"beta_{c}"] = b
        rec[f"effect_{c}"] = np.exp(b) if c == "chl" else np.exp(b / sd)

    # latent field decomposition on the training days of this window
    kern = {"spatial": [{"type": "rbf", "params": {"lengthscale": ls0, "variance": vs0}},
                        {"type": "rbf", "params": {"lengthscale": ls1, "variance": vs1}}],
            "temporal": [{"type": "rbf", "params": {"lengthscale": lt0, "variance": vt0}},
                         {"type": "periodic", "params": {"lengthscale": lt1, "variance": vt1, "period": per}}],
            "spatial_composition": "sum", "temporal_composition": "sum", "spacetime_composition": "product",
            "noise_var": ex("log_noise")}
    Z, m = st["Z"], st["m"]
    L = torch.linalg.cholesky(build_Kuu(Z, kern, 1e-6) + 1e-8 * torch.eye(Z.shape[0]))
    w = torch.cholesky_solve(m.unsqueeze(-1), L).squeeze(-1)
    days = pd.date_range(T0, pd.Timestamp(test_start) - pd.Timedelta(days=1))
    t_std = (((days - T0).days.to_numpy() / SPAN_DAYS) - td["t_raw_mean"]) / td["t_raw_std"]
    S, T = torch.tensor(cells_std), torch.tensor(t_std).unsqueeze(-1)
    Ks0, Ks1 = rbf_kernel(S, Z[:, :2], ls0, vs0), rbf_kernel(S, Z[:, :2], ls1, vs1)
    Kt0, Kp = rbf_kernel(T, Z[:, 2:3], lt0, vt0), periodic_kernel(T, Z[:, 2:3], per, lt1, vt1)
    Kconst = torch.full_like(Kp, float(vt1) * float(ive(0, 1.0 / float(lt1) ** 2)))
    comp = lambda Ks, Kt: torch.einsum("ck,dk,k->cd", Ks, Kt, w).numpy()  # noqa: E731
    static, cycle, short = comp(Ks0 + Ks1, Kconst), comp(Ks0 + Ks1, Kp - Kconst), comp(Ks0 + Ks1, Kt0)
    fbar = static + cycle + short
    direct = torch.einsum("cdk,k->cd", (Ks0 + Ks1).unsqueeze(1) * (Kt0 + Kp).unsqueeze(0), w).numpy()
    assert np.allclose(fbar, direct, atol=1e-8)
    rec.update({"f_mean": fbar.mean(), "share_static": static.var() / fbar.var(),
                "share_cycle": cycle.var() / fbar.var(), "share_short": short.var() / fbar.var()})

    # static map vs observed cell activity in the training period
    tr = data[data.date < pd.Timestamp(test_start)]
    by_cell = tr.groupby(["longitude", "latitude"]).ais_vessels_count
    key = pd.MultiIndex.from_arrays([cells[:, 0], cells[:, 1]])
    mean_all = by_cell.mean().reindex(key).to_numpy()
    mean_pos = tr[tr.ais_vessels_count > 0].groupby(["longitude", "latitude"]).ais_vessels_count.mean().reindex(key).to_numpy()
    smap = static.mean(1)
    ok = ~np.isnan(mean_pos)
    rec["spearman_map_vs_mean_count"] = spearmanr(smap, mean_all).statistic
    rec["spearman_map_vs_mean_positive_count"] = spearmanr(smap[ok], mean_pos[ok]).statistic
    rec["map_range_factor"] = float(np.exp(smap.max() - smap.min()))
    maps.append(pd.DataFrame({"seed": seed, "window": window, "month": month, "longitude": cells[:, 0],
                              "latitude": cells[:, 1], "f_static": smap}))

    # periodic component averaged over cells: weekly share of its 14-day profile and weekday profile
    cyc = pd.DataFrame({"phase": (days - T0).days.to_numpy() % 14, "weekday": days.dayofweek, "c": cycle.mean(0)})
    prof = cyc.groupby("phase").c.mean().to_numpy()
    weekly = np.tile((prof[:7] + prof[7:]) / 2, 2)
    rec["weekly_share_of_cycle"] = weekly.var() / prof.var()
    wd = cyc.groupby("weekday").c.mean()
    rec["weekday_peak_to_trough"] = float(np.exp(wd.max() - wd.min()))
    profiles.append(pd.DataFrame({"seed": seed, "window": window, "month": month, "weekday": wd.index, "cycle": wd.to_numpy()}))
    rows.append(rec)

res = pd.DataFrame(rows)
res.to_csv(OUT / "params_effects_all_runs.csv", index=False)
pd.concat(maps).to_csv(OUT / "static_maps_all_runs.csv", index=False)
prof = pd.concat(profiles)
prof.to_csv(OUT / "weekday_profiles_all_runs.csv", index=False)

pd.set_option("display.width", 220)
num = [c for c in res.columns if c not in ("seed", "window", "month")]
summ = res.groupby("month")[num].agg(["mean", lambda s: s.std(ddof=0)]).T
summ.index = [f"{a}_{'mean' if b == 'mean' else 'std'}" for a, b in summ.index]
summ.to_csv(OUT / "summary_by_month.csv")
print(summ.round(4).to_string())
print("\nSign consistency of beta over the 80 runs (positive / 80):")
print({c: int((res[f"beta_{c}"] > 0).sum()) for c in COVS})
print("\nRelative std of beta within month (std / |mean|):")
print(res.groupby("month")[[f"beta_{c}" for c in COVS]].agg(lambda s: s.std(ddof=0) / abs(s.mean())).round(3).to_string())
mp = pd.concat(maps).groupby(["month", "longitude", "latitude"]).f_static.mean().unstack(0)
print("\nCorrelation of the mean static maps, May vs Nov.:", round(np.corrcoef(mp["may"], mp["november"])[0, 1], 3))
print("\nWeekday profile of the periodic component (mean over runs, 0 = Monday):")
print(prof.groupby(["month", "weekday"]).cycle.mean().unstack(1).round(3).to_string())
