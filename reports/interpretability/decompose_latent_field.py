"""Additive decomposition of the LGCP posterior-mean latent field f(s, t).

K = (k_s0 + k_s1) * (k_t0 + k_t1), and the periodic kernel k_t1 splits into a
time-constant Fourier term c0 * var_t1 plus an oscillating remainder, so the
posterior mean f = K_xZ K_ZZ^{-1} m is a sum of separately computable pieces.
Evaluated on the 49 cells x all training days of each of the 80 definitive runs.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.special import ive

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.models.lgcp import build_Kuu, periodic_kernel, rbf_kernel  # noqa: E402

torch.set_default_dtype(torch.float64)

MU_LON, SD_LON = 13.39795918, 0.17917775
MU_LAT, SD_LAT = 45.63052718, 0.0622551
T0 = pd.Timestamp("2024-01-01")
SPAN_DAYS = 730

params = pd.read_csv(ROOT / "reports/interpretability/lgcp_params_all_runs.csv")
df = pd.read_parquet(ROOT / "data/processed/cpr_gfw_salinity_2024.parquet", columns=["longitude", "latitude"])
cells = df.drop_duplicates().sort_values(["latitude", "longitude"]).to_numpy().astype(np.float64)
cells_std = np.column_stack([(cells[:, 0] - MU_LON) / SD_LON, (cells[:, 1] - MU_LAT) / SD_LAT])

rows, static_maps, phase_profiles, weekday_profiles = [], [], [], []
for r in params.itertuples():
    run = ROOT / r.run_dir
    state = torch.load(run / "model.pt", map_location="cpu", weights_only=False)["model_state"]
    st = {k: v.double() for k, v in state.items()}
    td = yaml.safe_load(open(run / "time_diagnostics.yaml"))
    mu_t, sd_t = td["t_raw_mean"], td["t_raw_std"]

    ex = lambda k: torch.exp(st[k])
    vs0, vs1 = ex("log_spatial_0_variance"), ex("log_spatial_1_variance")
    ls0, ls1 = ex("log_spatial_0_lengthscale"), ex("log_spatial_1_lengthscale")
    vt0, vt1 = ex("log_temporal_0_variance"), ex("log_temporal_1_variance")
    lt0, lt1 = ex("log_temporal_0_lengthscale"), ex("log_temporal_1_lengthscale")
    per = ex("log_temporal_1_period")
    kern = {
        "spatial": [
            {"type": "rbf", "params": {"lengthscale": ls0, "variance": vs0}},
            {"type": "rbf", "params": {"lengthscale": ls1, "variance": vs1}},
        ],
        "temporal": [
            {"type": "rbf", "params": {"lengthscale": lt0, "variance": vt0}},
            {"type": "periodic", "params": {"lengthscale": lt1, "variance": vt1, "period": per}},
        ],
        "spatial_composition": "sum",
        "temporal_composition": "sum",
        "spacetime_composition": "product",
        "noise_var": ex("log_noise"),
    }
    Z, m = st["Z"], st["m"]
    Kuu = build_Kuu(Z, kern, 1e-6)
    L = torch.linalg.cholesky(Kuu + 1e-8 * torch.eye(Z.shape[0]))
    w = torch.cholesky_solve(m.unsqueeze(-1), L).squeeze(-1)

    days = pd.date_range(T0, pd.Timestamp(r.test_start) - pd.Timedelta(days=1))
    t_std = (((days - T0).days.to_numpy() / SPAN_DAYS) - mu_t) / sd_t
    S = torch.tensor(cells_std)
    T = torch.tensor(t_std).unsqueeze(-1)

    Ks0 = rbf_kernel(S, Z[:, :2], ls0, vs0)
    Ks1 = rbf_kernel(S, Z[:, :2], ls1, vs1)
    Kt0 = rbf_kernel(T, Z[:, 2:3], lt0, vt0)
    Kp = periodic_kernel(T, Z[:, 2:3], per, lt1, vt1)
    c0 = float(ive(0, 1.0 / float(lt1) ** 2))
    Kconst = torch.full_like(Kp, float(vt1) * c0)
    Kosc = Kp - Kconst

    comp = lambda Ks, Kt: torch.einsum("ck,dk,k->cd", Ks, Kt, w).numpy()
    parts = {
        "static_coarse": comp(Ks0, Kconst),
        "static_fine": comp(Ks1, Kconst),
        "cycle_coarse": comp(Ks0, Kosc),
        "cycle_fine": comp(Ks1, Kosc),
        "short_coarse": comp(Ks0, Kt0),
        "short_fine": comp(Ks1, Kt0),
    }
    f = sum(parts.values())
    Kfu = (Ks0 + Ks1).unsqueeze(1) * (Kt0 + Kp).unsqueeze(0)
    f_direct = torch.einsum("cdk,k->cd", Kfu, w).numpy()
    assert np.allclose(f, f_direct, atol=1e-8), np.abs(f - f_direct).max()

    tot_var = f.var()
    rec = {"seed": r.seed, "window": r.window, "month": r.month, "alpha": r.alpha,
           "f_mean": f.mean(), "f_var": tot_var, "f_sd": f.std(),
           "f_time_mean_sd_across_cells": f.mean(1).std()}
    for k, v in parts.items():
        rec[f"var_{k}"] = v.var()
        rec[f"share_{k}"] = v.var() / tot_var
    static = parts["static_coarse"] + parts["static_fine"]
    cycle = parts["cycle_coarse"] + parts["cycle_fine"]
    short = parts["short_coarse"] + parts["short_fine"]
    rec["share_static"] = static.var() / tot_var
    rec["share_cycle"] = cycle.var() / tot_var
    rec["share_short"] = short.var() / tot_var
    rec["corr_static_vs_logcellmean"] = np.nan
    rows.append(rec)

    static_maps.append(pd.DataFrame({"seed": r.seed, "window": r.window, "longitude": cells[:, 0],
                                     "latitude": cells[:, 1], "f_static": static.mean(1),
                                     "f_time_mean": f.mean(1)}))
    phase = ((days - T0).days.to_numpy() % 14)
    cyc_mean = cycle.mean(0)
    phase_profiles.append(pd.DataFrame({"seed": r.seed, "window": r.window, "phase": phase,
                                        "weekday": days.dayofweek, "cycle_cellmean": cyc_mean}))

out = ROOT / "reports/interpretability"
res = pd.DataFrame(rows)
res.to_csv(out / "latent_field_decomposition_all_runs.csv", index=False)
maps = pd.concat(static_maps)
maps.to_csv(out / "latent_static_maps_all_runs.csv", index=False)
prof = pd.concat(phase_profiles)
prof.to_csv(out / "latent_cycle_profiles_all_runs.csv", index=False)
print(res.groupby("month")[["f_mean", "f_sd", "share_static", "share_cycle", "share_short",
                            "share_static_coarse", "share_static_fine", "share_cycle_coarse",
                            "share_cycle_fine", "share_short_coarse", "share_short_fine"]].agg(["mean", "std"]).T.round(4))
