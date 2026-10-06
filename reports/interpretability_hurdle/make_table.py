"""Table of the LGCP parameters of the joint hurdle Red-LGCP as CSV (from params_effects_all_runs.csv).

One row per parameter: mean and population std over 10 seeds x 4 weekly windows per month and, for the
coefficients, the multiplicative effect on the intensity of active cells (mean over runs).
"""

from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parent
d = pd.read_csv(OUT / "params_effects_all_runs.csv")
MONTHS = ("may", "november")

ROWS = [("alpha", "alpha", "Intercept", "")]
ROWS += [(f"beta_{c}", f"beta_{c}", desc, unit) for c, desc, unit in [
    ("chl", "Chlorophyll-a", "per SD"), ("thetao", "Sea temperature", "per degree C"),
    ("salinity", "Salinity", "per psu"), ("fishing_block", "Fishing ban", "on vs off"),
    ("is_holiday", "Public holiday", "on vs off"), ("is_weekend", "Weekend", "on vs off"),
    ("cell_lag_1", "Count at t-1", "per count"), ("cell_lag_7", "Count at t-7", "per count")]]
ROWS += [("sigma2_s_broad", "var_s_broad", "Spatial RBF variance, lengthscale 0.2", ""),
         ("sigma2_s_loc", "var_s_loc", "Spatial RBF variance, lengthscale 0.01", ""),
         ("sigma2_t_RBF", "var_t_rbf", "Temporal RBF variance, lengthscale 0.01", ""),
         ("sigma2_t_per", "var_t_per", "Periodic kernel variance, lengthscale 0.8", ""),
         ("p", "period_days", "Period (days)", ""),
         ("share_static", "share_static", "Variance share of the time-constant cell map", ""),
         ("share_periodic", "share_cycle", "Variance share of the periodic component", ""),
         ("share_short_term", "share_short", "Variance share of the temporal RBF component", "")]

rows = []
for parameter, col, desc, unit in ROWS:
    row = {"parameter": parameter, "description": desc, "effect_unit": unit}
    for m in MONTHS:
        q = d[d.month == m]
        row[f"{m}_mean"] = q[col].mean()
        row[f"{m}_std"] = q[col].std(ddof=0)
    for m in MONTHS:
        effect = col.replace("beta_", "effect_")
        row[f"{m}_effect"] = d[d.month == m][effect].mean() if col.startswith("beta_") else None
    rows.append(row)

table = pd.DataFrame(rows)
table.to_csv(OUT / "lgcp_parameters_table.csv", index=False)
print(table.round(3).to_string(index=False))
