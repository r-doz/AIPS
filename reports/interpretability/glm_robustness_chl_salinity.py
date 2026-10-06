"""Poisson GLM robustness checks for the chlorophyll-a and salinity effects (appendix, interpretability).

Cell-specific intercepts + the LGCP covariates (standardized on the training rows), fitted on the
training data of the first May and November windows. For each specification, reports the coefficient
(per training SD) and a quasi-Poisson likelihood-ratio test for dropping chl / salinity.
"""

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.linear_model import PoissonRegressor

COVS = ["chl", "thetao", "fishing_block", "is_holiday", "is_weekend", "cell_lag_1", "cell_lag_7", "salinity"]

df = pd.concat([pd.read_parquet(f"data/processed/cpr_gfw_salinity_{y}.parquet") for y in (2024, 2025)])
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["longitude", "latitude", "date"])
grp = df.groupby(["longitude", "latitude"])["ais_vessels_count"]
df["cell_lag_1"] = grp.shift(1).fillna(0)
df["cell_lag_7"] = grp.shift(7).fillna(0)
df["chl_7d"] = df.groupby(["longitude", "latitude"])["chl"].transform(lambda s: s.rolling(7, min_periods=1).mean())
df["cell"] = df.groupby(["longitude", "latitude"]).ngroup()
df["dow"] = df.date.dt.dayofweek
df["month"] = df.date.dt.month
df["y2025"] = (df.date.dt.year == 2025).astype(float)


def design(tr, chl_col, dow, year, month):
    X = tr[[chl_col if c == "chl" else c for c in COVS]].astype(float)
    X.columns = COVS
    X = (X - X.mean()) / X.std(ddof=0)
    parts = [X, pd.get_dummies(tr.cell, prefix="c", drop_first=True).astype(float)]
    if dow:  # Tue-Fri and Sunday; Monday is the reference and Saturday is carried by is_weekend
        parts.append(pd.get_dummies(tr.dow, prefix="d").astype(float)[["d_1", "d_2", "d_3", "d_4", "d_6"]])
    if year:
        parts.append(tr[["y2025"]])
    if month:
        parts.append(pd.get_dummies(tr.month, prefix="m", drop_first=True).astype(float))
    return pd.concat(parts, axis=1)


def fit(X, y):
    m = PoissonRegressor(alpha=0, max_iter=5000, solver="newton-cholesky", tol=1e-9).fit(X, y)
    mu = m.predict(X)
    dev = 2 * np.sum(np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / mu), 0) - (y - mu))
    return m, dev, np.sum((y - mu) ** 2 / mu)


SPECS = [
    ("daily chl", "chl", 0, 0, 0), ("daily chl + weekday", "chl", 1, 0, 0),
    ("daily chl + weekday + year", "chl", 1, 1, 0), ("daily chl + weekday + year + month", "chl", 1, 1, 1),
    ("7d chl", "chl_7d", 0, 0, 0), ("7d chl + weekday", "chl_7d", 1, 0, 0),
    ("7d chl + weekday + year", "chl_7d", 1, 1, 0), ("7d chl + weekday + year + month", "chl_7d", 1, 1, 1),
]

for end in ("2025-05-01", "2025-11-01"):
    tr = df[df.date < end]
    y = tr.ais_vessels_count.to_numpy().astype(float)
    print(f"=== train < {end} (n={len(tr)}) ===")
    for name, chl_col, dow, year, month in SPECS:
        X = design(tr, chl_col, dow, year, month)
        m, dev, pearson = fit(X, y)
        phi = pearson / (len(y) - X.shape[1] - 1)
        coef = pd.Series(m.coef_, index=X.columns)
        out = []
        for c in ("chl", "salinity"):
            _, dev_reduced, _ = fit(X.drop(columns=[c]), y)
            stat = (dev_reduced - dev) / phi
            out.append(f"{c}: b={coef[c]:+.3f} p={chi2.sf(stat, 1):.1e}")
        print(f"  {name:36s} " + " | ".join(out))
