"""
src/data.py
Data loading, preprocessing, and metadata utilities for the LGCP pipeline.

Key design decisions
--------------------
* Scalers are fitted on TRAINING data only, then applied to test data.
  (The original code fitted scalers on the full dataset before splitting —
   a form of data leakage.)
* Train/test split is done at the day level (all grid cells for a given
  date go entirely to train or test), which matches the downstream
  evaluation strategy.
* t_norm = day_of_year / days_in_year ∈ (0, 1] is computed before
  standardization; the raw t_norm is stored in the DataFrame and in
  scalers for inverse-transform convenience.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import gammaln
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_leap_year(year: int) -> bool:
    return (year % 4 == 0) and ((year % 100 != 0) or (year % 400 == 0))


# ---------------------------------------------------------------------------
# Main data-preparation function
# ---------------------------------------------------------------------------


def prepare_data(
    parquet_path: str, train_fraction: float = 0.9, random_seed: int = 42
) -> tuple:
    """
    Load a parquet file and return train/test tensors ready for SparseLGCP.

    Expected columns
    ----------------
    date              : str or datetime (YYYY-MM-DD)
    latitude, longitude : float
    chl, thetao       : float covariates
    ais_vessels_count : int-like count

    Returns
    -------
    train_coords : [N_train, 3]   (lon_std, lat_std, t_std)
    train_covs   : [N_train, 2]   (chl_std, thetao_std)
    train_y      : [N_train]      integer counts
    test_coords  : [N_test, 3]
    test_covs    : [N_test, 2]
    test_y       : [N_test]
    scalers      : dict  {coord_scaler, cov_scaler, t_scaler}
    df           : original DataFrame (with t_norm column added)
    """
    df = pd.read_parquet(parquet_path)
    df["date"] = pd.to_datetime(df["date"])

    # ----- Compute fractional day-of-year ∈ (0, 1] ---------------------------
    year = int(df["date"].dt.year.iloc[0])
    days_in_year = 366.0 if _is_leap_year(year) else 365.0
    df["t_norm"] = df["date"].dt.dayofyear.astype(float) / days_in_year

    # ----- Day-level train/test split (no leakage between days) ---------------
    unique_dates = np.sort(df["date"].unique())
    rng = np.random.default_rng(random_seed)
    shuffled = rng.permutation(unique_dates)
    n_train = int(train_fraction * len(unique_dates))
    train_dates = set(shuffled[:n_train])
    test_dates = set(shuffled[n_train:])

    train_mask = df["date"].isin(train_dates).values
    test_mask = df["date"].isin(test_dates).values

    # ----- Raw arrays ---------------------------------------------------------
    coords_raw = df[["longitude", "latitude"]].values.astype(np.float32)
    t_raw = df["t_norm"].values.astype(np.float32).reshape(-1, 1)
    covs_raw = df[["chl", "thetao"]].values.astype(np.float32)
    y = df["ais_vessels_count"].fillna(0).astype(np.int32).values

    # ----- Fit scalers on TRAIN only, then transform both splits --------------
    coord_scaler = StandardScaler().fit(coords_raw[train_mask])
    t_scaler = StandardScaler().fit(t_raw[train_mask])
    cov_scaler = StandardScaler().fit(covs_raw[train_mask])

    coords_std = coord_scaler.transform(coords_raw)
    t_std = t_scaler.transform(t_raw)
    covs_std = cov_scaler.transform(covs_raw)

    # Full coords tensor: [lon_std, lat_std, t_std]
    coords_full = np.hstack([coords_std, t_std]).astype(np.float32)
    covs_full = covs_std.astype(np.float32)

    scalers = {
        "coord_scaler": coord_scaler,
        "cov_scaler": cov_scaler,
        "t_scaler": t_scaler,
    }

    return (
        coords_full[train_mask],
        covs_full[train_mask],
        y[train_mask],
        coords_full[test_mask],
        covs_full[test_mask],
        y[test_mask],
        scalers,
        df,
    )


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def compute_meta(
    df: pd.DataFrame, grid_x: int | None = None, grid_y: int | None = None
) -> dict:
    """
    Extract spatial and temporal metadata from the raw DataFrame.

    The meta dict is used by plotting routines; it stores *original*
    (unstandardized) bounds so that axes can be labeled in physical units.
    """
    df["date"] = pd.to_datetime(df["date"])

    dates_unique = np.sort(df["date"].unique())
    t_min, t_max = df["date"].min(), df["date"].max()

    lon_min, lon_max = float(df["longitude"].min()), float(df["longitude"].max())
    lat_min, lat_max = float(df["latitude"].min()), float(df["latitude"].max())

    if grid_x is None:
        grid_x = int(df["longitude"].nunique())
    if grid_y is None:
        grid_y = int(df["latitude"].nunique())

    return {
        "dates": dates_unique,
        "grid_x": grid_x,
        "grid_y": grid_y,
        "lon_min": lon_min,
        "lon_max": lon_max,
        "lat_min": lat_min,
        "lat_max": lat_max,
        "t_min": t_min,
        "t_max": t_max,
        # convenient scalar stats for inverse-transform
        "lon_mean": float(df["longitude"].mean()),
        "lon_std": float(df["longitude"].std()),
        "lat_mean": float(df["latitude"].mean()),
        "lat_std": float(df["latitude"].std()),
    }


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------


def evaluate_metrics(
    y_true: np.ndarray, rate_mean: np.ndarray, dates: pd.DatetimeIndex | np.ndarray
) -> dict:
    """
    Compute Poisson prediction quality metrics.

    Metrics
    -------
    mean_ll_obs    : mean Poisson log-likelihood per observation
    rmse_daily     : RMSE of daily totals (sum over grid cells)
    mean_ll_daily  : mean Poisson log-likelihood of daily totals

    Parameters
    ----------
    y_true     : [N]   observed counts
    rate_mean  : [N]   predicted Poisson rates
    dates      : [N]   corresponding dates (one per observation)
    """
    dates = pd.to_datetime(dates)
    rate_mean = np.clip(rate_mean, 0, None)  # safety

    # ---- Per-observation log-likelihood ----
    log_ll_obs = y_true * np.log(rate_mean + 1e-9) - rate_mean - gammaln(y_true + 1)
    mean_ll_obs = float(log_ll_obs.mean())

    # ---- Daily aggregates ----
    df_eval = pd.DataFrame({"date": dates, "y": y_true, "lambda_hat": rate_mean})
    daily = df_eval.groupby("date").sum().reset_index()

    rmse_daily = float(np.sqrt(np.mean((daily["y"] - daily["lambda_hat"]) ** 2)))

    ll_daily = (
        daily["y"].values * np.log(daily["lambda_hat"].values + 1e-9)
        - daily["lambda_hat"].values
        - gammaln(daily["y"].values + 1)
    )
    mean_ll_daily = float(ll_daily.mean())

    return {
        "mean_ll_obs": mean_ll_obs,
        "rmse_daily": rmse_daily,
        "mean_ll_daily": mean_ll_daily,
    }
