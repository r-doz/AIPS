"""
Evaluation metrics for Poisson count models.

These metrics are shared by the LGCP model and simple Poisson baselines.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import gammaln


def evaluate_metrics(
    y_true: np.ndarray, rate_mean: np.ndarray, dates: pd.DatetimeIndex | np.ndarray
) -> dict:
    """
    Compute Poisson prediction quality metrics.

    Metrics
    -------
    Observation-level metrics
    -------------------------
    mean_ll_obs : mean Poisson log-likelihood per observation/cell
    mae_obs     : mean absolute error per observation/cell
    rmse_obs    : root mean squared error per observation/cell

    Daily-level metrics
    -------------------
    mean_ll_daily : mean Poisson log-likelihood of daily totals
    mae_daily     : mean absolute error of daily totals
    rmse_daily    : root mean squared error of daily totals

    Parameters
    ----------
    y_true    : [N] observed counts
    rate_mean : [N] predicted Poisson rates
    dates     : [N] corresponding dates, one per observation
    """
    dates = pd.to_datetime(dates)

    y_true = np.asarray(y_true)
    rate_mean = np.asarray(rate_mean)
    rate_mean = np.clip(rate_mean, 0, None)

    eps = 1e-9

    # ------------------------------------------------------------------
    # Observation-level metrics
    # ------------------------------------------------------------------
    log_ll_obs = y_true * np.log(rate_mean + eps) - rate_mean - gammaln(y_true + 1)
    mean_ll_obs = float(log_ll_obs.mean())

    mae_obs = float(np.mean(np.abs(y_true - rate_mean)))
    rmse_obs = float(np.sqrt(np.mean((y_true - rate_mean) ** 2)))

    # ------------------------------------------------------------------
    # Daily-level metrics
    # ------------------------------------------------------------------
    df_eval = pd.DataFrame(
        {
            "date": dates,
            "y": y_true,
            "lambda_hat": rate_mean,
        }
    )

    daily = df_eval.groupby("date", as_index=False).sum()

    y_daily = daily["y"].values
    lambda_daily = daily["lambda_hat"].values

    log_ll_daily = (
        y_daily * np.log(lambda_daily + eps) - lambda_daily - gammaln(y_daily + 1)
    )
    mean_ll_daily = float(log_ll_daily.mean())

    mae_daily = float(np.mean(np.abs(y_daily - lambda_daily)))
    rmse_daily = float(np.sqrt(np.mean((y_daily - lambda_daily) ** 2)))

    return {
        # Observation-level metrics
        "mean_ll_obs": mean_ll_obs,
        "mae_obs": mae_obs,
        "rmse_obs": rmse_obs,
        # Daily-level metrics
        "mean_ll_daily": mean_ll_daily,
        "mae_daily": mae_daily,
        "rmse_daily": rmse_daily,
    }
