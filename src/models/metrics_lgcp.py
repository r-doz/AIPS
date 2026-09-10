"""
Evaluation metrics for Poisson count models.

These metrics are shared by the LGCP model and simple Poisson baselines.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import gammaln
from scipy.stats import wasserstein_distance


def evaluate_wasserstein(y_true, rate_mean) -> float:
    """Empirical 1-Wasserstein distance, following bpaf045, Table 2, Eq. 9.

    Integrates |CDF_observed(x) - CDF_predicted(x)| over the count axis.
    Each cell-day has equal weight; values retain their original count units.
    This compares value distributions, ignoring spatial/temporal ordering.
    Use final predictions before likelihood-specific epsilon floors.
    Empty or non-finite inputs yield NaN, rather than dropping observations.
    """
    observed = np.asarray(y_true, dtype=float).reshape(-1)
    predicted = np.asarray(rate_mean, dtype=float).reshape(-1)
    if not observed.size or not predicted.size:
        return float("nan")
    if not np.all(np.isfinite(observed)) or not np.all(np.isfinite(predicted)):
        return float("nan")
    return float(wasserstein_distance(observed, predicted))


def evaluate_activity_metrics(y_true, rate_mean) -> dict:
    """Binary activity scores per observation, following Fishes 2025, 10, 479.

    Observed and predicted activity mean strictly positive counts/rates (> 0).
    Use predictions before any numerical epsilon floor for log-likelihoods.
    Scores are fractions in [0, 1]; undefined precision/recall are 0.
    Empty inputs return NaN scores. Positive rates everywhere imply recall 1
    when observed activity exists, even if those rates are very small.
    """
    observed = np.asarray(y_true).reshape(-1) > 0
    predicted = np.asarray(rate_mean).reshape(-1) > 0
    if observed.shape != predicted.shape:
        raise ValueError("Observed counts and predicted rates must have equal size.")
    if not observed.size:
        return dict.fromkeys(("accuracy", "precision", "recall"), float("nan"))
    tp = int(np.count_nonzero(observed & predicted))
    predicted_positive = int(np.count_nonzero(predicted))
    observed_positive = int(np.count_nonzero(observed))
    return {
        "accuracy": float(np.mean(observed == predicted)),
        "precision": float(tp / predicted_positive) if predicted_positive else 0.0,
        "recall": float(tp / observed_positive) if observed_positive else 0.0,
    }


def _safe_corr(x, y):
    """
    Safe Pearson correlation.

    Returns np.nan if the correlation is not defined, for example when
    one of the two series is constant or too short.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 3:
        return np.nan

    if np.std(x) == 0.0 or np.std(y) == 0.0:
        return np.nan

    return float(np.corrcoef(x, y)[0, 1])


def evaluate_daily_trend_metrics(
    y_true,
    rate_mean,
    dates,
    eps_direction=1e-8,
):
    """
    Evaluate whether the model captures the daily temporal trend.

    Metrics:
    - daily_delta_corr:
        Pearson correlation between observed daily changes and predicted
        daily changes.

    - daily_direction_accuracy_moving:
        Fraction of non-flat observed daily transitions for which the model
        predicts the correct direction: up or down.
    """

    df = pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "y_true": np.asarray(y_true, dtype=float),
            "rate_mean": np.asarray(rate_mean, dtype=float),
        }
    )

    daily_df = (
        df.groupby("date", as_index=False)
        .agg(
            y_true=("y_true", "sum"),
            rate_mean=("rate_mean", "sum"),
        )
        .sort_values("date")
    )

    observed = daily_df["y_true"].values.astype(float)
    predicted = daily_df["rate_mean"].values.astype(float)

    observed_delta = np.diff(observed)
    predicted_delta = np.diff(predicted)

    daily_delta_corr = _safe_corr(observed_delta, predicted_delta)

    observed_sign = np.where(
        observed_delta > eps_direction,
        1,
        np.where(observed_delta < -eps_direction, -1, 0),
    )

    predicted_sign = np.where(
        predicted_delta > eps_direction,
        1,
        np.where(predicted_delta < -eps_direction, -1, 0),
    )

    # We ignore days where the observed daily total did not move.
    moving_mask = observed_sign != 0

    if moving_mask.sum() > 0:
        daily_direction_accuracy_moving = float(
            np.mean(observed_sign[moving_mask] == predicted_sign[moving_mask])
        )
    else:
        daily_direction_accuracy_moving = np.nan

    return {
        "daily_delta_corr": daily_delta_corr,
        "daily_direction_accuracy_moving": daily_direction_accuracy_moving,
    }


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
    wasserstein : 1-Wasserstein distance between cell-day value distributions
    accuracy, precision, recall : binary activity (> 0), per observation/cell;
        undefined precision/recall are 0 (see evaluate_activity_metrics)

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

    trend_metrics = evaluate_daily_trend_metrics(
        y_true=y_true,
        rate_mean=rate_mean,
        dates=dates,
    )

    metrics = {
        # Observation-level metrics
        "mean_ll_obs": mean_ll_obs,
        "mae_obs": mae_obs,
        "rmse_obs": rmse_obs,
        "wasserstein": evaluate_wasserstein(y_true, rate_mean),
        # Daily-level metrics
        "mean_ll_daily": mean_ll_daily,
        "mae_daily": mae_daily,
        "rmse_daily": rmse_daily,
    }

    metrics.update(trend_metrics)
    metrics.update(evaluate_activity_metrics(y_true, rate_mean))

    return metrics
