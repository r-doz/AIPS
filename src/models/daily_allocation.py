"""Allocate external predicted daily totals using LGCP spatial weights."""

import warnings

import numpy as np
import pandas as pd


def load_daily_totals(path, gamma=1.0):
    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError("daily_allocation.gamma must be finite and positive")
    frame = pd.read_csv(path)
    dates = pd.to_datetime(frame['date'], errors='raise').dt.normalize()
    totals = pd.to_numeric(frame['predicted_total'], errors='raise').to_numpy()
    if dates.isna().any() or dates.duplicated().any():
        raise ValueError("Daily totals require unique, non-missing dates")
    if not np.isfinite(totals).all() or (totals < 0).any():
        raise ValueError("Predicted totals must be finite and nonnegative")
    return pd.Series(totals, index=pd.DatetimeIndex(dates))


def validate_conflict_policy(policy):
    if policy not in ("error", "classifier", "daily_total"):
        raise ValueError(
            "daily_allocation.conflict_policy must be 'error', 'classifier', or 'daily_total'"
        )


def allocate_daily_totals(
    rates, dates, totals, gamma=1.0, *, conflict_policy="error", fallback_rates=None
):
    """Allocate totals, resolving positive totals with all-zero weights by policy.

    classifier: retain zeros; daily_total: use ungated fallback_rates (uniform
    if those are also all zero); error: raise. Other days keep gated weights.
    """
    validate_conflict_policy(conflict_policy)
    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError("daily_allocation.gamma must be finite and positive")
    rates = np.asarray(rates, dtype=float)
    dates = pd.DatetimeIndex(pd.to_datetime(dates)).normalize()
    if rates.ndim != 1 or len(rates) != len(dates):
        raise ValueError("Rates and dates must be aligned one-dimensional arrays")
    if dates.isna().any() or not np.isfinite(rates).all() or (rates < 0).any():
        raise ValueError("Allocation requires valid dates and finite nonnegative rates")
    if conflict_policy == "daily_total":
        fallback_rates = np.asarray(fallback_rates, dtype=float)
        if (fallback_rates.shape != rates.shape
                or not np.isfinite(fallback_rates).all()
                or (fallback_rates < 0).any()):
            raise ValueError("daily_total policy requires aligned finite nonnegative fallback_rates")
    missing = dates.unique().difference(totals.index)
    if len(missing):
        raise ValueError(f"Missing predicted daily totals for {missing.tolist()}")
    result = np.zeros_like(rates)
    for day in dates.unique():
        mask = dates == day
        total = float(totals.loc[day])
        if not np.isfinite(total) or total < 0:
            raise ValueError("Predicted totals must be finite and nonnegative")
        if total == 0:
            continue
        daily = rates[mask]
        if daily.max() == 0:
            if conflict_policy == "error":
                raise ValueError(
                    f"No positive allocation weights for {day}; "
                    "set daily_allocation.conflict_policy to 'classifier' or 'daily_total'"
                )
            warnings.warn(
                f"No positive allocation weights for {day} with predicted total {total}; "
                f"applying conflict_policy={conflict_policy}",
                RuntimeWarning, stacklevel=2,
            )
            if conflict_policy == "classifier":
                continue
            daily = fallback_rates[mask]
            if daily.max() == 0:
                daily = np.ones_like(daily)
        # Scaling before exponentiation avoids overflow at large rates/gamma.
        weights = (daily / daily.max()) ** gamma
        result[mask] = total * (weights / weights.sum())
    return result
