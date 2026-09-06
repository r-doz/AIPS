"""Allocate external predicted daily totals using LGCP spatial weights."""

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


def allocate_daily_totals(rates, dates, totals, gamma=1.0):
    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError("daily_allocation.gamma must be finite and positive")
    rates = np.asarray(rates, dtype=float)
    dates = pd.DatetimeIndex(pd.to_datetime(dates)).normalize()
    if rates.ndim != 1 or len(rates) != len(dates):
        raise ValueError("Rates and dates must be aligned one-dimensional arrays")
    if dates.isna().any() or not np.isfinite(rates).all() or (rates < 0).any():
        raise ValueError("Allocation requires valid dates and finite nonnegative rates")
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
            raise ValueError(f"No positive allocation weights for {day}; disable hard gating")
        # Scaling before exponentiation avoids overflow at large rates/gamma.
        weights = (daily / daily.max()) ** gamma
        result[mask] = total * (weights / weights.sum())
    return result
