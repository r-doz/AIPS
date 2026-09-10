"""Optional exponential recency weights, anchored to the last training date."""
import numpy as np
import pandas as pd


def recency_weights(dates, half_life_days=None):
    if half_life_days is None:
        return None
    if not np.isfinite(half_life_days) or half_life_days <= 0:
        raise ValueError('half_life_days must be positive or null')
    dates = pd.DatetimeIndex(dates)
    if dates.empty or dates.hasnans:
        raise ValueError('Training dates must be nonempty and valid')
    ages = (dates.max() - dates).days.to_numpy(dtype=float)
    weights = np.exp2(-ages / half_life_days)
    # Preserve average weight 1 so regularization remains comparable.
    return weights / weights.mean()
