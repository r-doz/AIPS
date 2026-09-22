"""
Zero-inflation gate for count models (e.g. the LGCP).

A small binary classifier (nonzero vs. zero) trained on the same
standardized coordinate + covariate features already used by the base
count model. At evaluation time, wherever the classifier predicts "zero",
the base model's predicted rate is forced to 0 (hard mode), or multiplied
by P(nonzero) (soft mode). This is post-processing of the base rate, not
a jointly trained hurdle model.

This corrects a known failure mode of a smooth process model like the
LGCP: it can still predict a small positive rate in a cell/date that is
always zero (e.g. a grid cell outside the fishing grounds), since the
spatial/temporal kernels smooth across neighboring nonzero observations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import warnings
from sklearn.neural_network import MLPClassifier


def build_zero_gate_features(coords: np.ndarray, covs: np.ndarray) -> np.ndarray:
    """
    Concatenate the standardized coordinate + covariate arrays already
    produced by prepare_data() -- the same feature space the count model
    sees -- into the classifier's input.
    """
    return np.hstack([np.asarray(coords), np.asarray(covs)])


def train_zero_gate(
    coords: np.ndarray,
    covs: np.ndarray,
    y: np.ndarray,
    cfg: dict | None = None,
    seed: int = 0,
) -> MLPClassifier:
    """
    Train a binary (y > 0) classifier on training data.

    Parameters
    ----------
    coords, covs : standardized arrays aligned with y (same rows/order as
        the count model's training tensors).
    y : raw (unstandardized) training counts.
    cfg : zero_gate config block, e.g.
        {"hidden_layer_sizes": [32], "activation": "relu", "alpha": 1e-4,
         "max_iter": 500, "learning_rate_init": 0.001}
    """
    cfg = cfg or {}
    X = build_zero_gate_features(coords, covs)
    is_nonzero = (np.asarray(y) > 0).astype(int)

    if len(np.unique(is_nonzero)) < 2:
        raise ValueError(
            "zero_gate: training target is all-zero or all-nonzero for this "
            "window/split, so a classifier cannot be trained. Disable "
            "zero_gate.enabled for this run."
        )

    clf = MLPClassifier(
        hidden_layer_sizes=tuple(cfg.get("hidden_layer_sizes", [32])),
        activation=cfg.get("activation", "relu"),
        alpha=float(cfg.get("alpha", 1e-4)),
        learning_rate_init=float(cfg.get("learning_rate_init", 0.001)),
        max_iter=int(cfg.get("max_iter", 500)),
        random_state=seed,
    )
    clf.fit(X, is_nonzero)
    return clf


def cell_grid_coordinates(spatial_coords):
    """Convert regular longitude/latitude cell centres to grid-step units.

    Use the full spatial grid, not a day's accepted cells, so missing cells
    do not collapse distances. Each axis may have its own spacing.
    """
    xy = np.asarray(spatial_coords, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all():
        raise ValueError("Spatial coordinates must be a finite (N, 2) array.")
    result = np.zeros_like(xy)
    for axis in range(2):
        levels = np.unique(xy[:, axis])
        if len(levels) > 1:
            step = np.diff(levels).min()
            positions = (xy[:, axis] - levels[0]) / step
            if not np.allclose(positions, np.round(positions), atol=1e-3, rtol=0):
                raise ValueError("Local redistribution requires a regular spatial grid.")
            result[:, axis] = np.round(positions)
    return result


def apply_zero_gate(
    clf: MLPClassifier,
    coords: np.ndarray,
    covs: np.ndarray,
    rate_mean: np.ndarray,
    threshold: float = 0.5,
    mode: str = "hard",
    dates=None,
    grid_coords=None,
    local_radius: float = 1,
    redistribution_gamma: float = 1.0,
    redistribution_scale: float = 1.0,
) -> np.ndarray:
    """
    Return a new array of gated rates. Hard mode zeros entries where
    P(nonzero) < threshold. Soft mode multiplies by P(nonzero) and ignores
    threshold. hard_redistribute rescales accepted rates within each calendar
    day to preserve the original total. If no positive rate survives on a day,
    retain its original predictions and warn. Requires aligned dates and scalar
    rates (a vector or single-column array).

    local_redistribuite (alias local_redistribute) moves each rejected rate to
    accepted cells on the same day within local_radius Chebyshev grid steps.
    If none exist, use all nearest accepted cells. Split in proportion to their
    original rates, or equally when all are zero. With no accepted cells, keep
    the original daily predictions and warn, as in global redistribution.
    grid_coords must contain spatial grid indices, never standardized features.

    confidence_redistribute zeros rejected cells but recovers only a fraction
    ``redistribution_scale * P(nonzero) / threshold`` of each rejected rate.
    Recovered mass is assigned, within each day, to accepted cells in
    proportion to ``rate * P(nonzero) ** redistribution_gamma``. Thus a
    confident rejection contributes little mass, while confident recipients
    receive more of it. Unlike hard_redistribute, this mode does not force the
    original daily total to be preserved.
    """
    valid_modes = {
        "hard", "soft", "hard_redistribute", "local_redistribuite",
        "local_redistribute", "confidence_redistribute",
    }
    if mode not in valid_modes:
        raise ValueError(
            "zero_gate.mode must be 'hard', 'soft', 'hard_redistribute', "
            "'local_redistribuite', or 'confidence_redistribute'."
        )
    X = build_zero_gate_features(coords, covs)
    nonzero_class_idx = list(clf.classes_).index(1)
    proba_nonzero = clf.predict_proba(X)[:, nonzero_class_idx]
    is_nonzero_pred = proba_nonzero >= threshold

    gated = np.asarray(rate_mean, dtype=float).copy()
    if gated.ndim == 0 or gated.shape[0] != len(proba_nonzero):
        raise ValueError("rate_mean must have one entry per classifier feature row.")
    if mode == "soft":
        weights = proba_nonzero.reshape((-1,) + (1,) * (gated.ndim - 1))
        return gated * weights
    gated[~is_nonzero_pred] = 0.0
    redistribute_modes = {
        "hard_redistribute", "local_redistribuite", "local_redistribute",
        "confidence_redistribute",
    }
    if mode in redistribute_modes:
        original = np.asarray(rate_mean, dtype=float)
        if original.ndim > 2 or (original.ndim == 2 and original.shape[1] != 1):
            raise ValueError("Redistribution requires one scalar rate per cell/date.")
        if not np.isfinite(original).all() or (original < 0).any():
            raise ValueError("Redistribution requires finite nonnegative rates.")
        if dates is None or np.asarray(dates).shape != (len(gated),):
            raise ValueError("Redistribution requires dates aligned with rates.")
        days = pd.DatetimeIndex(pd.to_datetime(dates)).normalize()
        if days.isna().any():
            raise ValueError("Redistribution dates must not be missing.")
        if mode == "confidence_redistribute":
            gamma = float(redistribution_gamma)
            scale = float(redistribution_scale)
            if not np.isfinite(threshold) or not 0 < threshold <= 1:
                raise ValueError("confidence redistribution requires 0 < threshold <= 1.")
            if not np.isfinite(gamma) or gamma < 0:
                raise ValueError("zero_gate.redistribution_gamma must be finite and nonnegative.")
            if not np.isfinite(scale) or not 0 <= scale <= 1:
                raise ValueError("zero_gate.redistribution_scale must be between 0 and 1.")
            flat_original = original.reshape(-1)
            flat_gated = gated.reshape(-1)
            for day in days.unique():
                mask = np.asarray(days == day)
                accepted = mask & is_nonzero_pred
                rejected = mask & ~is_nonzero_pred
                if not accepted.any():
                    continue
                recovered = np.sum(
                    flat_original[rejected]
                    * scale
                    * proba_nonzero[rejected]
                    / threshold
                )
                if recovered == 0:
                    continue
                recipient_weights = (
                    flat_original[accepted]
                    * np.power(proba_nonzero[accepted], gamma)
                )
                if recipient_weights.sum() == 0:
                    recipient_weights = np.power(proba_nonzero[accepted], gamma)
                recipient_weights = recipient_weights / recipient_weights.sum()
                flat_gated[accepted] += recovered * recipient_weights
            return gated
        local = mode != "hard_redistribute"
        if local:
            grid = np.asarray(grid_coords, dtype=float)
            if grid.shape != (len(gated), 2) or not np.isfinite(grid).all():
                raise ValueError("Local redistribution requires aligned finite grid_coords (N, 2).")
            if not np.isfinite(local_radius) or local_radius < 0:
                raise ValueError("zero_gate.local_radius must be finite and nonnegative.")
        for day in days.unique():
            mask = days == day
            if local and np.any(mask & is_nonzero_pred):
                accepted = np.flatnonzero(mask & is_nonzero_pred)
                discarded = np.flatnonzero(mask & ~is_nonzero_pred)
                for source in discarded:
                    mass = original.reshape(-1)[source]
                    if mass == 0:
                        continue
                    distances = np.max(np.abs(grid[accepted] - grid[source]), axis=1)
                    nearby = distances <= local_radius
                    if not nearby.any():
                        nearby = np.isclose(distances, distances.min(), rtol=0, atol=1e-10)
                    recipients = accepted[nearby]
                    weights = original.reshape(-1)[recipients].copy()
                    weights = weights / weights.sum() if weights.sum() > 0 else np.full(len(weights), 1 / len(weights))
                    gated.reshape(-1)[recipients] += mass * weights
                continue
            total = original[mask].sum()
            remaining = gated[mask].sum()
            if remaining > 0:
                gated[mask] = (gated[mask] / remaining) * total
            elif total > 0:
                gated[mask] = original[mask]
                warnings.warn(
                    f"zero_gate: no positive rate survived on {day.date()}; "
                    "keeping original LGCP rates to preserve the daily total.",
                    RuntimeWarning,
                    stacklevel=2,
                )
    return gated
