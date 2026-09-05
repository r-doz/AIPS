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
         "max_iter": 500}
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
        max_iter=int(cfg.get("max_iter", 500)),
        random_state=seed,
    )
    clf.fit(X, is_nonzero)
    return clf


def apply_zero_gate(
    clf: MLPClassifier,
    coords: np.ndarray,
    covs: np.ndarray,
    rate_mean: np.ndarray,
    threshold: float = 0.5,
    mode: str = "hard",
) -> np.ndarray:
    """
    Return a new array of gated rates. Hard mode zeros entries where
    P(nonzero) < threshold. Soft mode multiplies by P(nonzero) and ignores
    threshold. The first rate dimension must match the feature rows.
    """
    if mode not in {"hard", "soft"}:
        raise ValueError("zero_gate.mode must be 'hard' or 'soft'.")
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
    return gated
