"""
scripts/train_lgcp.py
Training script for the Sparse LGCP vessel-count model.

Usage
-----
  python scripts/train_lgcp.py [--config config.yaml]

All hyperparameters live in config.yaml (or the CONFIG dict below as
defaults).  Results (model checkpoint + metrics) are saved under
reports/<model_family>/<run_name>/.
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


# ---- Make src importable when script is run from the project root ----------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.multi_year import years_label
from src.models.data_pp_lgcp import prepare_data, compute_meta, make_day_split_masks
from src.models.metrics_lgcp import evaluate_metrics, evaluate_first_three_days
from src.models.lgcp import SparseLGCP
from src.models.zero_gate import train_zero_gate, apply_zero_gate, cell_grid_coordinates
from src.models.daily_allocation import load_daily_totals, allocate_daily_totals, validate_conflict_policy
from src.visualization.viz_lgcp import (
    plot_loss,
    plot_pp_overview,
    plot_test_window_daily_timeseries,
    plot_daily_observed_vs_predicted_maps,
    plot_daily_interpolated_spatial_maps_separate,
)

import pandas as pd

# ---------------------------------------------------------------------------
# Default configuration (overridden by config.yaml if present)
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = {
    # --- data ---
    "years": 2024,
    "parquet_path": "data/processed/cpr_gfw.parquet",
    "gulf_csv_path": "data/raw/ts_gulf_coords.csv",
    "train_fraction": 0.9,
    "data_seed": 42,
    "split_strategy": "random_day",
    "covariate_cols": ["chl", "thetao"],
    # --- model ---
    "M_inducing": 300,
    "np_seed": 0,
    "model": {
        "spatial_kernels": [
            {
                "type": "RBF",
                "hyperparameters": {
                    "lengthscale": 0.5,
                    "variance": 1.0,
                },
                "trainable": {
                    "lengthscale": True,
                    "variance": True,
                },
            }
        ],
        "temporal_kernels": [
            {
                "type": "periodic",
                "hyperparameters": {
                    "lengthscale": 0.3,
                    "variance": 0.5,
                    "period": 1.0,
                },
                "trainable": {
                    "lengthscale": True,
                    "variance": True,
                    "period": True,
                },
            }
        ],
    },
    # --- training ---
    "lr": 1e-4,
    "num_steps": 1000,  # to be increased
    "minibatch": 1024,
    "num_mc": 7,
    "grad_clip": 10.0,
    "device": "cpu",  # "cuda" if GPU is available
    # --- evaluation ---
    "num_pred_samples": 200,
    "num_vis_samples": 800,
    # --- output ---
    # --- output ---
    "report_root": "reports",
    "model_family": "basic_lgcp",
    "run_name": "lgcp_run_debug",
    "log_every": 50,
    "lag_features": {
        "enabled": False,
    },
    # Zero-inflation gate: a small MLP classifier (y > 0) trained on the
    # same standardized features as the LGCP. When enabled, test predictions
    # the classifier calls "zero" are forced to 0, correcting the LGCP's
    # tendency to smooth a small positive rate into always-zero cells/dates.
    "zero_gate": {
        "enabled": False,
        "mode": "hard",  # soft multiplies rates by P(nonzero); ignores threshold
        "hidden_layer_sizes": [32],
        "activation": "relu",
        "alpha": 1e-4,
        "max_iter": 500,
        "threshold": 0.5,
    },
}

KERNEL_TYPE_MAP = {
    "rbf": "rbf",
    "periodic": "periodic",
    "rationalquadratic": "rational_quadratic",
    "rational_quadratic": "rational_quadratic",
    "rq": "rational_quadratic",
    "quasiperiodic": "quasi_periodic",
    "quasi_periodic": "quasi_periodic",
    "quasi-periodic": "quasi_periodic",
}

REQUIRED_HYPERPARAMETERS = {
    "rbf": {"lengthscale", "variance"},
    "periodic": {"lengthscale", "variance", "period"},
    "rational_quadratic": {"lengthscale", "variance", "alpha"},
    "quasi_periodic": {
        "envelope_lengthscale",
        "periodic_lengthscale",
        "period",
        "variance",
    },
}


def load_config(config_path=None):
    """
    `years` may be a single year (2024) or a list of years ([2024, 2025]).
    The actual per-year parquet file (e.g. cpr_gfw_2024.parquet) is resolved
    lazily by prepare_data(), so `parquet_path` here stays the base path.
    """
    cfg = copy.deepcopy(DEFAULT_CONFIG)

    if config_path is not None:
        with open(config_path, "r") as f:
            user_cfg = yaml.safe_load(f)

        if user_cfg is not None:
            if not isinstance(user_cfg, dict):
                raise ValueError("Top-level YAML config must be a mapping.")
            cfg = _deep_merge(cfg, user_cfg)

    cfg["kernel_config"] = normalize_kernel_config(cfg)

    return cfg


def _deep_merge(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def _normalize_kernel_type(raw_name: str, location: str) -> str:
    if raw_name is None:
        raise ValueError(
            f"Missing kernel type at {location}. Supported kernels: RBF, periodic, RationalQuadratic."
        )

    key = str(raw_name).strip().replace("-", "_").replace(" ", "_").lower()
    key = key.replace("__", "_")
    if key not in KERNEL_TYPE_MAP:
        raise ValueError(
            f"Unsupported kernel type '{raw_name}' at {location}. "
            "Supported kernels: RBF, periodic, RationalQuadratic."
        )
    return KERNEL_TYPE_MAP[key]


def _normalize_kernel_block(
    block: dict,
    location: str,
    default_type: str | None = None,
) -> dict:
    if not isinstance(block, dict):
        raise ValueError(f"{location} must be a mapping.")

    kernel_type = block.get("type", default_type)
    normalized_type = _normalize_kernel_type(kernel_type, f"{location}.type")

    hyperparameters = block.get("hyperparameters")
    if not isinstance(hyperparameters, dict):
        raise ValueError(f"{location}.hyperparameters must be a mapping.")

    required = REQUIRED_HYPERPARAMETERS[normalized_type]
    trainable_keys = required
    if "period_days" in hyperparameters:
        if not location.startswith("model.temporal_kernels") or "period" not in required:
            raise ValueError(f"{location}: period_days is only supported for temporal periodic kernels.")
        if "period" in hyperparameters:
            raise ValueError(f"{location}: specify period or period_days, not both.")
        required = (required - {"period"}) | {"period_days"}
    unknown_hparams = set(hyperparameters) - required
    if unknown_hparams:
        raise ValueError(
            f"Unknown hyperparameters in {location}.hyperparameters: {sorted(unknown_hparams)}. "
            f"Required hyperparameters for {normalized_type}: {sorted(required)}"
        )

    missing_hparams = required - set(hyperparameters)
    if missing_hparams:
        raise ValueError(
            f"Missing hyperparameters in {location}.hyperparameters: {sorted(missing_hparams)}. "
            f"Required hyperparameters for {normalized_type}: {sorted(required)}"
        )

    normalized_hparams = {}
    for hp_name in sorted(required):
        value = hyperparameters[hp_name]
        if not isinstance(value, (int, float)):
            raise ValueError(
                f"{location}.hyperparameters.{hp_name} must be numeric, got {type(value).__name__}."
            )
        value = float(value)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(
                f"{location}.hyperparameters.{hp_name} must be > 0, got {value}."
            )
        normalized_hparams[hp_name] = value

    trainable = block.get("trainable", {})
    if not isinstance(trainable, dict):
        raise ValueError(f"{location}.trainable must be a mapping.")

    unknown_trainable = set(trainable) - trainable_keys
    if unknown_trainable:
        raise ValueError(
            f"Unknown trainable flags in {location}.trainable: {sorted(unknown_trainable)}. "
            f"Allowed keys: {sorted(trainable_keys)}"
        )

    normalized_trainable = {}
    for hp_name in sorted(trainable_keys):
        flag = trainable.get(hp_name, True)
        if not isinstance(flag, bool):
            raise ValueError(
                f"{location}.trainable.{hp_name} must be boolean, got {type(flag).__name__}."
            )
        normalized_trainable[hp_name] = flag

    return {
        "type": normalized_type,
        "hyperparameters": normalized_hparams,
        "trainable": normalized_trainable,
    }


def normalize_kernel_config(cfg: dict) -> dict:
    model_cfg = cfg.get("model")
    if not isinstance(model_cfg, dict):
        raise ValueError(
            "Missing required model config block. Expected spatial and temporal kernel configuration."
        )

    def _normalize_kernel_collection(
        singular_key: str,
        plural_key: str,
        location: str,
        default_type: str | None = None,
    ) -> list[dict]:
        if plural_key in model_cfg:
            raw_blocks = model_cfg[plural_key]
            if not isinstance(raw_blocks, list) or len(raw_blocks) == 0:
                raise ValueError(f"model.{plural_key} must be a non-empty list.")
        elif singular_key in model_cfg:
            raw_blocks = [model_cfg[singular_key]]
        else:
            raise ValueError(
                f"Missing required config block: model.{plural_key} (or model.{singular_key})."
            )

        normalized_blocks = []
        for idx, block in enumerate(raw_blocks):
            block_location = f"{location}[{idx}]"
            normalized_blocks.append(
                _normalize_kernel_block(
                    block, block_location, default_type=default_type
                )
            )
        return normalized_blocks

    spatial = _normalize_kernel_collection(
        singular_key="spatial_kernel",
        plural_key="spatial_kernels",
        location="model.spatial_kernels",
    )
    temporal = _normalize_kernel_collection(
        singular_key="temporal_kernel",
        plural_key="temporal_kernels",
        location="model.temporal_kernels",
        default_type="periodic",
    )

    spatial_composition = (
        str(model_cfg.get("spatial_composition", "sum")).strip().lower()
    )
    temporal_composition = (
        str(model_cfg.get("temporal_composition", "sum")).strip().lower()
    )

    allowed_compositions = {"sum", "product"}

    if spatial_composition not in allowed_compositions:
        raise ValueError(
            f"model.spatial_composition must be one of {allowed_compositions}, "
            f"got '{spatial_composition}'."
        )

    if temporal_composition not in allowed_compositions:
        raise ValueError(
            f"model.temporal_composition must be one of {allowed_compositions}, "
            f"got '{temporal_composition}'."
        )

    normalized = {
        "spatial": spatial,
        "temporal": temporal,
        "spatial_composition": spatial_composition,
        "temporal_composition": temporal_composition,
    }

    cfg["model"]["spatial_kernels"] = spatial
    cfg["model"]["temporal_kernels"] = temporal
    cfg["model"]["spatial_composition"] = spatial_composition
    cfg["model"]["temporal_composition"] = temporal_composition

    return normalized


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seeds(seed: int = 0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train(model: SparseLGCP, cfg: dict) -> list[float]:
    optim = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    losses = []

    n_batch = min(cfg["minibatch"], model.N)
    t0 = time.time()

    for step in range(1, cfg["num_steps"] + 1):
        optim.zero_grad()
        elbo, info = model.elbo_mc(num_mc=cfg["num_mc"], minibatch_size=n_batch)
        loss = -elbo
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        optim.step()

        losses.append(loss.item())

        if step % cfg["log_every"] == 0 or step == 1:
            elapsed = time.time() - t0
            print(
                f"[{step:>6}/{cfg['num_steps']}]  loss={loss.item():.4f}"
                f"  ll={info['elbo_ll']:.4f}  kl={info['kl']:.4f}"
                f"  ({elapsed:.1f}s)"
            )

    return losses


def get_kernel_parameters_dict(model):
    params = model.kernel_params()
    out = {}

    for group_name, group_value in params.items():
        if isinstance(group_value, list):
            out[group_name] = []

            for component in group_value:
                component_out = {
                    "type": component["type"],
                    "params": {},
                }

                for par_name, par_value in component["params"].items():
                    component_out["params"][par_name] = float(par_value.detach().cpu())

                out[group_name].append(component_out)

        else:
            try:
                out[group_name] = float(group_value.detach().cpu())
            except Exception:
                out[group_name] = group_value

    return out


def print_kernel_parameters(model, title="Kernel parameters"):
    print(f"\n=== {title} ===")

    params = get_kernel_parameters_dict(model)

    for group_name, group_value in params.items():
        print(f"\n{group_name}:")

        if isinstance(group_value, list):
            for i, component in enumerate(group_value):
                print(f"  component {i}: {component['type']}")

                for par_name, par_value in component["params"].items():
                    print(f"    {par_name}: {par_value:.6f}")
        else:
            if isinstance(group_value, float):
                print(f"  {group_value:.8f}")
            else:
                print(f"  {group_value}")


def save_kernel_parameters(model, save_path):
    params = get_kernel_parameters_dict(model)

    with open(save_path, "w") as f:
        yaml.dump(params, f, sort_keys=False)

    print(f"Kernel parameters saved → {save_path}")


def resolve_period_days(kernel_config, scalers, span_days):
    """Return model-ready kernels; preserve the requested config for provenance.

    prepare_data uses elapsed days / span_days, then a training-only scaler.
    Thus one standardized time unit spans span_days * scaler.scale_ days.
    """
    resolved = copy.deepcopy(kernel_config)
    days_per_unit = float(span_days) * float(scalers["t_scaler"].scale_[0])
    for block in resolved["temporal"]:
        hp = block["hyperparameters"]
        if "period_days" in hp:
            if not math.isfinite(days_per_unit) or days_per_unit <= 0:
                raise ValueError("Cannot convert period_days without a positive time span and scale.")
            hp["period"] = hp.pop("period_days") / days_per_unit
    return resolved


def compute_time_diagnostics(scalers, span_days, lengthscales=None, periods=None):
    if lengthscales is None:
        lengthscales = [0.01, 0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30]

    if periods is None:
        periods = [1.0, 3.85]

    if "t_scaler" not in scalers:
        raise KeyError(
            f"'t_scaler' not found in scalers. Available keys: {list(scalers.keys())}"
        )

    t_scaler = scalers["t_scaler"]

    t_mean = float(np.ravel(t_scaler.mean_)[0])
    t_scale = float(np.ravel(t_scaler.scale_)[0])

    if span_days <= 0:
        raise ValueError("Time diagnostics require a positive data span.")
    annual_period_raw = 365.25 / span_days
    annual_period_std = annual_period_raw / t_scale

    diagnostics = {
        "t_raw_mean": t_mean,
        "t_raw_std": t_scale,
        "span_days": int(span_days),
        "days_per_standardized_unit": float(t_scale * span_days),
        "annual_period_raw": annual_period_raw,
        "annual_period_standardized": annual_period_std,
        "lengthscale_to_days": {},
        "period_to_days": {},
    }

    print("\n=== Time scaler diagnostics ===")
    print(f"t_raw mean: {t_mean:.6f}")
    print(f"t_raw std:  {t_scale:.6f}")
    print(f"Annual period in standardized time: {annual_period_std:.6f}")

    print("\nLengthscale interpretation:")
    for l in lengthscales:
        days = l * t_scale * span_days
        diagnostics["lengthscale_to_days"][float(l)] = float(days)
        print(f"  lengthscale {l:.3f} ≈ {days:.2f} days")

    print("\nPeriod interpretation:")
    for p in periods:
        days = p * t_scale * span_days
        diagnostics["period_to_days"][float(p)] = float(days)
        print(f"  period {p:.3f} ≈ {days:.2f} days")

    print(
        f"\nCorrect annual period to use in standardized time: {annual_period_std:.6f}"
    )

    return diagnostics


def configure_log_noise(model, cfg):
    noise_cfg = cfg.get("kernel_noise", {})

    value = noise_cfg.get("value", None)
    trainable = noise_cfg.get("trainable", True)

    if value is not None:
        value = float(value)
        if value <= 0:
            raise ValueError(f"kernel_noise.value must be positive, got {value}")

        with torch.no_grad():
            model.log_noise.fill_(math.log(value))

    model.log_noise.requires_grad_(bool(trainable))

    noise_var = float(torch.exp(model.log_noise).detach().cpu())

    print("\n=== Kernel noise configuration ===")
    print(f"noise_var: {noise_var:.8f}")
    print(f"trainable: {model.log_noise.requires_grad}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(cfg: dict):
    allocation_cfg = cfg.get("daily_allocation") or {}
    allocation_totals = None
    if allocation_cfg.get("enabled", False):
        validate_conflict_policy(allocation_cfg.get("conflict_policy", "error"))
        allocation_totals = load_daily_totals(
            allocation_cfg["totals_csv"], float(allocation_cfg.get("gamma", 1.0))
        )
    gate_cfg = cfg.get("zero_gate") or {}
    valid_gate_modes = {"hard", "soft", "hard_redistribute", "local_redistribuite", "local_redistribute", "confidence_redistribute"}
    if gate_cfg.get("enabled", False) and gate_cfg.get("mode", "hard") not in valid_gate_modes:
        raise ValueError("zero_gate.mode must be 'hard', 'soft', 'hard_redistribute', 'local_redistribuite', or 'confidence_redistribute'.")
    if gate_cfg.get("enabled", False) and gate_cfg.get("mode") == "confidence_redistribute":
        gamma = float(gate_cfg.get("redistribution_gamma", 1))
        scale = float(gate_cfg.get("redistribution_scale", 1))
        threshold = float(gate_cfg.get("threshold", 0.5))
        if not np.isfinite(gamma) or gamma < 0:
            raise ValueError("zero_gate.redistribution_gamma must be finite and nonnegative.")
        if not np.isfinite(scale) or not 0 <= scale <= 1:
            raise ValueError("zero_gate.redistribution_scale must be between 0 and 1.")
        if not np.isfinite(threshold) or not 0 < threshold <= 1:
            raise ValueError("confidence redistribution requires 0 < zero_gate.threshold <= 1.")
    if gate_cfg.get("enabled", False) and gate_cfg.get("mode") in {"local_redistribuite", "local_redistribute"}:
        radius = float(gate_cfg.get("local_radius", 1))
        if not np.isfinite(radius) or radius < 0:
            raise ValueError("zero_gate.local_radius must be finite and nonnegative.")
    set_seeds(cfg.get("np_seed", 0))
    torch.set_default_dtype(torch.float32)

    out_dir = (
        Path(cfg["report_root"])
        / cfg["model_family"]
        / f"{years_label(cfg.get('years'))}_{cfg['run_name']}_{time.strftime('%Y%m%d-%H%M%S')}"
    )
    plots_dir = out_dir / "plots"

    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Save experiment parameters
    params_path = out_dir / "params.yaml"
    with open(params_path, "w") as f:
        yaml.dump(cfg, f, sort_keys=False)

    print(f"Experiment parameters saved → {params_path}")

    # ---- Data ----------------------------------------------------------------
    print("Loading data …")
    (train_coords, train_covs, train_y, test_coords, test_covs, test_y, scalers, df) = (
        prepare_data(
            cfg["parquet_path"],
            train_fraction=cfg["train_fraction"],
            random_seed=cfg["data_seed"],
            split_strategy=cfg.get("split_strategy", "random_day"),
            covariate_cols=cfg.get("covariate_cols"),
            test_start_date=cfg.get("test_start_date"),
            test_end_date=cfg.get("test_end_date"),
            lag_features=cfg.get("lag_features"),
            years=cfg.get("years"),
        )
    )
    meta = compute_meta(df)
    span_days = (df["date"].max() - df["date"].min()).days
    cfg["kernel_config"] = resolve_period_days(cfg["kernel_config"], scalers, span_days)
    # Record both the requested calendar-day settings and resolved model units.
    with open(params_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    periods = [
        block["hyperparameters"]["period"]
        for block in cfg["kernel_config"]["temporal"]
        if "period" in block["hyperparameters"]
    ]
    time_diagnostics = compute_time_diagnostics(scalers, span_days, periods=periods)

    time_diag_path = out_dir / "time_diagnostics.yaml"
    with open(time_diag_path, "w") as f:
        yaml.dump(time_diagnostics, f, sort_keys=False)

    print(f"Time diagnostics saved → {time_diag_path}")

    _, test_mask = make_day_split_masks(
        df=df,
        train_fraction=cfg["train_fraction"],
        random_seed=cfg["data_seed"],
        split_strategy=cfg.get("split_strategy", "random_day"),
        test_start_date=cfg.get("test_start_date"),
        test_end_date=cfg.get("test_end_date"),
    )

    test_dates = df.loc[test_mask, "date"].values
    local_grid_coords = (
        cell_grid_coordinates(df[["longitude", "latitude"]].values)[test_mask]
        if gate_cfg.get("enabled", False) and gate_cfg.get("mode") in {"local_redistribuite", "local_redistribute"}
        else None
    )

    print(
        f"  Train: {train_coords.shape[0]:,} obs   "
        f"Test: {test_coords.shape[0]:,} obs   "
        f"Coords dim: {train_coords.shape[1]}   "
        f"Covs dim: {train_covs.shape[1]}"
    )

    # Sanity checks
    assert train_coords.shape[0] == train_covs.shape[0] == train_y.shape[0]
    assert test_coords.shape[0] == test_covs.shape[0] == test_y.shape[0]
    if cfg["M_inducing"] > train_coords.shape[0]:
        raise ValueError(
            f"M_inducing ({cfg['M_inducing']}) cannot exceed train observations ({train_coords.shape[0]})."
        )

    # ---- Model ---------------------------------------------------------------
    print(f"Building SparseLGCP with M={cfg['M_inducing']} inducing points …")
    model = SparseLGCP(
        train_coords,
        train_covs,
        train_y,
        kernel_config=cfg["kernel_config"],
        M_inducing=cfg["M_inducing"],
        device=cfg["device"],
        np_seed=cfg["np_seed"],
    )

    configure_log_noise(model, cfg)
    print_kernel_parameters(model, "Kernel parameters BEFORE training")
    save_kernel_parameters(model, out_dir / "kernel_params_before.yaml")

    # ---- Training ------------------------------------------------------------
    print(f"Training for {cfg['num_steps']:,} steps on {cfg['device']} …")
    losses = train(model, cfg)

    print_kernel_parameters(model, "Kernel parameters AFTER training")
    save_kernel_parameters(model, out_dir / "kernel_params_after.yaml")

    # Save checkpoint
    ckpt_path = out_dir / "model.pt"
    torch.save({"model_state": model.state_dict(), "config": cfg}, ckpt_path)
    print(f"Checkpoint saved → {ckpt_path}")

    # Loss curve
    plot_loss(losses, save_path=plots_dir / "loss.png")

    # ---- Evaluation on test set ----------------------------------------------
    print("Evaluating on test set …")
    rate_mean_test, rate_p05_test, rate_p95_test = model.predict_rate(
        test_coords, test_covs, num_samples=cfg["num_pred_samples"]
    )

    ungated_rate_mean_test = rate_mean_test.copy()
    zero_gate_cfg = cfg.get("zero_gate") or {}
    if zero_gate_cfg.get("enabled", False):
        print("Training zero-inflation gate classifier …")
        zero_gate_clf = train_zero_gate(
            train_coords, train_covs, train_y, zero_gate_cfg, seed=cfg.get("np_seed", 0)
        )
        rate_mean_test = apply_zero_gate(
            zero_gate_clf, test_coords, test_covs, rate_mean_test,
            threshold=float(zero_gate_cfg.get("threshold", 0.5)),
            mode=zero_gate_cfg.get("mode", "hard"),
            dates=test_dates,
            grid_coords=local_grid_coords,
            local_radius=float(zero_gate_cfg.get("local_radius", 1)),
            redistribution_gamma=float(zero_gate_cfg.get("redistribution_gamma", 1)),
            redistribution_scale=float(zero_gate_cfg.get("redistribution_scale", 1)),
        )
        n_gated = int((rate_mean_test == 0).sum())
        print(f"  Zero-gate ({zero_gate_cfg.get('mode', 'hard')}): "
              f"{n_gated}/{len(rate_mean_test)} test predictions are zero.")

    if allocation_totals is not None:
        raw_allocation_rates = rate_mean_test.copy()
        rate_mean_test = allocate_daily_totals(
            rate_mean_test, test_dates, allocation_totals,
            gamma=float(allocation_cfg.get("gamma", 1.0)),
            conflict_policy=allocation_cfg.get("conflict_policy", "error"),
            fallback_rates=ungated_rate_mean_test,
        )
        pd.DataFrame({
            "date": test_dates,
            "rate_before_allocation": raw_allocation_rates,
            "allocated_rate": rate_mean_test,
        }).to_csv(out_dir / "allocation_predictions.csv", index=False)
        allocation_totals.rename("predicted_total").rename_axis("date").to_csv(
            out_dir / "allocation_daily_totals.csv"
        )
        print("Applied external daily totals with normalized spatial allocation.")

    # Preserve final cell values for exact, training-free spatial replotting.
    spatial_export = pd.DataFrame({
        "date": test_dates, "x": test_coords[:, 0], "y": test_coords[:, 1],
        "observed": test_y, "predicted": rate_mean_test,
    })
    for key in ("lon_mean", "lon_std", "lat_mean", "lat_std"):
        spatial_export[key] = meta[key]
    spatial_export.to_csv(out_dir / "spatial_predictions.csv", index=False)

    df_test_plot = df.loc[test_mask].copy().reset_index(drop=True)

    assert len(df_test_plot) == len(test_y) == len(rate_mean_test)

    plot_daily_interpolated_spatial_maps_separate(
        test_coords=test_coords,
        test_dates=test_dates,
        y_true=test_y,
        rate_mean=rate_mean_test,
        meta=meta,
        gulf_csv_path=cfg["gulf_csv_path"],
        out_dir=plots_dir / "spatial_interpolated_test_days",
        cmap="viridis",
        max_days=None,
        **(cfg.get("spatial_plot") or {}),
    )

    metrics = evaluate_metrics(test_y, rate_mean_test, test_dates)
    print("\n=== Test metrics ===")
    print("Observation-level metrics:")
    print(f"  Mean log-likelihood per obs: {metrics['mean_ll_obs']:.4f}")
    print(f"  MAE per obs:                 {metrics['mae_obs']:.4f}")
    print(f"  RMSE per obs:                {metrics['rmse_obs']:.4f}")
    print(f"  Wasserstein:                 {metrics['wasserstein']:.4f}")
    print(f"  Accuracy (activity): {metrics['accuracy']:.4f}")
    print(f"  Precision (activity): {metrics['precision']:.4f}")
    print(f"  Recall (activity): {metrics['recall']:.4f}")

    print("Daily-level metrics:")
    print(f"  Mean log-likelihood daily:   {metrics['mean_ll_daily']:.4f}")
    print(f"  MAE daily:                   {metrics['mae_daily']:.4f}")
    print(f"  RMSE daily:                  {metrics['rmse_daily']:.4f}")

    print("Daily trend metrics:")
    print(f"  Daily delta correlation:     {metrics['daily_delta_corr']:.4f}")
    print(
        f"  Direction accuracy moving:   "
        f"{metrics['daily_direction_accuracy_moving']:.4f}"
    )

    # Save metrics
    metrics_path = out_dir / "metrics.yaml"
    with open(metrics_path, "w") as f:
        yaml.dump(metrics, f)
    print(f"Metrics saved → {metrics_path}")

    # Reuse final predictions (including any gate/allocation) for the first
    # three test days. These are pooled window metrics, as in metrics.yaml.
    three_day_report = evaluate_first_three_days(test_y, rate_mean_test, test_dates)
    if three_day_report is not None:
        three_day_path = out_dir / "metrics_first_3_days.yaml"
        with open(three_day_path, "w") as f:
            yaml.safe_dump(three_day_report, f, sort_keys=False)
        print("\n=== First three test days ===")
        print(yaml.safe_dump(three_day_report, sort_keys=False))
        print(f"First-three-day metrics saved → {three_day_path}")
    else:
        print("Skipping first-three-day metrics: three or fewer test days.")

    # window plot
    test_window_plot_path = plots_dir / "daily_timeseries_test_window.png"

    plot_test_window_daily_timeseries(
        test_dates=test_dates,
        test_y=test_y,
        rate_mean_test=rate_mean_test,
        save_path=test_window_plot_path,
        title="LGCP daily totals — test window",
    )

    print(f"Test-window daily time-series plot saved → {test_window_plot_path}")

    # ---- Visualisation -------------------------------------------------------
    print("Generating overview plots …")

    # Merge + sort all data for the time-series plot
    all_coords = np.vstack([train_coords, test_coords])
    all_covs = np.vstack([train_covs, test_covs])
    all_y = np.concatenate([train_y, test_y])
    sort_idx = np.argsort(all_coords[:, 2])
    all_coords = all_coords[sort_idx]
    all_covs = all_covs[sort_idx]
    all_y = all_y[sort_idx]

    # Keep the full-period raw-LGCP time series; daily maps above show final predictions.
    plot_pp_overview(
        model=model,
        df=df,
        coords=all_coords,
        covariates=all_covs,
        y_true=all_y,
        meta=meta,
        scalers=scalers,
        gulf_csv_path=cfg["gulf_csv_path"],
        num_samples=cfg["num_vis_samples"],
        test_coords=test_coords,
        save_dir=plots_dir,
        include_spatial_map=False,
    )

    print("Done.")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config/train_basic_lgcp.yaml",
        help="Path to YAML config file.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    main(cfg)
