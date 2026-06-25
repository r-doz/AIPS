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
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml


# ---- Make src importable when script is run from the project root ----------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.data_pp_lgcp import prepare_data, compute_meta
from src.models.metrics_lgcp import evaluate_metrics
from src.models.lgcp import SparseLGCP
from src.visualization.viz_lgcp import plot_loss, plot_pp_overview

import pandas as pd

# ---------------------------------------------------------------------------
# Default configuration (overridden by config.yaml if present)
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = {
    # --- data ---
    "parquet_path": "data/processed/cpr_gfw_2024.parquet",
    "gulf_csv_path": "data/raw/ts_gulf_coords.csv",
    "train_fraction": 0.9,
    "data_seed": 42,
    "covariate_cols": ["chl", "thetao"],
    # --- model ---
    "M_inducing": 300,
    "np_seed": 0,
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
}


def resolve_parquet_path(cfg: dict) -> dict:
    """
    Resolve year-specific parquet path.

    Example
    -------
    years: 2024
    parquet_path: data/processed/cpr_gfw.parquet

    becomes:

    parquet_path: data/processed/cpr_gfw_2024.parquet
    """
    if "years" not in cfg or cfg["years"] is None:
        return cfg

    year = cfg["years"]

    if isinstance(year, list):
        if len(year) != 1:
            raise NotImplementedError(
                "Multiple years are not supported yet. "
                "For now, use a single year, e.g. years: 2024."
            )
        year = year[0]

    path = Path(cfg["parquet_path"])

    if f"_{year}" not in path.stem:
        resolved_path = path.with_name(f"{path.stem}_{year}{path.suffix}")
    else:
        resolved_path = path

    cfg["parquet_path"] = str(resolved_path)

    return cfg


def load_config(config_path=None):
    cfg = DEFAULT_CONFIG.copy()

    if config_path is not None:
        with open(config_path, "r") as f:
            user_cfg = yaml.safe_load(f)

        if user_cfg is not None:
            cfg.update(user_cfg)

    cfg = resolve_parquet_path(cfg)

    return cfg


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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(cfg: dict):
    set_seeds(cfg.get("np_seed", 0))
    torch.set_default_dtype(torch.float32)

    # out_dir = Path(cfg["report_root"]) / cfg["model_family"] / str(cfg["run_name"]) + "_" + str(cfg["year"]) + "_" + time.strftime("%Y%m%d-%H%M%S")
    out_dir = (
        Path(cfg["report_root"])
        / cfg["model_family"]
        / f"{cfg['years']}_{cfg['run_name']}_{time.strftime('%Y%m%d-%H%M%S')}"
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
    parquet_path = cfg["parquet_path"] + f"_{cfg['years']}.parquet"
    (train_coords, train_covs, train_y, test_coords, test_covs, test_y, scalers, df) = (
        prepare_data(
            cfg["parquet_path"],
            train_fraction=cfg["train_fraction"],
            random_seed=cfg["data_seed"],
            covariate_cols=cfg.get("covariate_cols"),
        )
    )
    meta = compute_meta(df)

    print(
        f"  Train: {train_coords.shape[0]:,} obs   "
        f"Test: {test_coords.shape[0]:,} obs   "
        f"Coords dim: {train_coords.shape[1]}   "
        f"Covs dim: {train_covs.shape[1]}"
    )

    # Sanity checks
    assert train_coords.shape[0] == train_covs.shape[0] == train_y.shape[0]
    assert test_coords.shape[0] == test_covs.shape[0] == test_y.shape[0]

    # ---- Model ---------------------------------------------------------------
    print(f"Building SparseLGCP with M={cfg['M_inducing']} inducing points …")
    model = SparseLGCP(
        train_coords,
        train_covs,
        train_y,
        M_inducing=cfg["M_inducing"],
        device=cfg["device"],
        np_seed=cfg["np_seed"],
    )

    # ---- Training ------------------------------------------------------------
    print(f"Training for {cfg['num_steps']:,} steps on {cfg['device']} …")
    losses = train(model, cfg)

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

    # Recover test dates for metric computation
    t_scaler = scalers["t_scaler"]
    t_norm_test = t_scaler.inverse_transform(test_coords[:, 2].reshape(-1, 1)).ravel()
    t_min, t_max = pd.Timestamp(meta["t_min"]), pd.Timestamp(meta["t_max"])
    test_dates = t_min + pd.to_timedelta(t_norm_test * (t_max - t_min))

    metrics = evaluate_metrics(test_y, rate_mean_test, test_dates)
    print("\n=== Test metrics ===")
    print("Observation-level metrics:")
    print(f"  Mean log-likelihood per obs: {metrics['mean_ll_obs']:.4f}")
    print(f"  MAE per obs:                 {metrics['mae_obs']:.4f}")
    print(f"  RMSE per obs:                {metrics['rmse_obs']:.4f}")

    print("Daily-level metrics:")
    print(f"  Mean log-likelihood daily:   {metrics['mean_ll_daily']:.4f}")
    print(f"  MAE daily:                   {metrics['mae_daily']:.4f}")
    print(f"  RMSE daily:                  {metrics['rmse_daily']:.4f}")

    # Save metrics
    metrics_path = out_dir / "metrics.yaml"
    with open(metrics_path, "w") as f:
        yaml.dump(metrics, f)
    print(f"Metrics saved → {metrics_path}")

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

    # Pick a random test date for the spatial map
    rng = np.random.default_rng(cfg.get("np_seed", 0))
    target_date = str(pd.to_datetime(rng.choice(test_dates)).date())
    print(f"  Spatial map target date: {target_date}")

    # print("\n=== DEBUG spatial dates ===")
    # print("df shape:", df.shape)
    # print("df date min/max:", df["date"].min(), df["date"].max())

    # daily_counts = df.groupby("date").size().sort_values(ascending=False)
    # print("Top 10 dates by number of rows:")
    # print(daily_counts.head(10))

    # print("Target date:", target_date)
    # print(
    #    "Rows in df for target_date:",
    #    (
    #        pd.to_datetime(df["date"]).dt.date == pd.to_datetime(target_date).date()
    #    ).sum(),
    # )

    # Check how many rows in all_coords correspond to target_date via t-scaler logic
    t_scaler = scalers["t_scaler"]
    t_min, t_max = pd.Timestamp(meta["t_min"]), pd.Timestamp(meta["t_max"])
    span = t_max - t_min

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
        target_date=target_date,
        test_coords=test_coords,
        save_dir=plots_dir,
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
