"""
Train/evaluate the ConvLSTM baseline.

Adapted from CATCH (Convolutional-LSTM approach for temporal catch
hotspots): Contreras Lopez et al., "Convolutional-LSTM approach for
temporal catch hotspots (CATCH): an AI-driven model for spatiotemporal
forecasting of fisheries catch probability densities",
https://pmc.ncbi.nlm.nih.gov/articles/PMC12203189/

See src/models/convlstm.py's module docstring for the full list of
adaptations from the original paper's setting to this project's.

For each grid cell, the model predicts:

    y(cell, t) ~ Poisson(lambda(cell, t))

by processing a sequence of the `seq_length` days immediately before t
(each day represented as a 2D grid of covariates + observed target,
covering the whole Gulf at once) through stacked ConvLSTM layers, then
concatenating day t's own covariate grid (known in advance, same as every
other baseline here) before a convolutional Poisson-rate output head.

Unlike the window Poisson GLM / GNN baselines, this model does not take
explicit `cell_lag_*` features -- the ConvLSTM's recurrence over the
lagged sequence is itself the autoregressive mechanism, so the fixed
covariate set here is just the 5 non-lag covariates (chl, thetao,
fishing_block, is_holiday, is_weekend).

Results are saved under:

    reports/convlstm/<run_name>/
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
import matplotlib.pyplot as plt

# ---- Make src importable when script is run from the project root ----------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.multi_year import load_parquet_years, years_label
from src.models.data_pp_lgcp import compute_meta, make_day_split_masks
from src.models.metrics_lgcp import evaluate_metrics
from src.models.convlstm import (
    ConvLSTMBaseline,
    build_cell_grid_index,
    poisson_nll_grid,
)
from src.visualization.viz_lgcp import plot_daily_interpolated_spatial_maps_separate


# ---------------------------------------------------------------------------
# Config utilities
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    # --- data ---
    "years": 2024,
    "parquet_path": "data/processed/cpr_gfw.parquet",
    "gulf_csv_path": "data/raw/ts_gulf_coords.csv",
    "target_col": "ais_vessels_count",
    # --- split ---
    "split_strategy": "fixed_test_window",
    "train_fraction": 0.9,
    "data_seed": 42,
    "test_start_date": "2024-12-01",
    "test_end_date": "2024-12-10",
    # --- features ---
    # No cell_lag_* here: the ConvLSTM's lagged-sequence input is itself
    # the autoregressive mechanism (see module docstring).
    "covariate_cols": ["chl", "thetao", "fishing_block", "is_holiday", "is_weekend"],
    "standardize_features": True,
    # --- sequence / model ---
    "seq_length": 7,  # T: number of past days fed into the ConvLSTM stack
    "hidden_channels": 4,
    "num_layers": 2,
    "kernel_size": 3,
    "dropout": 0.2,
    # --- training ---
    "lr": 1e-3,
    "num_epochs": 300,
    "torch_seed": 0,
    "device": "cpu",
    "log_every": 25,
    # --- output ---
    "report_root": "reports",
    "model_family": "convlstm",
    "run_name": "convlstm_dec01_dec10",
    "generate_spatial_plots": True,
}


def resolve_run_name(cfg: dict) -> dict:
    if "run_name" not in cfg:
        return cfg

    run_name_from_config = str(cfg["run_name"])
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    year_str = years_label(cfg.get("years"))

    if year_str:
        base_run_name = (
            run_name_from_config
            if run_name_from_config.startswith(f"{year_str}_")
            else f"{year_str}_{run_name_from_config}"
        )
    else:
        base_run_name = run_name_from_config

    cfg["run_name_base"] = run_name_from_config
    cfg["run_id"] = timestamp
    cfg["run_name"] = f"{base_run_name}_{timestamp}"
    return cfg


def _deep_merge(base: dict, updates: dict) -> dict:
    out = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(config_path=None):
    cfg = {k: (dict(v) if isinstance(v, dict) else v) for k, v in DEFAULT_CONFIG.items()}

    if config_path is not None:
        with open(config_path, "r") as f:
            user_cfg = yaml.safe_load(f)
        if user_cfg is not None:
            cfg = _deep_merge(cfg, user_cfg)

    cfg = resolve_run_name(cfg)
    return cfg


# ---------------------------------------------------------------------------
# Feature engineering / grid construction
# ---------------------------------------------------------------------------


def prepare_dataframe(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, list[str]]:
    covariate_cols = list(cfg["covariate_cols"])
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df[cfg["target_col"]] = df[cfg["target_col"]].fillna(0).astype(float)

    missing = [c for c in covariate_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing covariate columns in dataset: {missing}. Available: {list(df.columns)}")

    for col in covariate_cols:
        df[col] = pd.to_numeric(df[col], errors="raise")

    return df, covariate_cols


def build_grid_tensors(df: pd.DataFrame, covariate_cols: list[str], target_col: str):
    """
    Pivot the (cell, date) table into dense per-date [C, H, W] covariate
    grids and [H, W] target grids, on the smallest regular raster
    enclosing the (possibly irregular) set of grid cells.

    Requires a balanced panel: the same set of cells present on every date.
    """
    cell_coords = (
        df[["longitude", "latitude"]]
        .drop_duplicates()
        .sort_values(["longitude", "latitude"])
        .reset_index(drop=True)
    )
    n_cells = len(cell_coords)

    row_idx, col_idx, grid_shape = build_cell_grid_index(cell_coords[["longitude", "latitude"]].to_numpy())
    n_rows, n_cols = grid_shape

    cell_index = {
        (round(float(row.longitude), 6), round(float(row.latitude), 6)): i
        for i, row in cell_coords.iterrows()
    }
    df = df.copy()
    df["_cell_idx"] = [
        cell_index[(round(float(lon), 6), round(float(lat), 6))]
        for lon, lat in zip(df["longitude"], df["latitude"])
    ]

    counts_per_date = df.groupby("date")["_cell_idx"].nunique()
    if not (counts_per_date == n_cells).all():
        raise ValueError(
            "Unbalanced panel: not every date has all grid cells. "
            "The ConvLSTM pivot assumes a fixed set of cells across time."
        )

    dates = np.sort(df["date"].unique())
    date_to_pos = {d: i for i, d in enumerate(dates)}
    n_dates = len(dates)
    n_covs = len(covariate_cols)

    mask = np.zeros(grid_shape, dtype=np.float32)
    mask[row_idx, col_idx] = 1.0

    cov_grid = np.zeros((n_dates, n_covs, n_rows, n_cols), dtype=np.float32)
    y_grid = np.zeros((n_dates, n_rows, n_cols), dtype=np.float32)

    date_pos = df["date"].map(date_to_pos).to_numpy()
    r = row_idx[df["_cell_idx"].to_numpy()]
    c = col_idx[df["_cell_idx"].to_numpy()]

    cov_values = df[covariate_cols].to_numpy(dtype=np.float32)
    for k in range(n_covs):
        cov_grid[date_pos, k, r, c] = cov_values[:, k]
    y_grid[date_pos, r, c] = df[target_col].to_numpy(dtype=np.float32)

    return cov_grid, y_grid, dates, mask, cell_coords, grid_shape


def build_sequence_examples(cov_grid: np.ndarray, y_grid: np.ndarray, dates: np.ndarray, seq_length: int):
    """
    For every date t with >= seq_length days of history, build one example:
      x_seq    : [T, C+1, H, W]  -- covariates + target of days t-T .. t-1
      x_static : [C, H, W]       -- covariates of day t (known in advance)
      y        : [H, W]          -- target of day t
    """
    n_dates = len(dates)
    if n_dates <= seq_length:
        raise ValueError(f"Not enough history: {n_dates} dates <= seq_length={seq_length}.")

    x_seq_list, x_static_list, y_list, target_dates = [], [], [], []
    for t in range(seq_length, n_dates):
        cov_seq = cov_grid[t - seq_length : t]  # [T, C, H, W]
        y_seq = y_grid[t - seq_length : t][:, None, :, :]  # [T, 1, H, W]
        x_seq_list.append(np.concatenate([cov_seq, y_seq], axis=1))
        x_static_list.append(cov_grid[t])
        y_list.append(y_grid[t])
        target_dates.append(dates[t])

    X_seq = np.stack(x_seq_list).astype(np.float32)
    X_static = np.stack(x_static_list).astype(np.float32)
    Y = np.stack(y_list).astype(np.float32)
    target_dates = np.array(target_dates)
    return X_seq, X_static, Y, target_dates


def fit_channel_scaler(x_train: np.ndarray, channel_axis: int):
    axes = tuple(a for a in range(x_train.ndim) if a != channel_axis)
    mean = x_train.mean(axis=axes, keepdims=True)
    std = x_train.std(axis=axes, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return mean, std


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def plot_daily_predictions(test_dates, y_true, rate_mean, save_path=None):
    df_eval = pd.DataFrame({"date": pd.to_datetime(test_dates), "y": y_true, "lambda_hat": rate_mean})
    daily = df_eval.groupby("date", as_index=False).sum()

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(daily["date"], daily["y"], label="Observed", lw=2)
    ax.plot(daily["date"], daily["lambda_hat"], label="Predicted", lw=2)
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily total vessels")
    ax.set_title("ConvLSTM — daily totals")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_loss_curve(losses, save_path=None):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(losses)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train Poisson NLL")
    ax.set_title("ConvLSTM — training loss")
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(cfg: dict):
    out_dir = Path(cfg["report_root"]) / cfg["model_family"] / cfg["run_name"]
    plots_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "params.yaml", "w") as f:
        yaml.dump(cfg, f, sort_keys=False)

    torch.manual_seed(int(cfg.get("torch_seed", 0)))
    device = torch.device(cfg.get("device", "cpu"))

    # ---- Data ------------------------------------------------------------
    print("Loading data …")
    df = load_parquet_years(cfg["parquet_path"], cfg.get("years"))
    df, covariate_cols = prepare_dataframe(df, cfg)
    print(f"  Covariates: {covariate_cols}")

    cov_grid, y_grid, dates, mask, cell_coords, grid_shape = build_grid_tensors(
        df, covariate_cols, cfg["target_col"]
    )
    n_rows, n_cols = grid_shape
    print(f"  Grid: {grid_shape[0]}x{grid_shape[1]} ({int(mask.sum())}/{mask.size} cells valid), {len(dates)} dates")

    seq_length = int(cfg["seq_length"])
    X_seq, X_static, Y, target_dates = build_sequence_examples(cov_grid, y_grid, dates, seq_length)
    print(f"  Sequence examples: {len(target_dates)} (seq_length={seq_length})")

    # ---- Split (by TARGET date; input sequences may reach back into the
    #      training period, same convention as every other baseline's
    #      observed-past lag features) --------------------------------------
    train_mask, test_mask = make_day_split_masks(
        df=pd.DataFrame({"date": target_dates}),
        train_fraction=cfg.get("train_fraction", 0.9),
        random_seed=cfg.get("data_seed", 42),
        split_strategy=cfg.get("split_strategy", "fixed_test_window"),
        test_start_date=cfg.get("test_start_date"),
        test_end_date=cfg.get("test_end_date"),
    )

    X_seq_train, X_static_train, Y_train = X_seq[train_mask], X_static[train_mask], Y[train_mask]
    X_seq_test, X_static_test, Y_test = X_seq[test_mask], X_static[test_mask], Y[test_mask]
    dates_test = target_dates[test_mask]

    print(f"  Train: {X_seq_train.shape[0]} examples   Test: {X_seq_test.shape[0]} examples")

    # ---- Standardization (fit on train only) ------------------------------
    if cfg.get("standardize_features", True):
        seq_mean, seq_std = fit_channel_scaler(X_seq_train, channel_axis=1)
        static_mean, static_std = fit_channel_scaler(X_static_train, channel_axis=1)
        X_seq_train_m = (X_seq_train - seq_mean) / seq_std
        X_seq_test_m = (X_seq_test - seq_mean) / seq_std
        X_static_train_m = (X_static_train - static_mean) / static_std
        X_static_test_m = (X_static_test - static_mean) / static_std
    else:
        X_seq_train_m, X_seq_test_m = X_seq_train, X_seq_test
        X_static_train_m, X_static_test_m = X_static_train, X_static_test

    X_seq_train_t = torch.tensor(X_seq_train_m, dtype=torch.float32, device=device)
    X_static_train_t = torch.tensor(X_static_train_m, dtype=torch.float32, device=device)
    Y_train_t = torch.tensor(Y_train, dtype=torch.float32, device=device)
    X_seq_test_t = torch.tensor(X_seq_test_m, dtype=torch.float32, device=device)
    X_static_test_t = torch.tensor(X_static_test_m, dtype=torch.float32, device=device)
    mask_t = torch.tensor(mask, dtype=torch.float32, device=device)

    # ---- Model -------------------------------------------------------------
    model = ConvLSTMBaseline(
        seq_channels=X_seq_train_t.shape[2],
        static_channels=X_static_train_t.shape[1],
        hidden_channels=int(cfg["hidden_channels"]),
        num_layers=int(cfg["num_layers"]),
        kernel_size=int(cfg["kernel_size"]),
        dropout=float(cfg["dropout"]),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]))

    num_epochs = int(cfg["num_epochs"])
    log_every = int(cfg.get("log_every", 25))
    losses = []

    print(f"Training ConvLSTM for {num_epochs} epochs …")
    model.train()
    for epoch in range(1, num_epochs + 1):
        optimizer.zero_grad()
        rate_train = model(X_seq_train_t, X_static_train_t, mask_t)
        loss = poisson_nll_grid(rate_train, Y_train_t, mask_t)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

        if epoch == 1 or epoch % log_every == 0 or epoch == num_epochs:
            print(f"  epoch {epoch:5d}/{num_epochs}   train Poisson NLL = {loss.item():.4f}")

    # ---- Predict -------------------------------------------------------------
    model.eval()
    with torch.no_grad():
        rate_test = model(X_seq_test_t, X_static_test_t, mask_t).cpu().numpy()
    rate_test = np.clip(rate_test, 0, None)

    # Flatten to (obs) level, valid cells only, matching every other baseline's evaluate_metrics call.
    row_idx, col_idx, _ = build_cell_grid_index(cell_coords[["longitude", "latitude"]].to_numpy())
    n_test = rate_test.shape[0]
    y_test_flat = Y_test[:, row_idx, col_idx].reshape(-1)
    rate_test_flat = rate_test[:, row_idx, col_idx].reshape(-1)
    test_dates_flat = np.repeat(dates_test, len(row_idx))

    # ---- Metrics ---------------------------------------------------------
    metrics = evaluate_metrics(y_test_flat, rate_test_flat, test_dates_flat)

    print("\n=== Test metrics ===")
    print("Observation-level metrics:")
    print(f"  Mean log-likelihood per obs: {metrics['mean_ll_obs']:.4f}")
    print(f"  MAE per obs:                 {metrics['mae_obs']:.4f}")
    print(f"  RMSE per obs:                {metrics['rmse_obs']:.4f}")
    print("Daily-level metrics:")
    print(f"  Mean log-likelihood daily:   {metrics['mean_ll_daily']:.4f}")
    print(f"  MAE daily:                   {metrics['mae_daily']:.4f}")
    print(f"  RMSE daily:                  {metrics['rmse_daily']:.4f}")

    with open(out_dir / "metrics.yaml", "w") as f:
        yaml.dump(metrics, f, sort_keys=True)

    torch.save(model.state_dict(), out_dir / "model_state_dict.pt")

    plot_loss_curve(losses, save_path=plots_dir / "training_loss.png")
    plot_daily_predictions(test_dates_flat, y_test_flat, rate_test_flat, save_path=plots_dir / "daily_timeseries.png")

    if not cfg.get("generate_spatial_plots", True):
        print("Done (daily outputs only).")
        return

    meta = compute_meta(df)
    lon_std = (cell_coords["longitude"].to_numpy() - meta["lon_mean"]) / meta["lon_std"]
    lat_std = (cell_coords["latitude"].to_numpy() - meta["lat_mean"]) / meta["lat_std"]
    spatial_coords = np.tile(np.column_stack([lon_std, lat_std]), (len(dates_test), 1))

    print(f"  Saving observed vs predicted spatial maps for all {len(dates_test)} test days …")
    plot_daily_interpolated_spatial_maps_separate(
        test_coords=spatial_coords,
        test_dates=test_dates_flat,
        y_true=y_test_flat,
        rate_mean=rate_test_flat,
        meta=meta,
        gulf_csv_path=cfg["gulf_csv_path"],
        out_dir=plots_dir / "spatial_interpolated_test_days",
        cmap="viridis",
    )

    print(f"Plots saved → {plots_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    main(cfg)
