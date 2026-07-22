"""
Train/evaluate a spatial Graph Neural Network (GCN) baseline.

The ~50 grid cells of the Gulf of Trieste are treated as nodes of a fixed
spatial graph (rook/queen adjacency on the regular grid). For each date t,
the model predicts:

    y(cell, t) ~ Poisson(lambda(cell, t))

from node features built the same way as the other baselines:
    - spatial coordinates, time
    - selected covariates at t
    - optional observed-past lag features (cell-level and daily-total),
      shared with the lagged LGCP model (see src.models.data_pp_lgcp)

by propagating features through a stack of graph-convolution layers
(Kipf & Welling, 2017) shared across all cells at a given date.

Results are saved under:

    reports/gnn/<run_name>/
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

from sklearn.preprocessing import StandardScaler

# ---- Make src importable when script is run from the project root ----------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.data_pp_lgcp import (
    add_daily_chl_features,
    add_lag_features,
    make_day_split_masks,
)
from src.models.metrics_lgcp import evaluate_metrics
from src.models.gnn import (
    SpatioTemporalGNN,
    build_grid_adjacency,
    normalize_adjacency,
    poisson_nll,
)


# ---------------------------------------------------------------------------
# Config utilities
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    # --- data ---
    "years": 2024,
    "parquet_path": "data/processed/cpr_gfw.parquet",
    "target_col": "ais_vessels_count",
    # --- split ---
    "split_strategy": "fixed_test_window",  # "random_day" | "chronological" | "fixed_test_window"
    "train_fraction": 0.9,
    "data_seed": 42,
    "test_start_date": "2024-12-01",
    "test_end_date": "2024-12-10",
    # --- features ---
    "covariate_cols": [
        "chl",
        "thetao",
        "fishing_block",
        "is_holiday",
        "is_weekend",
        "cell_lag_1",
        "cell_lag_7",
        "cell_roll_mean_7",
        "daily_total_lag_1",
        "daily_total_lag_7",
        "daily_total_roll_mean_7",
    ],
    "include_coords": True,
    "include_time": True,
    "standardize_features": True,
    "lag_features": {
        "enabled": True,
        "mode": "observed_past",
        "target_col": "ais_vessels_count",
        "date_col": "date",
        "cell_cols": ["longitude", "latitude"],
        "cell_lags": [1, 7],
        "cell_rolling_windows": [7],
        "daily_total_lags": [1, 7],
        "daily_total_rolling_windows": [7],
        "fillna_value": 0.0,
    },
    # --- graph ---
    "adjacency": "queen",  # "rook" | "queen"
    # --- model ---
    "hidden_dim": 32,
    "num_layers": 2,
    "dropout": 0.1,
    # --- training ---
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "num_epochs": 300,
    "torch_seed": 0,
    "device": "cpu",
    "log_every": 25,
    # --- output ---
    "report_root": "reports",
    "model_family": "gnn",
    "run_name": "gnn_dec01_dec10",
}


def resolve_parquet_path(cfg: dict) -> dict:
    """
    Resolve year-specific parquet path.

    Example
    -------
    years: 2024
    parquet_path: data/processed/cpr_gfw.parquet

    becomes:
    data/processed/cpr_gfw_2024.parquet
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

    suffix = path.suffix
    stem = path.stem

    if suffix == "":
        suffix = ".parquet"

    if f"_{year}" not in stem:
        resolved_path = path.with_name(f"{stem}_{year}{suffix}")
    else:
        resolved_path = path.with_name(f"{stem}{suffix}")

    cfg["parquet_path"] = str(resolved_path)
    return cfg


def resolve_run_name(cfg: dict) -> dict:
    """
    Build a unique run name.

    Example
    -------
    years: 2024
    run_name: gnn_dec01_dec10

    becomes:

    run_name: 2024_gnn_dec01_dec10_20260625-162430
    """
    if "run_name" not in cfg:
        return cfg

    run_name_from_config = str(cfg["run_name"])
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    year = cfg.get("years", None)

    if isinstance(year, list):
        if len(year) != 1:
            raise NotImplementedError(
                "Multiple years are not supported yet. "
                "For now, use a single year, e.g. years: 2024."
            )
        year = year[0]

    if year is not None:
        year_str = str(year)

        if run_name_from_config.startswith(f"{year_str}_"):
            base_run_name = run_name_from_config
        else:
            base_run_name = f"{year_str}_{run_name_from_config}"
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

    cfg = resolve_parquet_path(cfg)
    cfg = resolve_run_name(cfg)

    return cfg


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------


def add_time_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add t_norm = day_of_year / days_in_year.
    """
    df = df.copy()
    year = int(df["date"].dt.year.iloc[0])
    is_leap = (year % 4 == 0) and ((year % 100 != 0) or (year % 400 == 0))
    days_in_year = 366.0 if is_leap else 365.0
    df["t_norm"] = df["date"].dt.dayofyear.astype(float) / days_in_year
    return df


def build_node_features(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, list[str]]:
    """
    Build the per-(cell, date) feature table used by the GNN.

    Reuses the same daily-chlorophyll and observed-past lag features as the
    lagged LGCP model, so the two are directly comparable.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    target_col = cfg["target_col"]
    covariate_cols = list(cfg["covariate_cols"])

    df = add_daily_chl_features(df, chl_col="chl", date_col="date")

    lag_cfg = cfg.get("lag_features")
    if lag_cfg is not None and lag_cfg.get("enabled", False):
        if cfg.get("split_strategy") == "random_day":
            raise ValueError(
                "lag_features with observed_past mode should not be used with "
                "split_strategy='random_day'. Use fixed_test_window or chronological."
            )
        df, generated_lag_cols = add_lag_features(df, lag_cfg)
        print(f"Generated lag features: {generated_lag_cols}")

    df[target_col] = df[target_col].fillna(0).astype(float)
    df = add_time_column(df)

    missing_cols = [col for col in covariate_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing covariate columns in dataset: {missing_cols}. "
            f"Available columns are: {list(df.columns)}"
        )

    feature_cols = []
    if cfg.get("include_coords", True):
        feature_cols += ["longitude", "latitude"]
    if cfg.get("include_time", True):
        feature_cols += ["t_norm"]
    feature_cols += covariate_cols

    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="raise")

    return df, feature_cols


def build_graph_tensors(df: pd.DataFrame, feature_cols: list[str], target_col: str):
    """
    Pivot the (cell, date) table into dense [n_dates, n_cells, ...] tensors.

    Requires a balanced panel: the same set of cells present on every date.
    """
    cell_coords = (
        df[["longitude", "latitude"]]
        .drop_duplicates()
        .sort_values(["longitude", "latitude"])
        .reset_index(drop=True)
    )
    n_cells = len(cell_coords)

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
            "The GNN pivot assumes a fixed set of nodes across time."
        )

    dates = np.sort(df["date"].unique())
    date_to_pos = {d: i for i, d in enumerate(dates)}

    n_dates = len(dates)
    n_features = len(feature_cols)

    X = np.zeros((n_dates, n_cells, n_features), dtype=np.float32)
    Y = np.zeros((n_dates, n_cells), dtype=np.float32)

    date_pos = df["date"].map(date_to_pos).to_numpy()
    cell_pos = df["_cell_idx"].to_numpy()

    X[date_pos, cell_pos, :] = df[feature_cols].to_numpy(dtype=np.float32)
    Y[date_pos, cell_pos] = df[target_col].to_numpy(dtype=np.float32)

    return X, Y, dates, cell_coords


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def plot_daily_predictions(test_dates, y_true, rate_mean, save_path=None):
    """
    Plot observed vs predicted daily total counts.
    """
    df_eval = pd.DataFrame(
        {
            "date": pd.to_datetime(test_dates),
            "y": y_true,
            "lambda_hat": rate_mean,
        }
    )

    daily = df_eval.groupby("date", as_index=False).sum()

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(daily["date"], daily["y"], label="Observed", lw=2)
    ax.plot(daily["date"], daily["lambda_hat"], label="Predicted", lw=2)

    ax.set_xlabel("Date")
    ax.set_ylabel("Daily total vessels")
    ax.set_title("GNN — daily totals")
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
    ax.set_title("GNN — training loss")
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

    params_path = out_dir / "params.yaml"
    with open(params_path, "w") as f:
        yaml.dump(cfg, f, sort_keys=False)

    print(f"Experiment parameters saved → {params_path}")

    torch.manual_seed(int(cfg.get("torch_seed", 0)))
    device = torch.device(cfg.get("device", "cpu"))

    # ---- Data ----------------------------------------------------------------
    print("Loading data …")
    df = pd.read_parquet(cfg["parquet_path"])
    df["date"] = pd.to_datetime(df["date"])

    print(f"  Raw data shape: {df.shape}")

    df_feat, feature_cols = build_node_features(df, cfg)

    print(f"  Number of features: {len(feature_cols)}")
    print("  Features:")
    for col in feature_cols:
        print(f"    - {col}")

    target_col = cfg["target_col"]
    X, Y, dates, cell_coords = build_graph_tensors(df_feat, feature_cols, target_col)

    n_dates, n_cells, n_features = X.shape
    print(f"  Graph: {n_cells} nodes, {n_dates} dates, {n_features} features/node")

    # ---- Graph -----------------------------------------------------------
    adj = build_grid_adjacency(
        cell_coords[["longitude", "latitude"]].to_numpy(),
        connectivity=cfg.get("adjacency", "queen"),
    )
    adj_norm = normalize_adjacency(adj)
    adj_norm_t = torch.tensor(adj_norm, dtype=torch.float32, device=device)

    print(f"  Adjacency ({cfg.get('adjacency', 'queen')}): {int(adj.sum())} directed edges")

    # ---- Split -------------------------------------------------------------
    train_mask, test_mask = make_day_split_masks(
        df=pd.DataFrame({"date": dates}),
        train_fraction=cfg.get("train_fraction", 0.9),
        random_seed=cfg.get("data_seed", 42),
        split_strategy=cfg.get("split_strategy", "fixed_test_window"),
        test_start_date=cfg.get("test_start_date"),
        test_end_date=cfg.get("test_end_date"),
    )

    X_train, Y_train = X[train_mask], Y[train_mask]
    X_test, Y_test = X[test_mask], Y[test_mask]
    dates_train, dates_test = dates[train_mask], dates[test_mask]

    print(f"  Train: {X_train.shape[0]} days   Test: {X_test.shape[0]} days")

    # ---- Standardization ---------------------------------------------------
    if cfg.get("standardize_features", True):
        scaler = StandardScaler()
        scaler.fit(X_train.reshape(-1, n_features))
        X_train_model = scaler.transform(X_train.reshape(-1, n_features)).reshape(
            X_train.shape
        )
        X_test_model = scaler.transform(X_test.reshape(-1, n_features)).reshape(
            X_test.shape
        )
    else:
        X_train_model = X_train
        X_test_model = X_test

    X_train_t = torch.tensor(X_train_model, dtype=torch.float32, device=device)
    Y_train_t = torch.tensor(Y_train, dtype=torch.float32, device=device)
    X_test_t = torch.tensor(X_test_model, dtype=torch.float32, device=device)

    # ---- Model ---------------------------------------------------------------
    model = SpatioTemporalGNN(
        in_dim=n_features,
        hidden_dim=int(cfg["hidden_dim"]),
        num_layers=int(cfg["num_layers"]),
        dropout=float(cfg["dropout"]),
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg["weight_decay"]),
    )

    print("Training GNN …")
    num_epochs = int(cfg["num_epochs"])
    log_every = int(cfg.get("log_every", 25))
    losses = []

    model.train()
    for epoch in range(1, num_epochs + 1):
        optimizer.zero_grad()
        rate_train = model(X_train_t, adj_norm_t)
        loss = poisson_nll(rate_train, Y_train_t)
        loss.backward()
        optimizer.step()

        losses.append(float(loss.item()))

        if epoch == 1 or epoch % log_every == 0 or epoch == num_epochs:
            print(f"  epoch {epoch:5d}/{num_epochs}   train Poisson NLL = {loss.item():.4f}")

    # ---- Predict ---------------------------------------------------------------
    model.eval()
    with torch.no_grad():
        rate_test = model(X_test_t, adj_norm_t).cpu().numpy()

    rate_test = np.clip(rate_test, 0, None)

    y_test_flat = Y_test.reshape(-1)
    rate_test_flat = rate_test.reshape(-1)
    test_dates_flat = np.repeat(dates_test, n_cells)

    # ---- Metrics -------------------------------------------------------------
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

    metrics_path = out_dir / "metrics.yaml"
    with open(metrics_path, "w") as f:
        yaml.dump(metrics, f, sort_keys=False)

    print(f"Metrics saved → {metrics_path}")

    # ---- Save model & training curve -----------------------------------------
    torch.save(model.state_dict(), out_dir / "model_state_dict.pt")

    plot_loss_curve(losses, save_path=plots_dir / "training_loss.png")

    # ---- Plot ----------------------------------------------------------------
    plot_daily_predictions(
        test_dates=test_dates_flat,
        y_true=y_test_flat,
        rate_mean=rate_test_flat,
        save_path=plots_dir / "daily_timeseries.png",
    )

    print(f"Plots saved → {plots_dir}")
    print("Done.")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config/15_gnn.yaml",
        help="Path to YAML config file.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    main(cfg)
