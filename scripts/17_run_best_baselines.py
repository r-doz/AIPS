"""
Run all three baselines (last-available Poisson, window Poisson GLM, GNN)
at their best hyperparameters -- as found via the grid searches in
notebooks/51_paper_lgcp_window_selection_baselines.ipynb -- on both paper
test weeks (2025-05-01..07 and 2025-11-01..07), and save every method's
metrics for both weeks to a single CSV.

Usage
-----
    python scripts/17_run_best_baselines.py [--output-csv PATH]

Each baseline is trained fresh here (not loaded from a checkpoint), on
years=[2024, 2025] data, using the same train/test split and evaluation
protocol as the notebook.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.metrics_lgcp import evaluate_metrics

# ---------------------------------------------------------------------------
# Config: best hyperparameters found in notebooks/51_paper_lgcp_window_selection_baselines.ipynb
# ---------------------------------------------------------------------------

YEARS = [2024, 2025]
SEED = 0

TEST_WEEKS = [
    {"label": "may", "test_start": "2025-05-01", "test_end": "2025-05-07"},
    {"label": "november", "test_start": "2025-11-01", "test_end": "2025-11-07"},
]

LAST_AVAILABLE_BEST_MODE = "one_step_ahead"

GLM_BEST_WINDOW_SIZE = 14
GLM_BEST_ALPHA = 0.0001

GNN_BEST_HIDDEN_DIM = 32
GNN_BEST_NUM_LAYERS = 2
GNN_BEST_LR = 0.01

BASE_GLM_CONFIG = PROJECT_ROOT / "config/14_window_poisson_glm.yaml"
BASE_GNN_CONFIG = PROJECT_ROOT / "config/15_gnn.yaml"

METRIC_KEYS = [
    "mean_ll_obs", "mae_obs", "rmse_obs",
    "mean_ll_daily", "mae_daily", "rmse_daily",
    "daily_delta_corr", "daily_direction_accuracy_moving",
]

DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "reports/paper/lgcp_two_week_2025_05_11/best_baselines_metrics.csv"


def load_script_module(name: str, relpath: str):
    """scripts/13_..., 14_..., 15_... start with a digit, so they can't be
    imported with a normal `import` statement."""
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / relpath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Per-method runners
# ---------------------------------------------------------------------------


def run_last_available(last_avail) -> list[dict]:
    la_cfg = last_avail.load_config(None)
    la_cfg["years"] = YEARS
    la_cfg["parquet_path"] = str(PROJECT_ROOT / la_cfg["parquet_path"])

    rows = []
    for week in TEST_WEEKS:
        label, test_start, test_end = week["label"], week["test_start"], week["test_end"]

        (_, _, _, _, _, test_y, _, df_la) = last_avail.prepare_data(
            la_cfg["parquet_path"], train_fraction=la_cfg["train_fraction"], random_seed=la_cfg["data_seed"],
            split_strategy="fixed_test_window", test_start_date=test_start, test_end_date=test_end,
            years=YEARS,
        )
        df_la["date"] = pd.to_datetime(df_la["date"])
        train_mask, test_mask = last_avail.make_day_split_masks(
            df=df_la, train_fraction=la_cfg["train_fraction"], random_seed=la_cfg["data_seed"],
            split_strategy="fixed_test_window", test_start_date=test_start, test_end_date=test_end,
        )
        test_dates = df_la.loc[test_mask, "date"].values
        preds = last_avail.build_last_available_predictions(
            df=df_la, train_mask=train_mask, test_mask=test_mask, mode=LAST_AVAILABLE_BEST_MODE
        )
        metrics = evaluate_metrics(test_y, preds, test_dates)
        print(f"[Last Available Poisson | {label}] mean_ll_obs={metrics['mean_ll_obs']:.4f}  rmse_daily={metrics['rmse_daily']:.4f}")
        rows.append({"method": "Last Available Poisson", "week": label, **{k: metrics[k] for k in METRIC_KEYS}})
    return rows


def run_window_glm(window_glm) -> list[dict]:
    glm_cfg = window_glm.load_config(str(BASE_GLM_CONFIG))
    glm_cfg["years"] = YEARS
    glm_cfg["parquet_path"] = str(PROJECT_ROOT / glm_cfg["parquet_path"])
    glm_cfg["window_size"] = GLM_BEST_WINDOW_SIZE

    df_glm_raw = window_glm.load_parquet_years(glm_cfg["parquet_path"], YEARS)
    df_glm_raw["date"] = pd.to_datetime(df_glm_raw["date"])

    rows = []
    for week in TEST_WEEKS:
        label, test_start, test_end = week["label"], week["test_start"], week["test_end"]

        df_win, feature_cols = window_glm.build_window_dataframe(df_glm_raw, glm_cfg)
        train_mask, test_mask = window_glm.make_split_masks(
            dates=df_win["date"], split_strategy="fixed_test_window",
            train_fraction=glm_cfg["train_fraction"], random_seed=glm_cfg["data_seed"],
            test_start_date=test_start, test_end_date=test_end,
        )
        X = df_win[feature_cols].values.astype(np.float32)
        y = df_win[glm_cfg["target_col"]].values.astype(np.float32)
        dts = df_win["date"].values
        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        test_dates = dts[test_mask]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = PoissonRegressor(alpha=GLM_BEST_ALPHA, max_iter=glm_cfg["max_iter"])
        model.fit(X_train_s, y_train)
        preds = np.clip(model.predict(X_test_s), 0, None)
        metrics = evaluate_metrics(y_test, preds, test_dates)
        print(f"[Window Poisson GLM | {label}] mean_ll_obs={metrics['mean_ll_obs']:.4f}  rmse_daily={metrics['rmse_daily']:.4f}")
        rows.append({"method": "Window Poisson GLM", "week": label, **{k: metrics[k] for k in METRIC_KEYS}})
    return rows


def run_gnn(gnn_mod) -> list[dict]:
    gnn_cfg = gnn_mod.load_config(str(BASE_GNN_CONFIG))
    gnn_cfg["years"] = YEARS
    gnn_cfg["parquet_path"] = str(PROJECT_ROOT / gnn_cfg["parquet_path"])
    device = torch.device(gnn_cfg.get("device", "cpu"))

    df_gnn = gnn_mod.load_parquet_years(gnn_cfg["parquet_path"], YEARS)
    df_gnn["date"] = pd.to_datetime(df_gnn["date"])
    df_feat, feature_cols = gnn_mod.build_node_features(df_gnn, gnn_cfg)
    X, Y, dates, cell_coords = gnn_mod.build_graph_tensors(df_feat, feature_cols, gnn_cfg["target_col"])
    n_dates, n_cells, n_features = X.shape

    adj = gnn_mod.build_grid_adjacency(
        cell_coords[["longitude", "latitude"]].to_numpy(), connectivity=gnn_cfg.get("adjacency", "queen")
    )
    adj_norm = gnn_mod.normalize_adjacency(adj)
    adj_norm_t = torch.tensor(adj_norm, dtype=torch.float32, device=device)

    rows = []
    for week in TEST_WEEKS:
        label, test_start, test_end = week["label"], week["test_start"], week["test_end"]

        train_mask, test_mask = gnn_mod.make_day_split_masks(
            df=pd.DataFrame({"date": dates}), train_fraction=gnn_cfg["train_fraction"], random_seed=gnn_cfg["data_seed"],
            split_strategy="fixed_test_window", test_start_date=test_start, test_end_date=test_end,
        )
        X_train, Y_train = X[train_mask], Y[train_mask]
        X_test, Y_test = X[test_mask], Y[test_mask]
        dates_test = dates[test_mask]

        scaler = StandardScaler()
        scaler.fit(X_train.reshape(-1, n_features))
        X_train_model = scaler.transform(X_train.reshape(-1, n_features)).reshape(X_train.shape)
        X_test_model = scaler.transform(X_test.reshape(-1, n_features)).reshape(X_test.shape)

        X_train_t = torch.tensor(X_train_model, dtype=torch.float32, device=device)
        Y_train_t = torch.tensor(Y_train, dtype=torch.float32, device=device)
        X_test_t = torch.tensor(X_test_model, dtype=torch.float32, device=device)

        torch.manual_seed(SEED)
        model = gnn_mod.SpatioTemporalGNN(
            in_dim=n_features, hidden_dim=GNN_BEST_HIDDEN_DIM, num_layers=GNN_BEST_NUM_LAYERS, dropout=gnn_cfg["dropout"],
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=GNN_BEST_LR, weight_decay=gnn_cfg["weight_decay"])

        model.train()
        for epoch in range(1, int(gnn_cfg["num_epochs"]) + 1):
            optimizer.zero_grad()
            rate_train = model(X_train_t, adj_norm_t)
            loss = gnn_mod.poisson_nll(rate_train, Y_train_t)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            rate_test = model(X_test_t, adj_norm_t).cpu().numpy()
        rate_test = np.clip(rate_test, 0, None)

        y_test_flat = Y_test.reshape(-1)
        rate_test_flat = rate_test.reshape(-1)
        test_dates_flat = np.repeat(dates_test, n_cells)
        metrics = evaluate_metrics(y_test_flat, rate_test_flat, test_dates_flat)
        print(f"[GNN | {label}] mean_ll_obs={metrics['mean_ll_obs']:.4f}  rmse_daily={metrics['rmse_daily']:.4f}")
        rows.append({"method": "GNN", "week": label, **{k: metrics[k] for k in METRIC_KEYS}})
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(output_csv: Path):
    last_avail = load_script_module("last_avail", "scripts/13_evaluate_last_available_poisson.py")
    window_glm = load_script_module("window_glm", "scripts/14_train_window_poisson_glm.py")
    gnn_mod = load_script_module("gnn_mod", "scripts/15_train_gnn.py")

    print("=" * 80)
    print("Running baselines at their best hyperparameters (see notebook 51)")
    print("=" * 80)
    print(f"Last Available Poisson: mode={LAST_AVAILABLE_BEST_MODE}")
    print(f"Window Poisson GLM:      window_size={GLM_BEST_WINDOW_SIZE}, poisson_alpha={GLM_BEST_ALPHA}")
    print(f"GNN:                     hidden_dim={GNN_BEST_HIDDEN_DIM}, num_layers={GNN_BEST_NUM_LAYERS}, lr={GNN_BEST_LR}")
    print()

    rows = []
    rows += run_last_available(last_avail)
    rows += run_window_glm(window_glm)
    rows += run_gnn(gnn_mod)

    results_df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_csv, index=False)

    print("\nPer-week metrics:")
    print(results_df.set_index(["method", "week"]))
    print(f"\nSaved → {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-csv",
        type=str,
        default=str(DEFAULT_OUTPUT_CSV),
        help="Path to save the metrics CSV.",
    )
    args = parser.parse_args()
    main(Path(args.output_csv))
