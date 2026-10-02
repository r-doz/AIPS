"""Re-run the Window Poisson GLM and GNN hyperparameter grid searches from
notebooks/51_paper_baseline_hyperparameter_selection.ipynb, on validation
weeks redefined as the two 7-day blocks immediately preceding each
reporting week (2025-05-05..11 and 2025-11-03..09), instead of the
original April/October blocks.

Grids, scoring (compute_relative_scores) and module-loading pattern are
copied verbatim from notebook 51 cells 2/3/5/9/11 -- only TEST_WEEKS and
RESULTS_DIR_NAME differ.

Usage:
    python scripts/51b_baseline_hparam_rerun_2weeks_before.py
"""

from __future__ import annotations

import importlib.util
import itertools
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

YEARS = [2024, 2025]
SEED = 0

# Two 7-day blocks immediately preceding each reporting week (2025-05-05..11
# and 2025-11-03..09), instead of notebook 51's April/October blocks.
TEST_WEEKS = [
    {"label": "before_may_w1", "test_start": "2025-04-21", "test_end": "2025-04-27"},
    {"label": "before_may_w2", "test_start": "2025-04-28", "test_end": "2025-05-04"},
    {"label": "before_nov_w1", "test_start": "2025-10-20", "test_end": "2025-10-26"},
    {"label": "before_nov_w2", "test_start": "2025-10-27", "test_end": "2025-11-02"},
]

METRIC_DIRECTIONS = {"mean_ll_obs": "max", "rmse_daily": "min"}

BASELINE_COVARIATE_COLS = [
    "salinity",
    "chl", "thetao", "fishing_block", "is_holiday", "is_weekend",
    "cell_lag_1", "cell_lag_7",
]

GLM_ALPHA_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
GLM_WINDOW_SIZE_GRID = [1, 3, 7, 14]

GNN_HIDDEN_DIM_GRID = [16, 32, 64]
GNN_NUM_LAYERS_GRID = [1, 2, 3]
GNN_LR_GRID = [1e-4, 1e-3, 1e-2, 1e-1]

BASE_GLM_CONFIG = "config/14_window_poisson_glm_salinity.yaml"
BASE_GNN_CONFIG = "config/15_gnn_salinity.yaml"
RESULTS_DIR_NAME = "reports/paper/hparam_revalidation_2weeks_before_salinity/baselines"

METRIC_KEYS = [
    "mean_ll_obs", "mae_obs", "rmse_obs",
    "mean_ll_daily", "mae_daily", "rmse_daily",
    "daily_delta_corr", "daily_direction_accuracy_moving",
]


def load_script_module(name, relpath):
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / relpath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compute_relative_scores(df: pd.DataFrame, week_col: str, group_cols: list[str], metrics_directions: dict) -> pd.DataFrame:
    group_cols = list(group_cols)
    valid_df = df.dropna(subset=list(metrics_directions)).copy()

    for metric, direction in metrics_directions.items():
        def _normalize(s, direction=direction):
            lo, hi = s.min(), s.max()
            if hi - lo < 1e-12:
                return pd.Series(1.0, index=s.index)
            return (s - lo) / (hi - lo) if direction == "max" else (hi - s) / (hi - lo)
        valid_df[f"relative_{metric}"] = valid_df.groupby(week_col)[metric].transform(_normalize)

    relative_cols = [f"relative_{m}" for m in metrics_directions]
    n_weeks_total = df[week_col].nunique()
    week_counts = valid_df.groupby(group_cols)[week_col].nunique().rename("n_weeks_valid")

    scores = valid_df.groupby(group_cols)[relative_cols].mean()
    scores["combined_relative_score"] = scores[relative_cols].mean(axis=1)
    scores = scores.join(week_counts).reset_index()

    complete = scores[scores["n_weeks_valid"] == n_weeks_total].copy()
    incomplete = scores[scores["n_weeks_valid"] != n_weeks_total]
    if len(incomplete):
        print(f"  Excluded {len(incomplete)} combo(s) that didn't converge on every week:")
        print(incomplete[group_cols + ["n_weeks_valid"]])
    return complete.sort_values("combined_relative_score", ascending=False)


def run_glm(window_glm, results_dir: Path):
    glm_cfg = window_glm.load_config(str(PROJECT_ROOT / BASE_GLM_CONFIG))
    glm_cfg["years"] = YEARS
    glm_cfg["parquet_path"] = str(PROJECT_ROOT / glm_cfg["parquet_path"])
    glm_cfg["covariate_cols"] = BASELINE_COVARIATE_COLS

    df_glm_raw = window_glm.load_parquet_years(glm_cfg["parquet_path"], YEARS)
    df_glm_raw["date"] = pd.to_datetime(df_glm_raw["date"])

    glm_rows = []
    for week in TEST_WEEKS:
        label, test_start, test_end = week["label"], week["test_start"], week["test_end"]
        for window_size in GLM_WINDOW_SIZE_GRID:
            cfg_ws = dict(glm_cfg)
            cfg_ws["window_size"] = window_size
            df_win, feature_cols = window_glm.build_window_dataframe(df_glm_raw, cfg_ws)
            train_mask, test_mask = window_glm.make_split_masks(
                dates=df_win["date"], split_strategy="fixed_test_window",
                train_fraction=cfg_ws["train_fraction"], random_seed=cfg_ws["data_seed"],
                test_start_date=test_start, test_end_date=test_end,
            )
            X = df_win[feature_cols].values.astype(np.float32)
            y = df_win[cfg_ws["target_col"]].values.astype(np.float32)
            dts = df_win["date"].values
            X_train, y_train = X[train_mask], y[train_mask]
            X_test, y_test = X[test_mask], y[test_mask]
            test_dates = dts[test_mask]

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            for alpha in GLM_ALPHA_GRID:
                model = PoissonRegressor(alpha=alpha, max_iter=cfg_ws["max_iter"])
                model.fit(X_train_s, y_train)
                preds = np.clip(model.predict(X_test_s), 0, None)
                metrics = evaluate_metrics(y_test, preds, test_dates)
                glm_rows.append({"week": label, "window_size": window_size, "poisson_alpha": alpha, **metrics})
        print(f"[week={label}] GLM grid done.", flush=True)

    glm_results_df = pd.DataFrame(glm_rows)
    glm_results_df.to_csv(results_dir / "glm_grid_results.csv", index=False)
    glm_scores_df = compute_relative_scores(glm_results_df, week_col="week", group_cols=["window_size", "poisson_alpha"], metrics_directions=METRIC_DIRECTIONS)
    glm_scores_df.to_csv(results_dir / "glm_scores.csv", index=False)
    best_glm_row = glm_scores_df.iloc[0]
    best_window_size = int(best_glm_row["window_size"])
    best_alpha = float(best_glm_row["poisson_alpha"])
    print(f"Selected GLM config: window_size={best_window_size}  poisson_alpha={best_alpha}")
    return best_window_size, best_alpha


def run_gnn(gnn_mod, results_dir: Path):
    gnn_cfg = gnn_mod.load_config(str(PROJECT_ROOT / BASE_GNN_CONFIG))
    gnn_cfg["years"] = YEARS
    gnn_cfg["parquet_path"] = str(PROJECT_ROOT / gnn_cfg["parquet_path"])
    gnn_cfg["covariate_cols"] = BASELINE_COVARIATE_COLS
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

    gnn_rows = []
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

        for hidden_dim, num_layers, lr in itertools.product(GNN_HIDDEN_DIM_GRID, GNN_NUM_LAYERS_GRID, GNN_LR_GRID):
            torch.manual_seed(SEED)
            model = gnn_mod.SpatioTemporalGNN(
                in_dim=n_features, hidden_dim=hidden_dim, num_layers=num_layers, dropout=gnn_cfg["dropout"],
            ).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=gnn_cfg["weight_decay"])

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
            gnn_rows.append({"week": label, "hidden_dim": hidden_dim, "num_layers": num_layers, "lr": lr, **metrics})
        print(f"[week={label}] GNN grid done.", flush=True)

    gnn_results_df = pd.DataFrame(gnn_rows)
    gnn_results_df.to_csv(results_dir / "gnn_grid_results.csv", index=False)
    gnn_scores_df = compute_relative_scores(gnn_results_df, week_col="week", group_cols=["hidden_dim", "num_layers", "lr"], metrics_directions=METRIC_DIRECTIONS)
    gnn_scores_df.to_csv(results_dir / "gnn_scores.csv", index=False)
    best_gnn_row = gnn_scores_df.iloc[0]
    best_hidden_dim = int(best_gnn_row["hidden_dim"])
    best_num_layers = int(best_gnn_row["num_layers"])
    best_lr = float(best_gnn_row["lr"])
    print(f"Selected GNN config: hidden_dim={best_hidden_dim}  num_layers={best_num_layers}  lr={best_lr}")
    return best_hidden_dim, best_num_layers, best_lr


def main():
    results_dir = PROJECT_ROOT / RESULTS_DIR_NAME
    results_dir.mkdir(parents=True, exist_ok=True)

    last_avail = load_script_module("last_avail", "scripts/13_evaluate_last_available_poisson.py")
    window_glm = load_script_module("window_glm", "scripts/14_train_window_poisson_glm.py")
    gnn_mod = load_script_module("gnn_mod", "scripts/15_train_gnn.py")

    print(f"Test weeks: {[w['label'] for w in TEST_WEEKS]}", flush=True)

    best_window_size, best_alpha = run_glm(window_glm, results_dir)
    best_hidden_dim, best_num_layers, best_lr = run_gnn(gnn_mod, results_dir)

    summary = {
        "test_weeks": TEST_WEEKS,
        "glm": {"window_size": best_window_size, "poisson_alpha": best_alpha},
        "gnn": {"hidden_dim": best_hidden_dim, "num_layers": best_num_layers, "lr": best_lr},
    }
    import yaml
    with open(results_dir / "selected_hyperparameters.yaml", "w") as f:
        yaml.safe_dump(summary, f, sort_keys=False)
    print(f"\nSaved selection summary -> {results_dir / 'selected_hyperparameters.yaml'}")


if __name__ == "__main__":
    main()
