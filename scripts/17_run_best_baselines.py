"""
Run all four baselines (last-available Poisson, global cell mean Poisson,
window Poisson GLM, GNN) at their best hyperparameters -- as found via the
grid searches in notebooks/51_paper_baseline_hyperparameter_selection.ipynb
(evaluated on 2025-04-17..30 and 2025-10-18..31), and (for global cell
mean, which has no notebook search) by comparing its two modes directly
here -- on the paper's reporting weeks (2025-05-01..07 and
2025-11-01..07). Hyperparameter selection and final reporting deliberately
use disjoint weeks.

For each method and week, predictions are computed once for the full
7-day test window, then scored at three forecast horizons to see how
prediction quality degrades with lead time: the full week, the first 3
days only, and the first day only. Three CSVs are saved, one per horizon.

Usage
-----
    python scripts/17_run_best_baselines.py [--output-dir DIR]
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
# Config: best hyperparameters found in
# notebooks/51_paper_baseline_hyperparameter_selection.ipynb (GLM, GNN) and
# by direct comparison here (last-available mode fixed; global cell mean
# mode chosen below).
# ---------------------------------------------------------------------------

YEARS = [2024, 2025]
SEED = 0

# Reporting weeks for the final tables -- deliberately NOT the weeks used
# to choose hyperparameters (2025-04-17..30 and 2025-10-18..31, see the
# notebook), so hyperparameter selection and final reporting use disjoint
# data.
TEST_WEEKS = [
    {"label": "may", "test_start": "2025-05-01", "test_end": "2025-05-07"},
    {"label": "november", "test_start": "2025-11-01", "test_end": "2025-11-07"},
]

# Forecast horizons: metrics are computed on the first `n_days` of each
# test week (predictions are generated once for the full week and then
# sliced -- not retrained per horizon), to see how prediction quality
# degrades with lead time.
HORIZONS = [
    ("full_week", 7),
    ("first_3_days", 3),
    ("first_day", 1),
]

# Covariates used by every baseline that takes covariates (GLM, GNN).
BASELINE_COVARIATE_COLS = [
    "chl", "thetao", "fishing_block", "is_holiday", "is_weekend",
    "cell_lag_1", "cell_lag_7",
]

LAST_AVAILABLE_BEST_MODE = "one_step_ahead"

# No notebook search exists for this baseline -- it only has two discrete
# modes, both compared directly in run_global_cell_mean() below.
GLOBAL_CELL_MEAN_MODES = ["frozen_train", "expanding"]

# From notebooks/51_paper_baseline_hyperparameter_selection.ipynb
# (selected on 2025-04-17..30 and 2025-10-18..31, week-by-week then averaged)
GLM_BEST_WINDOW_SIZE = 14
GLM_BEST_ALPHA = 0.01

GNN_BEST_HIDDEN_DIM = 64
GNN_BEST_NUM_LAYERS = 3
GNN_BEST_LR = 0.01

BASE_GLM_CONFIG = PROJECT_ROOT / "config/14_window_poisson_glm.yaml"
BASE_GNN_CONFIG = PROJECT_ROOT / "config/15_gnn.yaml"

METRIC_KEYS = [
    "mean_ll_obs", "mae_obs", "rmse_obs",
    "mean_ll_daily", "mae_daily", "rmse_daily",
    "daily_delta_corr", "daily_direction_accuracy_moving",
]

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports/paper/lgcp_two_week_2025_05_11"


def load_script_module(name: str, relpath: str):
    """scripts/12b_..., 13_..., 14_..., 15_... start with a digit (or, for
    12b, a digit+letter), so they can't be imported with a normal `import`
    statement."""
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / relpath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def horizon_rows(method: str, week_label: str, test_dates, y_true, preds, test_start: str) -> list[dict]:
    """Slices already-computed full-week predictions into each forecast
    horizon and scores them -- no retraining/re-predicting per horizon."""
    test_dates = pd.to_datetime(np.asarray(test_dates))
    y_true = np.asarray(y_true)
    preds = np.asarray(preds)

    rows = []
    for horizon_label, horizon_days in HORIZONS:
        cutoff = pd.Timestamp(test_start) + pd.Timedelta(days=horizon_days - 1)
        mask = test_dates <= cutoff
        metrics = evaluate_metrics(y_true[mask], preds[mask], test_dates[mask])
        rows.append({
            "method": method,
            "week": week_label,
            "horizon": horizon_label,
            "horizon_days": horizon_days,
            **{k: metrics[k] for k in METRIC_KEYS},
        })
    return rows


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
        week_rows = horizon_rows("Last Available Poisson", label, test_dates, test_y, preds, test_start)
        for r in week_rows:
            print(f"[Last Available Poisson | {label} | {r['horizon']}] mean_ll_obs={r['mean_ll_obs']:.4f}  rmse_daily={r['rmse_daily']:.4f}")
        rows += week_rows
    return rows


def run_global_cell_mean(global_cell_mean) -> list[dict]:
    """
    Both modes are evaluated on the full week to pick the better one (by
    mean_ll_obs averaged across weeks, as before); the winning mode's
    already-computed full-week predictions are then sliced into the three
    forecast horizons for reporting.
    """
    target_col = "ais_vessels_count"
    date_col = "date"
    cell_cols = ["longitude", "latitude"]
    parquet_path = str(PROJECT_ROOT / "data/processed/cpr_gfw.parquet")

    df = global_cell_mean.load_parquet_years(parquet_path, YEARS)
    df[date_col] = pd.to_datetime(df[date_col]).dt.normalize()

    per_week_preds = {mode: {} for mode in GLOBAL_CELL_MEAN_MODES}
    avg_ll_scores = {mode: [] for mode in GLOBAL_CELL_MEAN_MODES}

    for week in TEST_WEEKS:
        label, test_start, test_end = week["label"], week["test_start"], week["test_end"]

        train_mask, test_mask = global_cell_mean.make_day_split_masks(
            df=df, train_fraction=0.9, random_seed=42,
            split_strategy="fixed_test_window", test_start_date=test_start, test_end_date=test_end,
        )
        test_df = df.loc[test_mask]
        y_test = test_df[target_col].values.astype(float)
        test_dates = test_df[date_col].values

        for mode in GLOBAL_CELL_MEAN_MODES:
            if mode == "frozen_train":
                preds = global_cell_mean.build_frozen_train_cell_mean_predictions(
                    df=df, train_mask=train_mask, test_mask=test_mask,
                    target_col=target_col, cell_cols=cell_cols,
                )
            else:
                preds = global_cell_mean.build_expanding_cell_mean_predictions(
                    df=df, train_mask=train_mask, test_mask=test_mask,
                    target_col=target_col, date_col=date_col, cell_cols=cell_cols,
                )
            metrics = evaluate_metrics(y_test, preds, test_dates)
            print(f"[Global Cell Mean ({mode}) | {label} | full_week] mean_ll_obs={metrics['mean_ll_obs']:.4f}  rmse_daily={metrics['rmse_daily']:.4f}")
            per_week_preds[mode][label] = (test_dates, y_test, preds, test_start)
            avg_ll_scores[mode].append(metrics["mean_ll_obs"])

    avg_ll = {mode: float(np.mean(scores)) for mode, scores in avg_ll_scores.items()}
    best_mode = max(avg_ll.items(), key=lambda item: item[1])[0]
    print(
        "[Global Cell Mean] selected mode="
        + best_mode
        + "  (avg mean_ll_obs: "
        + ", ".join(f"{m}={v:.4f}" for m, v in avg_ll.items())
        + ")"
    )

    rows = []
    for label, (test_dates, y_test, preds, test_start) in per_week_preds[best_mode].items():
        rows += horizon_rows("Global Cell Mean Poisson", label, test_dates, y_test, preds, test_start)
    return rows


def run_window_glm(window_glm) -> list[dict]:
    glm_cfg = window_glm.load_config(str(BASE_GLM_CONFIG))
    glm_cfg["years"] = YEARS
    glm_cfg["parquet_path"] = str(PROJECT_ROOT / glm_cfg["parquet_path"])
    glm_cfg["window_size"] = GLM_BEST_WINDOW_SIZE
    glm_cfg["covariate_cols"] = BASELINE_COVARIATE_COLS

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

        week_rows = horizon_rows("Window Poisson GLM", label, test_dates, y_test, preds, test_start)
        for r in week_rows:
            print(f"[Window Poisson GLM | {label} | {r['horizon']}] mean_ll_obs={r['mean_ll_obs']:.4f}  rmse_daily={r['rmse_daily']:.4f}")
        rows += week_rows
    return rows


def run_gnn(gnn_mod) -> list[dict]:
    gnn_cfg = gnn_mod.load_config(str(BASE_GNN_CONFIG))
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

        week_rows = horizon_rows("GNN", label, test_dates_flat, y_test_flat, rate_test_flat, test_start)
        for r in week_rows:
            print(f"[GNN | {label} | {r['horizon']}] mean_ll_obs={r['mean_ll_obs']:.4f}  rmse_daily={r['rmse_daily']:.4f}")
        rows += week_rows
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(output_dir: Path):
    last_avail = load_script_module("last_avail", "scripts/13_evaluate_last_available_poisson.py")
    global_cell_mean = load_script_module("global_cell_mean", "scripts/12b_global_cell_mean.py")
    window_glm = load_script_module("window_glm", "scripts/14_train_window_poisson_glm.py")
    gnn_mod = load_script_module("gnn_mod", "scripts/15_train_gnn.py")

    print("=" * 80)
    print("Running baselines at their best hyperparameters (see notebook 51)")
    print("=" * 80)
    print(f"Last Available Poisson:  mode={LAST_AVAILABLE_BEST_MODE}")
    print(f"Global Cell Mean Poisson: mode selected below (compares {GLOBAL_CELL_MEAN_MODES})")
    print(f"Window Poisson GLM:      window_size={GLM_BEST_WINDOW_SIZE}, poisson_alpha={GLM_BEST_ALPHA}")
    print(f"GNN:                     hidden_dim={GNN_BEST_HIDDEN_DIM}, num_layers={GNN_BEST_NUM_LAYERS}, lr={GNN_BEST_LR}")
    print(f"Covariates (GLM/GNN):    {BASELINE_COVARIATE_COLS}")
    print(f"Horizons:                {[h[0] for h in HORIZONS]}")
    print()

    rows = []
    rows += run_last_available(last_avail)
    rows += run_global_cell_mean(global_cell_mean)
    rows += run_window_glm(window_glm)
    rows += run_gnn(gnn_mod)

    results_df = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)

    for horizon_label, _ in HORIZONS:
        horizon_df = results_df[results_df["horizon"] == horizon_label].drop(columns=["horizon", "horizon_days"])
        out_path = output_dir / f"best_baselines_metrics_{horizon_label}.csv"
        horizon_df.to_csv(out_path, index=False)
        print(f"\n[{horizon_label}] saved -> {out_path}")
        print(horizon_df.set_index(["method", "week"]))

    print(f"\nAll 3 tables saved under {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to save the three metrics CSVs (one per forecast horizon).",
    )
    args = parser.parse_args()
    main(Path(args.output_dir))
