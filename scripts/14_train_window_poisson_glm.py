"""
Train/evaluate a window-based Poisson GLM.

For each spatial cell and target date t, the model predicts:

    y(cell, t) ~ Poisson(lambda(cell, t))

using:
    - spatial coordinates
    - time
    - selected covariates at t
    - previous target values y(cell, t-1), ..., y(cell, t-W)

Results are saved under:

    reports/window_poisson_glm/<run_name>/
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt

from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler

# ---- Make src importable when script is run from the project root ----------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.multi_year import load_parquet_years, years_label
from src.models.data_pp_lgcp import add_lag_features
from src.models.metrics_lgcp import evaluate_metrics, evaluate_first_three_days


# ---------------------------------------------------------------------------
# Config utilities
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    # --- data ---
    "years": 2024,
    "parquet_path": "data/processed/cpr_gfw.parquet",
    "target_col": "ais_vessels_count",
    # --- split ---
    "split_strategy": "random_day",  # "random_day" or "chronological"
    "train_fraction": 0.9,
    "data_seed": 42,
    "test_start_date": None,
    "test_end_date": None,
    # --- window features ---
    "window_size": 7,
    "drop_incomplete_windows": True,
    "covariate_cols": ["chl", "thetao", "is_holiday", "is_weekend", "fishing_block"],
    "include_coords": True,
    "include_time": True,
    # --- model ---
    "poisson_alpha": 0.001,
    "max_iter": 1000,
    "standardize_features": True,
    "lag_features": {
        "enabled": False,
    },
    # --- output ---
    "report_root": "reports",
    "model_family": "window_poisson_glm",
    "run_name": "2024_window_poisson_glm_W7_random_day",
}


def resolve_run_name(cfg: dict) -> dict:
    """
    Build a unique run name.

    Example
    -------
    years: [2024, 2025]
    run_name: window_poisson_glm_W1_alpha005_random_day

    becomes:

    run_name: 2024+2025_window_poisson_glm_W1_alpha005_random_day_20260625-162430
    """
    if "run_name" not in cfg:
        return cfg

    run_name_from_config = str(cfg["run_name"])
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    year_str = years_label(cfg.get("years"))

    # Avoid duplicating the year if the config already contains it.
    if year_str:
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


def load_config(config_path=None):
    cfg = DEFAULT_CONFIG.copy()

    if config_path is not None:
        with open(config_path, "r") as f:
            user_cfg = yaml.safe_load(f)

        if user_cfg is not None:
            cfg.update(user_cfg)

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

    def _days_in_year(year: int) -> float:
        is_leap = (year % 4 == 0) and ((year % 100 != 0) or (year % 400 == 0))
        return 366.0 if is_leap else 365.0

    days_in_year = df["date"].dt.year.map(_days_in_year)
    df["t_norm"] = df["date"].dt.dayofyear.astype(float) / days_in_year
    return df


def build_window_dataframe(
    df: pd.DataFrame, cfg: dict
) -> tuple[pd.DataFrame, list[str]]:
    """
    Build tabular window features.

    For each spatial cell, create:
        lag_1, ..., lag_W
    where lag_k = target value k days before in the same cell.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    target_col = cfg["target_col"]
    covariate_cols = list(cfg["covariate_cols"])
    window_size = int(cfg["window_size"])

    # cell_lag_N is a generated feature, rather than a column expected in the
    # input parquet.  Keep this naming compatible with the LGCP/GNN configs.
    requested_cell_lags = {}
    input_covariate_cols = []
    for col in covariate_cols:
        match = re.fullmatch(r"cell_lag_([1-9]\d*)", col)
        if match:
            requested_cell_lags[int(match.group(1))] = col
        else:
            input_covariate_cols.append(col)

    required_cols = (
        ["date", "longitude", "latitude", target_col] + input_covariate_cols
    )

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing columns in dataset: {missing_cols}. "
            f"Available columns are: {list(df.columns)}"
        )

    df[target_col] = df[target_col].fillna(0).astype(float)
    df = add_time_column(df)

    # Sort by spatial cell and date before constructing lags.
    df = df.sort_values(["longitude", "latitude", "date"]).copy()
    grouped_target = df.groupby(["longitude", "latitude"])[target_col]

    for lag, col in requested_cell_lags.items():
        df[col] = grouped_target.shift(lag)

    lag_cols = []
    for k in range(1, window_size + 1):
        # Do not add the same predictor twice under two different names.
        if k in requested_cell_lags:
            continue
        col = f"{target_col}_lag_{k}"
        df[col] = grouped_target.shift(k)
        lag_cols.append(col)

    if cfg.get("drop_incomplete_windows", True):
        df = df.dropna(subset=lag_cols + list(requested_cell_lags.values())).copy()
    else:
        generated_lag_cols = lag_cols + list(requested_cell_lags.values())
        df[generated_lag_cols] = df[generated_lag_cols].fillna(0.0)

    feature_cols = []

    if cfg.get("include_coords", True):
        feature_cols += ["longitude", "latitude"]

    if cfg.get("include_time", True):
        feature_cols += ["t_norm"]

    feature_cols += covariate_cols
    feature_cols += lag_cols

    # Ensure features are numeric.
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="raise")

    df = df.sort_values("date").copy()

    return df, feature_cols


def make_split_masks(
    dates,
    split_strategy="chronological",
    train_fraction=0.9,
    random_seed=42,
    test_start_date=None,
    test_end_date=None,
):
    """
    Create train/test masks using the target date of each sample.

    dates must be the date column of the final modelling dataframe,
    i.e. after window features have been created.
    """

    dates = pd.Series(pd.to_datetime(dates)).dt.normalize()

    if split_strategy == "fixed_test_window":
        if test_start_date is None or test_end_date is None:
            raise ValueError(
                "For split_strategy='fixed_test_window', both "
                "'test_start_date' and 'test_end_date' must be provided."
            )

        test_start = pd.to_datetime(test_start_date).normalize()
        test_end = pd.to_datetime(test_end_date).normalize()

        train_mask = dates < test_start
        test_mask = (dates >= test_start) & (dates <= test_end)

        return train_mask.to_numpy(), test_mask.to_numpy()

    unique_dates = np.array(sorted(dates.unique()))

    if split_strategy == "chronological":
        n_train_days = int(len(unique_dates) * train_fraction)

        train_dates = unique_dates[:n_train_days]
        test_dates = unique_dates[n_train_days:]

        train_mask = dates.isin(train_dates)
        test_mask = dates.isin(test_dates)

        return train_mask.to_numpy(), test_mask.to_numpy()

    if split_strategy == "random_day":
        rng = np.random.default_rng(random_seed)

        shuffled_dates = unique_dates.copy()
        rng.shuffle(shuffled_dates)

        n_train_days = int(len(shuffled_dates) * train_fraction)

        train_dates = shuffled_dates[:n_train_days]
        test_dates = shuffled_dates[n_train_days:]

        train_mask = dates.isin(train_dates)
        test_mask = dates.isin(test_dates)

        return train_mask.to_numpy(), test_mask.to_numpy()

    raise ValueError(f"Unknown split_strategy: {split_strategy}")


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
    ax.set_title("Window Poisson GLM — daily totals")
    ax.legend()
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

    # ---- Data ----------------------------------------------------------------
    print("Loading data …")
    df = load_parquet_years(cfg["parquet_path"], cfg.get("years"))
    df["date"] = pd.to_datetime(df["date"])

    print(f"  Raw data shape: {df.shape}")

    df_win, feature_cols = build_window_dataframe(df, cfg)

    print(f"  Windowed data shape: {df_win.shape}")
    print(f"  Number of features: {len(feature_cols)}")
    print("  Features:")
    for col in feature_cols:
        print(f"    - {col}")

    reference_dates = df["date"].unique()
    train_mask, test_mask = make_split_masks(
        dates=df_win["date"],
        split_strategy=cfg.get("split_strategy", "chronological"),
        train_fraction=cfg.get("train_fraction", 0.9),
        random_seed=cfg.get("data_seed", 42),
        test_start_date=cfg.get("test_start_date"),
        test_end_date=cfg.get("test_end_date"),
    )

    target_col = cfg["target_col"]

    X = df_win[feature_cols].values.astype(np.float32)
    y = df_win[target_col].values.astype(np.float32)
    dates = df_win["date"].values

    X_train = X[train_mask]
    y_train = y[train_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]
    test_dates = dates[test_mask]

    print(f"  Train: {X_train.shape[0]:,} obs   Test: {X_test.shape[0]:,} obs")

    # ---- Standardization -----------------------------------------------------
    scaler = None
    if cfg.get("standardize_features", True):
        scaler = StandardScaler()
        X_train_model = scaler.fit_transform(X_train)
        X_test_model = scaler.transform(X_test)
    else:
        X_train_model = X_train
        X_test_model = X_test

    # ---- Model ---------------------------------------------------------------
    print("Training PoissonRegressor …")
    model = PoissonRegressor(
        alpha=float(cfg["poisson_alpha"]),
        max_iter=int(cfg["max_iter"]),
    )
    model.fit(X_train_model, y_train)

    rate_mean_test = model.predict(X_test_model)
    rate_mean_test = np.clip(rate_mean_test, 0, None)

    # ---- Metrics -------------------------------------------------------------
    metrics = evaluate_metrics(y_test, rate_mean_test, test_dates)

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

    print(f"  Daily delta correlation:     {metrics['daily_delta_corr']:.4f}")
    print(f"  Daily direction accuracy:    {metrics['daily_direction_accuracy_moving']:.4f}")

    three_day_report = evaluate_first_three_days(y_test, rate_mean_test, test_dates)
    if three_day_report is not None:
        with open(out_dir / "metrics_first_3_days.yaml", "w") as f:
            yaml.safe_dump(three_day_report, f, sort_keys=False)


    metrics_path = out_dir / "metrics.yaml"
    with open(metrics_path, "w") as f:
        yaml.dump(metrics, f, sort_keys=False)

    print(f"Metrics saved → {metrics_path}")

    # ---- Save coefficients ---------------------------------------------------
    coef_path = out_dir / "coefficients.yaml"
    coef_dict = {
        "intercept": float(model.intercept_),
        "coefficients": {
            name: float(value) for name, value in zip(feature_cols, model.coef_)
        },
    }
    with open(coef_path, "w") as f:
        yaml.dump(coef_dict, f, sort_keys=False)

    print(f"Coefficients saved → {coef_path}")

    # ---- Plot ----------------------------------------------------------------
    plot_daily_predictions(
        test_dates=test_dates,
        y_true=y_test,
        rate_mean=rate_mean_test,
        save_path=plots_dir / "daily_timeseries.png",
    )

    print(f"Plot saved → {plots_dir / 'daily_timeseries.png'}")
    print("Done.")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config/14_window_poisson_glm.yaml",
        help="Path to YAML config file.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    main(cfg)
