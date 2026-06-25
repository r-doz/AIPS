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
import sys
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

from src.models.metrics_lgcp import evaluate_metrics


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
    # --- output ---
    "report_root": "reports",
    "model_family": "window_poisson_glm",
    "run_name": "2024_window_poisson_glm_W7_random_day",
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
    Prefix run_name with the selected year.

    Example
    -------
    years: 2024
    run_name: window_poisson_glm_W7_random_day

    becomes:

    run_name: 2024_window_poisson_glm_W7_random_day
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

    run_name = str(cfg["run_name"])

    if not run_name.startswith(f"{year}_"):
        cfg["run_name"] = f"{year}_{run_name}"

    return cfg


def load_config(config_path=None):
    cfg = DEFAULT_CONFIG.copy()

    if config_path is not None:
        with open(config_path, "r") as f:
            user_cfg = yaml.safe_load(f)

        if user_cfg is not None:
            cfg.update(user_cfg)

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
    covariate_cols = cfg["covariate_cols"]
    window_size = int(cfg["window_size"])

    required_cols = ["date", "longitude", "latitude", target_col] + covariate_cols

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

    lag_cols = []
    for k in range(1, window_size + 1):
        col = f"{target_col}_lag_{k}"
        df[col] = df.groupby(["longitude", "latitude"])[target_col].shift(k)
        lag_cols.append(col)

    if cfg.get("drop_incomplete_windows", True):
        df = df.dropna(subset=lag_cols).copy()
    else:
        df[lag_cols] = df[lag_cols].fillna(0.0)

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
    df_windowed: pd.DataFrame,
    cfg: dict,
    reference_dates: np.ndarray | pd.Series | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create train/test masks at day level.

    If reference_dates is provided, the split is created using those dates
    instead of the dates available after window construction. This is useful
    to keep the same split as the LGCP/baseline models even after dropping
    the first W days due to incomplete windows.
    """
    split_strategy = cfg["split_strategy"]
    train_fraction = float(cfg["train_fraction"])

    if reference_dates is None:
        unique_dates = np.sort(pd.to_datetime(df_windowed["date"].unique()))
    else:
        unique_dates = np.sort(pd.to_datetime(reference_dates).unique())

    n_train = int(train_fraction * len(unique_dates))

    if split_strategy == "random_day":
        rng = np.random.default_rng(cfg["data_seed"])
        shuffled = rng.permutation(unique_dates)
        train_dates = set(shuffled[:n_train])
        test_dates = set(shuffled[n_train:])

    elif split_strategy == "chronological":
        train_dates = set(unique_dates[:n_train])
        test_dates = set(unique_dates[n_train:])

    else:
        raise ValueError(
            f"Unknown split_strategy: {split_strategy}. "
            "Use 'random_day' or 'chronological'."
        )

    train_mask = df_windowed["date"].isin(train_dates).values
    test_mask = df_windowed["date"].isin(test_dates).values

    return train_mask, test_mask


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
    df = pd.read_parquet(cfg["parquet_path"])
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
        df_windowed=df_win,
        cfg=cfg,
        reference_dates=reference_dates,
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

    print("Daily-level metrics:")
    print(f"  Mean log-likelihood daily:   {metrics['mean_ll_daily']:.4f}")
    print(f"  MAE daily:                   {metrics['mae_daily']:.4f}")
    print(f"  RMSE daily:                  {metrics['rmse_daily']:.4f}")

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
