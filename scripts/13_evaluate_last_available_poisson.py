"""
Evaluate the Last Available Poisson baseline.

For each test observation, this baseline predicts the Poisson rate as the last
observed count in the same spatial cell before the target date:

    lambda_hat(lon, lat, t) = y(lon, lat, previous available date)

If no previous observation exists for that cell, it falls back to the global
mean count computed on the training set.

Results are saved under:

    reports/last_available_poisson/<run_name>/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
import time


# ---- Make src importable when script is run from the project root ----------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.data_pp_lgcp import prepare_data
from src.models.metrics_lgcp import evaluate_metrics


# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = {
    # --- data ---
    "years": 2024,
    "parquet_path": "data/processed/cpr_gfw",
    "train_fraction": 0.9,
    "data_seed": 42,
    # --- output ---
    "report_root": "reports",
    "model_family": "last_available_poisson",
    "run_name": "last_available_debug",
}


def load_config(config_path=None):
    cfg = DEFAULT_CONFIG.copy()

    if config_path is not None:
        with open(config_path, "r") as f:
            shared_cfg = yaml.safe_load(f)

        if shared_cfg is not None:
            # Import shared data settings from the LGCP config.
            for key in ["parquet_path", "train_fraction", "data_seed", "report_root"]:
                if key in shared_cfg:
                    cfg[key] = shared_cfg[key]

            # Import baseline-specific run name, if available.
            if "last_available_run_name" in shared_cfg:
                cfg["run_name"] = shared_cfg["last_available_run_name"]

            # Keep track of the source config for reproducibility.
            cfg["data_config_path"] = config_path

    return cfg


def get_day_level_split_masks(
    df: pd.DataFrame,
    train_fraction: float,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reproduce the same day-level train/test split used in prepare_data().
    """
    unique_dates = np.sort(df["date"].unique())

    rng = np.random.default_rng(random_seed)
    shuffled = rng.permutation(unique_dates)

    n_train = int(train_fraction * len(unique_dates))
    train_dates = set(shuffled[:n_train])
    test_dates = set(shuffled[n_train:])

    train_mask = df["date"].isin(train_dates).values
    test_mask = df["date"].isin(test_dates).values

    return train_mask, test_mask


def build_last_available_predictions(
    df: pd.DataFrame,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
) -> np.ndarray:
    """
    Build last-available predictions for the test set.

    For each spatial cell, the prediction for date t is the previous observed
    count in the same cell. If unavailable, use the global train mean.
    """
    df_work = df.copy()
    df_work["date"] = pd.to_datetime(df_work["date"])

    global_train_mean = float(df_work.loc[train_mask, "ais_vessels_count"].mean())

    # Sort by spatial cell and time, then shift within each cell.
    df_work = df_work.sort_values(["longitude", "latitude", "date"]).copy()

    df_work["lambda_last_available"] = df_work.groupby(["longitude", "latitude"])[
        "ais_vessels_count"
    ].shift(1)

    # Fallback for the first available date in each cell.
    df_work["lambda_last_available"] = (
        df_work["lambda_last_available"].fillna(global_train_mean).clip(lower=0)
    )

    # Restore original row order so it matches prepare_data() output order.
    df_work = df_work.sort_index()

    rate_mean_test = df_work.loc[test_mask, "lambda_last_available"].values.astype(
        np.float32
    )

    return rate_mean_test


def plot_daily_predictions(
    test_dates,
    y_true,
    rate_mean,
    save_path=None,
):
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
    ax.set_title("Last Available Poisson baseline — daily totals")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def main(cfg: dict):
    # ---- Output folders ------------------------------------------------------
    #out_dir = Path(cfg["report_root"]) / cfg["model_family"] / cfg["run_name"]
    out_dir = (Path(cfg["report_root"]) / cfg["model_family"] / f"{cfg['run_name']}_{cfg['years']}_{time.strftime('%Y%m%d-%H%M%S')}")

    plots_dir = out_dir / "plots"

    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Save experiment parameters
    params_path = out_dir / "params.yaml"
    with open(params_path, "w") as f:
        yaml.dump(cfg, f, sort_keys=False)

    print(f"Experiment parameters saved → {params_path}")

    # ---- Data ----------------------------------------------------------------
    parquet_path = cfg["parquet_path"]+f"_{cfg['years']}.parquet"
    print("Loading data …")
    (
        train_coords,
        train_covs,
        train_y,
        test_coords,
        test_covs,
        test_y,
        scalers,
        df,
    ) = prepare_data(
        parquet_path,
        train_fraction=cfg["train_fraction"],
        random_seed=cfg["data_seed"],
    )

    df["date"] = pd.to_datetime(df["date"])

    train_mask, test_mask = get_day_level_split_masks(
        df,
        train_fraction=cfg["train_fraction"],
        random_seed=cfg["data_seed"],
    )

    test_df = df.loc[test_mask].copy()

    print(f"  Train: {train_y.shape[0]:,} obs   Test: {test_y.shape[0]:,} obs")

    # ---- Baseline prediction -------------------------------------------------
    rate_mean_test = build_last_available_predictions(
        df=df,
        train_mask=train_mask,
        test_mask=test_mask,
    )

    print("Last-available predictions computed.")

    # ---- Evaluation ----------------------------------------------------------
    test_dates = test_df["date"].values

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
        yaml.dump(metrics, f, sort_keys=False)

    print(f"Metrics saved → {metrics_path}")

    # ---- Plot ----------------------------------------------------------------
    plot_daily_predictions(
        test_dates=test_dates,
        y_true=test_y,
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
        default="config/train_basic_lgcp.yaml",
        help="Path to shared YAML config file.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(cfg)
    main(cfg)
