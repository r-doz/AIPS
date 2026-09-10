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

from src.data.multi_year import years_label
from src.models.data_pp_lgcp import prepare_data, make_day_split_masks
from src.models.metrics_lgcp import evaluate_metrics


# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = {
    # --- data ---
    "years": 2024,
    "parquet_path": "data/processed/cpr_gfw.parquet",
    "train_fraction": 0.9,
    "data_seed": 42,
    "split_strategy": "random_day",
    "test_start_date": None,
    "test_end_date": None,
    # --- output ---
    "report_root": "reports",
    "model_family": "last_available_poisson",
    "run_name": "debug",
    "last_available_mode": "frozen_train",
}


def load_config(config_path=None):
    cfg = DEFAULT_CONFIG.copy()

    if config_path is not None:
        with open(config_path, "r") as f:
            shared_cfg = yaml.safe_load(f)

        if shared_cfg is not None:
            # Import shared data and output settings, including batch-runner paths.
            for key in [
                "years",
                "parquet_path",
                "train_fraction",
                "data_seed",
                "split_strategy",
                "test_start_date",
                "test_end_date",
                "report_root",
                "model_family",
                "run_name",
                "last_available_mode",
            ]:
                if key in shared_cfg:
                    cfg[key] = shared_cfg[key]

            # Import baseline-specific run name, if available.
            if "last_available_run_name" in shared_cfg:
                cfg["run_name"] = shared_cfg["last_available_run_name"]

            # Keep track of the source config for reproducibility.
            cfg["data_config_path"] = config_path

    return cfg


def build_last_available_predictions(
    df: pd.DataFrame,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    mode: str = "frozen_train",
) -> np.ndarray:
    """
    Build last-available predictions for the test set.

    Modes:
    - frozen_train:
        For each spatial cell, use the last observed training value.
        The same value is used for the whole test window.
        This does not use true observations inside the test set.

    - one_step_ahead:
        For each spatial cell, use the previous observed value.
        Inside the test window, this means that Dec 2 can use true Dec 1,
        Dec 3 can use true Dec 2, and so on.
    """

    df_work = df.copy()
    df_work["date"] = pd.to_datetime(df_work["date"])

    target_col = "ais_vessels_count"
    cell_cols = ["longitude", "latitude"]

    global_train_mean = float(df_work.loc[train_mask, target_col].mean())

    if mode == "one_step_ahead":
        # Sort by spatial cell and time, then shift within each cell.
        df_work = df_work.sort_values(["longitude", "latitude", "date"]).copy()

        df_work["lambda_last_available"] = df_work.groupby(cell_cols)[target_col].shift(
            1
        )

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

    if mode == "frozen_train":
        train_df = df_work.loc[train_mask].copy()
        test_df = df_work.loc[test_mask].copy()

        train_df = train_df.sort_values(["longitude", "latitude", "date"])

        last_train_by_cell = train_df.groupby(cell_cols)[target_col].last()

        lambda_test = []

        for _, row in test_df.iterrows():
            key = (row["longitude"], row["latitude"])
            value = last_train_by_cell.get(key, global_train_mean)
            lambda_test.append(value)

        rate_mean_test = np.asarray(lambda_test, dtype=np.float32)
        rate_mean_test = np.clip(rate_mean_test, a_min=0.0, a_max=None)

        return rate_mean_test

    raise ValueError(
        f"Unknown last_available_mode: {mode}. Use 'frozen_train' or 'one_step_ahead'."
    )


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
    # out_dir = Path(cfg["report_root"]) / cfg["model_family"] / cfg["run_name"]
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
        cfg["parquet_path"],
        train_fraction=cfg["train_fraction"],
        random_seed=cfg["data_seed"],
        split_strategy=cfg.get("split_strategy", "random_day"),
        test_start_date=cfg.get("test_start_date"),
        test_end_date=cfg.get("test_end_date"),
        years=cfg.get("years"),
    )

    df["date"] = pd.to_datetime(df["date"])

    train_mask, test_mask = make_day_split_masks(
        df=df,
        train_fraction=cfg["train_fraction"],
        random_seed=cfg["data_seed"],
        split_strategy=cfg.get("split_strategy", "random_day"),
        test_start_date=cfg.get("test_start_date"),
        test_end_date=cfg.get("test_end_date"),
    )

    test_dates = df.loc[test_mask, "date"].values

    print(f"  Train: {train_y.shape[0]:,} obs   Test: {test_y.shape[0]:,} obs")

    # ---- Baseline prediction -------------------------------------------------
    print(f"Last available mode: {cfg.get('last_available_mode', 'frozen_train')}")

    rate_mean_test = build_last_available_predictions(
        df=df,
        train_mask=train_mask,
        test_mask=test_mask,
        mode=cfg.get("last_available_mode", "frozen_train"),
    )

    print("Last-available predictions computed.")

    # ---- Evaluation ----------------------------------------------------------
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
    main(cfg)
