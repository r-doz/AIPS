#!/usr/bin/env python3

from pathlib import Path
import argparse
import math
import yaml
import sys

# ---- Make src importable when script is run from the project root ----------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import gammaln

from src.data.multi_year import load_parquet_years, years_label
from src.models.data_pp_lgcp import make_day_split_masks


DEFAULT_CONFIG = {
    # data
    "parquet_path": "data/processed/cpr_gfw.parquet",
    "years": None,
    "target_col": "ais_vessels_count",
    "date_col": "date",
    "cell_cols": ["longitude", "latitude"],
    # split
    "train_fraction": 0.9,
    "data_seed": 42,
    "split_strategy": "random_day",
    "test_start_date": None,
    "test_end_date": None,
    # model
    # Options:
    # - "frozen_train": mean per cell computed on train only
    # - "expanding": mean per cell updated day by day inside the test window
    "cell_mean_mode": "expanding",
    # fallback for cells with no history
    "fallback": "global_train_mean",
    # output
    "report_root": "reports",
    "model_family": "global_cell_mean_poisson",
    "run_name": None,
    # numerical
    "eps": 1e-8,
}


def deep_merge(base, updates):
    out = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(config_path):
    cfg = dict(DEFAULT_CONFIG)

    if config_path is not None:
        with open(config_path, "r") as f:
            user_cfg = yaml.safe_load(f) or {}

        cfg = deep_merge(cfg, user_cfg)

        # Optional script-specific run name.
        # This avoids accidentally saving with the LGCP run name.
        if "global_cell_mean_run_name" in user_cfg:
            cfg["run_name"] = user_cfg["global_cell_mean_run_name"]

        # Optional script-specific mode.
        if "global_cell_mean_mode" in user_cfg:
            cfg["cell_mean_mode"] = user_cfg["global_cell_mean_mode"]

    if cfg["run_name"] is None:
        mode = cfg["cell_mean_mode"]
        split = cfg["split_strategy"]
        year_prefix = years_label(cfg.get("years"))
        prefix = f"{year_prefix}_" if year_prefix else ""
        cfg["run_name"] = f"{prefix}{mode}_{split}_dataseed{cfg['data_seed']}"

    return cfg


def poisson_metrics(y_true, rate_mean, dates, eps=1e-8):
    y_true = np.asarray(y_true, dtype=float)
    rate_mean = np.asarray(rate_mean, dtype=float)
    rate_mean = np.clip(rate_mean, eps, None)

    ll_obs = y_true * np.log(rate_mean) - rate_mean - gammaln(y_true + 1.0)

    mae_obs = float(np.mean(np.abs(y_true - rate_mean)))
    rmse_obs = float(np.sqrt(np.mean((y_true - rate_mean) ** 2)))
    mean_ll_obs = float(np.mean(ll_obs))

    daily_df = pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "y_true": y_true,
            "rate_mean": rate_mean,
        }
    )

    daily_df = (
        daily_df.groupby("date", as_index=False)
        .agg(
            y_true=("y_true", "sum"),
            rate_mean=("rate_mean", "sum"),
        )
        .sort_values("date")
    )

    daily_rate = np.clip(daily_df["rate_mean"].values.astype(float), eps, None)
    daily_y = daily_df["y_true"].values.astype(float)

    ll_daily = daily_y * np.log(daily_rate) - daily_rate - gammaln(daily_y + 1.0)

    mae_daily = float(np.mean(np.abs(daily_y - daily_rate)))
    rmse_daily = float(np.sqrt(np.mean((daily_y - daily_rate) ** 2)))
    mean_ll_daily = float(np.mean(ll_daily))

    metrics = {
        "mean_ll_obs": mean_ll_obs,
        "mae_obs": mae_obs,
        "rmse_obs": rmse_obs,
        "mean_ll_daily": mean_ll_daily,
        "mae_daily": mae_daily,
        "rmse_daily": rmse_daily,
    }

    return metrics, daily_df


def build_frozen_train_cell_mean_predictions(
    df,
    train_mask,
    test_mask,
    target_col,
    cell_cols,
):
    train_df = df.loc[train_mask].copy()
    test_df = df.loc[test_mask].copy()

    global_train_mean = float(train_df[target_col].mean())

    cell_mean = train_df.groupby(cell_cols)[target_col].mean()

    preds = []
    for _, row in test_df.iterrows():
        key = tuple(row[col] for col in cell_cols)
        value = cell_mean.get(key, global_train_mean)
        preds.append(value)

    return np.asarray(preds, dtype=np.float32)


def build_expanding_cell_mean_predictions(
    df,
    train_mask,
    test_mask,
    target_col,
    date_col,
    cell_cols,
):
    """
    Online expanding baseline.

    For each test day d:
    - predict each cell using the historical mean of that cell before d;
    - after predicting day d, update the history with the true observations of day d.

    This uses test observations from previous test days, but never from the same day
    or from future test days.
    """

    df_work = df.copy()
    df_work[date_col] = pd.to_datetime(df_work[date_col]).dt.normalize()

    test_df = df_work.loc[test_mask].copy().sort_values(date_col)
    test_dates = sorted(test_df[date_col].unique())

    if len(test_dates) == 0:
        raise ValueError("Empty test set.")

    first_test_date = test_dates[0]

    # Start from training rows strictly before the first test date.
    # This avoids accidental leakage if the split is not perfectly chronological.
    initial_history_mask = train_mask & (df_work[date_col] < first_test_date)
    history_df = df_work.loc[initial_history_mask].copy()

    if history_df.empty:
        raise ValueError(
            "No historical training data before the first test date. "
            "Use frozen_train mode or choose a chronological/fixed test window."
        )

    global_sum = float(history_df[target_col].sum())
    global_count = int(history_df[target_col].shape[0])

    cell_sum = history_df.groupby(cell_cols)[target_col].sum().to_dict()
    cell_count = history_df.groupby(cell_cols)[target_col].count().to_dict()

    pred_by_index = {}

    for current_date in test_dates:
        current_rows = test_df[test_df[date_col] == current_date]

        global_mean = global_sum / max(global_count, 1)

        # Predict current date using only previous history.
        for idx, row in current_rows.iterrows():
            key = tuple(row[col] for col in cell_cols)

            if key in cell_sum and cell_count.get(key, 0) > 0:
                pred = cell_sum[key] / cell_count[key]
            else:
                pred = global_mean

            pred_by_index[idx] = pred

        # Update history with true observations from current date.
        for _, row in current_rows.iterrows():
            key = tuple(row[col] for col in cell_cols)
            y = float(row[target_col])

            cell_sum[key] = cell_sum.get(key, 0.0) + y
            cell_count[key] = cell_count.get(key, 0) + 1

            global_sum += y
            global_count += 1

    preds = np.asarray(
        [pred_by_index[idx] for idx in df_work.loc[test_mask].index],
        dtype=np.float32,
    )

    return preds


def plot_daily_timeseries(daily_df, save_path, title):
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        daily_df["date"],
        daily_df["y_true"],
        marker="o",
        label="Observed",
    )
    ax.plot(
        daily_df["date"],
        daily_df["rate_mean"],
        marker="o",
        label="Predicted",
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("Daily total vessels")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


# ---------------MAIN------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    parquet_path = cfg["parquet_path"]
    target_col = cfg["target_col"]
    date_col = cfg["date_col"]
    cell_cols = cfg["cell_cols"]
    mode = cfg["cell_mean_mode"]

    print("=" * 80)
    print("Global Cell Mean Poisson baseline")
    print("=" * 80)
    print(f"Parquet path: {parquet_path}")
    print(f"Years: {cfg.get('years')}")
    print(f"Target column: {target_col}")
    print(f"Cell columns: {cell_cols}")
    print(f"Mode: {mode}")
    print(f"Split strategy: {cfg['split_strategy']}")
    print(f"Test start date: {cfg.get('test_start_date')}")
    print(f"Test end date: {cfg.get('test_end_date')}")

    df = load_parquet_years(parquet_path, cfg.get("years"))
    df[date_col] = pd.to_datetime(df[date_col]).dt.normalize()

    train_mask, test_mask = make_day_split_masks(
        df=df,
        train_fraction=cfg["train_fraction"],
        random_seed=cfg["data_seed"],
        split_strategy=cfg["split_strategy"],
        test_start_date=cfg.get("test_start_date"),
        test_end_date=cfg.get("test_end_date"),
    )

    n_train = int(train_mask.sum())
    n_test = int(test_mask.sum())

    print(f"Train: {n_train} obs")
    print(f"Test:  {n_test} obs")

    if n_train == 0 or n_test == 0:
        raise ValueError("Train or test set is empty.")

    if mode == "frozen_train":
        rate_mean_test = build_frozen_train_cell_mean_predictions(
            df=df,
            train_mask=train_mask,
            test_mask=test_mask,
            target_col=target_col,
            cell_cols=cell_cols,
        )

    elif mode == "expanding":
        rate_mean_test = build_expanding_cell_mean_predictions(
            df=df,
            train_mask=train_mask,
            test_mask=test_mask,
            target_col=target_col,
            date_col=date_col,
            cell_cols=cell_cols,
        )

    else:
        raise ValueError(
            f"Unknown cell_mean_mode: {mode}. Use 'frozen_train' or 'expanding'."
        )

    test_df = df.loc[test_mask].copy()
    y_test = test_df[target_col].values.astype(float)
    test_dates = test_df[date_col].values

    metrics, daily_df = poisson_metrics(
        y_true=y_test,
        rate_mean=rate_mean_test,
        dates=test_dates,
        eps=cfg["eps"],
    )

    report_dir = Path(cfg["report_root"]) / cfg["model_family"] / cfg["run_name"]
    plots_dir = report_dir / "plots"
    report_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    with open(report_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    with open(report_dir / "metrics.yaml", "w") as f:
        yaml.safe_dump(metrics, f, sort_keys=False)

    pred_df = test_df[[date_col] + cell_cols + [target_col]].copy()
    pred_df["rate_mean"] = rate_mean_test
    pred_df.to_csv(report_dir / "predictions.csv", index=False)

    daily_df.to_csv(report_dir / "daily_predictions.csv", index=False)

    plot_daily_timeseries(
        daily_df=daily_df,
        save_path=plots_dir / "daily_timeseries_test_window.png",
        title=f"Global Cell Mean Poisson ({mode}) — test window",
    )

    print("\nMetrics")
    print("-" * 80)
    for key, value in metrics.items():
        print(f"{key}: {value}")

    print("\nSaved outputs")
    print("-" * 80)
    print(f"Report dir: {report_dir}")
    print(f"Metrics:    {report_dir / 'metrics.yaml'}")
    print(f"Predictions:{report_dir / 'predictions.csv'}")
    print(f"Plot:       {plots_dir / 'daily_timeseries_test_window.png'}")


if __name__ == "__main__":
    main()
