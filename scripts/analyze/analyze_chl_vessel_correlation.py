#!/usr/bin/env python3

from pathlib import Path
import argparse
import yaml

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_CONFIG = {
    "years": 2024,
    "parquet_path": "data/processed/cpr_gfw.parquet",
    "date_col": "date",
    "target_col": "ais_vessels_count",
    "chl_col": "chl",
    "thetao_col": "thetao",
    "report_root": "reports/chl_vessel_diagnostics",
    "run_name": "2024_daily_chl_ais",
    "max_lag": 30,
    "rolling_window": 7,
}


def resolve_parquet_path(parquet_path, years=None):
    path = Path(parquet_path)

    if path.exists():
        return path

    if years is None:
        raise FileNotFoundError(f"File not found: {path}")

    if isinstance(years, list):
        if len(years) != 1:
            raise NotImplementedError("For now, use a single year.")
        years = years[0]

    candidate = path.with_name(f"{path.stem}_{years}{path.suffix}")

    if candidate.exists():
        return candidate

    raise FileNotFoundError(
        f"Could not find parquet file.\nBase path: {path}\nCandidate: {candidate}"
    )


def load_config(config_path=None):
    cfg = dict(DEFAULT_CONFIG)

    if config_path is not None:
        with open(config_path, "r") as f:
            user_cfg = yaml.safe_load(f) or {}
        cfg.update(user_cfg)

    cfg["parquet_path"] = str(
        resolve_parquet_path(cfg["parquet_path"], cfg.get("years"))
    )

    return cfg


def pearson_corr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)

    if mask.sum() < 3:
        return np.nan

    x = x[mask]
    y = y[mask]

    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan

    return float(np.corrcoef(x, y)[0, 1])


def spearman_corr(x, y):
    x = pd.Series(x)
    y = pd.Series(y)

    mask = x.notna() & y.notna()

    if mask.sum() < 3:
        return np.nan

    return float(x[mask].corr(y[mask], method="spearman"))


def build_daily_df(df, cfg):
    date_col = cfg["date_col"]
    target_col = cfg["target_col"]
    chl_col = cfg["chl_col"]
    thetao_col = cfg.get("thetao_col")

    agg_dict = {
        "daily_vessels": (target_col, "sum"),
        "daily_chl_mean": (chl_col, "mean"),
        "daily_chl_median": (chl_col, "median"),
        "daily_chl_std": (chl_col, "std"),
    }

    if thetao_col is not None and thetao_col in df.columns:
        agg_dict["daily_thetao_mean"] = (thetao_col, "mean")

    if "is_weekend" in df.columns:
        agg_dict["is_weekend"] = ("is_weekend", "max")

    if "is_holiday" in df.columns:
        agg_dict["is_holiday"] = ("is_holiday", "max")

    daily = df.groupby(date_col).agg(**agg_dict).reset_index().sort_values(date_col)

    daily["daily_vessels_diff"] = daily["daily_vessels"].diff()
    daily["daily_chl_diff"] = daily["daily_chl_mean"].diff()

    w = int(cfg["rolling_window"])
    daily[f"daily_vessels_roll_mean_{w}"] = (
        daily["daily_vessels"].rolling(w, min_periods=1).mean()
    )
    daily[f"daily_chl_roll_mean_{w}"] = (
        daily["daily_chl_mean"].rolling(w, min_periods=1).mean()
    )

    daily[f"daily_vessels_anomaly_{w}"] = (
        daily["daily_vessels"] - daily[f"daily_vessels_roll_mean_{w}"]
    )
    daily[f"daily_chl_anomaly_{w}"] = (
        daily["daily_chl_mean"] - daily[f"daily_chl_roll_mean_{w}"]
    )

    return daily


def compute_lag_correlations(daily, max_lag):
    rows = []

    for lag in range(-max_lag, max_lag + 1):
        # Positive lag:
        # corr(vessels_t, chl_{t-lag})
        chl_shifted = daily["daily_chl_mean"].shift(lag)

        rows.append(
            {
                "lag": lag,
                "meaning": (
                    "chl_leads_vessels"
                    if lag > 0
                    else "same_day"
                    if lag == 0
                    else "vessels_lead_chl"
                ),
                "pearson_vessels_chl": pearson_corr(
                    daily["daily_vessels"],
                    chl_shifted,
                ),
                "spearman_vessels_chl": spearman_corr(
                    daily["daily_vessels"],
                    chl_shifted,
                ),
            }
        )

    return pd.DataFrame(rows)


def compute_diff_lag_correlations(daily, max_lag):
    rows = []

    for lag in range(-max_lag, max_lag + 1):
        chl_diff_shifted = daily["daily_chl_diff"].shift(lag)

        rows.append(
            {
                "lag": lag,
                "meaning": (
                    "delta_chl_leads_delta_vessels"
                    if lag > 0
                    else "same_day_delta"
                    if lag == 0
                    else "delta_vessels_lead_delta_chl"
                ),
                "pearson_delta_vessels_delta_chl": pearson_corr(
                    daily["daily_vessels_diff"],
                    chl_diff_shifted,
                ),
                "spearman_delta_vessels_delta_chl": spearman_corr(
                    daily["daily_vessels_diff"],
                    chl_diff_shifted,
                ),
            }
        )

    return pd.DataFrame(rows)


def plot_daily_series(daily, save_path):
    fig, ax1 = plt.subplots(figsize=(14, 6))

    ax1.plot(
        daily["date"],
        daily["daily_vessels"],
        label="Daily AIS vessels",
    )
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Daily AIS vessels")

    ax2 = ax1.twinx()
    ax2.plot(
        daily["date"],
        daily["daily_chl_mean"],
        linestyle="--",
        label="Daily mean chlorophyll",
    )
    ax2.set_ylabel("Daily mean chlorophyll")

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()

    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")
    ax1.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_lag_correlations(lag_df, save_path):
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        lag_df["lag"],
        lag_df["pearson_vessels_chl"],
        marker="o",
        label="Pearson",
    )
    ax.plot(
        lag_df["lag"],
        lag_df["spearman_vessels_chl"],
        marker="o",
        label="Spearman",
    )

    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_xlabel("Lag")
    ax.set_ylabel("Correlation")
    ax.set_title("Lag correlation: vessels_t vs chlorophyll_{t-lag}")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_diff_lag_correlations(diff_lag_df, save_path):
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        diff_lag_df["lag"],
        diff_lag_df["pearson_delta_vessels_delta_chl"],
        marker="o",
        label="Pearson diff",
    )
    ax.plot(
        diff_lag_df["lag"],
        diff_lag_df["spearman_delta_vessels_delta_chl"],
        marker="o",
        label="Spearman diff",
    )

    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_xlabel("Lag")
    ax.set_ylabel("Correlation")
    ax.set_title("Lag correlation: Δvessels_t vs Δchlorophyll_{t-lag}")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional YAML config.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    out_dir = Path(cfg["report_root"]) / cfg["run_name"]
    plots_dir = out_dir / "plots"

    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading: {cfg['parquet_path']}")
    df = pd.read_parquet(cfg["parquet_path"])
    df[cfg["date_col"]] = pd.to_datetime(df[cfg["date_col"]]).dt.normalize()

    daily = build_daily_df(df, cfg)

    max_lag = int(cfg["max_lag"])

    lag_df = compute_lag_correlations(daily, max_lag=max_lag)
    diff_lag_df = compute_diff_lag_correlations(daily, max_lag=max_lag)

    daily.to_csv(out_dir / "daily_chl_vessels.csv", index=False)
    lag_df.to_csv(out_dir / "lag_correlations.csv", index=False)
    diff_lag_df.to_csv(out_dir / "diff_lag_correlations.csv", index=False)

    plot_daily_series(
        daily,
        save_path=plots_dir / "daily_vessels_chl.png",
    )
    plot_lag_correlations(
        lag_df,
        save_path=plots_dir / "lag_correlations.png",
    )
    plot_diff_lag_correlations(
        diff_lag_df,
        save_path=plots_dir / "diff_lag_correlations.png",
    )

    with open(out_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    print("\nTop Pearson correlations:")
    print(
        lag_df.sort_values(
            "pearson_vessels_chl",
            key=lambda s: s.abs(),
            ascending=False,
        )
        .head(10)
        .to_string(index=False)
    )

    print("\nTop Spearman correlations:")
    print(
        lag_df.sort_values(
            "spearman_vessels_chl",
            key=lambda s: s.abs(),
            ascending=False,
        )
        .head(10)
        .to_string(index=False)
    )

    print("\nTop Pearson correlations on daily differences:")
    print(
        diff_lag_df.sort_values(
            "pearson_delta_vessels_delta_chl",
            key=lambda s: s.abs(),
            ascending=False,
        )
        .head(10)
        .to_string(index=False)
    )

    print(f"\nSaved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
