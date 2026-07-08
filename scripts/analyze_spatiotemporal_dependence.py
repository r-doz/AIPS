import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def make_output_dir(cfg: dict) -> tuple[Path, Path]:
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    out_dir = Path(cfg["report_root"]) / f"{cfg['run_name']}_{timestamp}"
    plots_dir = out_dir / "plots"

    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    return out_dir, plots_dir


def save_yaml(obj: dict, path: Path):
    with open(path, "w") as f:
        yaml.dump(obj, f, sort_keys=False)


def load_and_prepare_data(cfg: dict) -> pd.DataFrame:
    parquet_path = Path(cfg["parquet_path"])

    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

    df = pd.read_parquet(parquet_path).copy()

    required_cols = [
        cfg["date_col"],
        cfg["lon_col"],
        cfg["lat_col"],
        cfg["target_col"],
    ]

    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in parquet: {missing_cols}")

    df[cfg["date_col"]] = pd.to_datetime(df[cfg["date_col"]])
    df = df.sort_values([cfg["date_col"], cfg["lon_col"], cfg["lat_col"]]).reset_index(
        drop=True
    )

    target_col = cfg["target_col"]

    df["y"] = df[target_col].astype(float)
    df["log1p_y"] = np.log1p(df["y"])

    return df


def compute_data_summary(df: pd.DataFrame, cfg: dict) -> dict:
    date_col = cfg["date_col"]
    lon_col = cfg["lon_col"]
    lat_col = cfg["lat_col"]
    target_col = cfg["target_col"]

    n_dates = df[date_col].nunique()
    n_cells = df[[lon_col, lat_col]].drop_duplicates().shape[0]

    summary = {
        "n_rows": int(df.shape[0]),
        "n_dates": int(n_dates),
        "n_cells": int(n_cells),
        "date_min": str(df[date_col].min().date()),
        "date_max": str(df[date_col].max().date()),
        "target_col": target_col,
        "target_sum": float(df[target_col].sum()),
        "target_mean": float(df[target_col].mean()),
        "target_std": float(df[target_col].std()),
        "target_min": float(df[target_col].min()),
        "target_max": float(df[target_col].max()),
        "zero_fraction": float((df[target_col] == 0).mean()),
    }

    return summary


def compute_daily_total_acf(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    date_col = cfg["date_col"]
    lags = cfg["temporal_lags"]

    daily = (
        df.groupby(date_col)
        .agg(
            daily_total_y=("y", "sum"),
            daily_total_log1p_y=("log1p_y", "sum"),
        )
        .sort_index()
    )

    rows = []

    for series_name in ["daily_total_y", "daily_total_log1p_y"]:
        series = daily[series_name]

        for lag in lags:
            corr = series.corr(series.shift(lag))

            rows.append(
                {
                    "series": series_name,
                    "lag_days": int(lag),
                    "correlation": float(corr) if pd.notna(corr) else np.nan,
                    "n_pairs": int(series.shift(lag).notna().sum()),
                }
            )

    return pd.DataFrame(rows)


def plot_daily_total_acf(acf_df: pd.DataFrame, save_path: Path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))

    for series_name, group in acf_df.groupby("series"):
        group = group.sort_values("lag_days")
        ax.plot(
            group["lag_days"],
            group["correlation"],
            marker="o",
            label=series_name,
        )

    ax.axhline(0.0, linestyle="--", linewidth=1)
    ax.set_xlabel("Lag (days)")
    ax.set_ylabel("Correlation")
    ax.set_title("Daily total autocorrelation")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def compute_temporal_acf_by_cell(
    df: pd.DataFrame, cfg: dict
) -> tuple[pd.DataFrame, pd.DataFrame]:
    date_col = cfg["date_col"]
    lon_col = cfg["lon_col"]
    lat_col = cfg["lat_col"]
    lags = cfg["temporal_lags"]
    transforms = cfg["transforms"]

    rows_by_cell = []

    for transform in transforms:
        pivot = df.pivot_table(
            index=date_col,
            columns=[lon_col, lat_col],
            values=transform,
            aggfunc="sum",
        ).sort_index()

        for lag in lags:
            shifted = pivot.shift(lag)
            corrs = pivot.corrwith(shifted, axis=0)

            for (lon, lat), corr in corrs.items():
                rows_by_cell.append(
                    {
                        "transform": transform,
                        "lag_days": int(lag),
                        "longitude": float(lon),
                        "latitude": float(lat),
                        "correlation": float(corr) if pd.notna(corr) else np.nan,
                    }
                )

    acf_by_cell = pd.DataFrame(rows_by_cell)

    acf_summary = (
        acf_by_cell.groupby(["transform", "lag_days"])
        .agg(
            mean_correlation=("correlation", "mean"),
            median_correlation=("correlation", "median"),
            q25_correlation=("correlation", lambda x: x.quantile(0.25)),
            q75_correlation=("correlation", lambda x: x.quantile(0.75)),
            n_cells=("correlation", "count"),
        )
        .reset_index()
    )

    return acf_by_cell, acf_summary


def plot_temporal_acf_summary(acf_summary: pd.DataFrame, save_path: Path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))

    for transform, group in acf_summary.groupby("transform"):
        group = group.sort_values("lag_days")

        x = group["lag_days"].to_numpy()
        median = group["median_correlation"].to_numpy()
        q25 = group["q25_correlation"].to_numpy()
        q75 = group["q75_correlation"].to_numpy()

        ax.plot(x, median, marker="o", label=f"{transform} median")
        ax.fill_between(x, q25, q75, alpha=0.2)

    ax.axhline(0.0, linestyle="--", linewidth=1)
    ax.set_xlabel("Lag (days)")
    ax.set_ylabel("Correlation")
    ax.set_title("Temporal autocorrelation per cell")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def compute_spatial_variogram(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    date_col = cfg["date_col"]
    lon_col = cfg["lon_col"]
    lat_col = cfg["lat_col"]
    transforms = cfg["transforms"]

    spatial_cfg = cfg["spatial"]
    n_bins = int(spatial_cfg["n_distance_bins"])
    max_pairs_per_day = int(spatial_cfg["max_pairs_per_day"])
    random_seed = int(spatial_cfg["random_seed"])

    rng = np.random.default_rng(random_seed)

    rows = []

    # Spatial points are fixed on the grid
    unique_points = (
        df[[lon_col, lat_col]]
        .drop_duplicates()
        .sort_values([lon_col, lat_col])
        .reset_index(drop=True)
    )

    coords = unique_points[[lon_col, lat_col]].to_numpy(dtype=float)
    n_points = coords.shape[0]

    # All possible spatial pairs i < j
    pair_i, pair_j = np.triu_indices(n_points, k=1)

    lon_i = coords[pair_i, 0]
    lat_i = coords[pair_i, 1]
    lon_j = coords[pair_j, 0]
    lat_j = coords[pair_j, 1]

    # Approximate distance in km
    mean_lat_rad = np.deg2rad((lat_i + lat_j) / 2.0)
    dx = (lon_i - lon_j) * 111.320 * np.cos(mean_lat_rad)
    dy = (lat_i - lat_j) * 110.574
    distances_km = np.sqrt(dx**2 + dy**2)

    max_dist = distances_km.max()
    bin_edges = np.linspace(0.0, max_dist, n_bins + 1)
    bin_ids = np.digitize(distances_km, bin_edges, right=False) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    point_index = {
        (row[lon_col], row[lat_col]): idx for idx, row in unique_points.iterrows()
    }

    for transform in transforms:
        print(f"  Spatial variogram transform: {transform}")

        for date, day_df in df.groupby(date_col):
            day_df = day_df.copy()

            values = np.full(n_points, np.nan, dtype=float)

            for _, row in day_df.iterrows():
                key = (row[lon_col], row[lat_col])
                idx = point_index[key]
                values[idx] = float(row[transform])

            valid_pair_mask = np.isfinite(values[pair_i]) & np.isfinite(values[pair_j])
            valid_indices = np.where(valid_pair_mask)[0]

            if valid_indices.size == 0:
                continue

            if valid_indices.size > max_pairs_per_day:
                valid_indices = rng.choice(
                    valid_indices,
                    size=max_pairs_per_day,
                    replace=False,
                )

            diffs = values[pair_i[valid_indices]] - values[pair_j[valid_indices]]
            semivariances = 0.5 * diffs**2

            for b in range(n_bins):
                mask = bin_ids[valid_indices] == b
                if not np.any(mask):
                    continue

                rows.append(
                    {
                        "transform": transform,
                        "date": str(pd.to_datetime(date).date()),
                        "distance_bin": int(b),
                        "distance_min_km": float(bin_edges[b]),
                        "distance_max_km": float(bin_edges[b + 1]),
                        "distance_mid_km": float(
                            0.5 * (bin_edges[b] + bin_edges[b + 1])
                        ),
                        "semivariance_mean": float(np.mean(semivariances[mask])),
                        "semivariance_median": float(np.median(semivariances[mask])),
                        "n_pairs": int(mask.sum()),
                    }
                )

    variogram_daily = pd.DataFrame(rows)

    variogram_summary = (
        variogram_daily.groupby(
            [
                "transform",
                "distance_bin",
                "distance_min_km",
                "distance_max_km",
                "distance_mid_km",
            ]
        )
        .agg(
            semivariance_mean=("semivariance_mean", "mean"),
            semivariance_median=("semivariance_median", "median"),
            q25_semivariance=("semivariance_mean", lambda x: x.quantile(0.25)),
            q75_semivariance=("semivariance_mean", lambda x: x.quantile(0.75)),
            total_pairs=("n_pairs", "sum"),
            n_days=("date", "nunique"),
        )
        .reset_index()
    )

    return variogram_daily, variogram_summary


def plot_spatial_variogram_summary(variogram_summary: pd.DataFrame, save_path: Path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))

    for transform, group in variogram_summary.groupby("transform"):
        group = group.sort_values("distance_mid_km")

        x = group["distance_mid_km"].to_numpy()
        y = group["semivariance_mean"].to_numpy()
        q25 = group["q25_semivariance"].to_numpy()
        q75 = group["q75_semivariance"].to_numpy()

        ax.plot(x, y, marker="o", label=transform)
        ax.fill_between(x, q25, q75, alpha=0.2)

    ax.set_xlabel("Spatial distance (km)")
    ax.set_ylabel("Semivariance")
    ax.set_title("Spatial variogram")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def compute_spatiotemporal_variogram(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    date_col = cfg["date_col"]
    lon_col = cfg["lon_col"]
    lat_col = cfg["lat_col"]
    transforms = cfg["transforms"]

    st_cfg = cfg["spatiotemporal"]
    temporal_lags = st_cfg["temporal_lags"]
    n_bins = int(st_cfg["n_distance_bins"])
    max_pairs_per_lag = int(st_cfg["max_pairs_per_lag"])
    random_seed = int(st_cfg["random_seed"])

    rng = np.random.default_rng(random_seed)

    rows = []

    unique_points = (
        df[[lon_col, lat_col]]
        .drop_duplicates()
        .sort_values([lon_col, lat_col])
        .reset_index(drop=True)
    )

    coords = unique_points[[lon_col, lat_col]].to_numpy(dtype=float)
    n_points = coords.shape[0]

    # All ordered spatial pairs, including same-cell pairs.
    pair_i, pair_j = np.meshgrid(
        np.arange(n_points),
        np.arange(n_points),
        indexing="ij",
    )
    pair_i = pair_i.ravel()
    pair_j = pair_j.ravel()

    lon_i = coords[pair_i, 0]
    lat_i = coords[pair_i, 1]
    lon_j = coords[pair_j, 0]
    lat_j = coords[pair_j, 1]

    mean_lat_rad = np.deg2rad((lat_i + lat_j) / 2.0)
    dx = (lon_i - lon_j) * 111.320 * np.cos(mean_lat_rad)
    dy = (lat_i - lat_j) * 110.574
    distances_km = np.sqrt(dx**2 + dy**2)

    max_dist = distances_km.max()
    bin_edges = np.linspace(0.0, max_dist, n_bins + 1)
    bin_ids = np.digitize(distances_km, bin_edges, right=False) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    for transform in transforms:
        print(f"  Spatio-temporal variogram transform: {transform}")

        pivot = df.pivot_table(
            index=date_col,
            columns=[lon_col, lat_col],
            values=transform,
            aggfunc="sum",
        ).sort_index()

        values = pivot.to_numpy(dtype=float)

        for lag in temporal_lags:
            lag = int(lag)

            if lag >= values.shape[0]:
                continue

            x_t = values[:-lag, :]
            x_t_lag = values[lag:, :]

            n_times = x_t.shape[0]
            n_spatial_pairs = pair_i.shape[0]
            total_pairs = n_times * n_spatial_pairs

            sample_size = min(max_pairs_per_lag, total_pairs)

            sampled_time_idx = rng.integers(
                low=0,
                high=n_times,
                size=sample_size,
            )

            sampled_pair_idx = rng.integers(
                low=0,
                high=n_spatial_pairs,
                size=sample_size,
            )

            sampled_i = pair_i[sampled_pair_idx]
            sampled_j = pair_j[sampled_pair_idx]

            z1 = x_t[sampled_time_idx, sampled_i]
            z2 = x_t_lag[sampled_time_idx, sampled_j]

            valid_mask = np.isfinite(z1) & np.isfinite(z2)

            if not np.any(valid_mask):
                continue

            sampled_pair_idx = sampled_pair_idx[valid_mask]
            z1 = z1[valid_mask]
            z2 = z2[valid_mask]

            semivariances = 0.5 * (z1 - z2) ** 2
            sampled_bin_ids = bin_ids[sampled_pair_idx]

            for b in range(n_bins):
                mask = sampled_bin_ids == b

                if not np.any(mask):
                    continue

                rows.append(
                    {
                        "transform": transform,
                        "lag_days": lag,
                        "distance_bin": int(b),
                        "distance_min_km": float(bin_edges[b]),
                        "distance_max_km": float(bin_edges[b + 1]),
                        "distance_mid_km": float(
                            0.5 * (bin_edges[b] + bin_edges[b + 1])
                        ),
                        "semivariance_mean": float(np.mean(semivariances[mask])),
                        "semivariance_median": float(np.median(semivariances[mask])),
                        "q25_semivariance": float(
                            np.quantile(semivariances[mask], 0.25)
                        ),
                        "q75_semivariance": float(
                            np.quantile(semivariances[mask], 0.75)
                        ),
                        "n_pairs": int(mask.sum()),
                    }
                )

    return pd.DataFrame(rows)


def plot_spatiotemporal_variogram(variogram_df: pd.DataFrame, plots_dir: Path):
    import matplotlib.pyplot as plt

    for transform, transform_df in variogram_df.groupby("transform"):
        fig, ax = plt.subplots(figsize=(8, 5))

        for lag, group in transform_df.groupby("lag_days"):
            group = group.sort_values("distance_mid_km")

            ax.plot(
                group["distance_mid_km"],
                group["semivariance_mean"],
                marker="o",
                label=f"lag {lag} days",
            )

        ax.set_xlabel("Spatial distance (km)")
        ax.set_ylabel("Semivariance")
        ax.set_title(f"Spatio-temporal variogram — {transform}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.tight_layout()

        save_path = plots_dir / f"spatiotemporal_variogram_{transform}.png"
        fig.savefig(save_path, dpi=200)
        plt.close(fig)

        print(f"Spatio-temporal variogram plot saved → {save_path}")


def plot_spatial_pair_counts(variogram_summary: pd.DataFrame, save_path: Path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))

    for transform, group in variogram_summary.groupby("transform"):
        group = group.sort_values("distance_mid_km")

        ax.plot(
            group["distance_mid_km"],
            group["total_pairs"],
            marker="o",
            label=transform,
        )

    ax.set_xlabel("Spatial distance (km)")
    ax.set_ylabel("Number of pairs")
    ax.set_title("Spatial variogram pair counts")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_spatiotemporal_pair_counts(variogram_df: pd.DataFrame, plots_dir: Path):
    import matplotlib.pyplot as plt

    for transform, transform_df in variogram_df.groupby("transform"):
        fig, ax = plt.subplots(figsize=(8, 5))

        for lag, group in transform_df.groupby("lag_days"):
            group = group.sort_values("distance_mid_km")

            ax.plot(
                group["distance_mid_km"],
                group["n_pairs"],
                marker="o",
                label=f"lag {lag} days",
            )

        ax.set_xlabel("Spatial distance (km)")
        ax.set_ylabel("Number of sampled pairs")
        ax.set_title(f"Spatio-temporal variogram pair counts — {transform}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.tight_layout()

        save_path = plots_dir / f"spatiotemporal_variogram_pair_counts_{transform}.png"
        fig.savefig(save_path, dpi=200)
        plt.close(fig)

        print(f"Spatio-temporal pair-count plot saved → {save_path}")


# -------------------------MAIN----------------------
def main(cfg: dict):
    out_dir, plots_dir = make_output_dir(cfg)

    save_yaml(cfg, out_dir / "params.yaml")
    print(f"Parameters saved → {out_dir / 'params.yaml'}")

    print("Loading data ...")
    df = load_and_prepare_data(cfg)

    print("Computing data summary ...")
    summary = compute_data_summary(df, cfg)
    save_yaml(summary, out_dir / "data_summary.yaml")

    print("Computing daily total autocorrelation ...")
    daily_acf = compute_daily_total_acf(df, cfg)

    if cfg.get("save_csv", True):
        daily_acf_path = out_dir / "daily_total_acf.csv"
        daily_acf.to_csv(daily_acf_path, index=False)
        print(f"Daily total ACF saved → {daily_acf_path}")

    if cfg.get("save_plots", True):
        daily_acf_plot_path = plots_dir / "daily_total_acf.png"
        plot_daily_total_acf(daily_acf, daily_acf_plot_path)
        print(f"Daily total ACF plot saved → {daily_acf_plot_path}")

    print("Computing temporal autocorrelation per cell ...")
    temporal_acf_by_cell, temporal_acf_summary = compute_temporal_acf_by_cell(df, cfg)

    if cfg.get("save_csv", True):
        temporal_by_cell_path = out_dir / "temporal_acf_by_cell.csv"
        temporal_summary_path = out_dir / "temporal_acf_summary.csv"

        temporal_acf_by_cell.to_csv(temporal_by_cell_path, index=False)
        temporal_acf_summary.to_csv(temporal_summary_path, index=False)

        print(f"Temporal ACF by cell saved → {temporal_by_cell_path}")
        print(f"Temporal ACF summary saved → {temporal_summary_path}")

    if cfg.get("save_plots", True):
        temporal_acf_plot_path = plots_dir / "temporal_acf_summary.png"
        plot_temporal_acf_summary(temporal_acf_summary, temporal_acf_plot_path)
        print(f"Temporal ACF summary plot saved → {temporal_acf_plot_path}")

    print("Computing spatial variogram ...")
    spatial_variogram_daily, spatial_variogram_summary = compute_spatial_variogram(
        df, cfg
    )

    if cfg.get("save_csv", True):
        spatial_daily_path = out_dir / "spatial_variogram_daily.csv"
        spatial_summary_path = out_dir / "spatial_variogram_summary.csv"

        spatial_variogram_daily.to_csv(spatial_daily_path, index=False)
        spatial_variogram_summary.to_csv(spatial_summary_path, index=False)

        print(f"Spatial variogram daily saved → {spatial_daily_path}")
        print(f"Spatial variogram summary saved → {spatial_summary_path}")

    if cfg.get("save_plots", True):
        spatial_variogram_plot_path = plots_dir / "spatial_variogram_summary.png"
        plot_spatial_variogram_summary(
            spatial_variogram_summary, spatial_variogram_plot_path
        )
        print(f"Spatial variogram plot saved → {spatial_variogram_plot_path}")

        spatial_pair_counts_path = plots_dir / "spatial_variogram_pair_counts.png"
        plot_spatial_pair_counts(spatial_variogram_summary, spatial_pair_counts_path)
        print(f"Spatial variogram pair-count plot saved → {spatial_pair_counts_path}")

    print("Computing spatio-temporal variogram ...")
    spatiotemporal_variogram = compute_spatiotemporal_variogram(df, cfg)

    if cfg.get("save_csv", True):
        st_variogram_path = out_dir / "spatiotemporal_variogram.csv"
        spatiotemporal_variogram.to_csv(st_variogram_path, index=False)
        print(f"Spatio-temporal variogram saved → {st_variogram_path}")

    if cfg.get("save_plots", True):
        plot_spatiotemporal_variogram(spatiotemporal_variogram, plots_dir)
        plot_spatiotemporal_pair_counts(spatiotemporal_variogram, plots_dir)

    # ------ summary ------
    print("\n=== Data summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

    print(f"\nData summary saved → {out_dir / 'data_summary.yaml'}")
    print(f"Output directory → {out_dir}")
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config/analyze_spatiotemporal_dependence.yaml",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    main(cfg)
