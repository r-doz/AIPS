"""
src/viz.py
Visualization utilities for the LGCP vessel-count model.

Fixed issues vs. original
--------------------------
1. Time-series plot: the loop now groups observations by their DATE (using
   the original df) rather than iterating over unique standardized time
   values and comparing against a separate `dates` array — the original
   code had a length-mismatch bug when duplicate normalized times existed.

2. t_norm inversion: uses t_scaler.inverse_transform() consistently for
   going from standardized → raw t_norm, then maps to datetime via
   t_min + t_norm * (t_max - t_min).  The original mixed two different
   normalization conventions.

3. Gulf coastline CSV path is passed as an argument (not hardcoded).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf
from scipy import ndimage
from matplotlib.path import Path
from pathlib import Path as FilePath

import torch


# ---------------------------------------------------------------------------
# Main overview plot
# ---------------------------------------------------------------------------


def plot_pp_overview(
    model,
    df: pd.DataFrame,
    coords: np.ndarray,
    covariates: np.ndarray,
    y_true: np.ndarray,
    meta: dict,
    scalers: dict,
    gulf_csv_path: str,
    num_samples: int = 200,
    target_date: str | None = None,
    test_coords: np.ndarray | None = None,
    cmap: str = "viridis",
    save_dir=None,
) -> None:
    """
    Two-panel figure:
      1. Time series of daily total ships (predicted vs. observed).
      2. Spatial maps of predicted intensity and observed counts for
         one selected date.

    Parameters
    ----------
    model         : trained SparseLGCP
    df            : original DataFrame (must contain 'date', 'longitude', 'latitude')
    coords        : [N, 3]  standardized (lon_std, lat_std, t_std)
    covariates    : [N, C]  standardized covariates
    y_true        : [N]     observed counts
    meta          : dict from compute_meta()
    scalers       : dict from prepare_data() — needs 't_scaler'
    gulf_csv_path : path to the Gulf of Venice boundary CSV
                    (columns: 'longitude', 'latitude')
    num_samples   : MC samples for rate prediction
    target_date   : 'YYYY-MM-DD' string; defaults to first date in meta['dates']
    test_coords   : if provided, test-date ticks are drawn on the time series
    cmap          : matplotlib colormap
    """
    # ---- Torch → numpy -------------------------------------------------------
    if torch.is_tensor(coords):
        coords = coords.detach().cpu().numpy()
    if torch.is_tensor(covariates):
        covariates = covariates.detach().cpu().numpy()
    if torch.is_tensor(y_true):
        y_true = y_true.detach().cpu().numpy()

    coords = np.asarray(coords, dtype=np.float32)
    covariates = np.asarray(covariates, dtype=np.float32)
    y_true = np.asarray(y_true, dtype=np.float32)

    # ---- Predict rates -------------------------------------------------------
    rate_mean, _, _ = model.predict_rate(coords, covariates, num_samples=num_samples)
    rate_mean = np.clip(rate_mean, 0, None)

    # =========================================================================
    # 1. Time series: daily totals
    # =========================================================================
    t_scaler = scalers["t_scaler"]

    # Invert standardized time → raw t_norm ∈ (0,1]
    t_std_col = coords[:, 2].reshape(-1, 1)
    t_norm_all = t_scaler.inverse_transform(t_std_col).ravel()

    t_min = pd.Timestamp(meta["t_min"])
    t_max = pd.Timestamp(meta["t_max"])
    span = t_max - t_min

    # Map t_norm → Timestamp
    dates_obs = t_min + pd.to_timedelta(t_norm_all * span)

    df_ts = pd.DataFrame(
        {
            "date": dates_obs,
            "y": y_true,
            "lambda": rate_mean,
        }
    )
    daily = df_ts.groupby("date").sum().reset_index().sort_values("date")

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(daily["date"], daily["y"], label="Observed", color="black", lw=2)
    ax.plot(daily["date"], daily["lambda"], label="Predicted", color="tab:blue", lw=2)

    if test_coords is not None:
        t_std_test = np.asarray(test_coords)[:, 2].reshape(-1, 1)
        t_norm_test = t_scaler.inverse_transform(t_std_test).ravel()
        test_dates = t_min + pd.to_timedelta(t_norm_test * span)
        ax.scatter(
            test_dates,
            np.zeros(len(test_dates)),
            color="green",
            marker="|",
            s=200,
            label="Test dates",
            zorder=5,
        )

    ax.set_xlabel("Date")
    ax.set_ylabel("Total ships per day")
    ax.set_title("Daily total ships — observed vs. predicted")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_dir is not None:
        fig.savefig(save_dir / "daily_timeseries.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

    # =========================================================================
    # 2. Spatial map for one date
    # =========================================================================
    dates_meta = pd.to_datetime(meta["dates"])

    if target_date is None:
        target_ts = dates_meta[0]
    else:
        target_ts = pd.Timestamp(target_date)

    # Find the closest available date
    deltas = np.abs((dates_meta - target_ts).total_seconds().values)
    closest_ts = dates_meta[np.argmin(deltas)]

    # Map closest date → t_norm → standardized t
    closest_tnorm = (closest_ts - t_min) / span
    closest_tstd = t_scaler.transform([[float(closest_tnorm)]])[0, 0]

    # Select the closest available standardized time value.
    # Important: if the exact transformed date is not found, we select the whole
    # nearest time slice, not just one nearest point.
    time_dist = np.abs(coords[:, 2] - closest_tstd)
    nearest_tstd = coords[np.argmin(time_dist), 2]

    mask = np.isclose(coords[:, 2], nearest_tstd, atol=1e-6)

    print(f"Rows in selected spatial slice: {mask.sum()}")
    print(f"Spatial map for: {closest_ts.date()}  (t_norm={float(closest_tnorm):.3f})")

    # ---- Standardized spatial coords for selected slice ---------------------
    x_sl = coords[mask, 0]  # lon_std
    y_sl = coords[mask, 1]  # lat_std
    rm_sl = rate_mean[mask]
    yt_sl = y_true[mask]

    # ---- Gulf boundary (standardized with SAME mean/std as training data) ---
    gulf_df = pd.read_csv(gulf_csv_path)
    lon_mean, lon_std = meta["lon_mean"], meta["lon_std"]
    lat_mean, lat_std = meta["lat_mean"], meta["lat_std"]

    gulf_x = (gulf_df["longitude"].values - lon_mean) / lon_std
    gulf_y = (gulf_df["latitude"].values - lat_mean) / lat_std

    # ---- Regular grid for interpolation ------------------------------------
    nx, ny = 150, 100
    xi = np.linspace(gulf_x.min(), gulf_x.max(), nx)
    yi = np.linspace(gulf_y.min(), gulf_y.max(), ny)
    Xi, Yi = np.meshgrid(xi, yi)

    # ---- Interpolation ---------------------------------------------------------
    valid = (
        np.isfinite(x_sl) & np.isfinite(y_sl) & np.isfinite(rm_sl) & np.isfinite(yt_sl)
    )

    x_sl = x_sl[valid]
    y_sl = y_sl[valid]
    rm_sl = rm_sl[valid]
    yt_sl = yt_sl[valid]

    n_unique_points = len(np.unique(np.column_stack([x_sl, y_sl]), axis=0))

    print(f"Valid spatial points for interpolation: {len(x_sl)}")
    print(f"Unique spatial points for interpolation: {n_unique_points}")

    if len(x_sl) < 3 or n_unique_points < 3:
        print(
            f"Skipping spatial RBF interpolation: only {len(x_sl)} valid points "
            f"and {n_unique_points} unique spatial points available."
        )
        return

    rbf_rm = Rbf(x_sl, y_sl, rm_sl, function="gaussian")
    rbf_yt = Rbf(x_sl, y_sl, yt_sl, function="gaussian")
    rm_grid = np.clip(rbf_rm(Xi, Yi), 0, None)
    yt_grid = np.clip(rbf_yt(Xi, Yi), 0, None)

    # Fill remaining NaNs with local mean
    for grid in (rm_grid, yt_grid):
        nan_mask = np.isnan(grid)
        if nan_mask.any():
            grid[nan_mask] = ndimage.generic_filter(grid, np.nanmean, size=3)[nan_mask]

    # ---- Mask outside Gulf polygon -----------------------------------------
    gulf_poly = Path(np.column_stack([gulf_x, gulf_y]))
    pts_flat = np.column_stack([Xi.ravel(), Yi.ravel()])
    inside = gulf_poly.contains_points(pts_flat).reshape(Xi.shape)
    rm_masked = np.where(inside, rm_grid, np.nan)
    yt_masked = np.where(inside, yt_grid, np.nan)

    # ---- Plot ---------------------------------------------------------------
    fig2, axs = plt.subplots(1, 2, figsize=(13, 6))

    for ax, grid, title in zip(
        axs,
        [rm_masked, yt_masked],
        [
            f"Predicted intensity ({closest_ts.date()})",
            f"Observed counts ({closest_ts.date()})",
        ],
    ):
        im = ax.imshow(
            grid,
            origin="lower",
            extent=[xi.min(), xi.max(), yi.min(), yi.max()],
            cmap=cmap,
            aspect="auto",
        )
        ax.plot(gulf_x, gulf_y, color="red", lw=1.2, label="Gulf boundary")
        ax.set_title(title)
        ax.set_xlabel("Longitude (standardized)")
        ax.set_ylabel("Latitude (standardized)")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()

    if save_dir is not None:
        fig2.savefig(save_dir / "spatial_map.png", dpi=200, bbox_inches="tight")
        plt.close(fig2)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Loss curve helper
# ---------------------------------------------------------------------------


def plot_test_window_daily_timeseries(
    test_dates,
    test_y,
    rate_mean_test,
    save_path,
    title="LGCP daily totals — test window",
):
    """
    Plot observed vs predicted daily totals on the test window only.

    Parameters
    ----------
    test_dates:
        Date associated with each test observation.
    test_y:
        Observed test counts at observation/cell level.
    rate_mean_test:
        Predicted Poisson rates at observation/cell level.
    save_path:
        Path where the plot will be saved.
    title:
        Plot title.
    """

    plot_df = pd.DataFrame(
        {
            "date": pd.to_datetime(test_dates),
            "observed": np.asarray(test_y, dtype=float),
            "predicted": np.asarray(rate_mean_test, dtype=float),
        }
    )

    daily_df = (
        plot_df.groupby("date", as_index=False)
        .agg(
            observed=("observed", "sum"),
            predicted=("predicted", "sum"),
        )
        .sort_values("date")
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        daily_df["date"],
        daily_df["observed"],
        marker="o",
        label="Observed",
    )

    ax.plot(
        daily_df["date"],
        daily_df["predicted"],
        marker="o",
        label="Predicted",
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("Daily total vessels")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_loss(
    losses: list[float], title: str = "Training loss", save_path=None
) -> None:
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(losses, color="tab:blue", lw=1.5)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("−ELBO")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_daily_observed_vs_predicted_maps(
    df_test,
    y_true,
    rate_mean,
    out_path,
    date_col="date",
    lon_col="longitude",
    lat_col="latitude",
    gulf_coords=None,
    max_days=None,
):
    """
    Plot observed vs predicted spatial maps for each test day.

    Left column  : observed counts per grid cell
    Right column : predicted Poisson rate per grid cell

    Parameters
    ----------
    df_test : pd.DataFrame
        Test dataframe with at least date, longitude, latitude.
    y_true : array-like
        Observed test counts.
    rate_mean : array-like
        Predicted Poisson rates.
    out_path : str or Path
        Output figure path.
    gulf_coords : pd.DataFrame or None
        Optional dataframe with longitude/latitude columns for Gulf boundary.
    max_days : int or None
        If not None, plot only the first max_days test days.
    """
    df_plot = df_test.copy()
    df_plot[date_col] = pd.to_datetime(df_plot[date_col]).dt.normalize()
    df_plot["observed"] = np.asarray(y_true, dtype=float)
    df_plot["predicted"] = np.asarray(rate_mean, dtype=float)

    test_dates = sorted(df_plot[date_col].unique())
    if max_days is not None:
        test_dates = test_dates[:max_days]

    n_days = len(test_dates)
    if n_days == 0:
        raise ValueError("No test dates available for plotting.")

    vmax = max(
        float(df_plot["observed"].max()),
        float(df_plot["predicted"].max()),
    )

    fig, axes = plt.subplots(
        n_days,
        2,
        figsize=(11, 3.2 * n_days),
        squeeze=False,
        sharex=True,
        sharey=True,
    )

    for row, day in enumerate(test_dates):
        day_df = df_plot[df_plot[date_col] == day]

        for col, value_col, title in [
            (0, "observed", "Observed"),
            (1, "predicted", "Predicted"),
        ]:
            ax = axes[row, col]

            sc = ax.scatter(
                day_df[lon_col],
                day_df[lat_col],
                c=day_df[value_col],
                s=90,
                vmin=0.0,
                vmax=vmax,
                edgecolor="black",
                linewidth=0.3,
            )

            if gulf_coords is not None:
                ax.plot(
                    gulf_coords[lon_col],
                    gulf_coords[lat_col],
                    linewidth=1.0,
                    color="black",
                )

            ax.set_title(f"{pd.Timestamp(day).date()} — {title}")
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.grid(alpha=0.3)

        obs_total = day_df["observed"].sum()
        pred_total = day_df["predicted"].sum()

        axes[row, 0].text(
            0.02,
            0.95,
            f"total = {obs_total:.1f}",
            transform=axes[row, 0].transAxes,
            va="top",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
        )

        axes[row, 1].text(
            0.02,
            0.95,
            f"total = {pred_total:.1f}",
            transform=axes[row, 1].transAxes,
            va="top",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
        )

    cbar = fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.7)
    cbar.set_label("Count / predicted rate")

    fig.suptitle("Observed vs predicted spatial intensity by test day", y=0.995)
    fig.subplots_adjust(
        left=0.08,
        right=0.90,
        top=0.96,
        bottom=0.04,
        hspace=0.45,
        wspace=0.20,
    )

    out_path = FilePath(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return out_path


def plot_daily_interpolated_spatial_maps_separate(
    test_coords,
    test_dates,
    y_true,
    rate_mean,
    meta,
    gulf_csv_path,
    out_dir,
    cmap="viridis",
    max_days=None,
    nx=150,
    ny=100,
):
    """
    Save one interpolated observed-vs-predicted spatial map per test day.

    Same style as plot_pp_overview spatial_map.png:
      - standardized longitude/latitude axes
      - RBF interpolation
      - Gulf polygon mask
      - red Gulf boundary
      - separate colorbar for predicted and observed
    """

    out_dir = FilePath(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    test_coords = np.asarray(test_coords, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    rate_mean = np.asarray(rate_mean, dtype=float)
    test_dates = pd.to_datetime(test_dates).normalize()

    if len(test_coords) != len(y_true) or len(test_coords) != len(rate_mean):
        raise ValueError(
            "Length mismatch: test_coords, y_true and rate_mean must have the same length."
        )

    # Gulf boundary in standardized coordinates
    gulf_df = pd.read_csv(gulf_csv_path)

    lon_mean, lon_std = meta["lon_mean"], meta["lon_std"]
    lat_mean, lat_std = meta["lat_mean"], meta["lat_std"]

    gulf_x = (gulf_df["longitude"].values - lon_mean) / lon_std
    gulf_y = (gulf_df["latitude"].values - lat_mean) / lat_std

    xi = np.linspace(gulf_x.min(), gulf_x.max(), nx)
    yi = np.linspace(gulf_y.min(), gulf_y.max(), ny)
    Xi, Yi = np.meshgrid(xi, yi)

    gulf_poly = Path(np.column_stack([gulf_x, gulf_y]))
    pts_flat = np.column_stack([Xi.ravel(), Yi.ravel()])
    inside = gulf_poly.contains_points(pts_flat).reshape(Xi.shape)

    global_vmax = max(
        float(np.nanmax(y_true)),
        float(np.nanmax(rate_mean)),
        1e-8,
    )

    print(f"Using shared color scale: vmin=0, vmax={global_vmax:.4f}")

    unique_dates = sorted(pd.unique(test_dates))

    if max_days is not None:
        unique_dates = unique_dates[:max_days]

    saved_paths = []

    for day in unique_dates:
        mask = test_dates == day

        x_sl = test_coords[mask, 0]
        y_sl = test_coords[mask, 1]
        rm_sl = rate_mean[mask]
        yt_sl = y_true[mask]

        valid = (
            np.isfinite(x_sl)
            & np.isfinite(y_sl)
            & np.isfinite(rm_sl)
            & np.isfinite(yt_sl)
        )

        x_sl = x_sl[valid]
        y_sl = y_sl[valid]
        rm_sl = rm_sl[valid]
        yt_sl = yt_sl[valid]

        n_unique_points = len(np.unique(np.column_stack([x_sl, y_sl]), axis=0))

        if len(x_sl) < 3 or n_unique_points < 3:
            print(
                f"Skipping {pd.Timestamp(day).date()}: only {len(x_sl)} valid points "
                f"and {n_unique_points} unique spatial points."
            )
            continue

        rbf_rm = Rbf(x_sl, y_sl, rm_sl, function="gaussian")
        rbf_yt = Rbf(x_sl, y_sl, yt_sl, function="gaussian")

        rm_grid = np.clip(rbf_rm(Xi, Yi), 0, None)
        yt_grid = np.clip(rbf_yt(Xi, Yi), 0, None)

        for grid in (rm_grid, yt_grid):
            nan_mask = np.isnan(grid)
            if nan_mask.any():
                grid[nan_mask] = ndimage.generic_filter(grid, np.nanmean, size=3)[
                    nan_mask
                ]

        rm_masked = np.where(inside, rm_grid, np.nan)
        yt_masked = np.where(inside, yt_grid, np.nan)

        day_str = pd.Timestamp(day).strftime("%Y-%m-%d")

        fig, axs = plt.subplots(1, 2, figsize=(13, 6))

        for ax, grid, title in zip(
            axs,
            [rm_masked, yt_masked],
            [
                f"Predicted intensity ({day_str})",
                f"Observed counts ({day_str})",
            ],
        ):
            im = ax.imshow(
                grid,
                origin="lower",
                extent=[xi.min(), xi.max(), yi.min(), yi.max()],
                cmap=cmap,
                aspect="auto",
                vmin=0.0,
                vmax=global_vmax,
            )

            ax.plot(gulf_x, gulf_y, color="red", lw=1.2, label="Gulf boundary")
            ax.set_title(title)
            ax.set_xlabel("Longitude (standardized)")
            ax.set_ylabel("Latitude (standardized)")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.tight_layout()

        out_path = out_dir / f"spatial_map_{day_str}.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

        saved_paths.append(out_path)

    return saved_paths
