"""Audit Ebro versus Trieste processed data for one year.

Example:
    python scripts/audit_ebro_comparability.py --year 2025
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.config import load_config  # noqa: E402
from src.data.paths import get_data_dirs  # noqa: E402

KEYS = ["date", "latitude", "longitude"]
NON_GEAR_COLUMNS = {
    *KEYS,
    "is_holiday",
    "holiday_name",
    "is_weekend",
    "fishing_block",
    "chl",
    "thetao",
    "salinity",
    "sar_vessels_count",
    "ais_vessels_count",
}


def load_region(
    config_path: str,
    year: int,
) -> tuple[dict, pd.DataFrame, Path]:
    config = load_config(config_path)
    config["year"] = year
    processed_path = get_data_dirs(config)["processed"] / f"cpr_gfw_{year}.parquet"
    if not processed_path.exists():
        raise FileNotFoundError(f"Processed dataset not found: {processed_path}")
    frame = pd.read_parquet(processed_path)
    missing_keys = [column for column in KEYS if column not in frame.columns]
    if missing_keys:
        raise ValueError(f"{processed_path} missing key columns: {missing_keys}")
    frame["date"] = pd.to_datetime(frame["date"])
    return config, frame, processed_path


def find_gfw_report_version(config: dict, year: int) -> list[str]:
    raw_dir = get_data_dirs(config)["raw"] / "gfw_ais"
    candidates = sorted(raw_dir.glob(f"*{year}.parquet"))
    versions = set()
    for candidate in candidates:
        try:
            raw = pd.read_parquet(candidate, columns=["report_dataset"])
        except (ValueError, KeyError):
            continue
        versions.update(raw["report_dataset"].dropna().astype(str).unique())
    return sorted(versions)


def region_summary(
    name: str,
    config: dict,
    frame: pd.DataFrame,
    processed_path: Path,
    year: int,
) -> dict:
    bbox = config["area"]
    coords = frame[["latitude", "longitude"]].drop_duplicates()
    dates = frame["date"].dt.normalize()
    observed_dates = sorted(dates.dt.strftime("%Y-%m-%d").unique().tolist())
    daily_cell_counts = (
        frame.assign(_date=dates)
        .groupby("_date")[["latitude", "longitude"]]
        .apply(lambda rows: rows.drop_duplicates().shape[0])
    )

    inside_bbox = coords["longitude"].between(
        bbox["minimum_longitude"], bbox["maximum_longitude"]
    ) & coords["latitude"].between(bbox["minimum_latitude"], bbox["maximum_latitude"])
    grid_latitudes = sorted(coords["latitude"].unique())
    grid_longitudes = sorted(coords["longitude"].unique())
    latitude_step = (
        float(pd.Series(grid_latitudes).diff().dropna().median())
        if len(grid_latitudes) > 1
        else None
    )
    longitude_step = (
        float(pd.Series(grid_longitudes).diff().dropna().median())
        if len(grid_longitudes) > 1
        else None
    )
    mean_latitude = float(coords["latitude"].mean())
    cell_day_count = int(len(frame))
    ais_total = (
        int(frame["ais_vessels_count"].sum()) if "ais_vessels_count" in frame else None
    )
    active_ais_cell_days = (
        int((frame["ais_vessels_count"] > 0).sum())
        if "ais_vessels_count" in frame
        else None
    )
    gear_columns = sorted(set(frame.columns) - NON_GEAR_COLUMNS)
    gear_sum_matches_ais = None
    if gear_columns and "ais_vessels_count" in frame:
        gear_sum_matches_ais = bool(
            frame[gear_columns].sum(axis=1).equals(frame["ais_vessels_count"])
        )

    expected_dates = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    missing_core = {
        column: int(frame[column].isna().sum())
        for column in [
            "chl",
            "thetao",
            "salinity",
            "ais_vessels_count",
            "sar_vessels_count",
        ]
        if column in frame.columns
    }
    duplicate_keys = int(frame.duplicated(KEYS).sum())
    bbox_width_km = (
        (bbox["maximum_longitude"] - bbox["minimum_longitude"])
        * 111.32
        * math.cos(math.radians(mean_latitude))
    )
    bbox_height_km = (bbox["maximum_latitude"] - bbox["minimum_latitude"]) * 111.32

    return {
        "region": name,
        "year": year,
        "processed_file": str(processed_path.relative_to(PROJECT_ROOT)),
        "shape": {"rows": int(len(frame)), "columns": int(frame.shape[1])},
        "period": {
            "first_date": observed_dates[0] if observed_dates else None,
            "last_date": observed_dates[-1] if observed_dates else None,
            "observed_days": len(observed_dates),
            "expected_days": len(expected_dates),
            "missing_days": int(
                len(set(expected_dates.strftime("%Y-%m-%d")) - set(observed_dates))
            ),
        },
        "grid": {
            "cells": int(len(coords)),
            "latitude_count": len(grid_latitudes),
            "longitude_count": len(grid_longitudes),
            "rectangular_occupancy": round(
                len(coords) / max(1, len(grid_latitudes) * len(grid_longitudes)), 4
            ),
            "cells_per_day_min": int(daily_cell_counts.min()),
            "cells_per_day_max": int(daily_cell_counts.max()),
            "median_latitude_step_deg": latitude_step,
            "median_longitude_step_deg": longitude_step,
            "approx_latitude_step_km": (
                round(latitude_step * 111.32, 2) if latitude_step is not None else None
            ),
            "approx_longitude_step_km": (
                round(
                    longitude_step * 111.32 * math.cos(math.radians(mean_latitude)),
                    2,
                )
                if longitude_step is not None
                else None
            ),
            "longitude_min": float(coords["longitude"].min()),
            "longitude_max": float(coords["longitude"].max()),
            "latitude_min": float(coords["latitude"].min()),
            "latitude_max": float(coords["latitude"].max()),
            "all_cells_inside_configured_bbox": bool(inside_bbox.all()),
            "cells_outside_bbox": int((~inside_bbox).sum()),
            "configured_bbox_approx_km": {
                "width": round(bbox_width_km, 1),
                "height": round(bbox_height_km, 1),
                "area": round(bbox_width_km * bbox_height_km, 1),
            },
        },
        "integrity": {
            "duplicate_date_cell_keys": duplicate_keys,
            "missing_core_values": missing_core,
        },
        "ais": {
            "configured_group_by": config.get("gfw_ais", {}).get("group_by"),
            "configured_dataset": config.get("gfw_ais", {}).get("product_name"),
            "raw_report_versions": find_gfw_report_version(config, year),
            "total_count": ais_total,
            "active_cell_days": active_ais_cell_days,
            "active_cell_day_rate": (
                round(active_ais_cell_days / cell_day_count, 4)
                if active_ais_cell_days is not None and cell_day_count
                else None
            ),
            "count_per_cell_day": (
                round(ais_total / cell_day_count, 4)
                if ais_total is not None and cell_day_count
                else None
            ),
            "gear_columns": gear_columns,
            "gear_count_sum_equals_ais_total_per_row": gear_sum_matches_ais,
        },
        "satellite": {
            "total_count": (
                int(frame["sar_vessels_count"].sum())
                if "sar_vessels_count" in frame
                else None
            ),
            "active_cell_days": (
                int((frame["sar_vessels_count"] > 0).sum())
                if "sar_vessels_count" in frame
                else None
            ),
        },
        "features": {
            "columns": list(frame.columns),
            "calendar_features": [
                column
                for column in [
                    "is_holiday",
                    "holiday_name",
                    "is_weekend",
                    "fishing_block",
                ]
                if column in frame.columns
            ],
            "holiday_country": config.get("calendar", {}).get("country"),
            "holiday_subdivision": config.get("calendar", {}).get("subdivision"),
        },
    }


def save_geometry_plot(
    regions: list[tuple[str, dict, pd.DataFrame]],
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    colors = ["#0072B2", "#D55E00"]
    for axis, (name, config, frame), color in zip(axes, regions, colors):
        bbox = config["area"]
        coords = frame[["latitude", "longitude"]].drop_duplicates()
        axis.add_patch(
            Rectangle(
                (bbox["minimum_longitude"], bbox["minimum_latitude"]),
                bbox["maximum_longitude"] - bbox["minimum_longitude"],
                bbox["maximum_latitude"] - bbox["minimum_latitude"],
                fill=False,
                edgecolor="#263746",
                linewidth=1.2,
                linestyle="--",
                label="Configured bounding box",
            )
        )
        axis.scatter(
            coords["longitude"],
            coords["latitude"],
            s=18,
            color=color,
            edgecolors="white",
            linewidths=0.35,
            alpha=0.85,
            label="Copernicus grid centers",
        )
        axis.set_xlim(bbox["minimum_longitude"], bbox["maximum_longitude"])
        axis.set_ylim(bbox["minimum_latitude"], bbox["maximum_latitude"])
        axis.set_aspect(
            1 / math.cos(math.radians(coords["latitude"].mean())),
            adjustable="box",
        )
        axis.set_title(f"{name}: {len(coords)} grid cells")
        axis.set_xlabel("Longitude (degrees east)")
        axis.set_ylabel("Latitude (degrees north)")
        axis.grid(color="#d8dee5", linewidth=0.6)
        axis.legend(loc="best", fontsize=8)

    figure.suptitle("Processed Copernicus grid centers and configured bounds")
    figure.savefig(output_path, dpi=170)
    plt.close(figure)


def save_activity_plot(
    regions: list[tuple[str, dict, pd.DataFrame]],
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True, constrained_layout=True
    )
    colors = ["#0072B2", "#D55E00"]
    for (name, _config, frame), color in zip(regions, colors):
        monthly = (
            frame.set_index("date")
            .resample("MS")
            .agg(
                ais_total=("ais_vessels_count", "sum"),
                cell_days=("ais_vessels_count", "size"),
                active_ais_cell_days=(
                    "ais_vessels_count",
                    lambda values: int((values > 0).sum()),
                ),
                sar_total=("sar_vessels_count", "sum"),
                active_sar_cell_days=(
                    "sar_vessels_count",
                    lambda values: int((values > 0).sum()),
                ),
            )
        )
        monthly["ais_count_per_cell_day"] = monthly["ais_total"] / monthly["cell_days"]
        monthly["ais_active_rate"] = (
            monthly["active_ais_cell_days"] / monthly["cell_days"]
        )
        monthly["sar_active_rate"] = (
            monthly["active_sar_cell_days"] / monthly["cell_days"]
        )
        axes[0].plot(
            monthly.index,
            monthly["ais_count_per_cell_day"],
            marker="o",
            linewidth=1.6,
            color=color,
            label=f"{name} AIS per cell-day",
        )
        axes[1].plot(
            monthly.index,
            monthly["ais_active_rate"],
            marker="o",
            linewidth=1.6,
            color=color,
            label=f"{name} AIS active cell-day rate",
        )
        axes[1].plot(
            monthly.index,
            monthly["sar_active_rate"],
            marker="x",
            linewidth=1.2,
            linestyle="--",
            color=color,
            alpha=0.65,
            label=f"{name} satellite active cell-day rate",
        )

    axes[0].set_title("Monthly AIS count per observed grid-cell-day")
    axes[0].set_ylabel("AIS count / cell-day")
    axes[1].set_title("Fraction of active cell-days")
    axes[1].set_ylabel("Active cell-day fraction")
    axes[1].set_xlabel("Month")
    for axis in axes:
        axis.grid(color="#d8dee5", linewidth=0.6)
        axis.legend(fontsize=8, ncol=2)
    figure.suptitle("Activity overview (AIS groupings differ by region)")
    figure.savefig(output_path, dpi=170)
    plt.close(figure)


def write_markdown_report(summary: dict, output_path: Path) -> None:
    ebro = summary["regions"]["Ebro Delta"]
    trieste = summary["regions"]["Trieste"]
    lines = [
        f"# Ebro Delta vs Trieste data audit ({summary['year']})",
        "",
        "## Comparability checks",
        "",
        f"- GFW effort version: Ebro {ebro['ais']['raw_report_versions']}; "
        f"Trieste {trieste['ais']['raw_report_versions']}.",
        f"- AIS grouping: Ebro `{ebro['ais']['configured_group_by']}`; "
        f"Trieste `{trieste['ais']['configured_group_by']}`. Absolute AIS counts "
        "are not strictly like-for-like because the observation units differ.",
        f"- Grid size: Ebro {ebro['grid']['cells']} cells; "
        f"Trieste {trieste['grid']['cells']} cells.",
        f"- Days: Ebro {ebro['period']['observed_days']}/{ebro['period']['expected_days']}; "
        f"Trieste {trieste['period']['observed_days']}/{trieste['period']['expected_days']}.",
        "- The geometry plot shows Copernicus grid centers and the configured bounding "
        "boxes. It does not show coastlines or independently validate a marine mask.",
        "",
        "## Region metrics",
        "",
        "| Metric | Ebro Delta | Trieste |",
        "|---|---:|---:|",
        f"| Rows | {ebro['shape']['rows']} | {trieste['shape']['rows']} |",
        f"| Grid cells | {ebro['grid']['cells']} | {trieste['grid']['cells']} |",
        f"| Duplicate date-cell keys | {ebro['integrity']['duplicate_date_cell_keys']} | {trieste['integrity']['duplicate_date_cell_keys']} |",
        f"| Cells outside configured bbox | {ebro['grid']['cells_outside_bbox']} | {trieste['grid']['cells_outside_bbox']} |",
        f"| AIS group-by | {ebro['ais']['configured_group_by']} | {trieste['ais']['configured_group_by']} |",
        f"| AIS total | {ebro['ais']['total_count']} | {trieste['ais']['total_count']} |",
        f"| AIS count per cell-day | {ebro['ais']['count_per_cell_day']} | {trieste['ais']['count_per_cell_day']} |",
        f"| AIS active cell-day rate | {ebro['ais']['active_cell_day_rate']} | {trieste['ais']['active_cell_day_rate']} |",
        f"| Satellite total | {ebro['satellite']['total_count']} | {trieste['satellite']['total_count']} |",
        "",
        "## Files",
        "",
        "- `summary.json`: full metrics and validation results.",
        "- `geometry_comparison.png`: grid centers and bounding boxes.",
        "- `activity_comparison.png`: monthly normalized activity overview.",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=None)
    parser.add_argument(
        "--ebro-config",
        default="config/data_ebro_vGearAndFlag.yaml",
    )
    parser.add_argument("--trieste-config", default="config/data.yaml")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    ebro_base_config = load_config(args.ebro_config)
    year = args.year or int(ebro_base_config["year"])
    ebro_config, ebro_frame, ebro_path = load_region(args.ebro_config, year)
    trieste_config, trieste_frame, trieste_path = load_region(
        args.trieste_config,
        year,
    )

    output_dir = Path(args.output_dir or f"reports/ebro_comparability/{year}")
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    regions = {
        "Ebro Delta": region_summary(
            "Ebro Delta", ebro_config, ebro_frame, ebro_path, year
        ),
        "Trieste": region_summary(
            "Trieste", trieste_config, trieste_frame, trieste_path, year
        ),
    }
    common_gears = sorted(
        set(regions["Ebro Delta"]["ais"]["gear_columns"])
        & set(regions["Trieste"]["ais"]["gear_columns"])
    )
    summary = {
        "year": year,
        "regions": regions,
        "comparison": {
            "same_gfw_effort_version": (
                regions["Ebro Delta"]["ais"]["raw_report_versions"]
                == regions["Trieste"]["ais"]["raw_report_versions"]
            ),
            "same_ais_grouping": (
                regions["Ebro Delta"]["ais"]["configured_group_by"]
                == regions["Trieste"]["ais"]["configured_group_by"]
            ),
            "common_gear_columns": common_gears,
            "same_calendar_country": (
                regions["Ebro Delta"]["features"]["holiday_country"]
                == regions["Trieste"]["features"]["holiday_country"]
            ),
        },
    }

    save_geometry_plot(
        [
            ("Ebro Delta", ebro_config, ebro_frame),
            ("Trieste", trieste_config, trieste_frame),
        ],
        output_dir / "geometry_comparison.png",
    )
    save_activity_plot(
        [
            ("Ebro Delta", ebro_config, ebro_frame),
            ("Trieste", trieste_config, trieste_frame),
        ],
        output_dir / "activity_comparison.png",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    write_markdown_report(summary, output_dir / "summary.md")
    print(f"[DONE] Ebro-Trieste audit saved to {output_dir}")


if __name__ == "__main__":
    main()
