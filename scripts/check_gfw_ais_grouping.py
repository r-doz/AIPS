"""Compare GFW AIS reports grouped by flag and gear type.

Example:
    python scripts/check_gfw_ais_grouping.py --start-date 2024-01-01 --days 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import gfwapiclient as gfw
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.config import load_config  # noqa: E402
from src.data.gfw_ais import get_gfw_ais_raw_path, read_gfw_token  # noqa: E402
from src.data.geo import bbox_to_geojson  # noqa: E402

VERSIONS = {
    "v4": "public-global-fishing-effort:v4.0",
    "v5": "public-global-fishing-effort:v5.0",
}
GROUP_KEYS = ["date", "lat", "lon", "flag", "gear_type"]


async def fetch_report(
    config: dict,
    dataset: str,
    start_date: date,
    api_end_date_exclusive: date,
) -> pd.DataFrame:
    """Fetch a report; FourWings treats the date-range end as exclusive."""
    ais_config = config["gfw_ais"]
    area = config["area"]
    geojson = bbox_to_geojson(
        area["minimum_longitude"],
        area["minimum_latitude"],
        area["maximum_longitude"],
        area["maximum_latitude"],
    )
    params = {
        "format": "JSON",
        "spatial-resolution": ais_config["spatial_resolution"],
        "group-by": "FLAGANDGEARTYPE",
        "temporal-resolution": ais_config["temporal_resolution"],
        "datasets[0]": dataset,
        "spatial-aggregation": "false",
        "date-range": (
            f"{start_date.isoformat()},{api_end_date_exclusive.isoformat()}"
        ),
    }

    client = gfw.Client(access_token=read_gfw_token(config))
    async with client._http_client as http_client:
        response = await http_client.post(
            "4wings/report",
            params=params,
            json={"geojson": geojson},
        )
        response.raise_for_status()
        payload = response.json()

    records = []
    for entry in payload.get("entries", []):
        if not isinstance(entry, dict):
            continue
        for report_dataset, rows in entry.items():
            if isinstance(rows, list):
                records.extend(
                    {"report_dataset": report_dataset, **row}
                    for row in rows
                    if isinstance(row, dict)
                )

    if not records:
        raise ValueError(f"GFW returned no report rows for {dataset}.")
    return pd.DataFrame(records)


def grouped_vessel_counts(
    frame: pd.DataFrame,
    count_column: str | None,
) -> pd.DataFrame:
    work_frame = frame.copy()
    aliases = {
        "geartype": "gear_type",
        "vesselIDs": "vessel_ids",
    }
    work_frame = work_frame.rename(
        columns={
            source: target
            for source, target in aliases.items()
            if target not in work_frame.columns and source in work_frame.columns
        }
    )

    missing = [column for column in GROUP_KEYS if column not in work_frame.columns]
    if missing:
        raise ValueError(f"Report is missing comparison columns: {missing}")
    if count_column is not None:
        count_column = aliases.get(count_column, count_column)
    if count_column is not None and count_column not in work_frame.columns:
        raise ValueError(f"Report is missing count column '{count_column}'.")

    work = work_frame[GROUP_KEYS].copy()
    work["date"] = pd.to_datetime(work["date"]).dt.strftime("%Y-%m-%d")
    work["lat"] = pd.to_numeric(work["lat"]).round(6)
    work["lon"] = pd.to_numeric(work["lon"]).round(6)
    work["flag"] = work["flag"].astype("string").fillna("__MISSING__").str.upper()
    work["gear_type"] = (
        work["gear_type"].astype("string").fillna("__MISSING__").str.lower()
    )

    if count_column is None:
        counts = work.groupby(GROUP_KEYS, dropna=False).size()
    else:
        work["vessel_count"] = pd.to_numeric(
            work_frame[count_column], errors="raise"
        ).fillna(0)
        counts = work.groupby(GROUP_KEYS, dropna=False)["vessel_count"].sum()

    return counts.rename("vessel_count").reset_index()


def compare_counts(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_name: str,
    right_name: str,
) -> tuple[pd.DataFrame, dict]:
    comparison = left.merge(
        right,
        on=GROUP_KEYS,
        how="outer",
        suffixes=(f"_{left_name}", f"_{right_name}"),
    )
    left_column = f"vessel_count_{left_name}"
    right_column = f"vessel_count_{right_name}"
    comparison[[left_column, right_column]] = comparison[
        [left_column, right_column]
    ].fillna(0)
    comparison[left_column] = comparison[left_column].astype(int)
    comparison[right_column] = comparison[right_column].astype(int)
    comparison["difference"] = comparison[left_column] - comparison[right_column]

    summary = {
        "groups": int(len(comparison)),
        "matching_groups": int((comparison["difference"] == 0).sum()),
        "different_groups": int((comparison["difference"] != 0).sum()),
        f"total_{left_name}": int(comparison[left_column].sum()),
        f"total_{right_name}": int(comparison[right_column].sum()),
    }
    return comparison, summary


def position_counts(frame: pd.DataFrame, count_column: str | None) -> pd.DataFrame:
    work_frame = frame.copy()
    aliases = {"geartype": "gear_type", "vesselIDs": "vessel_ids"}
    work_frame = work_frame.rename(
        columns={
            source: target
            for source, target in aliases.items()
            if target not in work_frame.columns and source in work_frame.columns
        }
    )

    required = ["date", "lat", "lon"]
    missing = [column for column in required if column not in work_frame.columns]
    if missing:
        raise ValueError(f"Position data is missing columns: {missing}")

    work = work_frame[required].copy()
    work["date"] = pd.to_datetime(work["date"]).dt.strftime("%Y-%m-%d")
    work["lat"] = pd.to_numeric(work["lat"]).round(6)
    work["lon"] = pd.to_numeric(work["lon"]).round(6)

    if count_column is None:
        counts = work.groupby(required, dropna=False).size()
    else:
        count_column = aliases.get(count_column, count_column)
        if count_column not in work_frame.columns:
            raise ValueError(f"Position data is missing '{count_column}'.")
        work["position_count"] = pd.to_numeric(
            work_frame[count_column], errors="raise"
        ).fillna(0)
        counts = work.groupby(required, dropna=False)["position_count"].sum()

    return counts.rename("position_count").reset_index()


def create_daily_maps(
    baseline: pd.DataFrame,
    downloaded: dict[str, pd.DataFrame],
    start_date: date,
    days: int,
    area: dict,
    output_dir: Path,
) -> list[Path]:
    layers = [
        (
            "Original v4 - VESSEL_ID",
            position_counts(baseline, count_column=None),
            "#0072B2",
        ),
        (
            "GFW v4.0 - FLAGANDGEARTYPE",
            position_counts(downloaded["v4"], count_column="vessel_ids")
            if "v4" in downloaded
            else pd.DataFrame(columns=["date", "lat", "lon", "position_count"]),
            "#D55E00",
        ),
        (
            "GFW v5.0 - FLAGANDGEARTYPE",
            position_counts(downloaded["v5"], count_column="vessel_ids")
            if "v5" in downloaded
            else pd.DataFrame(columns=["date", "lat", "lon", "position_count"]),
            "#009E73",
        ),
    ]

    map_dir = output_dir / "maps"
    map_dir.mkdir(parents=True, exist_ok=True)
    lon_min = area["minimum_longitude"]
    lon_max = area["maximum_longitude"]
    lat_min = area["minimum_latitude"]
    lat_max = area["maximum_latitude"]
    latitude_scale = 1 / np.cos(np.deg2rad((lat_min + lat_max) / 2))
    map_paths = []

    for day_offset in range(days):
        day = start_date + timedelta(days=day_offset)
        day_string = day.isoformat()
        figure, axes = plt.subplots(
            1,
            len(layers),
            figsize=(15, 5.2),
            sharex=True,
            sharey=True,
            constrained_layout=True,
        )

        for axis, (label, positions, color) in zip(axes, layers):
            daily = positions.loc[positions["date"] == day_string]
            axis.set_xlim(lon_min, lon_max)
            axis.set_ylim(lat_min, lat_max)
            axis.set_aspect(latitude_scale, adjustable="box")
            axis.set_title(label, fontsize=10)
            axis.set_xlabel("Longitude (E)")
            axis.set_ylabel("Latitude (N)")
            axis.grid(color="#d8dee5", linewidth=0.6, alpha=0.8)
            axis.tick_params(labelsize=8)

            if daily.empty:
                axis.text(
                    0.5,
                    0.5,
                    "No observations",
                    ha="center",
                    va="center",
                    transform=axis.transAxes,
                    color="#5b6570",
                )
                continue

            counts = daily["position_count"].astype(float)
            axis.scatter(
                daily["lon"],
                daily["lat"],
                s=18 + 22 * np.sqrt(counts),
                color=color,
                edgecolors="white",
                linewidths=0.45,
                alpha=0.82,
            )
            axis.text(
                0.03,
                0.97,
                f"{len(daily)} locations | count {int(counts.sum())}",
                ha="left",
                va="top",
                transform=axis.transAxes,
                fontsize=8,
                bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
            )

        figure.suptitle(f"Trieste AIS positions | {day_string}", fontsize=13)
        map_path = map_dir / f"positions_{day_string}.png"
        figure.savefig(map_path, dpi=160)
        plt.close(figure)
        map_paths.append(map_path)

    return map_paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/data.yaml")
    parser.add_argument("--start-date", default="2024-01-01")
    parser.add_argument("--days", type=int, default=5)
    parser.add_argument(
        "--output-dir",
        default="data/raw/gfw_ais/groupby_check",
    )
    parser.add_argument(
        "--maps-only",
        action="store_true",
        help="Create maps from saved reports without making GFW requests.",
    )
    args = parser.parse_args()

    if args.days < 1:
        parser.error("--days must be at least 1")

    start_date = date.fromisoformat(args.start_date)
    end_date = start_date + timedelta(days=args.days - 1)
    api_end_date_exclusive = end_date + timedelta(days=1)
    config = load_config(args.config)
    config["year"] = start_date.year

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_path = get_gfw_ais_raw_path(config)
    if not baseline_path.exists():
        raise FileNotFoundError(
            f"Trieste VESSEL_ID baseline not found: {baseline_path}"
        )

    baseline = pd.read_parquet(baseline_path)
    if "report_dataset" not in baseline.columns:
        raise ValueError("Baseline has no report_dataset column to verify its version.")
    baseline_versions = sorted(baseline["report_dataset"].dropna().unique().tolist())
    if baseline_versions != [VERSIONS["v4"]]:
        raise ValueError(
            "Expected a v4.0 Trieste baseline, found: "
            f"{baseline_versions or 'no report_dataset values'}"
        )

    baseline["date"] = pd.to_datetime(baseline["date"])
    baseline = baseline.loc[baseline["date"].dt.date.between(start_date, end_date)]
    baseline_counts = grouped_vessel_counts(baseline, count_column=None)

    summary = {
        "requested_start_date": start_date.isoformat(),
        "requested_end_date": end_date.isoformat(),
        "api_date_range_end_exclusive": api_end_date_exclusive.isoformat(),
        "baseline_path": str(baseline_path.relative_to(PROJECT_ROOT)),
        "baseline_report_dataset": baseline_versions,
        "baseline_rows_in_window": int(len(baseline)),
        "acquired_at_utc": datetime.now(timezone.utc).isoformat(),
        "requests": {},
        "comparisons": {},
    }
    downloaded: dict[str, pd.DataFrame] = {}

    if args.maps_only:
        for version, dataset in VERSIONS.items():
            report_path = output_dir / f"trieste_{version}_flagandgeartype.parquet"
            if not report_path.exists():
                raise FileNotFoundError(
                    f"Saved {version} report not found: {report_path}"
                )
            downloaded[version] = pd.read_parquet(report_path)
            summary["requests"][version] = {
                "status": "loaded_existing",
                "path": str(report_path.relative_to(PROJECT_ROOT)),
                "requested_dataset": dataset,
            }
    else:
        for version, dataset in VERSIONS.items():
            try:
                report = asyncio.run(
                    fetch_report(
                        config,
                        dataset,
                        start_date,
                        api_end_date_exclusive,
                    )
                )
                report_dates = pd.to_datetime(report["date"]).dt.date
                report = report.loc[report_dates.between(start_date, end_date)].copy()
                if report.empty:
                    raise ValueError(
                        f"GFW returned no rows in the requested window for {dataset}."
                    )
                returned_datasets = sorted(
                    report["report_dataset"].dropna().unique().tolist()
                )
                if returned_datasets != [dataset]:
                    raise ValueError(
                        f"Requested {dataset}, but GFW returned {returned_datasets}."
                    )
                report["aips_requested_dataset"] = dataset
                report["aips_acquired_at_utc"] = summary["acquired_at_utc"]
                report.to_parquet(
                    output_dir / f"trieste_{version}_flagandgeartype.parquet",
                    index=False,
                )
                downloaded[version] = report
                summary["requests"][version] = {
                    "status": "ok",
                    "returned_report_datasets": returned_datasets,
                    "rows": int(len(report)),
                    "columns": list(report.columns),
                }
            except Exception as error:
                summary["requests"][version] = {
                    "status": "error",
                    "error": f"{type(error).__name__}: {error}",
                }
                print(f"[ERROR] {version}: {summary['requests'][version]['error']}")

    if "v4" in downloaded:
        v4_counts = grouped_vessel_counts(downloaded["v4"], "vessel_ids")
        comparison, comparison_summary = compare_counts(
            v4_counts,
            baseline_counts,
            "v4_flagandgeartype",
            "original_vesselid",
        )
        comparison.to_csv(
            output_dir / "compare_v4_to_original_vesselid.csv", index=False
        )
        summary["comparisons"]["v4_flagandgeartype_vs_original_vesselid"] = (
            comparison_summary
        )

    if "v4" in downloaded and "v5" in downloaded:
        v4_counts = grouped_vessel_counts(downloaded["v4"], "vessel_ids")
        v5_counts = grouped_vessel_counts(downloaded["v5"], "vessel_ids")
        comparison, comparison_summary = compare_counts(
            v4_counts,
            v5_counts,
            "v4_flagandgeartype",
            "v5_flagandgeartype",
        )
        comparison.to_csv(
            output_dir / "compare_v4_to_v5_flagandgeartype.csv", index=False
        )
        summary["comparisons"]["v4_flagandgeartype_vs_v5_flagandgeartype"] = (
            comparison_summary
        )

    map_paths = create_daily_maps(
        baseline=baseline,
        downloaded=downloaded,
        start_date=start_date,
        days=args.days,
        area=config["area"],
        output_dir=output_dir,
    )
    summary["maps"] = [str(path.relative_to(PROJECT_ROOT)) for path in map_paths]

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(f"[DONE] Comparison summary: {summary_path}")
    if any(item["status"] == "error" for item in summary["requests"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
