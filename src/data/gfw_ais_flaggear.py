from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import gfwapiclient as gfw
import pandas as pd

from src.data.config import load_config
from src.data.geo import bbox_to_geojson, filter_configured_marine_area
from src.data.paths import ensure_dir, get_data_dirs
from src.data.gfw_ais import read_gfw_token


def get_gfw_ais_flaggear_raw_path(config: dict) -> Path:
    year = int(config["year"])
    ais_config = config["gfw_ais"]
    data_dirs = get_data_dirs(config)
    raw_dir = ensure_dir(data_dirs["raw"] / "gfw_ais")
    filename = (
        f"ais_{ais_config['spatial_resolution'].lower()}_"
        f"{ais_config['temporal_resolution'].lower()}_flagandgeartype_{year}.parquet"
    )
    return raw_dir / filename


def get_gfw_ais_flaggear_interim_path(config: dict) -> Path:
    year = int(config["year"])
    ais_config = config["gfw_ais"]
    data_dirs = get_data_dirs(config)
    interim_dir = ensure_dir(data_dirs["interim"] / "gfw_ais")
    filename = (
        f"cleaned_ais_{ais_config['spatial_resolution'].lower()}_"
        f"{ais_config['temporal_resolution'].lower()}_flagandgeartype_{year}.parquet"
    )
    return interim_dir / filename


async def _download_gfw_ais_flaggear_async(config: dict) -> pd.DataFrame:
    year = int(config["year"])
    area = config["area"]
    dates = config["dates"]
    ais_config = config["gfw_ais"]
    start_date = date.fromisoformat(
        f"{year}-{dates['start_month']}-{dates['start_day']}"
    )
    end_date = date.fromisoformat(f"{year}-{dates['end_month']}-{dates['end_day']}")
    api_end_date_exclusive = end_date + timedelta(days=1)

    params = {
        "format": "JSON",
        "spatial-resolution": ais_config["spatial_resolution"],
        "group-by": "FLAGANDGEARTYPE",
        "temporal-resolution": ais_config["temporal_resolution"],
        "datasets[0]": ais_config["product_name"],
        "spatial-aggregation": "false",
        "date-range": f"{start_date.isoformat()},{api_end_date_exclusive.isoformat()}",
    }
    geojson = bbox_to_geojson(
        area["minimum_longitude"],
        area["minimum_latitude"],
        area["maximum_longitude"],
        area["maximum_latitude"],
    )

    print(
        f"[START] GFW AIS {ais_config['product_name']} "
        f"{start_date} through {end_date} grouped by FLAGANDGEARTYPE"
    )

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
        raise ValueError("GFW returned no AIS rows for the configured date range.")

    df = pd.DataFrame(records)
    returned_datasets = sorted(df["report_dataset"].dropna().unique().tolist())
    if returned_datasets != [ais_config["product_name"]]:
        raise ValueError(
            f"Requested {ais_config['product_name']}, "
            f"but GFW returned {returned_datasets}."
        )

    df["date"] = pd.to_datetime(df["date"])
    df = df.loc[df["date"].dt.date.between(start_date, end_date)].copy()
    df["aips_requested_dataset"] = ais_config["product_name"]
    df["aips_acquired_at_utc"] = datetime.now(timezone.utc).isoformat()
    print(f"[INFO] Returned AIS aggregate rows within date range: {len(df)}")
    return df


def download_gfw_ais_flaggear_data(config: dict | None = None) -> Path:
    if config is None:
        config = load_config("config/data_ebro_vGearAndFlag.yaml")

    output_path = get_gfw_ais_flaggear_raw_path(config)
    if output_path.exists() and not config["gfw_ais"].get("overwrite", False):
        print(f"[SKIP] GFW AIS raw file already exists: {output_path}")
        return output_path

    df = asyncio.run(_download_gfw_ais_flaggear_async(config))
    df.to_parquet(output_path, index=False)
    print(f"[DONE] GFW AIS aggregate saved: {output_path}")
    return output_path


def clean_gfw_ais_flaggear_data(config: dict | None = None) -> Path:
    if config is None:
        config = load_config("config/data_ebro_vGearAndFlag.yaml")

    raw_path = get_gfw_ais_flaggear_raw_path(config)
    output_path = get_gfw_ais_flaggear_interim_path(config)
    if output_path.exists() and not config["gfw_ais"].get("overwrite_clean", False):
        print(f"[SKIP] Cleaned GFW AIS file already exists: {output_path}")
        return output_path
    if not raw_path.exists():
        raise FileNotFoundError(f"GFW AIS raw file not found: {raw_path}")

    df = pd.read_parquet(raw_path)
    df = df.rename(
        columns={
            "geartype": "gear_type",
            "vesselIDs": "vessel_count",
            "vessel_ids": "vessel_count",
            "lat": "latitude",
            "lon": "longitude",
        }
    )
    required = ["date", "latitude", "longitude", "flag", "gear_type", "vessel_count"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"GFW FLAGANDGEARTYPE file missing columns: {missing}")

    df = df[required + (["hours"] if "hours" in df.columns else [])].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["vessel_count"] = pd.to_numeric(df["vessel_count"], errors="raise").fillna(0)
    df["vessel_count"] = df["vessel_count"].astype("int64")
    df = df.loc[df["flag"] == config["gfw_ais"].get("flag", "ESP")].copy()
    df = filter_configured_marine_area(
        df,
        config,
        lat_col="latitude",
        lon_col="longitude",
    )

    df.to_parquet(output_path, index=False)
    print(f"[DONE] Cleaned GFW AIS grouped data saved: {output_path}")
    return output_path
