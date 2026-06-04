from pathlib import Path
import asyncio

import gfwapiclient as gfw
import pandas as pd

from src.data.config import load_config
from src.data.paths import get_data_dirs, ensure_dir, project_path
from src.data.geo import bbox_to_geojson


def get_gfw_ais_raw_path(config: dict) -> Path:
    """
    Build raw output path for GFW AIS data.
    """
    year = int(config["year"])
    ais_cfg = config["gfw_ais"]

    spatial_resolution = ais_cfg["spatial_resolution"].lower()
    temporal_resolution = ais_cfg["temporal_resolution"].lower()

    data_dirs = get_data_dirs(config)
    raw_ais_dir = ensure_dir(data_dirs["raw"] / "gfw_ais")

    filename = f"ais_{spatial_resolution}_{temporal_resolution}_ts_{year}.parquet"

    return raw_ais_dir / filename


def read_gfw_token(config: dict) -> str:
    """
    Read the Global Fishing Watch API token from the secrets folder.
    """
    token_path = project_path(config["gfw_ais"]["token_file"])

    if not token_path.exists():
        raise FileNotFoundError(
            f"GFW token file not found: {token_path}\n"
            "Create it and paste your token inside."
        )

    return token_path.read_text().strip()


async def _download_gfw_ais_async(config: dict) -> pd.DataFrame:
    """
    Internal async function that calls the GFW FourWings API.
    """
    year = int(config["year"])
    area = config["area"]
    dates = config["dates"]
    ais_cfg = config["gfw_ais"]

    start_date = f"{year}-{dates['start_month']}-{dates['start_day']}"
    end_date = f"{year}-{dates['end_month']}-{dates['end_day']}"

    token = read_gfw_token(config)
    gfw_client = gfw.Client(access_token=token)

    bbox = (
        area["minimum_longitude"],
        area["minimum_latitude"],
        area["maximum_longitude"],
        area["maximum_latitude"],
    )

    print("[START] Downloading GFW AIS data")
    print(f"[INFO] Date range: {start_date} - {end_date}")
    print(f"[INFO] Spatial resolution: {ais_cfg['spatial_resolution']}")
    print(f"[INFO] Temporal resolution: {ais_cfg['temporal_resolution']}")
    print(f"[INFO] Group by: {ais_cfg['group_by']}")

    ais_report = await gfw_client.fourwings.create_report(
        datasets=[ais_cfg["product_name"]],
        spatial_resolution=ais_cfg["spatial_resolution"],
        temporal_resolution=ais_cfg["temporal_resolution"],
        group_by=ais_cfg["group_by"],
        spatial_aggregation=False,
        start_date=start_date,
        end_date=end_date,
        geojson=bbox_to_geojson(*bbox),
    )

    df = ais_report.df()

    print(f"[INFO] Downloaded AIS rows: {df.shape[0]}")
    print(f"[INFO] AIS columns: {list(df.columns)}")

    return df


def download_gfw_ais_data(config: dict | None = None) -> Path:
    """
    Download GFW AIS data and save it in data/raw/gfw_ais.
    """
    if config is None:
        config = load_config()

    output_path = get_gfw_ais_raw_path(config)

    overwrite = bool(config["gfw_ais"].get("overwrite", False))

    if output_path.exists() and not overwrite:
        print(f"[SKIP] GFW AIS raw file already exists: {output_path}")
        return output_path

    df = asyncio.run(_download_gfw_ais_async(config))

    print(f"[INFO] Saving GFW AIS raw file: {output_path}")
    df.to_parquet(output_path, index=False)

    print(f"[DONE] GFW AIS data saved: {output_path}")

    return output_path


# ---
def get_gfw_ais_interim_path(config: dict) -> Path:
    """
    Build interim output path for cleaned GFW AIS data.
    """
    year = int(config["year"])
    ais_cfg = config["gfw_ais"]

    spatial_resolution = ais_cfg["spatial_resolution"].lower()
    temporal_resolution = ais_cfg["temporal_resolution"].lower()

    data_dirs = get_data_dirs(config)
    interim_ais_dir = ensure_dir(data_dirs["interim"] / "gfw_ais")

    filename = (
        f"cleaned_ais_{spatial_resolution}_{temporal_resolution}_ts_{year}.parquet"
    )

    return interim_ais_dir / filename


def clean_gfw_ais_data(config: dict | None = None) -> Path:
    """
    Clean raw GFW AIS data.

    Steps:
    - drop unnecessary metadata columns;
    - rename lat/lon to latitude/longitude;
    - keep Italian vessels only;
    - keep only points above the Italy marine boundary;
    - save to data/interim/gfw_ais.
    """
    if config is None:
        config = load_config()

    raw_path = get_gfw_ais_raw_path(config)
    output_path = get_gfw_ais_interim_path(config)

    overwrite = bool(config["gfw_ais"].get("overwrite_clean", False))

    if output_path.exists() and not overwrite:
        print(f"[SKIP] Cleaned GFW AIS file already exists: {output_path}")
        return output_path

    if not raw_path.exists():
        raise FileNotFoundError(f"Raw GFW AIS file not found: {raw_path}")

    bounds_path = project_path(config["static_files"]["marine_bounds"])

    if not bounds_path.exists():
        raise FileNotFoundError(f"Marine bounds file not found: {bounds_path}")

    print("[START] Cleaning GFW AIS data")

    df = pd.read_parquet(raw_path)

    cols_to_drop = [
        "detections",
        "vessel_ids",
        "entry_timestamp",
        "exit_timestamp",
        "last_transmission_date",
        "first_transmission_date",
        "imo",
        "call_sign",
        "dataset",
        "report_dataset",
        "vessel_type",
        "vessel_id",
    ]

    existing_cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    df = df.drop(columns=existing_cols_to_drop)

    df = df.rename(
        columns={
            "lat": "latitude",
            "lon": "longitude",
        }
    )

    cols = [
        "date",
        "latitude",
        "longitude",
        "gear_type",
        "mmsi",
        "ship_name",
        "hours",
        "flag",
    ]

    df = df[cols]

    df = df[df["flag"] == "ITA"].copy()

    bounds_df = pd.read_csv(bounds_path)

    from src.data.geo import filter_above_boundary

    df = filter_above_boundary(
        df=df,
        bounds_df=bounds_df,
        lat_col="latitude",
        lon_col="longitude",
    )

    print(f"[INFO] Cleaned GFW AIS shape: {df.shape}")
    print(f"[INFO] Saving cleaned GFW AIS file: {output_path}")

    df.to_parquet(output_path, index=False)

    print(f"[DONE] Cleaned GFW AIS data saved: {output_path}")

    return output_path
