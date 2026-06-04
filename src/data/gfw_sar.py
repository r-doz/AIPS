from pathlib import Path

import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account

from src.data.config import load_config
from src.data.paths import get_data_dirs, ensure_dir, project_path


def get_gfw_sar_raw_path(config: dict) -> Path:
    """
    Build raw output path for GFW SAR data.
    """
    year = int(config["year"])
    sar_cfg = config["gfw_sar"]

    data_dirs = get_data_dirs(config)
    raw_sar_dir = ensure_dir(data_dirs["raw"] / "gfw_sar")

    output_name = sar_cfg.get("output_name", "s2_vessel_detections")
    filename = f"{output_name}_{year}.parquet"

    return raw_sar_dir / filename


def get_bigquery_client(config: dict) -> bigquery.Client:
    """
    Create a BigQuery client using the service account key stored in secrets.
    """
    key_path = project_path(config["gfw_sar"]["key_file"])

    if not key_path.exists():
        raise FileNotFoundError(
            f"BigQuery key file not found: {key_path}\n"
            "Expected something like: secrets/gcp_bigquery_key.json"
        )

    creds = service_account.Credentials.from_service_account_file(
        key_path,
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )

    project_id = creds.project_id

    client = bigquery.Client(
        project=project_id,
        credentials=creds,
        location="US",
    )

    return client


def build_gfw_sar_query(config: dict) -> tuple[str, bigquery.QueryJobConfig]:
    """
    Build the BigQuery SQL query for GFW SAR detections.

    The column names follow the GFW SAR notebook:
    detect_timestamp, detect_lat, detect_lon.
    """
    year = int(config["year"])
    area = config["area"]
    dates = config["dates"]
    sar_cfg = config["gfw_sar"]

    start_datetime = f"{year}-{dates['start_month']}-{dates['start_day']}T00:00:00"
    end_datetime = f"{year}-{dates['end_month']}-{dates['end_day']}T23:59:59"

    table = sar_cfg["table"]

    query = f"""
    SELECT
        ssvid,
        detect_timestamp,
        detect_lat,
        detect_lon,
        speed_kn_inferred,
        heading_deg_inferred,
        length_m_inferred,
        presence_score,
        matching_score,
        cloud_score,
        likely_infrastructure,
        date
    FROM `{table}`
    WHERE
        detect_timestamp BETWEEN @start_datetime AND @end_datetime
        AND detect_lon BETWEEN @lon_min AND @lon_max
        AND detect_lat BETWEEN @lat_min AND @lat_max
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter(
                "start_datetime",
                "TIMESTAMP",
                start_datetime,
            ),
            bigquery.ScalarQueryParameter(
                "end_datetime",
                "TIMESTAMP",
                end_datetime,
            ),
            bigquery.ScalarQueryParameter(
                "lon_min",
                "FLOAT64",
                area["minimum_longitude"],
            ),
            bigquery.ScalarQueryParameter(
                "lon_max",
                "FLOAT64",
                area["maximum_longitude"],
            ),
            bigquery.ScalarQueryParameter(
                "lat_min",
                "FLOAT64",
                area["minimum_latitude"],
            ),
            bigquery.ScalarQueryParameter(
                "lat_max",
                "FLOAT64",
                area["maximum_latitude"],
            ),
        ]
    )

    return query, job_config


def download_gfw_sar_data(config: dict | None = None) -> Path:
    """
    Download GFW SAR detections from BigQuery and save them in data/raw/gfw_sar.
    """
    if config is None:
        config = load_config()

    output_path = get_gfw_sar_raw_path(config)

    overwrite = bool(config["gfw_sar"].get("overwrite", False))

    if output_path.exists() and not overwrite:
        print(f"[SKIP] GFW SAR raw file already exists: {output_path}")
        return output_path

    query, job_config = build_gfw_sar_query(config)

    print("[START] Downloading GFW SAR data")
    print("[INFO] Running BigQuery query")

    client = get_bigquery_client(config)

    job = client.query(query, job_config=job_config)
    df = job.result().to_dataframe(create_bqstorage_client=True)

    # Standardize temporal columns before saving.
    # This avoids BigQuery-specific pandas extension dtypes such as 'dbdate'.
    if "detect_timestamp" in df.columns:
        df["detect_timestamp"] = pd.to_datetime(df["detect_timestamp"])

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    print(f"[INFO] Downloaded SAR rows: {df.shape[0]}")
    print(f"[INFO] SAR columns: {list(df.columns)}")

    print(f"[INFO] Saving GFW SAR raw file: {output_path}")
    df.to_parquet(output_path, index=False)

    print(f"[DONE] GFW SAR data saved: {output_path}")

    return output_path


# ---


def get_gfw_sar_interim_path(config: dict) -> Path:
    """
    Build interim output path for cleaned GFW SAR data.
    """
    year = int(config["year"])
    sar_cfg = config["gfw_sar"]

    data_dirs = get_data_dirs(config)
    interim_sar_dir = ensure_dir(data_dirs["interim"] / "gfw_sar")

    output_name = sar_cfg.get("output_name", "s2_vessel_detections")
    filename = f"cleaned_{output_name}_{year}.parquet"

    return interim_sar_dir / filename


def clean_gfw_sar_data(config: dict | None = None) -> Path:
    """
    Clean raw GFW SAR data.

    Steps:
    - drop detect_timestamp;
    - rename detect_lat/detect_lon to latitude/longitude;
    - apply quality filters;
    - drop filtering metadata columns;
    - keep only points above the Italy marine boundary;
    - save to data/interim/gfw_sar.
    """
    if config is None:
        config = load_config()

    raw_path = get_gfw_sar_raw_path(config)
    output_path = get_gfw_sar_interim_path(config)

    overwrite = bool(config["gfw_sar"].get("overwrite_clean", False))

    if output_path.exists() and not overwrite:
        print(f"[SKIP] Cleaned GFW SAR file already exists: {output_path}")
        return output_path

    if not raw_path.exists():
        raise FileNotFoundError(f"Raw GFW SAR file not found: {raw_path}")

    bounds_path = project_path(config["static_files"]["marine_bounds"])

    if not bounds_path.exists():
        raise FileNotFoundError(f"Marine bounds file not found: {bounds_path}")

    print("[START] Cleaning GFW SAR data")

    df = pd.read_parquet(raw_path)

    if "detect_timestamp" in df.columns:
        df = df.drop(columns=["detect_timestamp"])

    df = df.rename(
        columns={
            "detect_lat": "latitude",
            "detect_lon": "longitude",
        }
    )

    cols_ordered = [
        "date",
        "latitude",
        "longitude",
        "ssvid",
        "speed_kn_inferred",
        "heading_deg_inferred",
        "length_m_inferred",
        "presence_score",
        "matching_score",
        "cloud_score",
        "likely_infrastructure",
    ]

    df = df[cols_ordered]

    n_total = len(df)

    spd_mask = df["speed_kn_inferred"] < 9
    len_mask = df["length_m_inferred"] < 25
    prs_mask = df["presence_score"] > 0.5
    cld_mask = df["cloud_score"] < 0.5
    inf_mask = df["likely_infrastructure"] == False

    masks = {
        "speed < 9": spd_mask,
        "length < 25": len_mask,
        "presence > 0.5": prs_mask,
        "cloud < 0.5": cld_mask,
        "not infrastructure": inf_mask,
    }

    for name, mask in masks.items():
        n_pass = int(mask.sum())
        n_removed = n_total - n_pass
        perc_removed = 100 * n_removed / n_total if n_total > 0 else 0
        print(f"[INFO] {name:25s} removed {n_removed:8d} rows ({perc_removed:6.2f}%)")

    full_mask = spd_mask & len_mask & prs_mask & cld_mask & inf_mask
    n_after = int(full_mask.sum())

    print("[INFO] Combined SAR filters:")
    print(f"[INFO] Records before filtering: {n_total}")
    print(f"[INFO] Records after filtering:  {n_after}")
    print(f"[INFO] Records removed:          {n_total - n_after}")

    df = df.loc[full_mask].copy()

    cols_to_drop = [
        "matching_score",
        "cloud_score",
        "likely_infrastructure",
    ]

    df = df.drop(columns=cols_to_drop)

    bounds_df = pd.read_csv(bounds_path)

    from src.data.geo import filter_above_boundary

    df = filter_above_boundary(
        df=df,
        bounds_df=bounds_df,
        lat_col="latitude",
        lon_col="longitude",
    )

    print(f"[INFO] Cleaned GFW SAR shape: {df.shape}")
    print(f"[INFO] Saving cleaned GFW SAR file: {output_path}")

    df.to_parquet(output_path, index=False)

    print(f"[DONE] Cleaned GFW SAR data saved: {output_path}")

    return output_path
