from pathlib import Path

import pandas as pd
import holidays

from src.data.config import load_config
from src.data.paths import get_data_dirs, ensure_dir
from src.data.geo import project_to_nearest


def get_processed_dataset_path(config: dict) -> Path:
    """
    Build output path for the final processed dataset.
    """
    year = int(config["year"])

    data_dirs = get_data_dirs(config)
    processed_dir = ensure_dir(data_dirs["processed"])

    return processed_dir / f"cpr_gfw_{year}.parquet"


def get_interim_paths(config: dict) -> dict:
    """
    Build paths to cleaned/interim datasets.
    """
    year = int(config["year"])

    copernicus_products = config["copernicus"]["products"]

    data_dirs = get_data_dirs(config)
    interim_dir = data_dirs["interim"]

    return {
        "copernicus_bgc": (
            interim_dir
            / "copernicus"
            / f"cleaned_{copernicus_products[1]}_{year}.parquet"
        ),
        "copernicus_phy": (
            interim_dir
            / "copernicus"
            / f"cleaned_{copernicus_products[0]}_{year}.parquet"
        ),
        "gfw_ais": (
            interim_dir / "gfw_ais" / f"cleaned_ais_high_daily_ts_{year}.parquet"
        ),
        "gfw_sar": (
            interim_dir / "gfw_sar" / f"cleaned_s2_vessel_detections_{year}.parquet"
        ),
    }


def read_interim_datasets(
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Read cleaned Copernicus, AIS and SAR datasets.
    """
    paths = get_interim_paths(config)

    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing interim dataset '{name}': {path}")

    cpr_bgc = pd.read_parquet(paths["copernicus_bgc"])
    cpr_phy = pd.read_parquet(paths["copernicus_phy"])
    gfw_ais = pd.read_parquet(paths["gfw_ais"])
    gfw_sar = pd.read_parquet(paths["gfw_sar"])

    return cpr_bgc, cpr_phy, gfw_ais, gfw_sar


def standardize_dates(
    cpr_bgc: pd.DataFrame,
    cpr_phy: pd.DataFrame,
    gfw_ais: pd.DataFrame,
    gfw_sar: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Convert date columns to pandas datetime.
    """
    cpr_bgc = cpr_bgc.copy()
    cpr_phy = cpr_phy.copy()
    gfw_ais = gfw_ais.copy()
    gfw_sar = gfw_sar.copy()

    cpr_bgc["date"] = pd.to_datetime(cpr_bgc["date"])
    cpr_phy["date"] = pd.to_datetime(cpr_phy["date"])
    gfw_ais["date"] = pd.to_datetime(gfw_ais["date"])
    gfw_sar["date"] = pd.to_datetime(gfw_sar["date"])

    return cpr_bgc, cpr_phy, gfw_ais, gfw_sar


def build_copernicus_grid(
    cpr_bgc: pd.DataFrame,
    cpr_phy: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the base spatio-temporal grid from Copernicus.

    The grid is defined by unique combinations of:
    date, latitude, longitude.
    """
    keys = ["date", "latitude", "longitude"]

    df = cpr_bgc.drop_duplicates(subset=keys)[keys].copy()

    df = pd.merge(
        left=df,
        right=cpr_bgc[keys + ["chl"]],
        how="left",
        on=keys,
    )

    df = pd.merge(
        left=df,
        right=cpr_phy[keys + ["thetao"]],
        how="left",
        on=keys,
    )

    return df


def project_vessels_to_copernicus_grid(
    vessels: pd.DataFrame,
    grid: pd.DataFrame,
) -> pd.DataFrame:
    """
    Project vessel detections to the nearest Copernicus grid point.
    """
    spatial_keys = ["latitude", "longitude"]

    vessels = vessels.copy()

    vessel_unique_coords = vessels.drop_duplicates(subset=spatial_keys)[
        spatial_keys
    ].copy()
    grid_unique_coords = grid.drop_duplicates(subset=spatial_keys)[spatial_keys].copy()

    coords_projection = project_to_nearest(
        coords_to_project=vessel_unique_coords,
        coords_target=grid_unique_coords,
        lat_col="latitude",
        lon_col="longitude",
        return_distance=False,
    )

    vessels = pd.merge(
        left=vessels,
        right=coords_projection,
        how="left",
        on=spatial_keys,
    )

    vessels = vessels.drop(columns=spatial_keys)

    vessels = vessels.rename(
        columns={
            "lat_nearest": "latitude",
            "lon_nearest": "longitude",
        }
    )

    return vessels


def add_sar_counts(
    df: pd.DataFrame,
    gfw_sar: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add SAR vessel counts per date and Copernicus grid cell.
    """
    keys = ["date", "latitude", "longitude"]

    gfw_sar_projected = project_vessels_to_copernicus_grid(
        vessels=gfw_sar,
        grid=df,
    )

    sar_counts = (
        gfw_sar_projected.groupby(keys).size().reset_index(name="sar_vessels_count")
    )

    out = pd.merge(
        left=df,
        right=sar_counts,
        how="left",
        on=keys,
    )

    out["sar_vessels_count"] = out["sar_vessels_count"].fillna(0).astype(int)

    return out


def add_ais_counts(
    df: pd.DataFrame,
    gfw_ais: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add AIS vessel counts and gear-type counts per date and Copernicus grid cell.
    """
    keys = ["date", "latitude", "longitude"]

    gfw_ais_projected = project_vessels_to_copernicus_grid(
        vessels=gfw_ais,
        grid=df,
    )

    ais_totals = gfw_ais_projected.groupby(keys).size().rename("ais_vessels_count")

    ais_gear_counts = (
        gfw_ais_projected.groupby(keys + ["gear_type"], dropna=False)
        .size()
        .unstack("gear_type", fill_value=0)
    )

    ais_counts = ais_totals.to_frame().join(ais_gear_counts).reset_index()

    out = pd.merge(
        left=df,
        right=ais_counts,
        how="left",
        on=keys,
    )

    count_cols = [
        col
        for col in out.columns
        if col not in ["date", "latitude", "longitude", "chl", "thetao"]
    ]

    for col in count_cols:
        out[col] = out[col].fillna(0)

    out["ais_vessels_count"] = out["ais_vessels_count"].astype(int)

    gear_cols = [
        col
        for col in count_cols
        if col != "ais_vessels_count" and col != "sar_vessels_count"
    ]

    for col in gear_cols:
        out[col] = out[col].astype(int)

    return out


def add_calendar_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Add Italian holidays, weekend and fishing block information.
    """
    year = int(config["year"])

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    it_holidays = holidays.country_holidays("IT", years=year)

    df["is_holiday"] = df["date"].dt.date.isin(it_holidays)
    df["holiday_name"] = df["date"].dt.date.map(lambda d: it_holidays.get(d))
    df["is_weekend"] = df["date"].dt.dayofweek >= 5

    df["fishing_block"] = False

    if year == 2024:
        df.loc[
            (df["date"] >= pd.Timestamp(f"{year}-07-31"))
            & (df["date"] <= pd.Timestamp(f"{year}-09-13")),
            "fishing_block",
        ] = True

    return df


def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reorder final dataset columns.

    Gear columns are kept dynamically, because the set of gear types
    may change across years or filters.
    """
    base_cols = [
        "date",
        "is_holiday",
        "holiday_name",
        "is_weekend",
        "fishing_block",
        "latitude",
        "longitude",
        "chl",
        "thetao",
        "sar_vessels_count",
        "ais_vessels_count",
    ]

    gear_cols = [col for col in df.columns if col not in base_cols]

    gear_cols = sorted(gear_cols)

    return df[base_cols + gear_cols]


def build_unique_dataset(config: dict | None = None) -> Path:
    """
    Build the final processed AIPS dataset.
    """
    if config is None:
        config = load_config()

    output_path = get_processed_dataset_path(config)

    overwrite = bool(config.get("processed", {}).get("overwrite", False))

    if output_path.exists() and not overwrite:
        print(f"[SKIP] Processed dataset already exists: {output_path}")
        return output_path

    print("[START] Building final AIPS dataset")

    cpr_bgc, cpr_phy, gfw_ais, gfw_sar = read_interim_datasets(config)

    print(f"[INFO] Copernicus BGC shape: {cpr_bgc.shape}")
    print(f"[INFO] Copernicus PHY shape: {cpr_phy.shape}")
    print(f"[INFO] GFW AIS shape:        {gfw_ais.shape}")
    print(f"[INFO] GFW SAR shape:        {gfw_sar.shape}")

    cpr_bgc, cpr_phy, gfw_ais, gfw_sar = standardize_dates(
        cpr_bgc=cpr_bgc,
        cpr_phy=cpr_phy,
        gfw_ais=gfw_ais,
        gfw_sar=gfw_sar,
    )

    df = build_copernicus_grid(
        cpr_bgc=cpr_bgc,
        cpr_phy=cpr_phy,
    )

    print(f"[INFO] Base Copernicus grid shape: {df.shape}")

    df = add_sar_counts(
        df=df,
        gfw_sar=gfw_sar,
    )

    print(f"[INFO] After SAR merge shape: {df.shape}")

    df = add_ais_counts(
        df=df,
        gfw_ais=gfw_ais,
    )

    print(f"[INFO] After AIS merge shape: {df.shape}")

    df = add_calendar_features(df, config)

    df = reorder_columns(df)

    print(f"[INFO] Final dataset shape: {df.shape}")
    print(f"[INFO] Final columns: {list(df.columns)}")

    print(f"[INFO] Saving processed dataset: {output_path}")
    df.to_parquet(output_path, index=False)

    print(f"[DONE] Processed dataset saved: {output_path}")

    return output_path
