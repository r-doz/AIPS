from pathlib import Path
import os

import pandas as pd
import xarray as xr
from copernicusmarine import subset

from src.data.config import load_config
from src.data.paths import get_data_dirs, ensure_dir


def get_copernicus_raw_paths(
    product_name: str,
    year: int,
    config: dict,
) -> tuple[Path, Path]:
    """
    Build raw output paths for a Copernicus product.

    Returns
    -------
    tuple[Path, Path]
        NetCDF output path and Parquet output path.
    """
    data_dirs = get_data_dirs(config)

    raw_copernicus_dir = ensure_dir(data_dirs["raw"] / "copernicus")

    nc_path = raw_copernicus_dir / f"{product_name}_{year}.nc"
    parquet_path = raw_copernicus_dir / f"{product_name}_{year}.parquet"

    return nc_path, parquet_path


def download_copernicus_product(
    product_name: str,
    config: dict | None = None,
) -> Path:
    """
    Download one Copernicus product, convert it to Parquet,
    and save it in data/raw/copernicus.

    The raw Parquet file is still minimally filtered:
    points where the marine variable is missing are removed.
    Full cleaning is done later in the cleaning step.
    """
    if config is None:
        config = load_config()

    year = int(config["year"])

    area = config["area"]
    dates = config["dates"]
    copernicus_cfg = config["copernicus"]

    nc_path, parquet_path = get_copernicus_raw_paths(
        product_name=product_name,
        year=year,
        config=config,
    )

    overwrite = bool(copernicus_cfg.get("overwrite", False))

    if parquet_path.exists() and not overwrite:
        print(f"[SKIP] Copernicus raw file already exists: {parquet_path}")
        return parquet_path

    print(f"[START] Downloading Copernicus product: {product_name}")

    subset(
        dataset_id=product_name,
        minimum_longitude=area["minimum_longitude"],
        maximum_longitude=area["maximum_longitude"],
        minimum_latitude=area["minimum_latitude"],
        maximum_latitude=area["maximum_latitude"],
        start_datetime=(f"{year}-{dates['start_month']}-{dates['start_day']}T00:00:00"),
        end_datetime=(f"{year}-{dates['end_month']}-{dates['end_day']}T23:59:59"),
        minimum_depth=copernicus_cfg["min_depth"],
        maximum_depth=copernicus_cfg["max_depth"],
        output_filename=str(nc_path),
    )

    print(f"[INFO] Opening NetCDF file: {nc_path}")
    ds = xr.open_dataset(nc_path)

    print("[INFO] Converting xarray Dataset to pandas DataFrame")
    df = ds.to_dataframe().reset_index()

    ds.close()

    marine_col = list(df.columns)[-1]

    print(f"[INFO] Removing rows with missing values in variable: {marine_col}")
    df = df[df[marine_col].notna()].copy()

    print(f"[INFO] Saving raw Parquet file: {parquet_path}")
    df.to_parquet(parquet_path, index=False)

    remove_nc = bool(copernicus_cfg.get("remove_nc_after_conversion", True))

    if remove_nc and nc_path.exists():
        print(f"[INFO] Removing temporary NetCDF file: {nc_path}")
        os.remove(nc_path)

    print(f"[DONE] Copernicus product saved: {parquet_path}")

    return parquet_path


def download_copernicus_data(config: dict | None = None) -> list[Path]:
    """
    Download all Copernicus products listed in config/data.yaml.
    """
    if config is None:
        config = load_config()

    products = config["copernicus"]["products"]

    output_paths = []

    for product_name in products:
        output_path = download_copernicus_product(
            product_name=product_name,
            config=config,
        )
        output_paths.append(output_path)

    return output_paths


# -----


def get_copernicus_interim_path(
    product_name: str,
    year: int,
    config: dict,
) -> Path:
    """
    Build interim output path for a cleaned Copernicus product.
    """
    data_dirs = get_data_dirs(config)

    interim_copernicus_dir = ensure_dir(data_dirs["interim"] / "copernicus")

    filename = f"cleaned_{product_name}_{year}.parquet"

    return interim_copernicus_dir / filename


def clean_copernicus_product(
    product_name: str,
    config: dict | None = None,
) -> Path:
    """
    Clean one raw Copernicus product.

    Steps:
    - read raw parquet;
    - check that only one depth level is present;
    - drop depth;
    - rename time to date;
    - keep only points above the Italy marine boundary;
    - save to data/interim/copernicus.
    """
    if config is None:
        config = load_config()

    year = int(config["year"])

    _, raw_path = get_copernicus_raw_paths(
        product_name=product_name,
        year=year,
        config=config,
    )

    output_path = get_copernicus_interim_path(
        product_name=product_name,
        year=year,
        config=config,
    )

    overwrite = bool(config["copernicus"].get("overwrite_clean", False))

    if output_path.exists() and not overwrite:
        print(f"[SKIP] Cleaned Copernicus file already exists: {output_path}")
        return output_path

    if not raw_path.exists():
        raise FileNotFoundError(f"Raw Copernicus file not found: {raw_path}")

    bounds_path = Path(config["static_files"]["marine_bounds"])

    if not bounds_path.is_absolute():
        from src.data.paths import project_path

        bounds_path = project_path(bounds_path)

    if not bounds_path.exists():
        raise FileNotFoundError(f"Marine bounds file not found: {bounds_path}")

    print(f"[START] Cleaning Copernicus product: {product_name}")

    df = pd.read_parquet(raw_path)

    if "depth" in df.columns:
        if len(df["depth"].dropna().unique()) != 1:
            raise ValueError(
                f"Copernicus product {product_name} contains multiple depth levels."
            )

        df = df.drop(columns=["depth"])

    df = df.rename(columns={"time": "date"})

    bounds_df = pd.read_csv(bounds_path)

    from src.data.geo import filter_above_boundary

    df = filter_above_boundary(
        df=df,
        bounds_df=bounds_df,
        lat_col="latitude",
        lon_col="longitude",
    )

    print(f"[INFO] Cleaned Copernicus shape: {df.shape}")
    print(f"[INFO] Saving cleaned Copernicus file: {output_path}")

    df.to_parquet(output_path, index=False)

    print(f"[DONE] Cleaned Copernicus product saved: {output_path}")

    return output_path


def clean_copernicus_data(config: dict | None = None) -> list[Path]:
    """
    Clean all Copernicus products listed in config/data.yaml.
    """
    if config is None:
        config = load_config()

    products = config["copernicus"]["products"]

    output_paths = []

    for product_name in products:
        output_path = clean_copernicus_product(
            product_name=product_name,
            config=config,
        )
        output_paths.append(output_path)

    return output_paths
