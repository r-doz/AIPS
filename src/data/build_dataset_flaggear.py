from pathlib import Path

import pandas as pd

from src.data.build_dataset import (
    add_calendar_features,
    add_sar_counts,
    build_copernicus_grid,
    get_interim_paths,
    get_processed_dataset_path,
    reorder_columns,
    standardize_dates,
)
from src.data.config import load_config
from src.data.gfw_ais_flaggear import get_gfw_ais_flaggear_interim_path
from src.data.paths import get_data_dirs


def add_grouped_ais_counts(
    grid: pd.DataFrame,
    ais_groups: pd.DataFrame,
) -> pd.DataFrame:
    """Sum GFW vesselIDs counts after projecting groups to Copernicus cells.

    Counts are summed across flag/gear buckets. This preserves the source
    bucket counts but does not deduplicate vessel identities across gear types.
    """
    keys = ["date", "latitude", "longitude"]
    ais = ais_groups.copy()
    ais["gear_type"] = ais["gear_type"].fillna("unknown").astype(str).str.upper()

    from src.data.build_dataset import project_vessels_to_copernicus_grid

    projected = project_vessels_to_copernicus_grid(vessels=ais, grid=grid)
    totals = projected.groupby(keys)["vessel_count"].sum().rename("ais_vessels_count")
    gear_counts = (
        projected.groupby(keys + ["gear_type"])["vessel_count"]
        .sum()
        .unstack("gear_type", fill_value=0)
    )
    counts = totals.to_frame().join(gear_counts).reset_index()

    out = grid.merge(counts, on=keys, how="left")
    count_columns = [
        column
        for column in out.columns
        if column not in ["date", "latitude", "longitude", "chl", "thetao"]
    ]
    out[count_columns] = out[count_columns].fillna(0)
    out["ais_vessels_count"] = out["ais_vessels_count"].astype("int64")
    for column in gear_counts.columns:
        out[column] = out[column].astype("int64")
    return out


def build_unique_dataset_flaggear(config: dict | None = None) -> Path:
    if config is None:
        config = load_config("config/data_ebro_vGearAndFlag.yaml")

    output_path = get_processed_dataset_path(config)
    if output_path.exists() and not config.get("processed", {}).get("overwrite", False):
        print(f"[SKIP] Processed dataset already exists: {output_path}")
        return output_path

    interim_paths = get_interim_paths(config)
    interim_paths["gfw_ais"] = get_gfw_ais_flaggear_interim_path(config)
    salinity_product = config["copernicus"].get("salinity_product")
    if not salinity_product:
        raise ValueError("Configure copernicus.salinity_product for this pipeline.")
    salinity_path = (
        get_data_dirs(config)["interim"]
        / "copernicus"
        / f"cleaned_{salinity_product}_{config['year']}.parquet"
    )
    interim_paths["copernicus_salinity"] = salinity_path
    for name, path in interim_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing interim dataset '{name}': {path}")

    cpr_bgc = pd.read_parquet(interim_paths["copernicus_bgc"])
    cpr_phy = pd.read_parquet(interim_paths["copernicus_phy"])
    cpr_salinity = pd.read_parquet(interim_paths["copernicus_salinity"])
    ais_groups = pd.read_parquet(interim_paths["gfw_ais"])
    gfw_sar = pd.read_parquet(interim_paths["gfw_sar"])

    cpr_bgc, cpr_phy, ais_groups, gfw_sar = standardize_dates(
        cpr_bgc=cpr_bgc,
        cpr_phy=cpr_phy,
        gfw_ais=ais_groups,
        gfw_sar=gfw_sar,
    )
    cpr_salinity["date"] = pd.to_datetime(cpr_salinity["date"])
    grid = build_copernicus_grid(cpr_bgc, cpr_phy)
    keys = ["date", "latitude", "longitude"]
    grid = grid.merge(
        cpr_salinity[keys + ["so"]].rename(columns={"so": "salinity"}),
        on=keys,
        how="left",
    )
    grid = add_sar_counts(grid, gfw_sar)
    grid = add_grouped_ais_counts(grid, ais_groups)
    grid = add_calendar_features(grid, config)
    dataset = reorder_columns(grid)
    columns = list(dataset.columns)
    columns.remove("salinity")
    columns.insert(columns.index("thetao") + 1, "salinity")
    dataset = dataset[columns]

    processed_dir = get_data_dirs(config)["processed"]
    processed_dir.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(output_path, index=False)
    print(f"[DONE] FLAGANDGEARTYPE dataset saved: {output_path}")
    return output_path
