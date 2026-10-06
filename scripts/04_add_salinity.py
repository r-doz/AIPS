"""Download + clean the Mediterranean Sea salinity reanalysis product
(cmems_mod_med_phy-sal_my_4.2km_P1D-m) for 2024 and 2025, then merge it
onto the processed datasets (data/processed/cpr_gfw_{year}.parquet, built by
scripts/03_build_dataset.py) by (date, latitude, longitude), producing
data/processed/cpr_gfw_salinity_{year}.parquet, the files used by all experiments.

Usage:
    python scripts/04_add_salinity.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.copernicus import download_copernicus_product, clean_copernicus_product  # noqa: E402

BASE_CONFIG = yaml.safe_load((PROJECT_ROOT / "config/data.yaml").read_text())
PRODUCT_NAME = "cmems_mod_med_phy-sal_my_4.2km_P1D-m"


def build_year_config(year: int) -> dict:
    cfg = dict(BASE_CONFIG)
    cfg["year"] = year
    cfg["copernicus"] = dict(BASE_CONFIG["copernicus"])
    cfg["copernicus"]["products"] = [PRODUCT_NAME]
    return cfg


def main():
    for year in (2024, 2025):
        print(f"=== Year {year} ===", flush=True)
        cfg = build_year_config(year)

        download_copernicus_product(PRODUCT_NAME, config=cfg)
        interim_path = clean_copernicus_product(PRODUCT_NAME, config=cfg)

        salinity_df = pd.read_parquet(interim_path)
        print(f"  Cleaned salinity shape: {salinity_df.shape}, columns: {list(salinity_df.columns)}")

        processed_path = PROJECT_ROOT / f"data/processed/cpr_gfw_{year}.parquet"
        processed_df = pd.read_parquet(processed_path)

        salinity_df["date"] = pd.to_datetime(salinity_df["date"])
        processed_df["date"] = pd.to_datetime(processed_df["date"])

        keys = ["date", "latitude", "longitude"]
        merged = pd.merge(
            processed_df,
            salinity_df[keys + ["so"]].rename(columns={"so": "salinity"}),
            how="left",
            on=keys,
        )

        n_missing = merged["salinity"].isna().sum()
        print(f"  Merged shape: {merged.shape}  |  missing salinity: {n_missing}/{len(merged)}")

        out_path = PROJECT_ROOT / f"data/processed/cpr_gfw_salinity_{year}.parquet"
        merged.to_parquet(out_path, index=False)
        print(f"  Saved -> {out_path}")


if __name__ == "__main__":
    main()
