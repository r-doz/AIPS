"""Build Red-LGCP datasets for new years or a new region (rebuttal), with the paper's pipeline.

Runs, for each year, the steps of scripts/01_get_data.py, 02_clean_data.py, 03_build_dataset.py and
04_add_salinity.py with the functions of src/data unchanged, on a data config of config/rebuttal/data_*.yaml:
  - Copernicus Marine Mediterranean products at the surface (0-1.5 m), daily, 4.2 km: temperature (thetao) and
    chlorophyll (chl) from the analysis-forecast system, salinity (so) from the reanalysis, as in the paper's
    data (see PRODUCTS); the grid cells are the product's sea points in the area (and above the marine
    boundary line, when the config has one);
  - Global Fishing Watch AIS fishing effort (public-global-fishing-effort), 0.01 deg, daily; Italian vessels; each
    (vessel, 0.01 deg cell, day) is projected to the nearest grid cell and counted (ais_vessels_count). The
    paper downloads one record per vessel (group_by VESSEL_ID). With ais_group_by: FLAG in the region config
    (GFW has paused per-vessel reports, Nov. 2026), one record per (flag, 0.01 deg cell, day) is downloaded
    and repeated vessel_ids times (its number of distinct vessels), which gives the same count; per-gear
    counts are then unavailable and their columns are omitted (no model uses them);
  - calendar: Italian holidays, weekend, and the trawling ban (fishing_block). The paper's builder uses
    31 July - 13 September every year; here the ban dates come from the config, per year.
Differences from the paper's pipeline, none of which affects the model's inputs:
  - the Sentinel-2 SAR detections (BigQuery) are not downloaded: sar_vessels_count is left empty (NaN);
    no model uses it;
  - a region without a marine boundary keeps every sea point of the area.
All files go to the config's own data folders (data/rebuttal/<name>/{raw,interim,processed}); existing raw
files are reused, not downloaded again.

Usage: python build_region_data.py --config config/rebuttal/data_<name>.yaml --years 2022 2023 [--no-download]
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import pandas as pd
import yaml

from svgp_residual import ROOT
from src.data import build_dataset as bd
from src.data.copernicus import clean_copernicus_product, download_copernicus_product, get_copernicus_raw_paths
from src.data.gfw_ais import clean_gfw_ais_data, download_gfw_ais_data, get_gfw_ais_raw_path, read_gfw_token
from src.data.geo import bbox_to_geojson
from src.data.paths import ensure_dir

# The products behind the paper's data (data/processed/cpr_gfw_salinity_2024.parquet is reproduced exactly from
# them): temperature and chlorophyll from the analysis-forecast products, salinity from the reanalysis
# (scripts/04_add_salinity.py). config/data.yaml lists the reanalysis products instead; a region config can
# override them with "products".
PRODUCTS = {"phy": "cmems_mod_med_phy-tem_anfc_4.2km_P1D-m", "bgc": "cmems_mod_med_bgc-pft_anfc_4.2km_P1D-m",
            "sal": "cmems_mod_med_phy-sal_my_4.2km_P1D-m"}
NO_BOUNDARY = "none"


def products(region: dict) -> dict:
    return {**PRODUCTS, **region.get("products", {})}


def year_config(region: dict, year: int) -> dict:
    """The paper's data config (config/data.yaml) with the region's area, folders, boundary and year."""
    cfg = yaml.safe_load(open(ROOT / "config/data.yaml"))
    cfg = copy.deepcopy(cfg)
    cfg["year"] = year
    cfg["area"] = region["area"]
    cfg["paths"].update({k: str(Path(region["data_root"]) / k) for k in ("raw", "interim", "processed")})
    cfg["static_files"]["marine_bounds"] = region.get("marine_bounds", NO_BOUNDARY)
    cfg["copernicus"]["products"] = [products(region)["phy"], products(region)["bgc"]]  # order used by build_dataset
    for block in ("copernicus", "gfw_ais"):
        cfg[block]["overwrite"] = False        # reuse raw downloads
        cfg[block]["overwrite_clean"] = True
    return cfg


def clean(product_or_ais: str, cfg: dict) -> Path:
    """Cleaning step of the paper; without a marine boundary, the boundary filter keeps every point."""
    if cfg["static_files"]["marine_bounds"] != NO_BOUNDARY:
        return clean_copernicus_product(product_or_ais, config=cfg) if product_or_ais != "ais" else clean_gfw_ais_data(cfg)
    import src.data.geo as geo
    keep = geo.filter_above_boundary
    geo.filter_above_boundary = lambda df, bounds_df, lat_col="latitude", lon_col="longitude": df
    tmp = ensure_dir(Path(cfg["paths"]["interim"])) / "no_boundary.csv"
    pd.DataFrame({"latitude": [0.0, 0.0], "longitude": [0.0, 1.0]}).to_csv(tmp, index=False)  # read but unused
    cfg = copy.deepcopy(cfg)
    cfg["static_files"]["marine_bounds"] = str(tmp)
    try:
        return clean_copernicus_product(product_or_ais, config=cfg) if product_or_ais != "ais" else clean_gfw_ais_data(cfg)
    finally:
        geo.filter_above_boundary = keep


def flag_ais_raw_path(cfg: dict) -> Path:
    return Path(cfg["paths"]["raw"]) / "gfw_ais" / f"ais_high_daily_flag_{cfg['year']}.parquet"


def download_ais_by_flag(cfg: dict) -> Path:
    """download_gfw_ais_data with group_by FLAG (same dataset, resolution, dates and area)."""
    import asyncio

    import gfwapiclient as gfw
    out = flag_ais_raw_path(cfg)
    if out.exists():
        return out
    a, y = cfg["area"], cfg["year"]

    async def run():
        client = gfw.Client(access_token=read_gfw_token(cfg))
        rep = await client.fourwings.create_report(
            datasets=[cfg["gfw_ais"]["product_name"]], spatial_resolution="HIGH", temporal_resolution="DAILY",
            group_by="FLAG", spatial_aggregation=False, start_date=f"{y}-01-01", end_date=f"{y}-12-31",
            geojson=bbox_to_geojson(a["minimum_longitude"], a["minimum_latitude"], a["maximum_longitude"], a["maximum_latitude"]))
        return rep.df()
    df = asyncio.run(run())
    ensure_dir(out.parent)
    df.to_parquet(out, index=False)
    print(f"[AIS by flag {y}] {len(df)} records → {out}", flush=True)
    return out


def clean_ais_by_flag(cfg: dict) -> pd.DataFrame:
    """clean_gfw_ais_data for flag records: Italian vessels, marine boundary, one row per vessel."""
    df = pd.read_parquet(flag_ais_raw_path(cfg)).rename(columns={"lat": "latitude", "lon": "longitude"})
    df = df[df["flag"] == "ITA"].copy()
    if cfg["static_files"]["marine_bounds"] != NO_BOUNDARY:
        from src.data.geo import filter_above_boundary
        df = filter_above_boundary(df=df, bounds_df=pd.read_csv(cfg["static_files"]["marine_bounds"]))
    df = df.loc[df.index.repeat(df["vessel_ids"].astype(int))]
    df["gear_type"] = "UNAVAILABLE"
    df["date"] = pd.to_datetime(df["date"])
    return df[["date", "latitude", "longitude", "gear_type", "hours", "flag"]].reset_index(drop=True)


def fishing_block(df: pd.DataFrame, periods: list) -> pd.Series:
    out = pd.Series(False, index=df.index)
    for start, end in periods:
        out |= (df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))
    return out


def build_year(region: dict, year: int, download: bool) -> Path:
    cfg = year_config(region, year)
    for k in ("raw", "interim", "processed"):
        ensure_dir(Path(cfg["paths"][k]))
    prods = products(region)
    by_flag = region.get("ais_group_by", "VESSEL_ID") == "FLAG"
    if download:
        for p in prods.values():
            download_copernicus_product(p, config=cfg)
        download_ais_by_flag(cfg) if by_flag else download_gfw_ais_data(cfg)
    for p in prods.values():
        assert get_copernicus_raw_paths(p, year, cfg)[1].exists(), f"missing raw {p} {year}"
        clean(p, cfg)
    paths = bd.get_interim_paths(cfg)
    if by_flag:
        ais = clean_ais_by_flag(cfg)
    else:
        assert get_gfw_ais_raw_path(cfg).exists(), f"missing raw AIS {year}"
        clean("ais", cfg)
        ais = pd.read_parquet(paths["gfw_ais"])
    bgc, phy = (pd.read_parquet(paths[k]) for k in ("copernicus_bgc", "copernicus_phy"))
    for d in (bgc, phy, ais):
        d["date"] = pd.to_datetime(d["date"])
    df = bd.build_copernicus_grid(cpr_bgc=bgc, cpr_phy=phy)
    df["sar_vessels_count"] = float("nan")   # SAR not downloaded (not used by any model)
    df = bd.add_ais_counts(df=df, gfw_ais=ais)
    df = bd.add_calendar_features(df, cfg)
    df["fishing_block"] = fishing_block(df, region["fishing_block"][year])
    df = bd.reorder_columns(df)
    if by_flag:
        df = df.drop(columns=["UNAVAILABLE"])  # per-gear counts are not available from flag records
    count_cols = ["sar_vessels_count", "ais_vessels_count"] + [c for c in df.columns if c.isupper()]  # gear counts
    df[count_cols] = df[count_cols].astype(float)  # float, as in the paper's files
    sal = pd.read_parquet(Path(cfg["paths"]["interim"]) / "copernicus" / f"cleaned_{prods['sal']}_{year}.parquet")
    sal["date"] = pd.to_datetime(sal["date"])
    keys = ["date", "latitude", "longitude"]
    df = df.merge(sal[keys + ["so"]].rename(columns={"so": "salinity"}), how="left", on=keys)
    out = Path(cfg["paths"]["processed"]) / f"cpr_gfw_salinity_{year}.parquet"
    df.to_parquet(out, index=False)
    print(f"[{region['name']} {year}] {df.shape}, cells {df[['latitude', 'longitude']].drop_duplicates().shape[0]}, "
          f"active share {(df.ais_vessels_count > 0).mean():.3f}, mean count {df.ais_vessels_count.mean():.3f}, "
          f"missing covariates {int(df[['chl', 'thetao', 'salinity']].isna().sum().sum())}, "
          f"ban days {df.loc[df.fishing_block, 'date'].nunique()} → {out}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--years", type=int, nargs="+", required=True)
    ap.add_argument("--no-download", action="store_true")
    args = ap.parse_args()
    region = yaml.safe_load(open(ROOT / args.config))
    region["data_root"] = str(ROOT / region["data_root"])
    if region.get("marine_bounds", NO_BOUNDARY) != NO_BOUNDARY:
        region["marine_bounds"] = str(ROOT / region["marine_bounds"])
    for y in args.years:
        build_year(region, y, download=not args.no_download)


if __name__ == "__main__":
    main()
