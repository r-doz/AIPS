from pathlib import Path
import sys

import pandas as pd
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))


def read_parquet_safe(path: Path) -> pd.DataFrame:
    """
    Read a parquet file robustly.

    Some BigQuery date columns may be stored with extension dtypes
    that pandas/pyarrow cannot always convert automatically.
    """
    try:
        return pd.read_parquet(path)
    except TypeError as error:
        print(f"[WARNING] Standard pandas read failed: {error}")
        print("[INFO] Trying pyarrow fallback...")

        table = pq.read_table(path)

        # Convert problematic extension columns as plain Python objects.
        return table.to_pandas(types_mapper=None)


def print_dataset_summary(path: Path, date_cols: list[str] | None = None) -> None:
    print("=" * 80)
    print(path)
    print("=" * 80)

    if not path.exists():
        print("[MISSING]")
        print()
        return

    df = read_parquet_safe(path)

    print("shape:", df.shape)
    print("columns:", list(df.columns))
    print()

    print(df.head())
    print()

    if date_cols is not None:
        for col in date_cols:
            if col in df.columns:
                print(f"{col} min:", df[col].min())
                print(f"{col} max:", df[col].max())
                print()

    print("missing values:")
    print(df.isna().sum())
    print()


def main() -> None:
    raw_dir = PROJECT_ROOT / "data" / "raw"

    paths = [
        (
            raw_dir
            / "copernicus"
            / "cmems_mod_med_phy-tem_anfc_4.2km_P1D-m_2024.parquet",
            ["time"],
        ),
        (
            raw_dir
            / "copernicus"
            / "cmems_mod_med_bgc-pft_anfc_4.2km_P1D-m_2024.parquet",
            ["time"],
        ),
        (
            raw_dir / "gfw_ais" / "ais_high_daily_ts_2024.parquet",
            ["date"],
        ),
        (
            raw_dir / "gfw_sar" / "s2_vessel_detections_2024.parquet",
            ["date", "detect_timestamp"],
        ),
    ]

    for path, date_cols in paths:
        print_dataset_summary(path, date_cols=date_cols)


if __name__ == "__main__":
    main()
