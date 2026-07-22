from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))


NEW_PATH = PROJECT_ROOT / "data" / "processed" / "cpr_gfw_2024.parquet"
OLD_PATH = PROJECT_ROOT / "data" / "processed" / "cpr_gfw_2024_from_notebook.parquet"


def summarize(df: pd.DataFrame, name: str) -> None:
    print("=" * 80)
    print(name)
    print("=" * 80)

    print("shape:", df.shape)
    print("date min:", df["date"].min())
    print("date max:", df["date"].max())

    for col in [
        "chl",
        "thetao",
        "sar_vessels_count",
        "ais_vessels_count",
        "TRAWLERS",
        "SET_GILLNETS",
        "FISHING",
    ]:
        if col in df.columns:
            print(f"{col}_sum:", df[col].sum())
            print(f"{col}_mean:", df[col].mean())

    print()


def main() -> None:
    new_df = pd.read_parquet(NEW_PATH)
    old_df = pd.read_parquet(OLD_PATH)

    new_df["date"] = pd.to_datetime(new_df["date"])
    old_df["date"] = pd.to_datetime(old_df["date"])

    common_start = max(new_df["date"].min(), old_df["date"].min())
    common_end = min(new_df["date"].max(), old_df["date"].max())

    print("Common period:")
    print(common_start, "->", common_end)
    print()

    new_common = new_df[
        (new_df["date"] >= common_start) & (new_df["date"] <= common_end)
    ].copy()

    old_common = old_df[
        (old_df["date"] >= common_start) & (old_df["date"] <= common_end)
    ].copy()

    summarize(new_common, "NEW PIPELINE - COMMON PERIOD")
    summarize(old_common, "OLD NOTEBOOK - COMMON PERIOD")

    print("=" * 80)
    print("DIFFERENCES")
    print("=" * 80)

    print("shape diff:", new_common.shape[0] - old_common.shape[0], "rows")

    for col in [
        "chl",
        "thetao",
        "sar_vessels_count",
        "ais_vessels_count",
        "TRAWLERS",
        "SET_GILLNETS",
        "FISHING",
    ]:
        if col in new_common.columns and col in old_common.columns:
            new_sum = new_common[col].sum()
            old_sum = old_common[col].sum()
            new_mean = new_common[col].mean()
            old_mean = old_common[col].mean()

            print()
            print(col)
            print("  sum diff: ", new_sum - old_sum)
            print("  mean diff:", new_mean - old_mean)


if __name__ == "__main__":
    main()
