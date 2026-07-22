from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))


NEW_PATH = PROJECT_ROOT / "data" / "processed" / "cpr_gfw_2024.parquet"

# Cambia questo path con il nome reale del file prodotto dai notebook
OLD_PATH = PROJECT_ROOT / "data" / "processed" / "cpr_gfw_2024_from_notebook.parquet"


def summarize(df: pd.DataFrame) -> dict:
    summary = {}

    summary["shape"] = df.shape
    summary["columns"] = list(df.columns)

    if "date" in df.columns:
        summary["date_min"] = df["date"].min()
        summary["date_max"] = df["date"].max()

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
            summary[f"{col}_sum"] = df[col].sum()
            summary[f"{col}_mean"] = df[col].mean()

    return summary


def main() -> None:
    if not NEW_PATH.exists():
        raise FileNotFoundError(f"New processed file not found: {NEW_PATH}")

    if not OLD_PATH.exists():
        raise FileNotFoundError(
            f"Old notebook file not found: {OLD_PATH}\nModify OLD_PATH in this script."
        )

    new_df = pd.read_parquet(NEW_PATH)
    old_df = pd.read_parquet(OLD_PATH)

    new_summary = summarize(new_df)
    old_summary = summarize(old_df)

    print("=" * 80)
    print("NEW PIPELINE FILE")
    print(NEW_PATH)
    print("=" * 80)
    for key, value in new_summary.items():
        if key != "columns":
            print(f"{key}: {value}")

    print()
    print("=" * 80)
    print("OLD NOTEBOOK FILE")
    print(OLD_PATH)
    print("=" * 80)
    for key, value in old_summary.items():
        if key != "columns":
            print(f"{key}: {value}")

    print()
    print("=" * 80)
    print("COLUMN CHECK")
    print("=" * 80)

    new_cols = set(new_df.columns)
    old_cols = set(old_df.columns)

    print("Columns only in new:", sorted(new_cols - old_cols))
    print("Columns only in old:", sorted(old_cols - new_cols))

    print()
    print("=" * 80)
    print("NUMERIC DIFFERENCES")
    print("=" * 80)

    common_numeric_cols = sorted(
        set(new_df.select_dtypes("number").columns)
        & set(old_df.select_dtypes("number").columns)
    )

    for col in common_numeric_cols:
        new_sum = new_df[col].sum()
        old_sum = old_df[col].sum()
        new_mean = new_df[col].mean()
        old_mean = old_df[col].mean()

        print(f"{col}")
        print(f"  sum new:  {new_sum}")
        print(f"  sum old:  {old_sum}")
        print(f"  diff sum: {new_sum - old_sum}")
        print(f"  mean new: {new_mean}")
        print(f"  mean old: {old_mean}")
        print(f"  diff mean:{new_mean - old_mean}")
        print()


if __name__ == "__main__":
    main()
