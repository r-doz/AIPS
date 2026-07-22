from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

NEW_PATH = PROJECT_ROOT / "data" / "processed" / "cpr_gfw_2024.parquet"
OLD_PATH = PROJECT_ROOT / "data" / "processed" / "cpr_gfw_2024_from_notebook.parquet"


def main() -> None:
    new_df = pd.read_parquet(NEW_PATH)
    old_df = pd.read_parquet(OLD_PATH)

    new_df["date"] = pd.to_datetime(new_df["date"])
    old_df["date"] = pd.to_datetime(old_df["date"])

    common_start = max(new_df["date"].min(), old_df["date"].min())
    common_end = min(new_df["date"].max(), old_df["date"].max())

    new_common = new_df[
        (new_df["date"] >= common_start) & (new_df["date"] <= common_end)
    ].copy()

    old_common = old_df[
        (old_df["date"] >= common_start) & (old_df["date"] <= common_end)
    ].copy()

    print("=" * 80)
    print("FILES")
    print("=" * 80)
    print("NEW:", NEW_PATH)
    print("OLD:", OLD_PATH)
    print()

    print("=" * 80)
    print("COMMON PERIOD")
    print("=" * 80)
    print(common_start, "->", common_end)
    print()

    print("=" * 80)
    print("SHAPES")
    print("=" * 80)
    print("New full shape:   ", new_df.shape)
    print("Old full shape:   ", old_df.shape)
    print("New common shape: ", new_common.shape)
    print("Old common shape: ", old_common.shape)
    print()

    print("=" * 80)
    print("COLUMNS")
    print("=" * 80)
    print("Only in new:", sorted(set(new_df.columns) - set(old_df.columns)))
    print("Only in old:", sorted(set(old_df.columns) - set(new_df.columns)))
    print()

    common_numeric_cols = sorted(
        set(new_common.select_dtypes("number").columns)
        & set(old_common.select_dtypes("number").columns)
    )

    rows = []

    for col in common_numeric_cols:
        new_sum = new_common[col].sum()
        old_sum = old_common[col].sum()
        new_mean = new_common[col].mean()
        old_mean = old_common[col].mean()

        rows.append(
            {
                "column": col,
                "new_sum": new_sum,
                "old_sum": old_sum,
                "diff_sum": new_sum - old_sum,
                "new_mean": new_mean,
                "old_mean": old_mean,
                "diff_mean": new_mean - old_mean,
            }
        )

    out = pd.DataFrame(rows)

    if not out.empty:
        out = out.sort_values(
            "diff_sum",
            key=lambda s: s.abs(),
            ascending=False,
        )

    print("=" * 80)
    print("NUMERIC DIFFERENCES ON COMMON PERIOD")
    print("=" * 80)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
