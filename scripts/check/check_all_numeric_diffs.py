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

    new_df = new_df[
        (new_df["date"] >= common_start) & (new_df["date"] <= common_end)
    ].copy()

    old_df = old_df[
        (old_df["date"] >= common_start) & (old_df["date"] <= common_end)
    ].copy()

    common_numeric_cols = sorted(
        set(new_df.select_dtypes("number").columns)
        & set(old_df.select_dtypes("number").columns)
    )

    print("Common period:", common_start, "->", common_end)
    print("New shape:", new_df.shape)
    print("Old shape:", old_df.shape)
    print()

    rows = []

    for col in common_numeric_cols:
        new_sum = new_df[col].sum()
        old_sum = old_df[col].sum()
        new_mean = new_df[col].mean()
        old_mean = old_df[col].mean()

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
    out = out.sort_values("diff_sum", key=lambda s: s.abs(), ascending=False)

    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
