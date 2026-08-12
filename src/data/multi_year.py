"""
Helpers for resolving and loading per-year parquet datasets.

Convention used across the project: a single "base" parquet path
(e.g. data/processed/cpr_gfw.parquet) has one sibling file per year
(cpr_gfw_2024.parquet, cpr_gfw_2025.parquet, ...). A config's `years`
field selects which year(s) to use; when it names more than one year,
the corresponding files are concatenated into a single DataFrame.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def resolve_years(years) -> list | None:
    """Normalize a config's `years` field to a list, or None if unset."""
    if years is None:
        return None
    if isinstance(years, (list, tuple)):
        return list(years)
    return [years]


def resolve_parquet_paths(parquet_path: str | Path, years=None) -> list[Path]:
    """
    Resolve one parquet path per year.

    Example
    -------
    parquet_path: data/processed/cpr_gfw.parquet
    years: [2024, 2025]
    ->  [data/processed/cpr_gfw_2024.parquet, data/processed/cpr_gfw_2025.parquet]

    If `years` is None, the base path is used as-is (must exist).
    """
    path = Path(parquet_path)
    years = resolve_years(years)

    if years is None:
        if not path.exists():
            raise FileNotFoundError(
                f"Parquet file not found: {path}. "
                "No 'years' field was provided to build a year-specific path."
            )
        return [path]

    suffix = path.suffix or ".parquet"
    stem = path.stem

    paths = []
    for year in years:
        candidate = (
            path.with_name(f"{stem}{suffix}")
            if f"_{year}" in stem
            else path.with_name(f"{stem}_{year}{suffix}")
        )
        if not candidate.exists():
            raise FileNotFoundError(
                f"Could not find parquet file for year {year}.\n"
                f"Base path: {path}\n"
                f"Expected: {candidate}"
            )
        paths.append(candidate)

    return paths


def load_parquet_years(parquet_path: str | Path, years=None) -> pd.DataFrame:
    """Load and concatenate one parquet file per year in `years`."""
    paths = resolve_parquet_paths(parquet_path, years)
    dfs = [pd.read_parquet(p) for p in paths]
    return dfs[0] if len(dfs) == 1 else pd.concat(dfs, ignore_index=True)


def years_label(years) -> str:
    """Format years for run names / output paths, e.g. [2024, 2025] -> '2024+2025'."""
    years = resolve_years(years)
    if not years:
        return ""
    return "+".join(str(y) for y in years)
