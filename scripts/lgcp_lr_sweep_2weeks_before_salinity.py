"""Run the 12 LGCP pre-gate configs (3 learning rates x 4 validation weeks,
see reports/paper/hparam_revalidation_2weeks_before_salinity/configs/pregate_*.yaml)
and select the best learning rate by the same relative-score convention
used for the other baselines (mean_ll_obs maximized, rmse_daily minimized,
min-max normalized per week, averaged across weeks).

Each run also leaves behind a pre-gate prediction cache (via
cache_pregate_path in its config) that the classifier hyperparameter
sweep (scripts/classifier_hparam_rerun_2weeks_before.py) reads afterward.

Usage:
    python scripts/lgcp_lr_sweep_2weeks_before.py --parallel 4
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_SCRIPT = PROJECT_ROOT / "scripts/11_train_lgcp.py"
CONFIG_DIR = PROJECT_ROOT / "reports/paper/hparam_revalidation_2weeks_before_salinity/configs"
OUT_DIR = PROJECT_ROOT / "reports/paper/hparam_revalidation_2weeks_before_salinity/lgcp_lr"

METRIC_DIRECTIONS = {"mean_ll_obs": "max", "rmse_daily": "min"}


def run_one(config_path: Path) -> tuple[str, int]:
    print(f"[{config_path.stem}] launching...", flush=True)
    result = subprocess.run(
        [sys.executable, str(TRAIN_SCRIPT), "--config", str(config_path)],
        cwd=PROJECT_ROOT,
    )
    print(f"[{config_path.stem}] exit={result.returncode}", flush=True)
    return config_path.stem, result.returncode


def relative_score(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    valid = df.dropna(subset=list(METRIC_DIRECTIONS)).copy()
    for metric, direction in METRIC_DIRECTIONS.items():
        def _normalize(s, direction=direction):
            lo, hi = s.min(), s.max()
            if hi - lo < 1e-12:
                return pd.Series(1.0, index=s.index)
            return (s - lo) / (hi - lo) if direction == "max" else (hi - s) / (hi - lo)
        valid[f"relative_{metric}"] = valid.groupby("week")[metric].transform(_normalize)
    relative_cols = [f"relative_{m}" for m in METRIC_DIRECTIONS]
    scores = valid.groupby(group_cols)[relative_cols].mean()
    scores["combined_relative_score"] = scores[relative_cols].mean(axis=1)
    return scores.reset_index().sort_values("combined_relative_score", ascending=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parallel", type=int, default=4)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    config_paths = sorted(CONFIG_DIR.glob("pregate_*.yaml"))
    if not config_paths:
        sys.exit(f"No pregate configs found under {CONFIG_DIR}")

    with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as executor:
        futures = {executor.submit(run_one, p): p for p in config_paths}
        failures = []
        for future in as_completed(futures):
            stem, code = future.result()
            if code:
                failures.append(stem)
    if failures:
        print(f"FAILED runs: {failures}", file=sys.stderr)

    # Collect metrics.yaml for each (week, lr) from the run_name search dirs.
    pattern = re.compile(r"pregate_(?P<week>.+)_lr(?P<lr>[0-9.e+-]+)$")
    rows = []
    for config_path in config_paths:
        cfg = yaml.safe_load(config_path.read_text())
        run_name = cfg["run_name"]
        match = pattern.match(run_name)
        week, lr_tag = match.group("week"), match.group("lr")
        search_root = PROJECT_ROOT / cfg["report_root"] / cfg["model_family"]
        metrics_paths = sorted(search_root.glob(f"*{run_name}_*/metrics.yaml"))
        if not metrics_paths:
            print(f"WARNING: no metrics.yaml for {run_name}", file=sys.stderr)
            continue
        metrics = yaml.safe_load(metrics_paths[-1].read_text())
        rows.append({"week": week, "lr": cfg["lr"], **metrics})

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "lgcp_lr_grid_results.csv", index=False)
    scored = relative_score(df, group_cols=["lr"])
    scored.to_csv(OUT_DIR / "lgcp_lr_scores.csv", index=False)
    print("\nScores by lr:")
    print(scored[["lr", "combined_relative_score"]])

    best_lr = float(scored.iloc[0]["lr"])
    with open(OUT_DIR / "selected_hyperparameters.yaml", "w") as f:
        yaml.safe_dump({"selected": {"lr": best_lr}}, f, sort_keys=False)
    print(f"\nSelected LGCP learning rate: {best_lr}")


if __name__ == "__main__":
    main()
