"""Per-run Red-LGCP configs for the new datasets (rebuttal): more years of the Gulf of Trieste and a new region.

Every config is a definitive per-run config of Red-LGCP (reports/seeded_monthly_windows_hurdle/seed_*/configs,
10 seeds x the paper's 8 test weeks of 2025) with only the data changed:
  trieste_4y   data/rebuttal/trieste_reanalysis, years 2022-2025 (4 years of history before each test week)
  trieste_2y   the same data, years 2024-2025 (the paper's history; control for the data rebuild)
  venice       data/rebuttal/venice (Venice-Chioggia), years 2024-2025
Time is standardized with the training days, so a fixed temporal lengthscale covers more days when the history
is longer: one standardized unit is the standard deviation of the training days (about 140 days with 2 years,
350 with 4). For trieste_4y the short-term temporal RBF lengthscale is rescaled per test week so that it
covers the same number of days as in the paper (lengthscale x std_2y / std_4y). The periodic kernel's period
is given in days and its lengthscale is unitless, so they need no change.

Output: <out>/configs/<experiment>/seed_N/<window>.yaml (report_root <out>/runs/<experiment>/seed_N/runs) and
<out>/jobs.txt (absolute config paths).
Usage: python newdata_make_configs.py --out reports/rebuttal/new_data [--experiments venice ...]
"""

from __future__ import annotations

import argparse
import copy
import glob
import os

import numpy as np
import pandas as pd
import yaml

from svgp_residual import ROOT

EXPERIMENTS = {
    "venice": {"parquet_path": "data/rebuttal/venice/processed/cpr_gfw_salinity.parquet", "years": [2024, 2025]},
    "trieste_4y": {"parquet_path": "data/rebuttal/trieste_reanalysis/processed/cpr_gfw_salinity.parquet",
                   "years": [2022, 2023, 2024, 2025]},
    "trieste_2y": {"parquet_path": "data/rebuttal/trieste_reanalysis/processed/cpr_gfw_salinity.parquet", "years": [2024, 2025]},
}


def train_day_std(first_year: int, test_start: str) -> float:
    days = pd.date_range(f"{first_year}-01-01", pd.Timestamp(test_start) - pd.Timedelta(days=1))
    return float(np.std((days - days[0]).days.to_numpy(float)))  # population std, as StandardScaler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--experiments", nargs="*", default=list(EXPERIMENTS))
    args = ap.parse_args()
    out = os.path.abspath(os.path.join(ROOT, args.out))
    files = sorted(glob.glob(str(ROOT / "reports/seeded_monthly_windows_hurdle/seed_*/configs/*.yaml")))
    assert len(files) == 80, len(files)
    jobs = []
    for exp in args.experiments:
        spec = EXPERIMENTS[exp]
        for f in files:
            seed, win = f.split("/")[-3], os.path.basename(f)[:-5]
            c = copy.deepcopy(yaml.safe_load(open(f)))
            assert c["years"] == [2024, 2025], c["years"]
            c["parquet_path"], c["years"] = spec["parquet_path"], spec["years"]
            if spec["years"][0] != 2024:
                ratio = train_day_std(2024, c["test_start_date"]) / train_day_std(spec["years"][0], c["test_start_date"])
                rbf = [k for k in c["model"]["temporal_kernels"] if k["type"].lower() == "rbf"]
                assert len(rbf) == 1
                rbf[0]["hyperparameters"]["lengthscale"] = float(rbf[0]["hyperparameters"]["lengthscale"]) * ratio
                c["rebuttal_note"] = f"temporal RBF lengthscale x {ratio:.4f} (same length in days as with 2024-2025)"
            c["report_root"] = os.path.join(out, "runs", exp, seed, "runs")
            path = os.path.join(out, "configs", exp, seed, f"{win}.yaml")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            yaml.safe_dump(c, open(path, "w"), sort_keys=False)
            jobs.append(path)
    with open(os.path.join(out, "jobs.txt"), "w") as fh:
        fh.write("\n".join(jobs) + "\n")
    print(f"{len(jobs)} configs → {out}/configs")


if __name__ == "__main__":
    main()
