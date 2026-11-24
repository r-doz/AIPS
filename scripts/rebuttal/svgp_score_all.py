"""Score every run of the SVGP-residual experiment in parallel (see svgp_score.py).

Run sets (under reports/rebuttal/svgp_residual/):
  definitive/  the paper's runs (copy of reports/seeded_monthly_windows_hurdle)  -> "SoR (paper)"
  runs_sor/    the paper's model retrained with the same configs and seeds       -> "SoR (retrained)"
  runs_svgp/   the same with the SVGP residual variance                          -> "SVGP (retrained)"

Output: reports/rebuttal/svgp_residual/scores/all_scores.csv
Usage: python svgp_score_all.py [--workers N]
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from svgp_residual import ROOT
from svgp_score import score

BASE = ROOT / "reports/rebuttal/svgp_residual"
SETS = {"definitive": "SoR (paper)", "runs_sor": "SoR (retrained)", "runs_svgp": "SVGP (retrained)"}


def _one(job):
    run, label = job
    try:
        return score(run, label)
    except Exception as exc:  # report and continue
        print(f"FAILED {run}: {exc!r}", flush=True)
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    args = ap.parse_args()
    jobs = [(p.parent, label) for folder, label in SETS.items()
            for p in sorted((BASE / folder).glob("seed_*/runs/*/*/model.pt"))]
    print({label: sum(j[1] == label for j in jobs) for label in SETS.values()}, flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        rows = [r for part in ex.map(_one, jobs, chunksize=1) for r in part]
    out = BASE / "scores"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "all_scores.csv", index=False)
    print(f"Saved {len(rows)} rows → {out / 'all_scores.csv'}")


if __name__ == "__main__":
    main()
