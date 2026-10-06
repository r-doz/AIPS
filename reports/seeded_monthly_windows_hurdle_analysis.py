"""Pool the 10 joint-hurdle seeds and compare them with the two-stage model.

Writes per_window_per_seed_all_rules.csv (all gating rules) and per_window_per_seed_metrics.csv
(tau + sigma rule, the one reported in the paper), used by the tables and tests in reports/.
"""

import glob
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import wilcoxon

OUT = Path("reports/seeded_monthly_windows_hurdle")
RED = "reports/seeded_monthly_windows_salinity/11_train_lgcp/per_window_per_seed_metrics.csv"
METRICS = [("mae_obs", "lower"), ("rmse_obs", "lower"), ("wasserstein", "lower"), ("accuracy", "higher"), ("f1", "higher")]
RULES = ["nogate", "fixed", "lower", "higher"]

rows = []
for seed in range(1, 11):
    for run in sorted(glob.glob(str(OUT / f"seed_{seed}/runs/*/*/"))):
        window = Path(run).parent.name
        month = "may" if "_may_" in window else "november"
        for rule in RULES:
            m = yaml.safe_load(open(Path(run) / f"metrics_{rule}.yaml"))
            rows.append({"seed": seed, "name": window, "month": month, "rule": rule, **{k: m[k] for k, _ in METRICS}})
df = pd.DataFrame(rows)
counts = df[df.rule == "higher"].groupby("seed").size()
assert (counts == 8).all() and len(counts) == 10, counts
df.to_csv(OUT / "per_window_per_seed_all_rules.csv", index=False)
df[df.rule == "higher"].drop(columns="rule").to_csv(OUT / "per_window_per_seed_metrics.csv", index=False)

red = pd.read_csv(RED)
pd.set_option("display.width", 200)
print("=== 10 seeds x 4 weeks per month: mean ± population std ===")
for month in ("may", "november"):
    print(f"--- {month}")
    for label, q in [("Red-LGCP", red[red.month == month])] + [
            (f"joint {r}", df[(df.month == month) & (df.rule == r)]) for r in RULES]:
        print(f"{label:16s} " + " | ".join(f"{k}: {q[k].mean():.3f}±{q[k].std(ddof=0):.3f}" for k, _ in METRICS))

print("\n=== Joint (tau + sigma) vs Red-LGCP: paired over the 8 weeks (seed-averaged), exact Wilcoxon ===")
h = df[df.rule == "higher"].groupby("name")[[k for k, _ in METRICS]].mean()
r = red.groupby("name")[[k for k, _ in METRICS]].mean().loc[h.index]
for k, direction in METRICS:
    d = (r[k] - h[k]) if direction == "lower" else (h[k] - r[k])
    p = wilcoxon(d.to_numpy(), alternative="two-sided", method="exact").pvalue
    print(f"{k:12s} joint {h[k].mean():.3f} vs Red-LGCP {r[k].mean():.3f}  joint better in {int((d > 0).sum())}/8 weeks  p={p:.4f}")
