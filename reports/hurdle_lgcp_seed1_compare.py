"""Compare the joint hurdle Red-LGCP (seed 1, four gating rules) with Red-LGCP (seed 1)."""

import glob

import numpy as np
import pandas as pd
import yaml

HURDLE = "reports/hurdle_lgcp_seed1"
RED = "reports/seeded_monthly_windows_salinity/11_train_lgcp/seed_1/per_window_metrics.csv"
METRICS = [("mae_obs", "lower"), ("rmse_obs", "lower"), ("wasserstein", "lower"), ("accuracy", "higher"), ("f1", "higher")]
VARIANTS = ["nogate", "fixed", "lower", "higher"]

red = pd.read_csv(RED).set_index("name")
rows = []
flips = []
for window in red.index:
    run_dirs = glob.glob(f"{HURDLE}/runs/{window}/*/")
    assert len(run_dirs) == 1, (window, run_dirs)
    d = run_dirs[0]
    for v in VARIANTS:
        m = yaml.safe_load(open(f"{d}metrics_{v}.yaml"))
        rows.append({"window": window, "month": red.loc[window, "month"], "model": f"hurdle-{v}",
                     **{k: m[k] for k, _ in METRICS}})
    rows.append({"window": window, "month": red.loc[window, "month"], "model": "Red-LGCP",
                 **{k: red.loc[window, k] for k, _ in METRICS}})
    pr = pd.read_csv(f"{d}hurdle_predictions.csv")
    acc_fixed = pr.p_active_mean >= pr.tau_fixed
    flips.append({"window": window,
                  "p_std_median": pr.p_active_std.median(), "p_std_max": pr.p_active_std.max(),
                  "accepted_fixed": int(acc_fixed.sum()),
                  "accepted_lower": int((pr.p_active_mean >= pr.tau_lower).sum()),
                  "accepted_higher": int((pr.p_active_mean >= pr.tau_higher).sum()),
                  "observed_active": int((pr.observed > 0).sum())})

df = pd.DataFrame(rows)
df.to_csv(f"{HURDLE}/comparison_per_window.csv", index=False)
pd.set_option("display.width", 200)
print("=== Month means (seed 1, 4 weeks per month) ===")
print(df.groupby(["month", "model"])[[k for k, _ in METRICS]].mean().round(3).to_string())

print("\n=== Weeks (of 8) in which each hurdle rule beats Red-LGCP ===")
piv = {k: df.pivot(index="window", columns="model", values=k) for k, _ in METRICS}
for v in VARIANTS:
    out = []
    for k, direction in METRICS:
        diff = piv[k][f"hurdle-{v}"] - piv[k]["Red-LGCP"]
        wins = int((diff < 0).sum()) if direction == "lower" else int((diff > 0).sum())
        out.append(f"{k}: {wins}/8")
    print(f"hurdle-{v:6s} " + " | ".join(out))

print("\n=== Classifier uncertainty and accepted cells (out of 343 per week) ===")
print(pd.DataFrame(flips).round(3).to_string(index=False))
