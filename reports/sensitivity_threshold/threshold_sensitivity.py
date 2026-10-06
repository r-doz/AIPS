"""Sensitivity of Red-LGCP to the gate threshold tau.

The gate and the redistribution only post-process the saved predictions of each run (mean
activity probability, its Monte Carlo dropout std and the expected count), so they are
re-applied here for a grid of tau without retraining, with z = 1, gamma = 1 and rho = 1 as in
the full model. Log-likelihood and CRPS score the hurdle predictive distribution, which does
not depend on the gate, so only the five point metrics are reported.

Before the sweep, tau = 0.3 is checked to reproduce the saved predictions and metrics of
every run. Baseline references are the per-month means of the best baseline per metric
(excluding Last Available Poisson for the Wasserstein distance, as in the main table).
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.models.hurdle_lgcp import apply_modulated_gate  # noqa: E402
from src.models.metrics_lgcp import evaluate_metrics  # noqa: E402

RUNS = ROOT / "reports/seeded_monthly_windows_hurdle"
OUT = ROOT / "reports/sensitivity_threshold"
TAUS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7]
TAU_MAIN, Z, GAMMA, RHO = 0.3, 1.0, 1.0, 1.0
METRICS = [("mae_obs", "MAE", "min"), ("rmse_obs", "RMSE", "min"), ("wasserstein", "Wasserstein", "min"),
           ("accuracy", "Accuracy", "max"), ("f1", "F1", "max")]
BASELINES = {
    "LAP": "reports/seeded_monthly_windows/13_evaluate_last_available_poisson",
    "GCMP": "reports/seeded_monthly_windows/12b_global_cell_mean",
    "WGLM": "reports/seeded_monthly_windows_salinity/14_train_window_poisson_glm",
    "GNN": "reports/seeded_monthly_windows_salinity_gnn_cellpoisson/15_train_gnn",
    "ConvLSTM": "reports/seeded_monthly_windows_salinity/18_train_convlstm",
}


def gate(df, tau):
    tau_vec = np.clip(tau + Z * df.p_active_std.to_numpy(), 1e-6, 1.0)
    return apply_modulated_gate(df.expected_count.to_numpy(), df.p_active_mean.to_numpy(), tau_vec,
                                df.date.to_numpy(), gamma=GAMMA, scale=RHO)


rows, max_gap = [], 0.0
files = sorted(RUNS.glob("seed_*/runs/*/*/hurdle_predictions.csv"))
assert len(files) == 80, len(files)
for f in files:
    seed, name = int(f.parts[-5].split("_")[1]), f.parts[-3]
    df = pd.read_csv(f)
    y, dates = df.observed.to_numpy(), pd.to_datetime(df.date)
    check = gate(df, TAU_MAIN)
    assert np.allclose(check, df.pred_higher.to_numpy(), rtol=1e-4, atol=1e-6), f
    saved = yaml.safe_load(open(f.parent / "metrics_higher.yaml"))
    max_gap = max(max_gap, abs(evaluate_metrics(y, check, dates)["mae_obs"] - saved["mae_obs"]))
    for tau in TAUS:
        m = evaluate_metrics(y, gate(df, tau), dates)
        rows.append({"seed": seed, "name": name, "month": name.split("_")[1], "tau": tau,
                     **{k: m[k] for k, _, _ in METRICS}})
print(f"tau = {TAU_MAIN} reproduces the saved runs (max MAE gap {max_gap:.2e})")
res = pd.DataFrame(rows)
res.to_csv(OUT / "per_window_per_seed.csv", index=False)

keys = [k for k, _, _ in METRICS]
summary = res.groupby(["month", "tau"])[keys].agg(["mean", lambda s: s.std(ddof=0)])
summary.columns = [f"{k}_{'mean' if s == 'mean' else 'std'}" for k, s in summary.columns]
summary.reset_index().to_csv(OUT / "summary.csv", index=False)

# best baseline per month and metric (Wasserstein: predictive baselines only)
base_week = {b: pd.read_csv(ROOT / d / "per_window_per_seed_metrics.csv").groupby("name")[keys].mean()
             for b, d in BASELINES.items()}
ref = []
for month in ("may", "november"):
    for k, label, direction in METRICS:
        cands = {b: w.loc[[n for n in w.index if month in n], k].mean() for b, w in base_week.items()
                 if not (k == "wasserstein" and b == "LAP")}
        b = (min if direction == "min" else max)(cands, key=cands.get)
        ref.append({"month": month, "metric": k, "best_baseline": b, "value": cands[b]})
ref = pd.DataFrame(ref)
ref.to_csv(OUT / "best_baselines.csv", index=False)

# robustness: weeks (seed-averaged) in which Red-LGCP beats every baseline (predictive ones for W)
week = res.groupby(["tau", "name"])[keys].mean()
wins = []
for tau in TAUS:
    w = week.loc[tau]
    row = {"tau": tau}
    for k, _, direction in METRICS:
        better = np.ones(len(w), bool)
        for b, bw in base_week.items():
            if k == "wasserstein" and b == "LAP":
                continue
            diff = bw.loc[w.index, k].to_numpy() - w[k].to_numpy()
            better &= (diff > 0) if direction == "min" else (diff < 0)
        row[k] = int(better.sum())
    wins.append(row)
wins = pd.DataFrame(wins)
wins.to_csv(OUT / "weeks_best.csv", index=False)

pd.set_option("display.width", 200)
print("\nMean over 4 weeks x 10 seeds:")
print(res.groupby(["month", "tau"])[keys].mean().round(3).to_string())
print("\nBest baseline per month and metric:")
print(ref.round(3).to_string(index=False))
print("\nWeeks (of 8) in which Red-LGCP beats every baseline (predictive baselines for Wasserstein):")
print(wins.to_string(index=False))

# figure: one panel per metric, May and November, best baseline as dashed reference
plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "legend.fontsize": 8})
fig, axes = plt.subplots(2, 3, figsize=(6.75, 3.9))
colors = {"may": "#1f77b4", "november": "#d62728"}
mean = res.groupby(["month", "tau"])[keys].mean()
for ax, (k, label, direction) in zip(axes.flat, METRICS):
    for month, mlabel in (("may", "May"), ("november", "Nov.")):
        ax.plot(TAUS, mean.loc[month, k].to_numpy(), marker="o", ms=2.5, lw=1.2, color=colors[month],
                label=f"Red-LGCP, {mlabel}")
        r = ref[(ref.month == month) & (ref.metric == k)].iloc[0]
        ax.axhline(r.value, ls="--", lw=1.2, color=colors[month],
                   label=f"best baseline, {mlabel}")
    ax.axvline(TAU_MAIN, ls=":", lw=1.0, color="0.4", label=rf"$\tau={TAU_MAIN}$ (main results)")
    ax.set_title(f"{label} ({'lower' if direction == 'min' else 'higher'} is better)")
    ax.set_xlabel(r"threshold $\tau$")
    ax.grid(alpha=0.25, lw=0.5)
handles, labels = axes.flat[0].get_legend_handles_labels()
axes.flat[5].axis("off")
axes.flat[5].legend(handles, labels, loc="center", frameon=False)
fig.tight_layout()
fig.savefig(OUT / "threshold_sensitivity.pdf")
fig.savefig(OUT / "threshold_sensitivity.png", dpi=200)
print(f"\nSaved figure and tables to {OUT.relative_to(ROOT)}")
