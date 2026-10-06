"""Paired significance tests: LGCP vs each baseline over the 8 weekly test windows.

Unit of analysis: test window (4 in May + 4 in November 2025). Stochastic methods (LGCP, GNN,
ConvLSTM) are first averaged over their 10 seeds within each window. For every metric and baseline,
the 8 paired differences are tested with an exact two-sided Wilcoxon signed-rank test, and p-values
are Holm-corrected across the five baselines within each metric. Also reports how many windows
LGCP wins and, as a seed-level robustness check, the share of individual (seed, window) comparisons
LGCP wins.
"""

import argparse
import itertools

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

METHODS = {
    "Last Available Poisson": "reports/seeded_monthly_windows/13_evaluate_last_available_poisson",
    "Global Cell Mean Poisson": "reports/seeded_monthly_windows/12b_global_cell_mean",
    "Window Poisson GLM": "reports/seeded_monthly_windows_salinity/14_train_window_poisson_glm",
    "GNN": "reports/seeded_monthly_windows_salinity_gnn_cellpoisson/15_train_gnn",
    "ConvLSTM": "reports/seeded_monthly_windows_salinity/18_train_convlstm",
    "LGCP": "reports/seeded_monthly_windows_hurdle",
}
METRICS = {"mae_obs": "lower", "rmse_obs": "lower", "wasserstein": "lower", "accuracy": "higher", "f1": "higher"}
BASELINES = [m for m in METHODS if m != "LGCP"]


def holm(pvals):
    order = np.argsort(pvals)
    adj = np.empty(len(pvals))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(pvals) - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


parser = argparse.ArgumentParser()
parser.add_argument("--gnn-dir", default=METHODS["GNN"], help="Seeded results directory of the GNN to test.")
parser.add_argument("--out", default="reports/statistical_tests/paired_window_tests_hurdle.csv")
parser.add_argument("--main-dir", default=METHODS["LGCP"], help="Seeded results directory of the main model.")
args = parser.parse_args()
METHODS["GNN"] = args.gnn_dir
METHODS["LGCP"] = args.main_dir

raw = {name: pd.read_csv(f"{d}/per_window_per_seed_metrics.csv") for name, d in METHODS.items()}
window_means = {name: df.groupby("name")[list(METRICS)].mean() for name, df in raw.items()}
windows = sorted(window_means["LGCP"].index)
assert all(sorted(w.index) == windows for w in window_means.values()) and len(windows) == 8

rows = []
for metric, better in METRICS.items():
    sign = 1.0 if better == "lower" else -1.0
    pvals = []
    for base in BASELINES:
        lg = window_means["LGCP"].loc[windows, metric].to_numpy()
        bs = window_means[base].loc[windows, metric].to_numpy()
        d = sign * (bs - lg)  # > 0 means LGCP is better
        p = wilcoxon(d, alternative="two-sided", method="exact").pvalue
        pvals.append(p)
        # seed-level robustness: every LGCP run against every baseline run in the same window
        wins = total = 0
        for w in windows:
            a = raw["LGCP"].loc[raw["LGCP"].name == w, metric].to_numpy()
            b = raw[base].loc[raw[base].name == w, metric].to_numpy()
            diff = sign * (b[None, :] - a[:, None])
            wins += (diff > 0).sum(); total += diff.size
        may = [w for w in windows if "may" in w]
        rows.append({
            "metric": metric, "baseline": base,
            "lgcp_mean": lg.mean(), "baseline_mean": bs.mean(),
            "rel_improvement_%": 100 * sign * (bs.mean() - lg.mean()) / bs.mean(),
            "windows_won": int((d > 0).sum()),
            "may_won": int(sum(d[i] > 0 for i, w in enumerate(windows) if w in may)),
            "nov_won": int(sum(d[i] > 0 for i, w in enumerate(windows) if w not in may)),
            "p_wilcoxon": p, "seed_level_win_%": 100 * wins / total,
        })
    for r, padj in zip(rows[-len(BASELINES):], holm(np.array(pvals))):
        r["p_holm"] = padj

res = pd.DataFrame(rows)
res.to_csv(args.out, index=False)
pd.set_option("display.width", 200)
print(res.round({"lgcp_mean": 3, "baseline_mean": 3, "rel_improvement_%": 1, "p_wilcoxon": 4, "p_holm": 4, "seed_level_win_%": 1}).to_string(index=False))
