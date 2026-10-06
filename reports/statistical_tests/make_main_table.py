"""Regenerate the main results table (tab:results-main) from the per-window/per-seed metric files."""

import argparse
import re
from decimal import ROUND_HALF_UP, Decimal

import pandas as pd

METHODS = [
    ("Last Available Poisson", "reports/seeded_monthly_windows/13_evaluate_last_available_poisson"),
    ("Global Cell Mean Poisson", "reports/seeded_monthly_windows/12b_global_cell_mean"),
    ("Window Poisson GLM", "reports/seeded_monthly_windows_salinity/14_train_window_poisson_glm"),
    ("GNN", "reports/seeded_monthly_windows_salinity_gnn_cellpoisson/15_train_gnn"),
    ("ConvLSTM", "reports/seeded_monthly_windows_salinity/18_train_convlstm"),
    ("Red-LGCP (ours)", "reports/seeded_monthly_windows_hurdle"),
]
METRICS = [("mae_obs", "min"), ("rmse_obs", "min"), ("wasserstein", "min"), ("accuracy", "max"), ("f1", "max"),
           ("log_lik", "max"), ("crps", "min")]
SCORES = "reports/probabilistic_scores/per_window_per_seed_scores.csv"
SCORE_NAME = {"Red-LGCP (ours)": "Red-LGCP (joint, final)"}
MONTHS = [("may", "May"), ("november", "Nov.")]
SHORT = {"Last Available Poisson": "LAP", "Global Cell Mean Poisson": "GCMP", "Window Poisson GLM": "WGLM", "Red-LGCP (ours)": "Red-LGCP"}


def r3(x):
    s = str(Decimal(repr(float(x))).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))
    return s.replace("-", "$-$", 1) if s.startswith("-") else s


parser = argparse.ArgumentParser()
parser.add_argument("--out", default="reports/statistical_tests/main_results_table.tex")
parser.add_argument("--font", default="small")
args = parser.parse_args()

stats = {}
scores = pd.read_csv(SCORES)
for name, d in METHODS:
    df = pd.read_csv(f"{d}/per_window_per_seed_metrics.csv")
    sc = scores[scores.method == SCORE_NAME.get(name, name)][["seed", "name", "log_lik", "crps"]]
    if "seed" not in df:
        df["seed"] = 1
    merged = df.merge(sc, on=["seed", "name"], how="left", validate="one_to_one")
    assert len(merged) == len(df) and merged[["log_lik", "crps"]].notna().all().all(), name
    df = merged
    for month, _ in MONTHS:
        q = df[df.month == month]
        for m, _ in METRICS:
            stats[(name, month, m)] = (q[m].mean(), q[m].std(ddof=0))

predictive = [n for n, _ in METHODS if n != "Last Available Poisson"]
best, best_pred = {}, {}
for month, _ in MONTHS:
    for m, direction in METRICS:
        pick = min if direction == "min" else max
        best[(month, m)] = pick((n for n, _ in METHODS), key=lambda n: round(stats[(n, month, m)][0], 3))
        best_pred[(month, m)] = pick(predictive, key=lambda n: round(stats[(n, month, m)][0], 3))

lines = []
for i, (name, _) in enumerate(METHODS):
    for j, (month, label) in enumerate(MONTHS):
        cells = []
        for m, _ in METRICS:
            mu, sd = stats[(name, month, m)]
            mean = f"\\textbf{{{r3(mu)}}}" if best[(month, m)] == name else r3(mu)
            cell = f"{mean}\\,{{\\scriptsize$\\pm${r3(sd)}}}"
            if m == "wasserstein" and best_pred[(month, m)] == name and best[(month, m)] != name:
                cell += "$^{\\boldsymbol{\\dagger}}$"
            cells.append(cell)
        head = f"\\multirow{{2}}{{*}}{{{SHORT.get(name, name)}}}" if j == 0 else ""
        lines.append(f"{head} & {label} & " + " & ".join(cells) + " \\\\")
    if i < len(METHODS) - 1:
        lines.append("\\midrule")

body = "\n".join(lines)
tex = r"""\begin{table*}[t]
\centering
""" + "\\" + args.font + r"""
\setlength{\tabcolsep}{3pt}
\begin{tabular}{llccccccc}
\toprule
Method & Month & MAE & RMSE & Wasserstein & Accuracy & F1 & Log-lik. & CRPS \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}
\caption{Results on May and Nov.\ 2025 (4 weekly test windows per month: 01--07, 08--14, 15--21, 22--28) for Red-LGCP (ours), Last Available Poisson (LAP), Global Cell Mean Poisson (GCMP), Window Poisson GLM (WGLM), GNN and ConvLSTM; all methods except LAP and GCMP use the same covariates, salinity included. Mean $\pm$ population std pooled over all (window $\times$ seed) values -- 10 seeds for Red-LGCP, GNN and ConvLSTM, 1 seed (seed-invariant) for the other methods. Log-likelihood (per cell-day, higher is better) and CRPS (lower is better) evaluate each method's predictive distribution: a Poisson distribution with the predicted rate for the baselines, and the hurdle predictive distribution for Red-LGCP. Bold marks the best value per column within each month. LAP's low Wasserstein distance is not a sign of good calibration: it simply reuses real observed counts shifted by one day, so its predicted value distribution trivially matches the true one by construction; $^{\boldsymbol{\dagger}}$ marks the best Wasserstein distance among the remaining, genuinely predictive models.}
\label{tab:results-main}
\end{table*}
"""
open(args.out, "w").write(tex)
print(tex)
