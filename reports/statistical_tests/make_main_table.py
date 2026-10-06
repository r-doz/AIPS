"""Main results table (tab:results-main) as CSV, from the per-window/per-seed metric and score files.

One row per method and month: mean and population std, pooled over all (window x seed) values, of the
five point metrics and of the log-likelihood and CRPS (10 seeds for Red-LGCP, GNN and ConvLSTM,
1 seed for the deterministic baselines).
"""

import argparse

import pandas as pd

METHODS = [
    ("Last Available Poisson", "reports/seeded_monthly_windows/13_evaluate_last_available_poisson"),
    ("Global Cell Mean Poisson", "reports/seeded_monthly_windows/12b_global_cell_mean"),
    ("Window Poisson GLM", "reports/seeded_monthly_windows_salinity/14_train_window_poisson_glm"),
    ("GNN", "reports/seeded_monthly_windows_salinity_gnn_cellpoisson/15_train_gnn"),
    ("ConvLSTM", "reports/seeded_monthly_windows_salinity/18_train_convlstm"),
    ("Red-LGCP (ours)", "reports/seeded_monthly_windows_hurdle"),
]
METRICS = ["mae_obs", "rmse_obs", "wasserstein", "accuracy", "f1", "log_lik", "crps"]
SCORES = "reports/probabilistic_scores/per_window_per_seed_scores.csv"
SCORE_NAME = {"Red-LGCP (ours)": "Red-LGCP (joint, final)"}
SHORT = {"Last Available Poisson": "LAP", "Global Cell Mean Poisson": "GCMP", "Window Poisson GLM": "WGLM",
         "Red-LGCP (ours)": "Red-LGCP"}

parser = argparse.ArgumentParser()
parser.add_argument("--out", default="reports/statistical_tests/main_results_table.csv")
args = parser.parse_args()

scores = pd.read_csv(SCORES)
rows = []
for name, d in METHODS:
    df = pd.read_csv(f"{d}/per_window_per_seed_metrics.csv")
    sc = scores[scores.method == SCORE_NAME.get(name, name)][["seed", "name", "log_lik", "crps"]]
    if "seed" not in df:
        df["seed"] = 1
    merged = df.merge(sc, on=["seed", "name"], how="left", validate="one_to_one")
    assert len(merged) == len(df) and merged[["log_lik", "crps"]].notna().all().all(), name
    for month in ("may", "november"):
        q = merged[merged.month == month]
        row = {"method": SHORT.get(name, name), "month": month}
        for m in METRICS:
            row[f"{m}_mean"] = q[m].mean()
            row[f"{m}_std"] = q[m].std(ddof=0)
        rows.append(row)

table = pd.DataFrame(rows)
table.to_csv(args.out, index=False)
print(table.round(3).to_string(index=False))
