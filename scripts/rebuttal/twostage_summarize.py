"""Tables of the two-stage ablation ladder (reads twostage_score.py output and the paper's ablation scores).

Rows: S0 the paper's two-stage model (Table 5), S1-S3 the steps of twostage_score.py, and the joint Red-LGCP
(Table 5, "full"). Values: mean and population std over 4 weeks x 10 seeds per month. Paired tests follow
App. B.4 (seed-averaged weekly values, two-sided exact Wilcoxon over the 8 test weeks), uncorrected.

Output: reports/rebuttal/twostage/results/
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from r1_common import paired_test
from svgp_residual import ROOT

BASE = ROOT / "reports/rebuttal/twostage"
RES = BASE / "results"
METRICS = ["mae_obs", "rmse_obs", "wasserstein", "accuracy", "f1", "log_lik", "crps"]
GAP_METRICS = ["wasserstein", "accuracy", "log_lik", "crps"]
HIGHER = {"log_lik": True, "crps": False, "mae_obs": False, "rmse_obs": False, "wasserstein": False, "accuracy": True, "f1": True}
LABELS = {"mae_obs": "MAE ↓", "rmse_obs": "RMSE ↓", "wasserstein": "Wass. ↓", "accuracy": "Acc. ↑", "f1": "F1 ↑",
          "log_lik": "Log-lik. ↑", "crps": "CRPS ↓"}
NAMES = {"S0": "Two-stage (paper, Table 5)", "S1": "+ hurdle scoring", "S2": "+ MC-dropout classifier, gate τ + σ",
         "S3": "+ zero-truncated LGCP (Red-LGCP trained in two stages)", "joint": "Red-LGCP, joint (paper)"}


def pstd(s: pd.Series) -> float:
    return s.std(ddof=0)


def main():
    RES.mkdir(parents=True, exist_ok=True)
    sc = pd.read_csv(BASE / "scores/all_scores.csv")
    assert len(sc) == 240 and set(sc.step) == {"S1", "S2", "S3"}, (len(sc), set(sc.step))
    ab = pd.read_csv(ROOT / "reports/probabilistic_scores/ablation_base.csv").rename(columns={"name": "week"})
    paper = {"S0": ab[ab.variant == "w/o joint training"], "joint": ab[ab.variant == "full"]}
    per = pd.concat([paper["S0"].assign(step="S0"), sc, paper["joint"].assign(step="joint")], ignore_index=True)
    per = per[["step", "seed", "week", "month"] + METRICS]
    per.to_csv(RES / "per_window_per_seed.csv", index=False)

    # checks against the paper's two-stage runs
    s0 = paper["S0"].set_index(["seed", "week"])
    c = sc[sc.step == "S1"].set_index(["seed", "week"])
    checks = {
        "S0 log-lik. recomputed vs Table 5 (max abs gap)": float((c.check_s0_log_lik - s0.loc[c.index, "log_lik"]).abs().max()),
        "S0 CRPS recomputed vs Table 5 (max abs gap)": float((c.check_s0_crps - s0.loc[c.index, "crps"]).abs().max()),
        "S0 MAE recomputed with the retrained gate classifier vs saved (max abs gap)": float((c.check_s0_mae_recomputed - c.check_s0_mae_saved).abs().max()),
        "S0 MAE recomputed vs saved (mean abs gap)": float((c.check_s0_mae_recomputed - c.check_s0_mae_saved).abs().mean()),
        "S0 gate decisions reproduced (min share of cell-days per run)": float(c.check_s0_zero_pattern_agreement.min()),
        "S0 gate decisions reproduced (mean share)": float(c.check_s0_zero_pattern_agreement.mean()),
        "rows with extended pmf support (total)": int(sc.rows_extended_support.sum()),
    }
    pd.Series(checks).to_csv(RES / "checks.csv", header=["value"])

    order = list(NAMES)
    summ = per.groupby(["step", "month"])[METRICS].agg(["mean", pstd])
    summ = summ.reindex(pd.MultiIndex.from_product([order, ["may", "november"]]))
    summ.to_csv(RES / "summary.csv")

    weekly = {s: per[per.step == s].groupby("week")[METRICS].mean() for s in order}
    tests = []
    for a, b in (("S1", "S0"), ("S2", "S1"), ("S3", "S2"), ("S3", "joint"), ("S0", "joint")):
        for k in METRICS:
            tests.append({"comparison": f"{a} vs {b}", "metric": k, "mean_a": weekly[a][k].mean(), "mean_b": weekly[b][k].mean(),
                          **paired_test(weekly[a][k], weekly[b][k], HIGHER[k])})
    tests = pd.DataFrame(tests)
    tests.to_csv(RES / "paired_tests.csv", index=False)

    # share of the S0 -> joint gap closed by each step (means over all 80 runs), for the metrics where the
    # paper's two-stage model is significantly worse than the joint one (App. B.5); elsewhere the gap is noise
    m = {s: per[per.step == s][GAP_METRICS].mean() for s in order}
    gap = m["joint"] - m["S0"]
    closed = pd.DataFrame({s: (m[s] - m[p]) / gap for p, s in (("S0", "S1"), ("S1", "S2"), ("S2", "S3"))}).T
    closed.loc["remaining (joint - S3)"] = (m["joint"] - m["S3"]) / gap
    closed.to_csv(RES / "gap_closed.csv")

    md = ["## Checks", ""] + [f"- {k}: {v:.3g}" for k, v in checks.items()]
    md += ["", "## Ladder (mean ± population std over 4 weeks × 10 seeds)", "",
           "| Step | Month | " + " | ".join(LABELS[k] for k in METRICS) + " |", "|" + "---|" * (len(METRICS) + 2)]
    for s in order:
        for month in ("may", "november"):
            r = summ.loc[(s, month)]
            md.append(f"| {NAMES[s]} | {month[:3].capitalize()} | " + " | ".join(
                f"{r[(k, 'mean')]:.3f} ± {r[(k, 'pstd')]:.3f}" for k in METRICS) + " |")
    md += ["", "## Paired tests (weeks out of 8 where the first is better; exact Wilcoxon p, uncorrected)", "",
           "| Comparison | " + " | ".join(LABELS[k] for k in METRICS) + " |", "|" + "---|" * (len(METRICS) + 1)]
    for comp, g in tests.groupby("comparison", sort=False):
        g = g.set_index("metric")
        md.append(f"| {comp} | " + " | ".join(
            "identical" if np.isnan(g.loc[k, "p"]) else f"{int(g.loc[k, 'weeks_won'])}/8 (p={g.loc[k, 'p']:.3f})" for k in METRICS) + " |")
    md += ["", "## Share of the gap between the paper's two-stage model and joint Red-LGCP closed by each step", "",
           "| Step | " + " | ".join(LABELS[k] for k in GAP_METRICS) + " |", "|" + "---|" * (len(GAP_METRICS) + 1)]
    md += [f"| {i} | " + " | ".join(f"{v:+.0%}" for v in r) + " |" for i, r in closed.iterrows()]
    md += ["", "Mean difference joint − two-stage (the gap): " + ", ".join(f"{LABELS[k]} {gap[k]:+.4f}" for k in GAP_METRICS)]
    (RES / "tables.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
