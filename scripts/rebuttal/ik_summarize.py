"""Tables of the inducing-point and kernel ablation (rebuttal); reads ik_score.py and ik_timing.py outputs.

Inducing points: M = 50, 100, 200, 300 (the paper's runs), 500, 1000, paper's kernel. Kernels: the paper's
kernel at M = 300, one component removed at a time, the static field, the additive space-time kernel, and the
w/o-kernel model of Table 5 (no Gaussian process). Values: mean and population std over 4 weeks x 10 seeds
per month. Paired tests: each variant vs the paper's model (seed-averaged weekly values, two-sided exact
Wilcoxon over the 8 test weeks), Holm-corrected over the variants of each ablation, per metric.

Convergence in M: for each seed and week, the RMSE and correlation between the posterior-mean log-intensity
alpha + C beta + A m of a run and that of the M = 1000 run, at the test cell-days.

Output: reports/rebuttal/inducing_kernels/results/
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yaml

from r1_common import holm, paired_test
from svgp_residual import ROOT

BASE = ROOT / "reports/rebuttal/inducing_kernels"
RES = BASE / "results"
METRICS = ["mae_obs", "rmse_obs", "wasserstein", "accuracy", "f1", "log_lik", "crps"]
HIGHER = {"log_lik": True, "crps": False, "mae_obs": False, "rmse_obs": False, "wasserstein": False, "accuracy": True, "f1": True}
LABELS = {"mae_obs": "MAE ↓", "rmse_obs": "RMSE ↓", "wasserstein": "Wass. ↓", "accuracy": "Acc. ↑", "f1": "F1 ↑",
          "log_lik": "Log-lik. ↑", "crps": "CRPS ↓", "captured_share_train": "share of k(x,x) captured",
          "train_minutes": "train min (wall)", "eta_rmse_M1000": "RMSE of log-int. vs M=1000",
          "eta_corr_M1000": "corr. of log-int. with M=1000"}


def pstd(s: pd.Series) -> float:
    return s.std(ddof=0)


def table(df, order, names, cols, fmt="{:.3f}"):
    out = ["| Variant | Month | " + " | ".join(LABELS.get(c, c) for c in cols) + " |", "|" + "---|" * (len(cols) + 2)]
    for v in order:
        for month in ("may", "november"):
            g = df[(df.variant == v) & (df.month == month)]
            if len(g):
                out.append(f"| {names[v]} | {month[:3].capitalize()} | " + " | ".join(
                    fmt.format(g[c].mean()) + " ± " + fmt.format(pstd(g[c])) if g[c].notna().any() else "–" for c in cols) + " |")
    return "\n".join(out)


def tests(per, order, family):
    weekly = {v: per[per.variant == v].groupby("week")[METRICS].mean() for v in order}
    rows = [{"family": family, "variant": v, "metric": k, "mean_variant": weekly[v][k].mean(), "mean_paper": weekly["paper"][k].mean(),
             **paired_test(weekly[v][k], weekly["paper"][k], HIGHER[k])} for v in order if v != "paper" for k in METRICS]
    t = pd.DataFrame(rows)
    t["p_holm"] = t.groupby("metric", group_keys=False).p.apply(lambda p: holm(p.fillna(1.0)))
    return t


def main():
    RES.mkdir(parents=True, exist_ok=True)
    spec = yaml.safe_load(open(ROOT / "config/rebuttal/inducing_kernels.yaml"))
    sc = pd.read_csv(BASE / "scores/all_scores.csv")
    counts = sc.variant.value_counts()
    m_order = ["M50", "M100", "M200", "paper", "M500", "M1000"]
    k_order = ["paper", *spec["kernels"], "no_kernel"]
    m_names = {"paper": "M = 300 (paper)", **{k: f"M = {v}" for k, v in spec["inducing_points"].items()}}
    k_names = {"paper": "paper kernel (M = 300)", **{k: v["label"] for k, v in spec["kernels"].items()},
               "no_kernel": "w/o kernel (Table 5)"}

    # w/o kernel of Table 5 (no GP)
    nk = pd.read_csv(ROOT / "reports/probabilistic_scores/ablation_nokernel.csv").rename(columns={"name": "week"})
    per = pd.concat([sc, nk.assign(variant="no_kernel")], ignore_index=True)

    # consistency of the paper's runs with reports/probabilistic_scores
    ref = pd.read_csv(ROOT / "reports/probabilistic_scores/per_window_per_seed_scores.csv")
    ref = ref[ref.method == "Red-LGCP (joint, final)"].rename(columns={"name": "week"}).set_index(["seed", "week"])
    mine = per[per.variant == "paper"].set_index(["seed", "week"])
    gap = float(np.abs(mine[["log_lik", "crps"]].values - ref.loc[mine.index, ["log_lik", "crps"]].values).max())

    # convergence of the posterior-mean log-intensity with M
    conv = []
    for _, r in per[per.variant.isin(m_order)].iterrows():
        f = BASE / "scores/fields" / r.variant / f"seed_{int(r.seed)}_{r.week}.npz"
        g = BASE / "scores/fields/M1000" / f"seed_{int(r.seed)}_{r.week}.npz"
        if f.exists() and g.exists():
            a, b = np.load(f)["eta"], np.load(g)["eta"]
            conv.append({"variant": r.variant, "seed": r.seed, "week": r.week,
                         "eta_rmse_M1000": float(np.sqrt(np.mean((a - b) ** 2))), "eta_corr_M1000": float(np.corrcoef(a, b)[0, 1])})
    if conv:
        per = per.merge(pd.DataFrame(conv), on=["variant", "seed", "week"], how="left")
    else:
        per["eta_rmse_M1000"] = per["eta_corr_M1000"] = np.nan
    per.to_csv(RES / "per_window_per_seed.csv", index=False)

    t = pd.concat([tests(per, [v for v in m_order if v in counts], "inducing points"),
                   tests(per, [v for v in k_order if v in counts or v == "no_kernel"], "kernels")], ignore_index=True)
    t.to_csv(RES / "paired_tests.csv", index=False)
    summ = per.groupby(["variant", "month"])[METRICS + ["captured_share_train", "train_minutes", "eta_rmse_M1000", "eta_corr_M1000"]].agg(["mean", pstd])
    summ.to_csv(RES / "summary.csv")

    md = [f"Runs per variant: {counts.to_dict()}", "",
          f"Consistency: the paper's runs reproduce the log-likelihood and CRPS of reports/probabilistic_scores to {gap:.1e}.", ""]
    timing = BASE / "timing.csv"
    if timing.exists():
        tm = pd.read_csv(timing)
        md += ["## Cost against M (one core, lovelace; median over 20 training steps after warm-up)", "",
               "| M | s / step | est. training min (5000 steps) | prediction s (test week) |", "|---|---|---|---|"]
        md += [f"| {int(r.M)} | {r.sec_per_step:.3f} | {r.train_min_estimate:.1f} | {r.predict_sec:.1f} |" for r in tm.itertuples()]
        md += [""]
    md += ["## Inducing points (paper's kernel)", "",
           table(per, [v for v in m_order if v in counts], m_names, METRICS), "",
           table(per, [v for v in m_order if v in counts], m_names,
                 ["captured_share_train", "train_minutes", "eta_rmse_M1000", "eta_corr_M1000"]), "",
           "## Kernels (M = 300)", "", table(per, [v for v in k_order if v in counts or v == "no_kernel"], k_names, METRICS), "",
           table(per, [v for v in k_order if v in counts], k_names, ["captured_share_train"]), "",
           "## Paired tests vs the paper's model (weeks out of 8 where the variant is better; exact Wilcoxon p, Holm-corrected within each ablation)", ""]
    for fam, g in t.groupby("family", sort=False):
        names = m_names if fam == "inducing points" else k_names
        md += [f"**{fam}**", "", "| Variant | " + " | ".join(LABELS[k] for k in METRICS) + " |", "|" + "---|" * (len(METRICS) + 1)]
        for v, gv in g.groupby("variant", sort=False):
            gv = gv.set_index("metric")
            md.append(f"| {names[v]} | " + " | ".join(
                "identical" if np.isnan(gv.loc[k, "p"]) else f"{int(gv.loc[k, 'weeks_won'])}/8 (p={gv.loc[k, 'p_holm']:.3f})" for k in METRICS) + " |")
        md += [""]
    (RES / "tables.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
