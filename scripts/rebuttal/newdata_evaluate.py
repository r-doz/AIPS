"""Evaluate Red-LGCP and the cheap baselines on the new datasets (rebuttal).

Red-LGCP runs (newdata_make_configs.py + run_local.sh) are scored with ik_score.score: log-likelihood and CRPS
of the hurdle predictive as in reports/probabilistic_scores/compute_scores.py, point metrics of the paper's gate
from metrics_higher.yaml. Baselines come from newdata_baselines.py (validation-tuned thresholds, as in the
reviewer-1 tables). Paired tests follow App. B.4 (seed-averaged weekly values, two-sided exact Wilcoxon over
the 8 test weeks), Holm-corrected per metric over the baselines of each dataset.

Output: reports/rebuttal/new_data/results/
Usage: python newdata_evaluate.py [--workers N] [--skip-scoring]
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import ik_score
from r1_common import holm, paired_test
from svgp_residual import ROOT

BASE = ROOT / "reports/rebuttal/new_data"
RES = BASE / "results"
ik_score.BASE = BASE  # posterior fields go to new_data/scores/fields
METRICS = ["mae_obs", "rmse_obs", "wasserstein", "accuracy", "f1", "log_lik", "crps"]
HIGHER = {"log_lik": True, "crps": False, "mae_obs": False, "rmse_obs": False, "wasserstein": False, "accuracy": True, "f1": True}
LABELS = {"mae_obs": "MAE ↓", "rmse_obs": "RMSE ↓", "wasserstein": "Wass. ↓", "accuracy": "Acc. ↑", "f1": "F1 ↑",
          "log_lik": "Log-lik. ↑", "crps": "CRPS ↓"}
BASELINES = ["LAP", "GCMP", "GLM", "Hurdle GLM"]
DATASETS = {"venice": "Venice–Chioggia, 2024–2025", "trieste_4y": "Gulf of Trieste, 2022–2025 (4-year history)"}
# trieste_2y (2-year control on the rebuilt data) was stopped and is not evaluated


def _one(job):
    try:
        return [ik_score.score(job)]
    except Exception as exc:  # report and continue
        print(f"FAILED {job}: {exc!r}", flush=True)
        return []


def score_red_lgcp(workers: int) -> pd.DataFrame:
    jobs = [(p.parts[-6], p.parent) for ds in DATASETS for p in sorted((BASE / "runs" / ds).glob("seed_*/runs/*/*/model.pt"))]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        rows = [r for part in ex.map(_one, jobs, chunksize=1) for r in part]
    df = pd.DataFrame(rows).rename(columns={"variant": "dataset"}).assign(model="Red-LGCP")
    (BASE / "scores").mkdir(parents=True, exist_ok=True)
    df.to_csv(BASE / "scores/red_lgcp_scores.csv", index=False)
    print(f"Scored {len(df)} Red-LGCP runs (of {len(jobs)})", flush=True)
    return df


def paper_red_lgcp() -> pd.DataFrame:
    """The paper's Red-LGCP (Gulf of Trieste, 2024-2025, paper's data), as reference."""
    pt = pd.read_csv(ROOT / "reports/seeded_monthly_windows_hurdle/per_window_per_seed_metrics.csv")
    pr = pd.read_csv(ROOT / "reports/probabilistic_scores/per_window_per_seed_scores.csv")
    pr = pr[pr.method == "Red-LGCP (joint, final)"][["seed", "name", "log_lik", "crps"]]
    return (pt.merge(pr, on=["seed", "name"], validate="one_to_one").rename(columns={"name": "week"})
            .assign(model="Red-LGCP (paper data)", dataset="paper")[["dataset", "model", "seed", "week", "month"] + METRICS])


def pstd(s):
    return s.std(ddof=0)


def table(per: pd.DataFrame, models: list[str]) -> list[str]:
    out = ["| Model | Month | " + " | ".join(LABELS[k] for k in METRICS) + " |", "|" + "---|" * (len(METRICS) + 2)]
    for month in ("may", "november"):
        sub = per[per.month == month]
        means = sub.groupby("model")[METRICS].mean()
        best = {k: (means[k].max() if HIGHER[k] else means[k].min()) for k in METRICS}
        for m in models:
            g = sub[sub.model == m]
            if not len(g):
                continue
            cells = []
            for k in METRICS:
                v = f"{g[k].mean():.3f}"
                cells.append(f"**{v}**" if round(g[k].mean(), 3) == round(best[k], 3) else v)
            out.append(f"| {m} | {month[:3].capitalize()} | " + " | ".join(cells) + " |")
    return out


def tests(per: pd.DataFrame, ours: str, others: list[str]) -> pd.DataFrame:
    w = {m: per[per.model == m].groupby("week")[METRICS].mean() for m in [ours] + others}
    rows = [{"model": m, "metric": k, "ours": w[ours][k].mean(), "theirs": w[m][k].mean(),
             **paired_test(w[ours][k], w[m][k], HIGHER[k])} for m in others for k in METRICS]
    t = pd.DataFrame(rows)
    t["p_holm"] = t.groupby("metric", group_keys=False).p.apply(lambda p: holm(p.fillna(1.0)))
    return t


def test_lines(t: pd.DataFrame, title: str) -> list[str]:
    out = [title, "", "| vs | " + " | ".join(LABELS[k] for k in METRICS) + " |", "|" + "---|" * (len(METRICS) + 1)]
    for m, g in t.groupby("model", sort=False):
        g = g.set_index("metric")
        out.append(f"| {m} | " + " | ".join("identical" if np.isnan(g.loc[k, "p"]) else
                                             f"{int(g.loc[k, 'weeks_won'])}/8 (p={g.loc[k, 'p_holm']:.3f})" for k in METRICS) + " |")
    return out + [""]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--skip-scoring", action="store_true")
    args = ap.parse_args()
    RES.mkdir(parents=True, exist_ok=True)
    red = pd.read_csv(BASE / "scores/red_lgcp_scores.csv") if args.skip_scoring else score_red_lgcp(args.workers)
    base = pd.concat([pd.read_csv(BASE / f"baselines/{ds}_per_window.csv") for ds in DATASETS], ignore_index=True)
    per = pd.concat([red[["dataset", "model", "seed", "week", "month"] + METRICS],
                     base[["dataset", "model", "seed", "week", "month"] + METRICS], paper_red_lgcp()], ignore_index=True)
    per.to_csv(RES / "per_window_per_seed.csv", index=False)
    per.groupby(["dataset", "model", "month"])[METRICS].agg(["mean", pstd]).to_csv(RES / "summary.csv")

    md, all_tests = ["Runs per dataset and model: " + str(per.groupby(["dataset", "model"]).size().to_dict()), ""], []
    for ds, title in DATASETS.items():
        d = per[per.dataset == ds]
        if not (d.model == "Red-LGCP").any():
            continue
        t = tests(d, "Red-LGCP", BASELINES).assign(dataset=ds)
        all_tests.append(t)
        md += [f"## {title}", "", "Mean over 4 test weeks × 10 seeds (Red-LGCP) or 4 test weeks (deterministic baselines); "
               "baselines with validation-tuned thresholds (as in the reviewer-1 tables); best in bold.", ""]
        md += table(d, ["Red-LGCP"] + BASELINES) + [""]
        md += test_lines(t, "Red-LGCP vs each baseline: weeks out of 8 where Red-LGCP is better; exact Wilcoxon p, Holm-corrected over the 4 baselines")
    # longer history: 4-year Red-LGCP vs the paper's Red-LGCP (2 years, paper's data) as reference
    h = per[(per.model == "Red-LGCP") & (per.dataset == "trieste_4y")]
    if len(h):
        hh = pd.concat([h.assign(model="Red-LGCP, 4 years"), per[per.dataset == "paper"].assign(model="Red-LGCP, 2 years (paper)")])
        t = tests(hh, "Red-LGCP, 4 years", ["Red-LGCP, 2 years (paper)"]).assign(dataset="history")
        all_tests.append(t)
        md += ["## Gulf of Trieste: 4 years of history vs the paper's 2 years", ""] + table(hh, ["Red-LGCP, 4 years",
                                                                                             "Red-LGCP, 2 years (paper)"]) + [""]
        md += test_lines(t, "4 years vs paper: weeks out of 8 where the 4-year model is better; exact Wilcoxon p")
    if all_tests:
        pd.concat(all_tests, ignore_index=True).to_csv(RES / "paired_tests.csv", index=False)
    (RES / "tables.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
