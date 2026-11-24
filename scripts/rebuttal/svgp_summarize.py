"""Tables of the SVGP residual-variance experiment (reads the outputs of svgp_turing.slurm).

Variants (model trained with / predictive distribution with the residual variance):
  SoR (paper)                 the definitive runs (trained on Demetra/lovelace), no residual
  SoR + residual at prediction  the definitive runs, residual added to the predictive distribution only
  SoR (retrained)             the paper's model retrained on turing-01 with the same configs and seeds
  SVGP (retrained)            residual in training and prediction, retrained on turing-01
  SVGP, no residual at prediction

The main comparison is SVGP (retrained) vs SoR (retrained): same machine, configs and seeds; only the
residual variance differs. Paired tests follow App. B.4 (seed-averaged weekly values, two-sided exact
Wilcoxon over the 8 test weeks); p-values are not corrected for the 7 metrics.

Output: reports/rebuttal/svgp_residual/results/
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from r1_common import paired_test
from svgp_residual import ROOT

BASE = ROOT / "reports/rebuttal/svgp_residual"
RES = BASE / "results"
METRICS = ["log_lik", "crps", "mae_obs", "rmse_obs", "wasserstein", "accuracy", "f1"]
HIGHER = {"log_lik": True, "crps": False, "mae_obs": False, "rmse_obs": False, "wasserstein": False, "accuracy": True, "f1": True}
LABELS = {"log_lik": "Log-lik. ↑", "crps": "CRPS ↓", "mae_obs": "MAE ↓", "rmse_obs": "RMSE ↓", "wasserstein": "Wass. ↓",
          "accuracy": "Acc. ↑", "f1": "F1 ↑"}
VARIANTS = [("SoR (paper)", "no", "SoR (paper)"), ("SoR (paper)", "yes", "SoR + residual at prediction"),
            ("SoR (retrained)", "no", "SoR (retrained)"), ("SVGP (retrained)", "yes", "SVGP (retrained)"),
            ("SVGP (retrained)", "no", "SVGP, no residual at prediction")]


def table(df: pd.DataFrame, cols: list[str], labels: dict, fmt: str = "{:.3f}") -> str:
    out = ["| Variant | Month | " + " | ".join(labels.get(c, c) for c in cols) + " |", "|" + "---|" * (len(cols) + 2)]
    for (v, month), g in df.groupby(["variant", "month"], sort=False):
        out.append(f"| {v} | {month[:3].capitalize()} | " + " | ".join(
            f"{fmt.format(g[c].mean())} ± {fmt.format(g[c].std(ddof=0))}" for c in cols) + " |")
    return "\n".join(out)


def main():
    RES.mkdir(parents=True, exist_ok=True)
    sc = pd.read_csv(BASE / "scores/all_scores.csv")
    parts = []
    for model, resid, name in VARIANTS:
        d = sc[(sc.model == model) & (sc.residual_in_prediction == resid)].copy()
        assert len(d) == 80, (name, len(d))
        parts.append(d.assign(variant=name))
    per = pd.concat(parts, ignore_index=True)
    per["resid_share_train"] = per.resid_var_train / per.k0
    per["latent_var_test"] = per.proj_var_test + np.where(per.residual_in_prediction == "yes", per.resid_var_test, 0.0)
    per.to_csv(RES / "per_window_per_seed.csv", index=False)

    # consistency with the paper's probabilistic scores
    ref = pd.read_csv(ROOT / "reports/probabilistic_scores/per_window_per_seed_scores.csv")
    ref = ref[ref.method == "Red-LGCP (joint, final)"].set_index(["seed", "name"])[["log_lik", "crps"]]
    mine = per[per.variant == "SoR (paper)"].set_index(["seed", "week"])[["log_lik", "crps"]]
    gap = float(np.abs(mine.values - ref.loc[mine.index].values).max())

    month_order = {"may": 0, "november": 1}
    per = per.sort_values(["variant", "month"], key=lambda s: s.map(month_order) if s.name == "month" else s.map(
        {v[2]: i for i, v in enumerate(VARIANTS)}))
    summ = per.groupby(["variant", "month"], sort=False)[METRICS].agg(["mean", lambda s: s.std(ddof=0)])
    summ.to_csv(RES / "predictive_summary.csv")

    tests = []
    for a, b in (("SVGP (retrained)", "SoR (retrained)"), ("SVGP (retrained)", "SoR (paper)"),
                 ("SoR + residual at prediction", "SoR (paper)"), ("SoR (retrained)", "SoR (paper)")):
        wa = per[per.variant == a].groupby("week")[METRICS].mean()
        wb = per[per.variant == b].groupby("week")[METRICS].mean()
        for k in METRICS:
            tests.append({"comparison": f"{a} vs {b}", "metric": k, "mean_a": wa[k].mean(), "mean_b": wb[k].mean(),
                          **paired_test(wa[k], wb[k], HIGHER[k])})
    tests = pd.DataFrame(tests)
    tests.to_csv(RES / "paired_tests.csv", index=False)

    # GP contribution
    gp_cols = ["var_s_broad", "var_s_loc", "var_t_rbf", "var_t_per", "k0", "resid_share_train", "proj_var_train",
               "var_fbar_train", "latent_var_test", "alpha"]
    gp = per[per.variant.isin(["SoR (paper)", "SoR (retrained)", "SVGP (retrained)"])]
    interp = {"SoR (paper)": ROOT / "reports/interpretability_hurdle/params_effects_all_runs.csv",
              "SoR (retrained)": BASE / "interpretability_sor/params_effects_all_runs.csv",
              "SVGP (retrained)": BASE / "interpretability_svgp/params_effects_all_runs.csv"}
    eff_cols = ["share_static", "share_cycle", "share_short", "period_days", "spearman_map_vs_mean_positive_count",
                "effect_chl", "effect_thetao", "effect_salinity", "effect_fishing_block", "effect_is_holiday",
                "effect_is_weekend", "effect_cell_lag_1", "effect_cell_lag_7"]
    eff = []
    for v, path in interp.items():
        if path.exists():
            eff.append(pd.read_csv(path).rename(columns={"window": "week"}).assign(variant=v))
    eff = pd.concat(eff, ignore_index=True) if eff else pd.DataFrame(columns=["variant", "seed", "week", "month"] + eff_cols)
    gpe = gp.merge(eff[["variant", "seed", "week"] + eff_cols], on=["variant", "seed", "week"], how="left")
    gpe.groupby(["variant", "month"], sort=False)[gp_cols + eff_cols].agg(["mean", lambda s: s.std(ddof=0)]).to_csv(RES / "gp_contribution.csv")

    lab = {**LABELS, "var_s_broad": "σ²_s,broad", "var_s_loc": "σ²_s,loc", "var_t_rbf": "σ²_t,RBF", "var_t_per": "σ²_t,per",
           "k0": "prior var. k(x,x)", "resid_share_train": "share of k(x,x) not captured by u",
           "proj_var_train": "proj. var. a'Sa", "var_fbar_train": "var. of f̄ = Am", "latent_var_test": "latent var. at test",
           "alpha": "α", "share_static": "static share", "share_cycle": "periodic share", "share_short": "short-term share",
           "period_days": "period (days)", "spearman_map_vs_mean_positive_count": "Spearman map vs counts",
           **{f"effect_{c}": f"×{c}" for c in ("chl", "thetao", "salinity", "fishing_block", "is_holiday", "is_weekend", "cell_lag_1", "cell_lag_7")}}
    md = [f"Consistency check: SoR (paper) log-likelihood and CRPS reproduce reports/probabilistic_scores to {gap:.1e}.", "",
          "## Predictive performance (mean ± population std over 4 weeks × 10 seeds)", "", table(per, METRICS, LABELS), "",
          "## Paired tests (weeks out of 8 where the first variant is better; exact Wilcoxon p, uncorrected)", ""]
    # p is undefined when the two variants coincide in every week (accuracy and F1 depend only on the classifier)
    piv = tests.assign(cell=lambda d: np.where(d.p.isna(), "identical",
                                               d.weeks_won.astype(str) + "/8 (p=" + d.p.map(lambda p: f"{p:.3f}") + ")"))
    p2 = piv.pivot_table(index="comparison", columns="metric", values="cell", aggfunc="first", sort=False)[METRICS]
    md += ["| Comparison | " + " | ".join(LABELS[c] for c in METRICS) + " |", "|" + "---|" * (len(METRICS) + 1)]
    md += [f"| {i} | " + " | ".join(r) + " |" for i, r in p2.iterrows()]
    md += ["", "## GP contribution", "", table(gpe, gp_cols[:4] + ["alpha"], lab), "",
           table(gpe, ["k0", "resid_share_train", "proj_var_train", "var_fbar_train", "latent_var_test"], lab), "",
           table(gpe, eff_cols[:5], lab), "", table(gpe, eff_cols[5:], lab)]
    (RES / "tables.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
