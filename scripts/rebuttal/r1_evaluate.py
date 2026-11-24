"""Reviewer-1 evaluation: validation-tuned thresholds, test metrics, paired tests against Red-LGCP.

Activity threshold (all models). A cell-day is predicted active when P(Y > 0) >= tau, with
P(Y > 0) = 1 - exp(-rate) for Poisson models and p for hurdle models. For every model, tau is chosen
on the validation-week predictions (seeds and weeks pooled, grid 0.01..0.99) twice: tau_acc maximizes
the mean accuracy and tau_f1 the mean F1. Ties go to the smallest tau.

Experiment 1 (Poisson baselines of Table 1): accuracy at tau_acc and F1 at tau_f1 replace the paper's
1[rate > 0]. Also reported: all activity scores at tau_f1, and MAE, RMSE and Wasserstein distance of the
point forecast gated at tau_f1 (rate if active, 0 otherwise), which the paper does not use for baselines.

Experiments 2 and 3 (hurdle GLM, GNN-hurdle, ConvLSTM-lag(-hurdle)): the point forecast of a hurdle
model is Red-LGCP's gate with tau_f1 and redistribution (rho = gamma = 1; sigma = 0, as these models have
no dropout uncertainty), applied to the hurdle mean p lam / (1 - exp(-lam)). Log-likelihood and CRPS
score the hurdle predictive distribution. ConvLSTM-lag (Poisson head) is evaluated like experiment 1.

Red-LGCP is the definitive model (reports/seeded_monthly_windows_hurdle and
reports/probabilistic_scores), unchanged. Paired tests follow App. B.4: metrics averaged over seeds
within each test week, two-sided exact Wilcoxon over the 8 weeks, Holm correction within each experiment.

Outputs: reports/rebuttal/reviewer1/results/
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from r1_common import OUT, PRED_DIR, ROOT, TAU_GRID, gate, holm, hurdle_mean, hurdle_scores, paired_test, point_metrics, poisson_scores

RES = OUT / "results"
POISSON = ["LAP", "GCMP", "GLM", "GNN", "ConvLSTM"]
HURDLE = ["Hurdle GLM", "GNN-hurdle", "ConvLSTM-lag", "ConvLSTM-lag-hurdle"]
HIGHER = {"accuracy": True, "f1": True, "log_lik": True, "mae_obs": False, "rmse_obs": False, "wasserstein": False, "crps": False}
METRICS = ["mae_obs", "rmse_obs", "wasserstein", "accuracy", "f1", "log_lik", "crps"]


def prob_active(df):
    return df["p"].to_numpy() if "p" in df and df["p"].notna().all() else -np.expm1(-df["rate"].to_numpy())


def tune_tau(val: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Validation accuracy and F1 for every tau, averaged over (seed, week)."""
    acc, f1 = [], []
    for _, g in val.groupby(["seed", "week"]):
        obs = g.y.to_numpy() > 0
        pred = prob_active(g)[:, None] >= TAU_GRID[None, :]
        tp = (pred & obs[:, None]).sum(0)
        npred, nobs = pred.sum(0), obs.sum()
        prec = np.divide(tp, npred, out=np.zeros(len(TAU_GRID)), where=npred > 0)
        rec = tp / nobs if nobs else np.zeros(len(TAU_GRID))
        acc.append((pred == obs[:, None]).mean(0))
        f1.append(np.divide(2 * prec * rec, prec + rec, out=np.zeros(len(TAU_GRID)), where=(prec + rec) > 0))
    curve = pd.DataFrame({"tau": TAU_GRID, "val_accuracy": np.mean(acc, 0), "val_f1": np.mean(f1, 0)})
    ia, i1 = int(np.argmax(curve.val_accuracy)), int(np.argmax(curve.val_f1))
    return {"tau_acc": float(TAU_GRID[ia]), "val_accuracy": float(curve.val_accuracy[ia]),
            "tau_f1": float(TAU_GRID[i1]), "val_f1": float(curve.val_f1[i1])}, curve


def evaluate_model(name: str, pred: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    tau, curve = tune_tau(pred[pred.split == "val"])
    rows = []
    for (seed, week, month), g in pred[pred.split == "test"].groupby(["seed", "week", "month"]):
        y, dates, pa = g.y.to_numpy(), g.date.to_numpy(), prob_active(g)
        hurdle = "p" in g and g["p"].notna().all()
        if hurdle:
            mu = hurdle_mean(g.p, g.lam)
            point = gate(mu, g.p, tau["tau_f1"], dates, redistribute=True)
            ll, cr = hurdle_scores(y, g.p, g.lam)
            gate_only = point_metrics(y, gate(mu, g.p, tau["tau_f1"], dates, redistribute=False), dates)
        else:
            rate = g.rate.to_numpy()
            point = np.where(pa >= tau["tau_f1"], rate, 0.0)
            ll, cr = poisson_scores(y, rate)
            paper = point_metrics(y, rate, dates)
        m = point_metrics(y, point, dates)
        at_acc = point_metrics(y, (pa >= tau["tau_acc"]).astype(float), dates)
        row = {"model": name, "seed": seed, "week": week, "month": month,
               "mae_obs": m["mae_obs"], "rmse_obs": m["rmse_obs"], "wasserstein": m["wasserstein"],
               "accuracy": at_acc["accuracy"], "f1": m["f1"], "accuracy_at_tau_f1": m["accuracy"],
               "precision_at_tau_f1": m["precision"], "recall_at_tau_f1": m["recall"],
               "log_lik": float(ll.mean()), "crps": float(cr.mean())}
        if hurdle:
            row.update({f"{k}_gate_only": gate_only[k] for k in ("mae_obs", "rmse_obs", "wasserstein")})
            row.update({f"{k}_no_gate": v for k, v in point_metrics(y, mu, dates).items() if k in ("mae_obs", "rmse_obs", "wasserstein")})
        else:
            row.update({f"{k}_paper": paper[k] for k in ("mae_obs", "rmse_obs", "wasserstein", "accuracy", "f1")})
        rows.append(row)
    return {"model": name, **tau}, pd.DataFrame(rows), curve.assign(model=name)


def red_lgcp() -> pd.DataFrame:
    pt = pd.read_csv(ROOT / "reports/seeded_monthly_windows_hurdle/per_window_per_seed_metrics.csv")
    pr = pd.read_csv(ROOT / "reports/probabilistic_scores/per_window_per_seed_scores.csv")
    pr = pr[pr.method == "Red-LGCP (joint, final)"][["seed", "name", "log_lik", "crps"]]
    df = pt.merge(pr, on=["seed", "name"], validate="one_to_one")
    assert len(df) == 80
    return df.rename(columns={"name": "week"}).assign(model="Red-LGCP")[["model", "seed", "week", "month"] + METRICS]


def summarize(per: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    g = per.groupby(["model", "month"], sort=False)[cols]
    return g.mean().add_suffix("_mean").join(g.agg(lambda s: s.std(ddof=0)).add_suffix("_std")).reset_index()


def tests(per: pd.DataFrame, ref: pd.DataFrame, models: list[str], metrics: list[str], family: str) -> pd.DataFrame:
    ours = ref.groupby("week")[metrics].mean()
    rows = []
    for m in models:
        theirs = per[per.model == m].groupby("week")[metrics].mean()
        for k in metrics:
            rows.append({"family": family, "metric": k, "baseline": m, "red_lgcp": ours[k].mean(),
                         "baseline_mean": theirs[k].mean(), **paired_test(ours[k], theirs[k], HIGHER[k])})
    t = pd.DataFrame(rows)
    t["p_holm"] = t.groupby("metric", group_keys=False).p.apply(holm)
    return t


def md(df: pd.DataFrame, index: bool = False) -> str:
    """Minimal markdown table (avoids the optional tabulate dependency of DataFrame.to_markdown)."""
    d = df.reset_index() if index else df
    f = lambda v: f"{v:.3f}" if isinstance(v, (float, np.floating)) else str(v)  # noqa: E731
    head = "| " + " | ".join(map(str, d.columns)) + " |"
    return "\n".join([head, "|" + "---|" * len(d.columns)] + ["| " + " | ".join(f(v) for v in r) + " |" for r in d.itertuples(index=False)])


def fmt_table(summary: pd.DataFrame, cols: list[str], labels: dict) -> str:
    lines = ["| Model | Month | " + " | ".join(labels[c] for c in cols) + " |", "|" + "---|" * (len(cols) + 2)]
    for _, r in summary.iterrows():
        cells = [f"{r[c + '_mean']:.3f} ± {r[c + '_std']:.3f}" for c in cols]
        lines.append(f"| {r.model} | {r.month[:3].capitalize()} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    RES.mkdir(parents=True, exist_ok=True)
    preds = {}
    for f in ("poisson_baselines", "hurdle_glm", "neural_hurdle"):
        path = PRED_DIR / f"{f}.parquet"
        if path.exists():
            d = pd.read_parquet(path)
            for m, g in d.groupby("model"):
                preds[m] = g
        else:
            print(f"(missing {path.name}: skipped)")

    taus, per, curves = [], [], []
    for name in [m for m in POISSON + HURDLE if m in preds]:
        t, p, c = evaluate_model(name, preds[name])
        taus.append(t), per.append(p), curves.append(c)
        print(f"{name:22s} tau_acc={t['tau_acc']:.2f} (val acc {t['val_accuracy']:.3f})  tau_f1={t['tau_f1']:.2f} (val F1 {t['val_f1']:.3f})")
    taus, per, curves = pd.DataFrame(taus), pd.concat(per, ignore_index=True), pd.concat(curves, ignore_index=True)
    ref = red_lgcp()
    taus.to_csv(RES / "thresholds.csv", index=False)
    curves.to_csv(RES / "threshold_validation_curves.csv", index=False)
    per.to_csv(RES / "per_window_per_seed.csv", index=False)

    labels = {"mae_obs": "MAE ↓", "rmse_obs": "RMSE ↓", "wasserstein": "Wasserstein ↓", "accuracy": "Accuracy ↑",
              "f1": "F1 ↑", "log_lik": "Log-lik. ↑", "crps": "CRPS ↓", "accuracy_paper": "Acc. (paper) ↑",
              "f1_paper": "F1 (paper) ↑", "accuracy_at_tau_f1": "Acc. at τ_F1 ↑"}
    out = []

    # ---- experiment 1
    e1 = [m for m in POISSON if m in preds]
    s1 = summarize(pd.concat([per[per.model.isin(e1)], ref]), ["accuracy", "f1", "accuracy_at_tau_f1", "mae_obs", "rmse_obs", "wasserstein"])
    s1p = summarize(per[per.model.isin(e1)], ["accuracy_paper", "f1_paper"])
    s1 = s1.merge(s1p, on=["model", "month"], how="left")
    s1.to_csv(RES / "exp1_thresholded_baselines.csv", index=False)
    t1 = tests(per, ref, e1, ["accuracy", "f1", "mae_obs", "wasserstein"], "exp1")
    out += ["## Experiment 1: Poisson baselines with a validation-tuned activity threshold\n",
            md(taus[taus.model.isin(e1)]), "",
            fmt_table(s1, ["accuracy_paper", "f1_paper", "accuracy", "f1", "accuracy_at_tau_f1"], labels), "",
            "Gated point forecast at τ_F1 (rate if P(Y>0) ≥ τ_F1, else 0); the paper's baselines are ungated:", "",
            fmt_table(s1, ["mae_obs", "rmse_obs", "wasserstein"], labels), ""]

    # ---- experiments 2 and 3
    e23 = [m for m in HURDLE if m in preds]
    if e23:
        s23 = summarize(pd.concat([per[per.model.isin(e23)], ref]), METRICS)
        s23.to_csv(RES / "exp23_hurdle_models.csv", index=False)
        t23 = tests(per, ref, e23, METRICS, "exp2+3")
        hcols = [c for c in per.columns if c.endswith("_gate_only") or c.endswith("_no_gate")]
        if hcols:
            summarize(per[per.model.isin(e23)], hcols).to_csv(RES / "exp23_point_forecast_variants.csv", index=False)
        out += ["## Experiments 2 and 3: hurdle GLM and neural baselines with lags and a hurdle head\n",
                md(taus[taus.model.isin(e23)]), "",
                fmt_table(s23, METRICS, labels), ""]
    else:
        t23 = pd.DataFrame()
    tt = pd.concat([t1, t23], ignore_index=True)
    tt.to_csv(RES / "paired_tests.csv", index=False)
    out += ["## Paired tests (Red-LGCP vs baseline; weeks won out of 8, Holm-adjusted p within each experiment)\n"]
    piv = tt.assign(cell=lambda d: d.weeks_won.astype(str) + "/8 (" + d.p_holm.map(lambda p: f"{p:.3f}") + ")")
    out.append(md(piv.pivot_table(index=["family", "baseline"], columns="metric", values="cell", aggfunc="first", sort=False), index=True))

    # ---- platform check of the GNN
    if "GNN (rerun)" in preds:
        g = preds["GNN (rerun)"][preds["GNN (rerun)"].split == "test"]
        rows = []
        for (seed, week, month), d in g.groupby(["seed", "week", "month"]):
            m = point_metrics(d.y, d.rate, d.date)
            ll, cr = poisson_scores(d.y, d.rate)
            rows.append({"model": "GNN (rerun on Demetra)", "seed": seed, "week": week, "month": month,
                         **{k: m[k] for k in ("mae_obs", "rmse_obs", "wasserstein")}, "log_lik": ll.mean(), "crps": cr.mean()})
        defi = per[per.model == "GNN"][["seed", "week", "month", "mae_obs_paper", "rmse_obs_paper", "wasserstein_paper", "log_lik", "crps"]]
        defi = defi.rename(columns=lambda c: c.replace("_paper", "")).assign(model="GNN (definitive, Table 1)")
        chk = summarize(pd.concat([defi, pd.DataFrame(rows)]), ["mae_obs", "rmse_obs", "wasserstein", "log_lik", "crps"])
        chk.to_csv(RES / "gnn_platform_check.csv", index=False)
        out += ["", "## GNN retrained on Demetra (paper environment) vs the definitive runs\n",
                fmt_table(chk, ["mae_obs", "rmse_obs", "wasserstein", "log_lik", "crps"], labels)]

    (RES / "tables.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
