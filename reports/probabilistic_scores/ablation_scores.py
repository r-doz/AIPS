"""Ablation of the joint Red-LGCP: point metrics, log-likelihood and CRPS for each variant.

Variants and where they come from:
  full                 joint hurdle model, threshold tau + sigma            (seeded_monthly_windows_hurdle)
  w/o uncertainty      same runs, fixed threshold tau                       (metrics_fixed.yaml)
  w/o redistribution   same runs, zero rejected cells without recovery      (recomputed from hurdle_predictions.csv)
  w/o gate             same runs, expected counts without gating            (metrics_nogate.yaml)
  w/o joint training   LGCP + post-hoc classifier (previous Red-LGCP)       (seeded_monthly_windows_salinity/11_train_lgcp)
  w/o classifier       LGCP alone                                           (seeded_monthly_windows_ablation/no_classifier)
  w/o kernel           joint hurdle model without Gaussian process          (seeded_monthly_windows_hurdle_ablation/no_kernel)

Log-likelihood and CRPS score each variant's predictive distribution: the hurdle predictive for the
joint models (identical for the four gating rules, which only change point predictions) and the LGCP
posterior predictive E_q[Poisson(y | lambda)] for the two models without joint training (the post-hoc
gate changes point predictions only).
"""

import argparse
import glob
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.special import gammaln

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.models.data_pp_lgcp import prepare_data  # noqa: E402
from src.models.hurdle_lgcp import HurdleSparseLGCP  # noqa: E402
from src.models.lgcp import SparseLGCP, build_Kfu, build_Kuu  # noqa: E402
from src.models.metrics_lgcp import evaluate_metrics  # noqa: E402

OUT = ROOT / "reports/probabilistic_scores"
HURDLE = ROOT / "reports/seeded_monthly_windows_hurdle"
NOKERNEL = ROOT / "reports/seeded_monthly_windows_hurdle_ablation/no_kernel"
OLD = ROOT / "reports/seeded_monthly_windows_salinity/11_train_lgcp"
NOCLF = ROOT / "reports/seeded_monthly_windows_ablation/no_classifier"
POINT = ["mae_obs", "rmse_obs", "wasserstein", "accuracy", "f1"]
WINDOWS = [f"2025_{m}_{d:02d}_{d + 6:02d}" for m in ("may", "november") for d in (1, 8, 15, 22)]


def month_of(window):
    return "may" if "_may_" in window else "november"


def definitive_run(root, seed, window):
    mm = yaml.safe_load(open(root / f"seed_{seed}/monthly_metrics.yaml"))
    p = [w["metrics_path"] for w in mm["per_window"] if w["name"] == window][0]
    return ROOT / p.split("projects/AIPS/")[-1] if "projects/AIPS/" in p else Path(p)


def pmf_scores(y, pmf):
    y = np.asarray(y, int)
    ll = np.log(np.clip(pmf[np.arange(len(y)), y], 1e-300, None))
    cdf = np.cumsum(pmf, axis=1)
    ind = (np.arange(pmf.shape[1])[None, :] >= y[:, None]).astype(float)
    return ll.mean(), ((cdf - ind) ** 2).sum(1).mean()


def rebuild(model_cls, run, **extra):
    ck = torch.load(run / "model.pt", map_location="cpu", weights_only=False)
    cfg = ck["config"]
    trc, trv, try_, tec, tev, tey, _, _ = prepare_data(
        str(ROOT / cfg["parquet_path"]), train_fraction=cfg["train_fraction"], random_seed=cfg["data_seed"],
        split_strategy=cfg["split_strategy"], covariate_cols=cfg["covariate_cols"],
        test_start_date=cfg["test_start_date"], test_end_date=cfg["test_end_date"],
        lag_features=cfg["lag_features"], years=cfg["years"])
    kwargs = {}
    if model_cls is HurdleSparseLGCP:
        h = cfg.get("hurdle", {})
        kwargs = dict(clf_hidden=int(h.get("clf_hidden", 32)), clf_dropout=float(h.get("clf_dropout", 0.1)),
                      clf_alpha=float(h.get("clf_alpha", 1e-4)))
    model = model_cls(trc, trv, try_, kernel_config=cfg["kernel_config"], M_inducing=cfg["M_inducing"],
                      device="cpu", np_seed=cfg["np_seed"], **kwargs)
    model.load_state_dict(ck["model_state"])
    return model, tec, tev, tey


@torch.no_grad()
def lambda_samples(model, tec, tev, n=300):
    Xn, Cn = torch.tensor(tec, dtype=torch.float32), torch.tensor(tev, dtype=torch.float32)
    kern = model.kernel_params()
    eye = torch.eye(model.M)
    chol = torch.linalg.cholesky(build_Kuu(model.Z, kern, 1e-6) + 1e-8 * eye)
    A = torch.cholesky_solve(build_Kfu(Xn, model.Z, kern).T, chol).T
    Lq = torch.linalg.cholesky(model.get_S() + 1e-8 * eye)
    U = model.m.unsqueeze(0) + torch.randn(n, model.M) @ Lq.T
    return torch.exp(model.alpha + (Cn @ model.beta).unsqueeze(0) + U @ A.T).numpy().astype(float), Xn, Cn


def lgcp_scores(run, seed):
    model, tec, tev, tey = rebuild(SparseLGCP, run)
    torch.manual_seed(2000 + seed)
    lam, _, _ = lambda_samples(model, tec, tev)
    K = int(min(400, max(tey.max() + 10, np.quantile(lam, 0.999) * 3 + 30)))
    k = np.arange(K + 1)
    logp = k[None, None, :] * np.log(np.clip(lam, 1e-300, None))[:, :, None] - lam[:, :, None] - gammaln(k + 1)[None, None, :]
    pmf = np.exp(logp).mean(0)
    assert pmf.sum(1).min() > 0.9999
    return pmf_scores(tey, pmf)


def hurdle_scores(run, seed):
    model, tec, tev, tey = rebuild(HurdleSparseLGCP, run)
    torch.manual_seed(1000 + seed)
    lam, Xn, Cn = lambda_samples(model, tec, tev)
    with torch.no_grad():
        model.classifier.train()
        feats = torch.cat([Xn, Cn], dim=1)
        p_bar = torch.stack([torch.sigmoid(model.classifier(feats).squeeze(-1)) for _ in range(100)]).mean(0).numpy().astype(float)
    K = int(min(400, max(tey.max() + 10, np.quantile(lam, 0.999) * 3 + 30)))
    k = np.arange(1, K + 1)
    log1mexp = np.where(lam < 1e-6, np.log(np.clip(lam, 1e-300, None)), np.log(-np.expm1(-lam)))
    log_tp = (k[None, None, :] * np.log(np.clip(lam, 1e-300, None))[:, :, None] - lam[:, :, None]
              - gammaln(k + 1)[None, None, :] - log1mexp[:, :, None])
    pmf = np.concatenate([(1 - p_bar)[:, None], p_bar[:, None] * np.exp(log_tp).mean(0)], axis=1)
    assert pmf.sum(1).min() > 0.9999
    return pmf_scores(tey, pmf)


def r3(x):
    s = str(Decimal(repr(float(x))).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))
    return s.replace("-", "$-$", 1) if s.startswith("-") else s


parser = argparse.ArgumentParser()
parser.add_argument("--stage", choices=["base", "nokernel", "table"], required=True)
args = parser.parse_args()
torch.set_num_threads(4)

if args.stage == "base":
    rows = []
    full_scores = pd.read_csv(OUT / "per_window_per_seed_scores.csv")
    full_scores = full_scores[full_scores.method == "Red-LGCP (joint, final)"].set_index(["seed", "name"])
    for seed in range(1, 11):
        for w in WINDOWS:
            (run,) = glob.glob(str(HURDLE / f"seed_{seed}/runs/{w}/*/"))
            run = Path(run)
            ll, cr = full_scores.loc[(seed, w), ["log_lik", "crps"]]
            pr = pd.read_csv(run / "hurdle_predictions.csv")
            hard = np.where(pr.p_active_mean >= pr.tau_higher, pr.expected_count, 0.0)
            variants = {
                "full": yaml.safe_load(open(run / "metrics_higher.yaml")),
                "w/o uncertainty": yaml.safe_load(open(run / "metrics_fixed.yaml")),
                "w/o redistribution": evaluate_metrics(pr.observed.to_numpy(), hard, pd.to_datetime(pr.date).to_numpy()),
                "w/o gate": yaml.safe_load(open(run / "metrics_nogate.yaml")),
            }
            for v, m in variants.items():
                rows.append({"variant": v, "seed": seed, "name": w, "month": month_of(w), **{k: m[k] for k in POINT},
                             "log_lik": ll, "crps": cr})
            old = definitive_run(OLD, seed, w)
            noc = definitive_run(NOCLF, seed, w)
            ll_l, cr_l = lgcp_scores(noc.parent, seed)
            for v, mpath in (("w/o joint training", old), ("w/o classifier", noc)):
                m = yaml.safe_load(open(mpath))
                rows.append({"variant": v, "seed": seed, "name": w, "month": month_of(w), **{k: m[k] for k in POINT},
                             "log_lik": ll_l, "crps": cr_l})
        print(f"seed {seed} done", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "ablation_base.csv", index=False)

elif args.stage == "nokernel":
    rows = []
    for seed in range(1, 11):
        for w in WINDOWS:
            (run,) = glob.glob(str(NOKERNEL / f"seed_{seed}/runs/{w}/*/"))
            run = Path(run)
            m = yaml.safe_load(open(run / "metrics_higher.yaml"))
            ll, cr = hurdle_scores(run, seed)
            rows.append({"variant": "w/o kernel", "seed": seed, "name": w, "month": month_of(w),
                         **{k: m[k] for k in POINT}, "log_lik": ll, "crps": cr})
        print(f"seed {seed} done", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "ablation_nokernel.csv", index=False)

else:
    parts = [pd.read_csv(OUT / "ablation_base.csv")]
    if (OUT / "ablation_nokernel.csv").exists():
        parts.append(pd.read_csv(OUT / "ablation_nokernel.csv"))
    df = pd.concat(parts)
    order = ["full", "w/o uncertainty", "w/o redistribution", "w/o gate", "w/o joint training", "w/o classifier", "w/o kernel"]
    labels = {"full": "Red-LGCP (full)", "w/o uncertainty": "w/o uncertainty", "w/o redistribution": "w/o redistribution",
              "w/o gate": "w/o gate", "w/o joint training": "two-stage", "w/o classifier": "w/o classifier",
              "w/o kernel": "w/o kernel"}
    order = [v for v in order if v in set(df.variant)]
    metrics = [("mae_obs", min), ("rmse_obs", min), ("wasserstein", min), ("accuracy", max), ("f1", max),
               ("log_lik", max), ("crps", min)]
    stats = df.groupby(["variant", "month"])[[m for m, _ in metrics]].agg(["mean", lambda s: s.std(ddof=0)])
    print(df.groupby(["month", "variant"])[[m for m, _ in metrics]].mean().reindex(order, level=1).round(3).to_string())
    lines = []
    for i, v in enumerate(order):
        for j, (month, mlabel) in enumerate((("may", "May"), ("november", "Nov."))):
            cells = []
            for m, pick in metrics:
                means = {u: round(stats.loc[(u, month), (m, "mean")], 3) for u in order}
                best = pick(means.values())
                mu, sd = stats.loc[(v, month), (m, "mean")], stats.loc[(v, month), (m, "<lambda_0>")]
                mean = f"\\textbf{{{r3(mu)}}}" if means[v] == best else r3(mu)
                cells.append(f"{mean}\\,{{\\scriptsize$\\pm${r3(sd)}}}")
            head = f"\\multirow{{2}}{{*}}{{{labels[v]}}}" if j == 0 else ""
            lines.append(f"{head} & {mlabel} & " + " & ".join(cells) + " \\\\")
        if i < len(order) - 1:
            lines.append("\\midrule")
    tex = ("\\begin{table*}[t]\n\\centering\n\\footnotesize\n\\setlength{\\tabcolsep}{2.5pt}\n\\begin{tabular}{llccccccc}\n\\toprule\n"
           "Variant & Month & MAE & RMSE & Wasserstein & Accuracy & F1 & Log-lik. & CRPS \\\\\n\\midrule\n"
           + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n"
           "\\caption{Ablation of Red-LGCP on May and Nov.\\ 2025 (same windows and covariates as Table~\\ref{tab:results-main}; "
           "mean $\\pm$ population std over 10 seeds $\\times$ 4 weekly windows per month). Each variant removes or replaces one component of the full model: "
           "``w/o uncertainty'' uses the fixed threshold $\\tau$ instead of $\\tau+\\sigma_{j,t}$; ``w/o redistribution'' zeroes rejected cells without recovering any of their expected counts; "
           "``w/o gate'' reports the expected counts without zeroing; ``two-stage'' is not jointly optimized: the LGCP is fitted alone, with a Poisson likelihood, and the classifier is trained afterwards and used as a post-hoc gate with the fixed threshold $\\tau$; "
           "``w/o classifier'' is the LGCP alone, with a Poisson likelihood; ``w/o kernel'' removes the latent Gaussian process, keeping the covariate effects and the classifier. "
           "The first four rows share the same trained model, so their log-likelihood and CRPS, which score the hurdle predictive distribution, coincide; "
           "for ``two-stage'' and ``w/o classifier'' these scores refer to the predictive distribution of the Poisson LGCP, since a post-hoc gate only changes point predictions. "
           "Bold marks the best value per column within each month.}\n\\label{tab:results-ablation}\n\\end{table*}\n")
    (ROOT / "reports/statistical_tests/ablation_table.tex").write_text(tex)
    print(tex)
