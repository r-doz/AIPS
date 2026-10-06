"""Test log-likelihood and CRPS of every method's predictive distribution.

Baselines are Poisson models: their predictive distribution is Poisson(rate), with the
rate clipped at 1e-8 as in src/models/metrics_lgcp.py. The final model (joint hurdle
Red-LGCP) uses its hurdle predictive distribution, P(y=0) = 1 - E[p] and
P(y=k) = E[p] E_q[Poisson(k | lambda) / (1 - exp(-lambda))] for k >= 1, estimated with
300 latent-field samples and 100 dropout samples per run. CRPS is the discrete
(ranked probability) form for counts: sum_k (F(k) - 1{y <= k})^2.

Every reconstructed forecast is checked against the MAE reported by the original run.
"""

import glob
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.special import gammaln
from scipy.stats import poisson, wilcoxon
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.models.data_pp_lgcp import prepare_data, make_day_split_masks  # noqa: E402
from src.models.hurdle_lgcp import HurdleSparseLGCP  # noqa: E402

OUT = ROOT / "reports/probabilistic_scores"
WINDOWS = [(m, d) for m in ("may", "november") for d in (1, 8, 15, 22)]
MONTH_NUM = {"may": 5, "november": 11}
EPS = 1e-8


def load_module(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def window_info(month, day):
    start = pd.Timestamp(2025, MONTH_NUM[month], day)
    return f"2025_{month}_{day:02d}_{day + 6:02d}", start, start + pd.Timedelta(days=6)


def cell_key(lon, lat):
    return np.round(np.asarray(lon, float) * 48).astype(int), np.round(np.asarray(lat, float) * 48).astype(int)


def poisson_scores(y, lam):
    y = np.asarray(y, float)
    lam = np.clip(np.asarray(lam, float), EPS, None)
    ll = y * np.log(lam) - lam - gammaln(y + 1)
    K = int(max(y.max(), (lam + 10 * np.sqrt(lam)).max()) + 30)
    cdf = poisson.cdf(np.arange(K + 1)[None, :], lam[:, None])
    ind = (np.arange(K + 1)[None, :] >= y[:, None]).astype(float)
    return ll, ((cdf - ind) ** 2).sum(1)


def pmf_scores(y, pmf):
    y = np.asarray(y, int)
    ll = np.log(np.clip(pmf[np.arange(len(y)), y], 1e-300, None))
    cdf = np.cumsum(pmf, axis=1)
    ind = (np.arange(pmf.shape[1])[None, :] >= y[:, None]).astype(float)
    return ll, ((cdf - ind) ** 2).sum(1)


# ---------------------------------------------------------------- data and keys
df = pd.concat([pd.read_parquet(ROOT / f"data/processed/cpr_gfw_salinity_{y}.parquet") for y in (2024, 2025)])
df["date"] = pd.to_datetime(df["date"]).dt.normalize()
df["ci"], df["cj"] = cell_key(df.longitude, df.latitude)
df = df.sort_values(["ci", "cj", "date"]).reset_index(drop=True)
y_all = df.ais_vessels_count.astype(float)
df["la_rate"] = df.groupby(["ci", "cj"]).ais_vessels_count.shift(1)
df["gcm_rate"] = (df.groupby(["ci", "cj"]).ais_vessels_count.cumsum() - y_all) / df.groupby(["ci", "cj"]).cumcount()
truth = df.set_index(["date", "ci", "cj"]).ais_vessels_count.astype(int)

rows = []


def add_row(method, seed, window, month, y, ll, crps, mae, mae_ref):
    if mae_ref is not None:
        assert abs(mae - mae_ref) < 2e-4, (method, seed, window, mae, mae_ref)
    rows.append({"method": method, "seed": seed, "name": window, "month": month,
                 "log_lik": float(np.mean(ll)), "crps": float(np.mean(crps)), "mae_check": float(mae)})


def ref_mae(path, window, seed=None):
    t = pd.read_csv(path)
    if seed is not None and "seed" in t:
        t = t[t.seed == seed]
    return float(t.set_index("name").loc[window, "mae_obs"])


# ---------------------------------------------------------------- Last Available, Global Cell Mean
for month, day in WINDOWS:
    name, start, end = window_info(month, day)
    q = df[(df.date >= start) & (df.date <= end)]
    for method, col, ref in [
        ("Last Available Poisson", "la_rate", "reports/seeded_monthly_windows/13_evaluate_last_available_poisson/per_window_per_seed_metrics.csv"),
        ("Global Cell Mean Poisson", "gcm_rate", "reports/seeded_monthly_windows/12b_global_cell_mean/per_window_per_seed_metrics.csv"),
    ]:
        lam = q[col].to_numpy(float)
        ll, cr = poisson_scores(q.ais_vessels_count, lam)
        add_row(method, 1, name, month, q.ais_vessels_count, ll, cr,
                np.abs(q.ais_vessels_count - lam).mean(), ref_mae(ROOT / ref, name))
print("Last Available and Global Cell Mean done", flush=True)

# ---------------------------------------------------------------- Window Poisson GLM (deterministic refit)
glm = load_module("glm", "scripts/14_train_window_poisson_glm.py")
gcfg = glm.load_config(str(ROOT / "config/14_window_poisson_glm_salinity.yaml"))
gcfg["parquet_path"] = str(ROOT / gcfg["parquet_path"])
graw = glm.load_parquet_years(gcfg["parquet_path"], gcfg.get("years"))
graw["date"] = pd.to_datetime(graw["date"])
gwin, gfeat = glm.build_window_dataframe(graw, gcfg)
for month, day in WINDOWS:
    name, start, end = window_info(month, day)
    tr, te = glm.make_split_masks(dates=gwin["date"], split_strategy="fixed_test_window",
                                  train_fraction=gcfg["train_fraction"], random_seed=gcfg["data_seed"],
                                  test_start_date=str(start.date()), test_end_date=str(end.date()))
    X = gwin[gfeat].values.astype(np.float32)
    yv = gwin[gcfg["target_col"]].values.astype(np.float32)
    sc = StandardScaler().fit(X[tr])
    model = PoissonRegressor(alpha=float(gcfg["poisson_alpha"]), max_iter=int(gcfg["max_iter"])).fit(sc.transform(X[tr]), yv[tr])
    lam = np.clip(model.predict(sc.transform(X[te])), 0, None)
    ll, cr = poisson_scores(yv[te], lam)
    add_row("Window Poisson GLM", 1, name, month, yv[te], ll, cr, np.abs(yv[te] - lam).mean(),
            ref_mae(ROOT / "reports/seeded_monthly_windows_salinity/14_train_window_poisson_glm/per_window_per_seed_metrics.csv", name))
print("GLM done", flush=True)

# ---------------------------------------------------------------- GNN (saved per-cell predictions)
gnn_root = ROOT / "reports/seeded_monthly_windows_salinity_gnn_cellpoisson/15_train_gnn"
for seed in range(1, 11):
    for month, day in WINDOWS:
        name, _, _ = window_info(month, day)
        (run,) = glob.glob(str(gnn_root / f"seed_{seed}/runs/{name}/*/"))
        sp = pd.read_csv(Path(run) / "spatial_predictions.csv")
        y, lam = sp.observed.to_numpy(float), sp.predicted.to_numpy(float)
        ll, cr = poisson_scores(y, lam)
        add_row("GNN", seed, name, month, y, ll, cr, np.abs(y - lam).mean(),
                ref_mae(gnn_root / "per_window_per_seed_metrics.csv", name, seed))
print("GNN done", flush=True)

# ---------------------------------------------------------------- ConvLSTM (reload saved weights)
conv = load_module("conv", "scripts/18_train_convlstm.py")
conv_root = ROOT / "reports/seeded_monthly_windows_salinity/18_train_convlstm"
torch.set_num_threads(4)
cache = {}
for seed in range(1, 11):
    mm = yaml.safe_load(open(conv_root / f"seed_{seed}/monthly_metrics.yaml"))
    paths = {w["name"]: ROOT / w["metrics_path"].split("projects/AIPS/")[-1] for w in mm["per_window"]}
    for month, day in WINDOWS:
        name, _, _ = window_info(month, day)
        run = paths[name].parent
        cfg = yaml.safe_load(open(run / "params.yaml"))
        key = (cfg["parquet_path"], tuple(cfg["covariate_cols"]), int(cfg["seq_length"]))
        if key not in cache:
            d0 = conv.load_parquet_years(str(ROOT / cfg["parquet_path"]), cfg.get("years"))
            d0, covs = conv.prepare_dataframe(d0, cfg)
            cov_grid, y_grid, dates, mask, cell_coords, _ = conv.build_grid_tensors(d0, covs, cfg["target_col"])
            cache[key] = (conv.build_sequence_examples(cov_grid, y_grid, dates, int(cfg["seq_length"])), mask, cell_coords)
        (X_seq, X_static, Y, tdates), mask, cell_coords = cache[key]
        tr, te = make_day_split_masks(df=pd.DataFrame({"date": tdates}), train_fraction=cfg.get("train_fraction", 0.9),
                                      random_seed=cfg.get("data_seed", 42), split_strategy="fixed_test_window",
                                      test_start_date=cfg["test_start_date"], test_end_date=cfg["test_end_date"])
        sm, ss = conv.fit_channel_scaler(X_seq[tr], 1)
        tm, ts = conv.fit_channel_scaler(X_static[tr], 1)
        model = conv.ConvLSTMBaseline(seq_channels=X_seq.shape[2], static_channels=X_static.shape[1],
                                      hidden_channels=int(cfg["hidden_channels"]), num_layers=int(cfg["num_layers"]),
                                      kernel_size=int(cfg["kernel_size"]), dropout=float(cfg["dropout"]))
        state = torch.load(run / "model_state_dict.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(state.get("model_state_dict", state) if isinstance(state, dict) and "model_state_dict" in state else state)
        model.eval()
        with torch.no_grad():
            rate = model(torch.tensor((X_seq[te] - sm) / ss, dtype=torch.float32),
                         torch.tensor((X_static[te] - tm) / ts, dtype=torch.float32),
                         torch.tensor(mask, dtype=torch.float32)).numpy()
        rate = np.clip(rate, 0, None)
        ri, cj, _ = conv.build_cell_grid_index(cell_coords[["longitude", "latitude"]].to_numpy())
        lam = rate[:, ri, cj].reshape(-1)
        y = Y[te][:, ri, cj].reshape(-1)
        ll, cr = poisson_scores(y, lam)
        add_row("ConvLSTM", seed, name, month, y, ll, cr, np.abs(y - lam).mean(),
                yaml.safe_load(open(paths[name]))["mae_obs"])
print("ConvLSTM done", flush=True)

# ---------------------------------------------------------------- Final model: joint hurdle Red-LGCP
hroot = ROOT / "reports/seeded_monthly_windows_hurdle"
torch.set_default_dtype(torch.float32)
gap = []
for seed in range(1, 11):
    for month, day in WINDOWS:
        name, _, _ = window_info(month, day)
        (run,) = glob.glob(str(hroot / f"seed_{seed}/runs/{name}/*/"))
        run = Path(run)
        ck = torch.load(run / "model.pt", map_location="cpu", weights_only=False)
        cfg, h = ck["config"], ck["config"].get("hurdle", {})
        trc, trv, try_, tec, tev, tey, _, _ = prepare_data(
            str(ROOT / cfg["parquet_path"]), train_fraction=cfg["train_fraction"], random_seed=cfg["data_seed"],
            split_strategy=cfg["split_strategy"], covariate_cols=cfg["covariate_cols"],
            test_start_date=cfg["test_start_date"], test_end_date=cfg["test_end_date"],
            lag_features=cfg["lag_features"], years=cfg["years"])
        model = HurdleSparseLGCP(trc, trv, try_, kernel_config=cfg["kernel_config"], M_inducing=cfg["M_inducing"],
                                 device="cpu", np_seed=cfg["np_seed"], clf_hidden=int(h.get("clf_hidden", 32)),
                                 clf_dropout=float(h.get("clf_dropout", 0.1)), clf_alpha=float(h.get("clf_alpha", 1e-4)))
        model.load_state_dict(ck["model_state"])
        torch.manual_seed(1000 + seed)
        with torch.no_grad():
            from src.models.lgcp import build_Kfu, build_Kuu
            Xn = torch.tensor(tec, dtype=torch.float32)
            Cn = torch.tensor(tev, dtype=torch.float32)
            kern = model.kernel_params()
            eye = torch.eye(model.M)
            chol = torch.linalg.cholesky(build_Kuu(model.Z, kern, 1e-6) + 1e-8 * eye)
            A = torch.cholesky_solve(build_Kfu(Xn, model.Z, kern).T, chol).T
            Lq = torch.linalg.cholesky(model.get_S() + 1e-8 * eye)
            U = model.m.unsqueeze(0) + torch.randn(300, model.M) @ Lq.T
            lam = torch.exp(model.alpha + (Cn @ model.beta).unsqueeze(0) + U @ A.T).numpy().astype(float)
            model.classifier.train()
            feats = torch.cat([Xn, Cn], dim=1)
            p_bar = torch.stack([torch.sigmoid(model.classifier(feats).squeeze(-1)) for _ in range(100)]).mean(0).numpy().astype(float)
        K = int(min(400, max(tey.max() + 10, np.quantile(lam, 0.999) * 3 + 30)))
        k = np.arange(1, K + 1)
        log1mexp = np.where(lam < 1e-6, np.log(np.clip(lam, 1e-300, None)), np.log(-np.expm1(-lam)))
        log_tp = (k[None, None, :] * np.log(np.clip(lam, 1e-300, None))[:, :, None] - lam[:, :, None]
                  - gammaln(k + 1)[None, None, :] - log1mexp[:, :, None])
        tp = np.exp(log_tp).mean(0)
        pmf = np.concatenate([(1 - p_bar)[:, None], p_bar[:, None] * tp], axis=1)
        assert pmf.sum(1).min() > 0.9999, (seed, name, K, pmf.sum(1).min())
        y = tey.astype(float)
        ll, cr = pmf_scores(y, pmf)
        expected = (pmf * np.arange(K + 1)[None, :]).sum(1)
        saved = pd.read_csv(run / "hurdle_predictions.csv").expected_count.to_numpy()
        gap.append(np.abs(expected - saved).mean() / saved.mean())
        add_row("Red-LGCP (joint, final)", seed, name, month, y, ll, cr, np.abs(y - expected).mean(), None)
        gated = pd.read_csv(run / "hurdle_predictions.csv").pred_higher.to_numpy()
        ll_g, cr_g = poisson_scores(y, gated)
        add_row("Red-LGCP (joint, gated point forecast as Poisson)", seed, name, month, y, ll_g, cr_g,
                np.abs(y - gated).mean(), yaml.safe_load(open(run / "metrics_higher.yaml"))["mae_obs"])
print(f"Joint hurdle done; MC gap of expected counts vs saved run: mean relative {np.mean(gap):.3%}", flush=True)

# ---------------------------------------------------------------- summaries and paired tests
res = pd.DataFrame(rows)
res.to_csv(OUT / "per_window_per_seed_scores.csv", index=False)
pd.set_option("display.width", 220)
order = ["Last Available Poisson", "Global Cell Mean Poisson", "Window Poisson GLM", "GNN", "ConvLSTM",
         "Red-LGCP (joint, final)", "Red-LGCP (joint, gated point forecast as Poisson)"]
summ = res.groupby(["method", "month"]).agg(ll_mean=("log_lik", "mean"), ll_std=("log_lik", lambda s: s.std(ddof=0)),
                                            crps_mean=("crps", "mean"), crps_std=("crps", lambda s: s.std(ddof=0)),
                                            n=("crps", "size")).reindex(order, level=0)
summ.to_csv(OUT / "monthly_scores.csv")
print(summ.round(4).to_string())

final = res[res.method == "Red-LGCP (joint, final)"].groupby("name")[["log_lik", "crps"]].mean()
tests = []
for base in order[:5]:
    b = res[res.method == base].groupby("name")[["log_lik", "crps"]].mean().loc[final.index]
    for metric, sign in (("log_lik", 1), ("crps", -1)):
        d = sign * (final[metric] - b[metric])
        tests.append({"metric": metric, "baseline": base, "final": final[metric].mean(), "baseline_mean": b[metric].mean(),
                      "weeks_won": int((d > 0).sum()), "p": wilcoxon(d.to_numpy(), method="exact").pvalue})
tests = pd.DataFrame(tests)
for metric in ("log_lik", "crps"):
    sel = tests.metric == metric
    order_p = tests[sel].p.sort_values()
    adj, run_max = {}, 0.0
    for rank, (i, p) in enumerate(order_p.items()):
        run_max = max(run_max, (len(order_p) - rank) * p)
        adj[i] = min(1.0, run_max)
    tests.loc[sel, "p_holm"] = pd.Series(adj)
tests.to_csv(OUT / "paired_tests.csv", index=False)
print(tests.round(4).to_string(index=False))
