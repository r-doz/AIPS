"""
Hyperparameter sweep for the LGCP model over (lr, num_mc).

Two-phase search, so the real test window stays an honest held-out
evaluation:

  1. "search" phase — every (lr, num_mc) combo is trained on data strictly
     before a validation window (a short window carved out of the training
     period, distinct from the real test window) for a reduced number of
     steps. Combos are ranked by a validation metric.

  2. "confirm" phase — the top-K combos from the search are retrained on
     the *full* training set (exactly as scripts/11_train_lgcp.py does) for
     the full num_steps, and evaluated on the real test window. The winner
     is whichever confirm run scores best.

num_mc directly multiplies the per-step training cost (SparseLGCP.elbo_mc
loops over MC samples), so combos are not equal-cost — that's expected and
is exactly what the two-phase search is for for: cheap ranking first, full
-length confirmation only for the promising few.

Usage
-----
    python scripts/16_sweep_lgcp_hparams.py --config config/sweep_lgcp_hparams.yaml

Results are saved under:
    <report_root>/<model_family>/hparam_sweep_<run_name>_<timestamp>/
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import itertools
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

# ---- Make src importable when script is run from the project root ----------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.metrics_lgcp import evaluate_metrics

# ---- Load scripts/11_train_lgcp.py as a module ------------------------------
# (its filename starts with a digit, so it can't be `import`-ed normally)
_spec = importlib.util.spec_from_file_location(
    "train_lgcp", PROJECT_ROOT / "scripts" / "11_train_lgcp.py"
)
train_lgcp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(train_lgcp)


# Whether a higher value of each evaluate_metrics() key is better.
HIGHER_IS_BETTER = {
    "mean_ll_obs": True,
    "mean_ll_daily": True,
    "daily_delta_corr": True,
    "daily_direction_accuracy_moving": True,
    "mae_obs": False,
    "rmse_obs": False,
    "mae_daily": False,
    "rmse_daily": False,
}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def load_sweep_config(config_path: str) -> tuple[dict, dict]:
    with open(config_path, "r") as f:
        sweep_yaml = yaml.safe_load(f)

    if "base_config" not in sweep_yaml:
        raise ValueError(
            "Sweep config must set 'base_config' to the LGCP training config "
            "to sweep on top of (e.g. config/train_basic_lgcp_multi_kernel.yaml)."
        )

    base_cfg = train_lgcp.load_config(sweep_yaml["base_config"])
    sweep_opts = sweep_yaml.get("sweep", {})

    for key in ("test_start_date", "test_end_date"):
        if base_cfg.get(key) is None:
            raise ValueError(
                f"base_config must set '{key}' (split_strategy=fixed_test_window) "
                "so a validation window can be carved out before it."
            )

    return base_cfg, sweep_opts


def resolve_validation_window(cfg: dict, sweep_opts: dict) -> tuple[str, str]:
    """
    Pick the validation window used for the cheap search phase.

    Defaults to the `val_days` days immediately before test_start_date, so
    it's close in time/seasonality to the real test window without
    overlapping it. Can be overridden explicitly via val_start_date/
    val_end_date in the sweep config.
    """
    if "val_start_date" in sweep_opts and "val_end_date" in sweep_opts:
        return str(sweep_opts["val_start_date"]), str(sweep_opts["val_end_date"])

    test_start = pd.Timestamp(cfg["test_start_date"])
    val_days = int(sweep_opts.get("val_days", 10))

    val_end = test_start - pd.Timedelta(days=1)
    val_start = val_end - pd.Timedelta(days=val_days - 1)

    return str(val_start.date()), str(val_end.date())


# ---------------------------------------------------------------------------
# Train / evaluate one combo
# ---------------------------------------------------------------------------


def train_and_evaluate(
    cfg: dict,
    lr: float,
    num_mc: int,
    num_steps: int,
    train_coords,
    train_covs,
    train_y,
    eval_coords,
    eval_covs,
    eval_y,
    eval_dates,
    seed: int,
) -> tuple[dict, float]:
    m_inducing = min(cfg["M_inducing"], train_coords.shape[0])

    train_lgcp.set_seeds(seed)
    model = train_lgcp.SparseLGCP(
        train_coords,
        train_covs,
        train_y,
        kernel_config=cfg["kernel_config"],
        M_inducing=m_inducing,
        device=cfg["device"],
        np_seed=seed,
    )
    train_lgcp.configure_log_noise(model, cfg)

    run_cfg = dict(cfg)
    run_cfg["lr"] = lr
    run_cfg["num_mc"] = num_mc
    run_cfg["num_steps"] = num_steps
    run_cfg["log_every"] = max(num_steps, 1)  # keep the sweep log quiet

    t0 = time.time()
    train_lgcp.train(model, run_cfg)
    elapsed = time.time() - t0

    rate_mean, _, _ = model.predict_rate(
        eval_coords, eval_covs, num_samples=cfg.get("num_pred_samples", 200)
    )
    metrics = evaluate_metrics(eval_y, rate_mean, eval_dates)

    return metrics, elapsed


def rank_leaderboard(rows: list[dict], metric: str) -> list[dict]:
    if metric not in HIGHER_IS_BETTER:
        raise ValueError(
            f"Unknown ranking metric '{metric}'. Choose one of: "
            f"{sorted(HIGHER_IS_BETTER)}"
        )
    higher_is_better = HIGHER_IS_BETTER[metric]
    return sorted(rows, key=lambda r: r[metric], reverse=higher_is_better)


def format_metrics_line(metrics: dict, metric: str, elapsed: float) -> str:
    extras = [m for m in ("mean_ll_obs", "rmse_daily") if m != metric]
    parts = [f"{metric}={metrics[metric]:.4f}"]
    parts += [f"{m}={metrics[m]:.4f}" for m in extras]
    return "  " + "   ".join(parts) + f"   ({elapsed:.1f}s)"


def save_leaderboard(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(config_path: str):
    cfg, sweep_opts = load_sweep_config(config_path)

    lr_grid = sweep_opts.get("lr_grid", [1e-4, 3e-4, 1e-3, 3e-3])
    num_mc_grid = sweep_opts.get("num_mc_grid", [8, 16, 32])
    metric = sweep_opts.get("metric", "mean_ll_obs")
    num_steps_search = int(sweep_opts.get("num_steps_search", 1500))
    num_steps_confirm = int(sweep_opts.get("num_steps_confirm", cfg["num_steps"]))
    top_k_confirm = int(sweep_opts.get("top_k_confirm", 2))
    seed = int(sweep_opts.get("sweep_seed", cfg.get("np_seed", 0)))

    val_start_date, val_end_date = resolve_validation_window(cfg, sweep_opts)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = (
        Path(cfg["report_root"])
        / cfg["model_family"]
        / f"hparam_sweep_{cfg['run_name']}_{timestamp}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("LGCP hyperparameter sweep: lr x num_mc")
    print("=" * 80)
    print(f"Base config:        {sweep_opts.get('base_config', config_path)}")
    print(f"lr grid:            {lr_grid}")
    print(f"num_mc grid:        {num_mc_grid}")
    print(f"Ranking metric:     {metric}")
    print(f"Search num_steps:   {num_steps_search}")
    print(f"Confirm num_steps:  {num_steps_confirm}")
    print(f"Top-K to confirm:   {top_k_confirm}")
    print(f"Validation window:  {val_start_date} .. {val_end_date}")
    print(f"Real test window:   {cfg['test_start_date']} .. {cfg['test_end_date']}")
    print(f"Output dir:         {out_dir}")

    with open(out_dir / "sweep_config.yaml", "w") as f:
        yaml.dump(
            {
                "base_config": sweep_opts.get("base_config"),
                "lr_grid": lr_grid,
                "num_mc_grid": num_mc_grid,
                "metric": metric,
                "num_steps_search": num_steps_search,
                "num_steps_confirm": num_steps_confirm,
                "top_k_confirm": top_k_confirm,
                "val_start_date": val_start_date,
                "val_end_date": val_end_date,
                "sweep_seed": seed,
            },
            f,
            sort_keys=False,
        )

    # ---- Phase 1: cheap search on the carved-out validation window ----------
    print("\nLoading data for the search phase (train < validation window) …")
    (
        search_train_coords,
        search_train_covs,
        search_train_y,
        val_coords,
        val_covs,
        val_y,
        _search_scalers,
        search_df,
    ) = train_lgcp.prepare_data(
        cfg["parquet_path"],
        train_fraction=cfg["train_fraction"],
        random_seed=cfg["data_seed"],
        split_strategy="fixed_test_window",
        covariate_cols=cfg.get("covariate_cols"),
        test_start_date=val_start_date,
        test_end_date=val_end_date,
        lag_features=cfg.get("lag_features"),
        years=cfg.get("years"),
    )

    _, val_mask = train_lgcp.make_day_split_masks(
        df=search_df,
        train_fraction=cfg["train_fraction"],
        random_seed=cfg["data_seed"],
        split_strategy="fixed_test_window",
        test_start_date=val_start_date,
        test_end_date=val_end_date,
    )
    val_dates = search_df.loc[val_mask, "date"].values

    print(
        f"  Search-train: {search_train_coords.shape[0]:,} obs   "
        f"Validation: {val_coords.shape[0]:,} obs"
    )

    search_rows = []
    combos = list(itertools.product(lr_grid, num_mc_grid))
    for i, (lr, num_mc) in enumerate(combos, start=1):
        print(f"\n[search {i}/{len(combos)}] lr={lr}  num_mc={num_mc}")
        metrics, elapsed = train_and_evaluate(
            cfg,
            lr=lr,
            num_mc=num_mc,
            num_steps=num_steps_search,
            train_coords=search_train_coords,
            train_covs=search_train_covs,
            train_y=search_train_y,
            eval_coords=val_coords,
            eval_covs=val_covs,
            eval_y=val_y,
            eval_dates=val_dates,
            seed=seed,
        )
        print(format_metrics_line(metrics, metric, elapsed))
        search_rows.append({"lr": lr, "num_mc": num_mc, "elapsed_s": elapsed, **metrics})

    search_rows = rank_leaderboard(search_rows, metric)
    save_leaderboard(search_rows, out_dir / "search_leaderboard.csv")

    print("\n" + "=" * 80)
    print(f"Search phase done. Top {top_k_confirm} by {metric}:")
    for row in search_rows[:top_k_confirm]:
        print(f"  lr={row['lr']}  num_mc={row['num_mc']}  {metric}={row[metric]:.4f}")

    # ---- Phase 2: confirm top-K on the real train/test split ----------------
    print("\nLoading data for the confirm phase (real train/test split) …")
    (
        full_train_coords,
        full_train_covs,
        full_train_y,
        test_coords,
        test_covs,
        test_y,
        _full_scalers,
        full_df,
    ) = train_lgcp.prepare_data(
        cfg["parquet_path"],
        train_fraction=cfg["train_fraction"],
        random_seed=cfg["data_seed"],
        split_strategy=cfg.get("split_strategy", "fixed_test_window"),
        covariate_cols=cfg.get("covariate_cols"),
        test_start_date=cfg.get("test_start_date"),
        test_end_date=cfg.get("test_end_date"),
        lag_features=cfg.get("lag_features"),
        years=cfg.get("years"),
    )

    _, test_mask = train_lgcp.make_day_split_masks(
        df=full_df,
        train_fraction=cfg["train_fraction"],
        random_seed=cfg["data_seed"],
        split_strategy=cfg.get("split_strategy", "fixed_test_window"),
        test_start_date=cfg.get("test_start_date"),
        test_end_date=cfg.get("test_end_date"),
    )
    test_dates = full_df.loc[test_mask, "date"].values

    print(
        f"  Full train: {full_train_coords.shape[0]:,} obs   "
        f"Test: {test_coords.shape[0]:,} obs"
    )

    confirm_rows = []
    for i, row in enumerate(search_rows[:top_k_confirm], start=1):
        lr, num_mc = row["lr"], row["num_mc"]
        print(f"\n[confirm {i}/{top_k_confirm}] lr={lr}  num_mc={num_mc}")
        metrics, elapsed = train_and_evaluate(
            cfg,
            lr=lr,
            num_mc=num_mc,
            num_steps=num_steps_confirm,
            train_coords=full_train_coords,
            train_covs=full_train_covs,
            train_y=full_train_y,
            eval_coords=test_coords,
            eval_covs=test_covs,
            eval_y=test_y,
            eval_dates=test_dates,
            seed=seed,
        )
        print(format_metrics_line(metrics, metric, elapsed))
        confirm_rows.append({"lr": lr, "num_mc": num_mc, "elapsed_s": elapsed, **metrics})

    confirm_rows = rank_leaderboard(confirm_rows, metric)
    save_leaderboard(confirm_rows, out_dir / "confirm_leaderboard.csv")

    best = confirm_rows[0]
    with open(out_dir / "best_hparams.yaml", "w") as f:
        yaml.dump(
            {
                "lr": best["lr"],
                "num_mc": best["num_mc"],
                "metric": metric,
                "test_metrics": {k: v for k, v in best.items() if k not in ("lr", "num_mc")},
            },
            f,
            sort_keys=False,
        )

    print("\n" + "=" * 80)
    print(f"Winner (confirmed on the real test window, ranked by {metric}):")
    print(f"  lr={best['lr']}  num_mc={best['num_mc']}  {metric}={best[metric]:.4f}")
    print(f"\nLeaderboards + best_hparams.yaml saved → {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config/sweep_lgcp_hparams.yaml",
        help="Path to sweep YAML config.",
    )
    args = parser.parse_args()
    main(args.config)
