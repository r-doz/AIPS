"""
Train and evaluate the joint hurdle variant of Red-LGCP.

The LGCP and the activity classifier are trained together in one ELBO with a
hurdle likelihood (src/models/hurdle_lgcp.py). At prediction time the expected
count is P(active) * E[count | active], and four gating rules are evaluated from
the same trained model:

    nogate : no gate (expected counts as they are)
    fixed  : confidence-weighted redistribution with the fixed threshold tau
    lower  : threshold tau - z * sigma (zero a cell only if confidently inactive)
    higher : threshold tau + z * sigma (zero uncertain cells more readily)

where sigma is the Monte Carlo dropout std of P(active). Each rule's metrics go to
metrics_<rule>.yaml; metrics.yaml holds the rule named by hurdle.primary_variant.

Usage
-----
  python scripts/11c_train_hurdle_lgcp.py --config config/exp4_salinity_hurdle.yaml
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.multi_year import years_label
from src.models.data_pp_lgcp import prepare_data, make_day_split_masks
from src.models.hurdle_lgcp import HurdleSparseLGCP, apply_modulated_gate
from src.models.metrics_lgcp import evaluate_metrics

_spec = importlib.util.spec_from_file_location("lgcp_base", PROJECT_ROOT / "scripts" / "11_train_lgcp.py")
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)

VARIANTS = ("nogate", "fixed", "lower", "higher")


def main(cfg: dict):
    gate_cfg = cfg.get("zero_gate") or {}
    hcfg = cfg.get("hurdle") or {}
    tau = float(gate_cfg.get("threshold", 0.3))
    gamma = float(gate_cfg.get("redistribution_gamma", 1.0))
    scale = float(gate_cfg.get("redistribution_scale", 1.0))
    z = float(hcfg.get("uncertainty_z", 1.0))
    primary = hcfg.get("primary_variant", "lower")
    if primary not in VARIANTS:
        raise ValueError(f"hurdle.primary_variant must be one of {VARIANTS}")

    base.set_seeds(cfg.get("np_seed", 0))
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.set_default_dtype(torch.float32)

    out_dir = (
        Path(cfg["report_root"]) / cfg["model_family"]
        / f"{years_label(cfg.get('years'))}_{cfg['run_name']}_{time.strftime('%Y%m%d-%H%M%S')}"
    )
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data …")
    (train_coords, train_covs, train_y, test_coords, test_covs, test_y, scalers, df) = prepare_data(
        cfg["parquet_path"],
        train_fraction=cfg["train_fraction"],
        random_seed=cfg["data_seed"],
        split_strategy=cfg.get("split_strategy", "random_day"),
        covariate_cols=cfg.get("covariate_cols"),
        test_start_date=cfg.get("test_start_date"),
        test_end_date=cfg.get("test_end_date"),
        lag_features=cfg.get("lag_features"),
        years=cfg.get("years"),
    )
    span_days = (df["date"].max() - df["date"].min()).days
    cfg["kernel_config"] = base.resolve_period_days(cfg["kernel_config"], scalers, span_days)
    with open(out_dir / "params.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    periods = [b["hyperparameters"]["period"] for b in cfg["kernel_config"]["temporal"] if "period" in b["hyperparameters"]]
    with open(out_dir / "time_diagnostics.yaml", "w") as f:
        yaml.dump(base.compute_time_diagnostics(scalers, span_days, periods=periods), f, sort_keys=False)

    _, test_mask = make_day_split_masks(
        df=df, train_fraction=cfg["train_fraction"], random_seed=cfg["data_seed"],
        split_strategy=cfg.get("split_strategy", "random_day"),
        test_start_date=cfg.get("test_start_date"), test_end_date=cfg.get("test_end_date"),
    )
    test_dates = df.loc[test_mask, "date"].values
    print(f"  Train: {train_coords.shape[0]:,} obs   Test: {test_coords.shape[0]:,} obs")

    print(f"Building HurdleSparseLGCP with M={cfg['M_inducing']} inducing points …")
    model = HurdleSparseLGCP(
        train_coords, train_covs, train_y,
        kernel_config=cfg["kernel_config"],
        M_inducing=cfg["M_inducing"],
        device=cfg["device"],
        np_seed=cfg["np_seed"],
        clf_hidden=int(hcfg.get("clf_hidden", 32)),
        clf_dropout=float(hcfg.get("clf_dropout", 0.1)),
        clf_alpha=float(hcfg.get("clf_alpha", 1e-4)),
    )
    base.configure_log_noise(model, cfg)
    base.save_kernel_parameters(model, out_dir / "kernel_params_before.yaml")

    print(f"Training for {cfg['num_steps']:,} steps …")
    losses = base.train(model, cfg)
    base.save_kernel_parameters(model, out_dir / "kernel_params_after.yaml")
    torch.save({"model_state": model.state_dict(), "config": cfg}, out_dir / "model.pt")
    base.plot_loss(losses, save_path=plots_dir / "loss.png")

    print("Evaluating on test set …")
    p_mean, p_std, cond_mean, lam_mean = model.predict_hurdle(
        test_coords, test_covs,
        num_samples=int(cfg["num_pred_samples"]),
        num_dropout=int(hcfg.get("num_dropout_samples", 100)),
    )
    mu = p_mean * cond_mean
    thresholds = {
        "fixed": np.full_like(p_mean, tau),
        "lower": np.clip(tau - z * p_std, 1e-6, 1.0),
        "higher": np.clip(tau + z * p_std, 1e-6, 1.0),
    }
    predictions = {"nogate": mu}
    for name, tau_vec in thresholds.items():
        predictions[name] = apply_modulated_gate(mu, p_mean, tau_vec, test_dates, gamma=gamma, scale=scale)

    for name, pred in predictions.items():
        metrics = evaluate_metrics(test_y, pred, test_dates)
        with open(out_dir / f"metrics_{name}.yaml", "w") as f:
            yaml.dump(metrics, f)
        print(f"  [{name:6s}] MAE={metrics['mae_obs']:.4f} RMSE={metrics['rmse_obs']:.4f} "
              f"W={metrics['wasserstein']:.4f} acc={metrics['accuracy']:.4f} F1={metrics['f1']:.4f} "
              f"zeros={int((pred == 0).sum())}/{len(pred)}")
        if name == primary:
            with open(out_dir / "metrics.yaml", "w") as f:
                yaml.dump(metrics, f)

    pd.DataFrame({
        "date": test_dates, "x": test_coords[:, 0], "y": test_coords[:, 1], "observed": test_y,
        "p_active_mean": p_mean, "p_active_std": p_std, "cond_mean": cond_mean, "lambda_mean": lam_mean,
        "expected_count": mu, **{f"pred_{k}": v for k, v in predictions.items() if k != "nogate"},
        **{f"tau_{k}": v for k, v in thresholds.items()},
    }).to_csv(out_dir / "hurdle_predictions.csv", index=False)
    print(f"Done → {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(base.load_config(args.config))
