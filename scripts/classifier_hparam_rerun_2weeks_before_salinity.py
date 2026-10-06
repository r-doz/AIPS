"""Re-run the zero-gate MLP classifier's sequential tuning chain (hidden
layer sizes -> activation -> alpha -> max_iter -> threshold -> confidence-
redistribution gamma/scale) on validation weeks redefined as the two 7-day
blocks immediately preceding each reporting week, instead of the original
May 24-30 / Nov 24-30 (which are *after* the reporting weeks, not before).

Unlike the original manual chain (which retrained the full 5000-step LGCP
for every single classifier setting), this loads a cached pre-gate LGCP
prediction per validation week (see scripts/11_train_lgcp.py's
cache_pregate_path option, and the configs under
reports/paper/hparam_revalidation_2weeks_before_salinity/configs/pregate_*.yaml)
and only retrains the cheap MLPClassifier for each grid point, which is
mathematically equivalent but orders of magnitude faster.

Usage:
    python scripts/classifier_hparam_rerun_2weeks_before_salinity.py --lr 0.001
    (--lr selects which pregate cache, i.e. which winning LGCP learning
    rate, to use; defaults to reading selected_hyperparameters.yaml from
    the LGCP lr sweep if present.)
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.metrics_lgcp import evaluate_metrics
from src.models.zero_gate import train_zero_gate, apply_zero_gate

CACHE_DIR = PROJECT_ROOT / "reports/paper/hparam_revalidation_2weeks_before_salinity/lgcp_cache"
RESULTS_DIR = PROJECT_ROOT / "reports/paper/hparam_revalidation_2weeks_before_salinity/classifier"

WEEK_LABELS = ["apr_21_27", "apr_28_may04", "oct_20_26", "oct_27_nov02"]

METRIC_DIRECTIONS = {"mean_ll_obs": "max", "rmse_daily": "min"}


def lr_tag(lr: float) -> str:
    return f"{lr:.0e}".replace("e-0", "e-")


def load_week_caches(lr: float) -> dict[str, dict]:
    caches = {}
    for label in WEEK_LABELS:
        path = CACHE_DIR / f"{label}_lr{lr_tag(lr)}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"Missing pre-gate cache: {path}")
        data = np.load(path, allow_pickle=True)
        caches[label] = {k: data[k] for k in data.files}
    return caches


def relative_score(rows: list[dict], group_cols: list[str]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    # hidden_layer_sizes may be a list; make it hashable/groupable.
    df = df.copy()
    for col in group_cols:
        if df[col].apply(lambda v: isinstance(v, list)).any():
            df[col] = df[col].apply(tuple)

    valid = df.dropna(subset=list(METRIC_DIRECTIONS)).copy()
    for metric, direction in METRIC_DIRECTIONS.items():
        def _normalize(s, direction=direction):
            lo, hi = s.min(), s.max()
            if hi - lo < 1e-12:
                return pd.Series(1.0, index=s.index)
            return (s - lo) / (hi - lo) if direction == "max" else (hi - s) / (hi - lo)
        valid[f"relative_{metric}"] = valid.groupby("week")[metric].transform(_normalize)
    relative_cols = [f"relative_{m}" for m in METRIC_DIRECTIONS]
    scores = valid.groupby(group_cols)[relative_cols].mean()
    scores["combined_relative_score"] = scores[relative_cols].mean(axis=1)
    return scores.reset_index().sort_values("combined_relative_score", ascending=False)


def evaluate_setting(caches: dict, zero_gate_cfg: dict) -> list[dict]:
    rows = []
    for label, cache in caches.items():
        clf = train_zero_gate(
            cache["train_coords"], cache["train_covs"], cache["train_y"],
            zero_gate_cfg, seed=int(cache["np_seed"]),
        )
        gated = apply_zero_gate(
            clf, cache["test_coords"], cache["test_covs"], cache["ungated_rate_mean_test"],
            threshold=float(zero_gate_cfg.get("threshold", 0.5)),
            mode=zero_gate_cfg.get("mode", "hard"),
            dates=cache["test_dates"],
            grid_coords=cache["local_grid_coords"],
            redistribution_gamma=float(zero_gate_cfg.get("redistribution_gamma", 1.0)),
            redistribution_scale=float(zero_gate_cfg.get("redistribution_scale", 1.0)),
        )
        metrics = evaluate_metrics(cache["test_y"], gated, cache["test_dates"])
        rows.append({"week": label, **{k: v for k, v in zero_gate_cfg.items() if k != "enabled"}, **metrics})
    return rows


def sweep_stage(caches, base_cfg, param_name, grid, all_rows):
    results = []
    for value in grid:
        cfg = dict(base_cfg)
        cfg[param_name] = value
        rows = evaluate_setting(caches, cfg)
        all_rows.extend(rows)
        results.extend(rows)
    scored = relative_score(results, group_cols=[param_name])
    best_row = scored.iloc[0]
    best_value = best_row[param_name]
    if isinstance(best_value, tuple):
        best_value = list(best_value)
    print(f"  stage {param_name}: grid={grid} -> best={best_value} (score={best_row['combined_relative_score']:.4f})")
    return best_value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=None, help="LGCP learning rate whose pre-gate cache to use.")
    parser.add_argument(
        "--fix-activation", type=str, default=None,
        help="Fix activation to this value and skip the activation sweep stage entirely.",
    )
    args = parser.parse_args()

    lr = args.lr
    if lr is None:
        lr_selection_path = PROJECT_ROOT / "reports/paper/hparam_revalidation_2weeks_before_salinity/lgcp_lr/selected_hyperparameters.yaml"
        if not lr_selection_path.is_file():
            parser.error("--lr not given and no LGCP lr selection found; run the LGCP lr sweep first.")
        lr = yaml.safe_load(lr_selection_path.read_text())["selected"]["lr"]

    print(f"Using pre-gate caches for lr={lr}")
    caches = load_week_caches(lr)

    results_dir = RESULTS_DIR
    if args.fix_activation:
        results_dir = RESULTS_DIR.parent / f"classifier_fixed_{args.fix_activation}"
    results_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []

    base_cfg = {
        "mode": "hard",
        "hidden_layer_sizes": [16],
        "activation": "relu",
        "alpha": 1e-4,
        "max_iter": 500,
        "threshold": 0.3,
    }

    print("Stage 1: hidden_layer_sizes")
    best_layer = sweep_stage(caches, base_cfg, "hidden_layer_sizes", [[16], [32], [64], [32, 16]], all_rows)
    base_cfg["hidden_layer_sizes"] = list(best_layer) if not isinstance(best_layer, list) else best_layer

    if args.fix_activation:
        base_cfg["activation"] = args.fix_activation
        print(f"Stage 2: activation -- fixed to {args.fix_activation!r}, skipping sweep")
    else:
        print("Stage 2: activation")
        base_cfg["activation"] = sweep_stage(caches, base_cfg, "activation", ["relu", "tanh"], all_rows)

    print("Stage 3: alpha")
    base_cfg["alpha"] = sweep_stage(caches, base_cfg, "alpha", [1e-5, 1e-4, 1e-3, 1e-2], all_rows)

    print("Stage 4: max_iter")
    base_cfg["max_iter"] = int(sweep_stage(caches, base_cfg, "max_iter", [500, 1000], all_rows))

    print("Stage 5: threshold")
    base_cfg["threshold"] = sweep_stage(caches, base_cfg, "threshold", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6], all_rows)

    print("Stage 6: confidence_redistribute gamma/scale")
    base_cfg["mode"] = "confidence_redistribute"
    stage6_rows = []
    for gamma in [1.0, 2.0]:
        for scale in [0.5, 1.0]:
            cfg = dict(base_cfg)
            cfg["redistribution_gamma"] = gamma
            cfg["redistribution_scale"] = scale
            rows = evaluate_setting(caches, cfg)
            stage6_rows.extend(rows)
            all_rows.extend(rows)
    scored6 = relative_score(stage6_rows, group_cols=["redistribution_gamma", "redistribution_scale"])
    best_row6 = scored6.iloc[0]
    base_cfg["redistribution_gamma"] = float(best_row6["redistribution_gamma"])
    base_cfg["redistribution_scale"] = float(best_row6["redistribution_scale"])
    print(f"  stage gamma/scale: best gamma={base_cfg['redistribution_gamma']} scale={base_cfg['redistribution_scale']}")

    fieldnames = sorted({k for row in all_rows for k in row})
    with open(results_dir / "classifier_grid_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    def _to_native(value):
        if isinstance(value, (np.generic,)):
            return value.item()
        if isinstance(value, (list, tuple)):
            return [_to_native(v) for v in value]
        return value

    base_cfg = {k: _to_native(v) for k, v in base_cfg.items()}

    with open(results_dir / "selected_hyperparameters.yaml", "w") as f:
        yaml.safe_dump({"lgcp_lr": _to_native(lr), "selected": base_cfg}, f, sort_keys=False)

    print(f"\nFinal selected zero-gate config (lgcp_lr={lr}):")
    print(yaml.safe_dump(base_cfg, sort_keys=False))
    print(f"Saved -> {results_dir / 'selected_hyperparameters.yaml'}")


if __name__ == "__main__":
    main()
