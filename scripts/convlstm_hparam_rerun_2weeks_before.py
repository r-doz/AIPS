"""Re-run the ConvLSTM (seq_length x lr) hyperparameter sweep on validation
weeks redefined as the two 7-day blocks immediately preceding each
reporting week (2025-05-05..11 and 2025-11-03..09), instead of the
original April/October blocks used in reports/paper/convlstm_tuning/
(whose driver script no longer exists in the working tree).

Grid matches the original sweep: seq_length in {3, 7, 14}, lr in
{0.001, 0.01}, scored by mean mean_ll_obs across the 4 validation weeks
(same convention as the original convlstm_search_results.csv).

Usage:
    python scripts/convlstm_hparam_rerun_2weeks_before.py
"""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_SCRIPT = PROJECT_ROOT / "scripts/18_train_convlstm.py"
BASE_CONFIG = PROJECT_ROOT / "config/18_convlstm.yaml"
RESULTS_DIR = PROJECT_ROOT / "reports/paper/hparam_revalidation_2weeks_before/convlstm"

TEST_WEEKS = [
    {"label": "before_may_w1", "test_start": "2025-04-21", "test_end": "2025-04-27"},
    {"label": "before_may_w2", "test_start": "2025-04-28", "test_end": "2025-05-04"},
    {"label": "before_nov_w1", "test_start": "2025-10-20", "test_end": "2025-10-26"},
    {"label": "before_nov_w2", "test_start": "2025-10-27", "test_end": "2025-11-02"},
]

SEQ_LENGTH_GRID = [3, 7, 14]
LR_GRID = [0.001, 0.01]


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    configs_dir = RESULTS_DIR / "configs"
    runs_dir = RESULTS_DIR / "runs"
    configs_dir.mkdir(exist_ok=True)
    runs_dir.mkdir(exist_ok=True)

    base_cfg = yaml.safe_load(BASE_CONFIG.read_text())

    rows = []
    for week in TEST_WEEKS:
        label, start, end = week["label"], week["test_start"], week["test_end"]
        for seq_length in SEQ_LENGTH_GRID:
            for lr in LR_GRID:
                run_name = f"{label}_seq{seq_length}_lr{lr}"
                cfg = dict(base_cfg)
                cfg["test_start_date"] = start
                cfg["test_end_date"] = end
                cfg["seq_length"] = seq_length
                cfg["lr"] = lr
                cfg["report_root"] = str(runs_dir)
                cfg["model_family"] = "sweep"
                cfg["run_name"] = run_name
                cfg["generate_spatial_plots"] = False

                cfg_path = configs_dir / f"{run_name}.yaml"
                cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

                print(f"[{run_name}] running...", flush=True)
                result = subprocess.run(
                    [sys.executable, str(TRAIN_SCRIPT), "--config", str(cfg_path)],
                    cwd=PROJECT_ROOT,
                )
                if result.returncode:
                    print(f"[{run_name}] FAILED (exit {result.returncode})", file=sys.stderr)
                    continue

                metrics_paths = sorted((runs_dir / "sweep").rglob(f"*{run_name}*/metrics.yaml"))
                if not metrics_paths:
                    print(f"[{run_name}] no metrics.yaml found", file=sys.stderr)
                    continue
                metrics = yaml.safe_load(metrics_paths[-1].read_text())
                rows.append({"week": label, "seq_length": seq_length, "lr": lr, **metrics})

    fieldnames = sorted({k for row in rows for k in row})
    with open(RESULTS_DIR / "convlstm_grid_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Average mean_ll_obs across the 4 weeks per (seq_length, lr), same
    # selection convention as the original convlstm_search_results.csv.
    import statistics
    combos = {}
    for row in rows:
        key = (row["seq_length"], row["lr"])
        combos.setdefault(key, []).append(row["mean_ll_obs"])
    averaged = sorted(
        ((k, statistics.mean(v)) for k, v in combos.items()),
        key=lambda item: item[1], reverse=True,
    )
    print("\nAveraged mean_ll_obs by (seq_length, lr), best first:")
    for (seq_length, lr), avg in averaged:
        print(f"  seq_length={seq_length:<3} lr={lr:<6} mean_ll_obs={avg:.4f}")

    best_seq_length, best_lr = averaged[0][0]
    summary = {
        "test_weeks": TEST_WEEKS,
        "selected": {"seq_length": best_seq_length, "lr": best_lr},
    }
    with open(RESULTS_DIR / "selected_hyperparameters.yaml", "w") as f:
        yaml.safe_dump(summary, f, sort_keys=False)
    print(f"\nSelected ConvLSTM config: seq_length={best_seq_length}  lr={best_lr}")


if __name__ == "__main__":
    main()
