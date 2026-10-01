"""Run scripts/run_monthly_windows.py across multiple seeds (in parallel),
then pool per-window metrics across both the 4 weekly windows and the seeds
into per-month mean/std tables.

For each seed, every seed-relevant field already present in the base config
(torch_seed, np_seed, data_seed) is overridden to that seed value, then
scripts/run_monthly_windows.py runs the usual 8 (4 May + 4 November)
fixed-test-window jobs for it. Afterwards, each seed's per_window_metrics.csv
is concatenated and, per month, per metric, the mean and population std are
computed over all (window x seed) values.

Example:
    python scripts/run_seeded_monthly_windows.py \
        --script scripts/15_train_gnn.py \
        --config config/15_gnn.yaml \
        --seeds 1-10 \
        --parallel 3
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

# Force single-threaded BLAS/OpenMP in every child process. Running several
# seeds concurrently otherwise lets each process's math library spawn one
# thread per core (observed: 56 threads for a single LGCP run), so N
# concurrent seeds oversubscribe the machine by N x core_count and can be an
# order of magnitude slower than running sequentially instead of faster.
_SINGLE_THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SEED_FIELDS = ["torch_seed", "np_seed", "data_seed"]
META_COLS = {"name", "month", "test_start_date", "test_end_date", "seed"}


def parse_seeds(spec: str) -> list[int]:
    seeds: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-")
            seeds.extend(range(int(lo), int(hi) + 1))
        else:
            seeds.append(int(part))
    return seeds


def resolve_input(value: str, folder: str) -> Path:
    path = Path(value).expanduser()
    for candidate in (path, PROJECT_ROOT / path, PROJECT_ROOT / folder / path):
        if candidate.is_file():
            return candidate.resolve()
    raise ValueError(f"File not found: {value}")


def run_one_seed(script: Path, base_config: dict, seed: int, base_output: Path, year: int | None) -> tuple[int, int, Path]:
    cfg = dict(base_config)
    for field in SEED_FIELDS:
        if field in cfg:
            cfg[field] = seed
    seed_config_path = base_output / f"config_seed_{seed}.yaml"
    seed_config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    seed_out = base_output / f"seed_{seed}"
    command = [
        sys.executable, str(PROJECT_ROOT / "scripts/run_monthly_windows.py"),
        "--script", str(script), "--config", str(seed_config_path),
        "--output-dir", str(seed_out),
    ]
    if year is not None:
        command += ["--year", str(year)]
    print(f"[seed {seed}] launching: {' '.join(command)}", flush=True)
    env = {**os.environ, **_SINGLE_THREAD_ENV}
    result = subprocess.run(command, cwd=PROJECT_ROOT, env=env)
    print(f"[seed {seed}] exit={result.returncode}", flush=True)
    return seed, result.returncode, seed_out


def pool_metrics(seed_dirs: dict[int, Path], output: Path) -> None:
    all_rows = []
    for seed, seed_out in seed_dirs.items():
        csv_path = seed_out / "per_window_metrics.csv"
        if not csv_path.is_file():
            print(f"Warning: missing {csv_path}", file=sys.stderr)
            continue
        with open(csv_path, newline="") as stream:
            for row in csv.DictReader(stream):
                row["seed"] = seed
                all_rows.append(row)

    metric_names = sorted({key for row in all_rows for key in row if key not in META_COLS})
    months = sorted({row["month"] for row in all_rows})

    monthly = {}
    for month in months:
        month_rows = [row for row in all_rows if row["month"] == month]
        monthly[month] = {}
        for metric in metric_names:
            values = []
            for row in month_rows:
                raw = row.get(metric, "")
                if raw == "":
                    continue
                value = float(raw)
                if math.isfinite(value):
                    values.append(value)
            if not values:
                continue
            monthly[month][metric] = {
                "mean": statistics.mean(values),
                "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
                "n": len(values),
            }

    summary = {
        "aggregation": (
            "Unweighted mean/pstdev pooled over all (window x seed) values per "
            "month. n = n_windows_per_month (4) x n_seeds, minus any missing runs."
        ),
        "seeds": sorted(seed_dirs),
        "monthly_metrics": monthly,
    }
    (output / "seeded_monthly_metrics.yaml").write_text(yaml.safe_dump(summary, sort_keys=False))

    rows = []
    for month, metrics in monthly.items():
        for metric, values in metrics.items():
            rows.append({"month": month, "metric": metric, **values})
    with open(output / "seeded_monthly_metrics.csv", "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["month", "metric", "mean", "std", "n"])
        writer.writeheader()
        writer.writerows(rows)

    with open(output / "per_window_per_seed_metrics.csv", "w", newline="") as stream:
        fieldnames = ["seed", "name", "month", "test_start_date", "test_end_date", *metric_names]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nSaved: {output / 'seeded_monthly_metrics.yaml'}")
    for month, metrics in monthly.items():
        print(f"\n{month.capitalize()}")
        for metric, values in metrics.items():
            print(f"  {metric}: {values['mean']:.4f} +/- {values['std']:.4f} (n={values['n']})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--script", required=True, help="Training script path or filename in scripts/.")
    parser.add_argument("--config", required=True, help="Base YAML path or filename in config/.")
    parser.add_argument("--seeds", default="1-10", help="e.g. '1-10' or '1,2,5'.")
    parser.add_argument("--parallel", type=int, default=3, help="Max concurrent seed runs.")
    parser.add_argument("--year", type=int, help="Forwarded to run_monthly_windows.py.")
    parser.add_argument("--output-dir", type=Path, help="Batch directory to create.")
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    script = resolve_input(args.script, "scripts")
    source_config = resolve_input(args.config, "config")
    base_config = yaml.safe_load(source_config.read_text())
    if not isinstance(base_config, dict):
        raise SystemExit("Config must contain a YAML mapping.")

    default_output = (
        PROJECT_ROOT / "reports" / "seeded_monthly_windows" / script.stem
        / datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    output = (args.output_dir or default_output).resolve()
    output.mkdir(parents=True, exist_ok=False)

    seed_dirs: dict[int, Path] = {}
    failures = []
    with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as executor:
        futures = {
            executor.submit(run_one_seed, script, base_config, seed, output, args.year): seed
            for seed in seeds
        }
        for future in as_completed(futures):
            seed, returncode, seed_out = future.result()
            seed_dirs[seed] = seed_out
            if returncode:
                failures.append(seed)

    if failures:
        print(f"Seeds failed: {sorted(failures)}", file=sys.stderr)

    pool_metrics(seed_dirs, output)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
