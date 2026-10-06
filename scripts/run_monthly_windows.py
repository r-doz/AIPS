"""Run a configurable training script over four May and four November weeks.

Example:
    python scripts/run_monthly_windows.py \
        --script scripts/15_train_gnn.py \
        --config config/15_gnn_salinity_cellpoisson.yaml
"""

from __future__ import annotations

import argparse
import copy
import csv
import math
import shlex
import statistics
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
WINDOWS = {
    "may": (5, (1, 8, 15, 22)),
    "november": (11, (1, 8, 15, 22)),
}


def resolve_input(value: str, folder: str) -> Path:
    path = Path(value).expanduser()
    for candidate in (path, PROJECT_ROOT / path, PROJECT_ROOT / folder / path):
        if candidate.is_file():
            return candidate.resolve()
    raise ValueError(f"File not found: {value}")


def build_windows(config: dict, year: int) -> list[dict]:
    """Return eight jobs with independent copies of the base configuration."""
    jobs = []
    for month_name, (month_number, start_days) in WINDOWS.items():
        for start_day in start_days:
            end_day = start_day + 6
            name = f"{year}_{month_name}_{start_day:02d}_{end_day:02d}"
            window_config = copy.deepcopy(config)
            window_config.update(
                split_strategy="fixed_test_window",
                test_start_date=date(year, month_number, start_day).isoformat(),
                test_end_date=date(year, month_number, end_day).isoformat(),
                run_name=name,
            )
            jobs.append(
                {
                    "name": name,
                    "month": month_name,
                    "test_start_date": window_config["test_start_date"],
                    "test_end_date": window_config["test_end_date"],
                    "config": window_config,
                }
            )
    return jobs


def _finite_metrics(path: Path) -> dict[str, float]:
    metrics = yaml.safe_load(path.read_text())
    if not isinstance(metrics, dict):
        raise ValueError(f"Metrics must be a YAML mapping: {path}")
    return {
        key: float(value)
        for key, value in metrics.items()
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    }


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_metrics(output: Path) -> Path:
    """Create per-window data and separate unweighted May/November summaries."""
    manifest_path = output / "manifest.yaml"
    if not manifest_path.is_file():
        raise ValueError(f"Manifest not found: {manifest_path}")
    manifest = yaml.safe_load(manifest_path.read_text())
    if not isinstance(manifest, list):
        raise ValueError(f"Manifest must contain a list: {manifest_path}")

    per_window = []
    missing = []
    for job in manifest:
        name = job["name"]
        paths = sorted((output / "runs" / name).rglob("metrics.yaml"))
        if not paths:
            missing.append(name)
            continue
        if len(paths) != 1:
            raise ValueError(f"Expected one metrics.yaml for {name}, found {len(paths)}")
        per_window.append(
            {
                "name": name,
                "month": job["month"],
                "test_start_date": job["test_start_date"],
                "test_end_date": job["test_end_date"],
                "metrics_path": str(paths[0]),
                "metrics": _finite_metrics(paths[0]),
            }
        )

    monthly = {}
    for month_name in WINDOWS:
        month_runs = [item for item in per_window if item["month"] == month_name]
        metric_names = sorted({key for item in month_runs for key in item["metrics"]})
        monthly[month_name] = {}
        for metric in metric_names:
            values = [item["metrics"][metric] for item in month_runs if metric in item["metrics"]]
            monthly[month_name][metric] = {
                "mean": statistics.mean(values),
                "std": statistics.pstdev(values),
                "n_windows": len(values),
            }

    summary = {
        "aggregation": (
            "Unweighted mean by month; std is population standard deviation. "
            "Non-finite and non-numeric values are excluded. RMSE is the mean "
            "of weekly RMSE values, not a pooled RMSE."
        ),
        "expected_windows_per_month": 4,
        "missing_windows": missing,
        "monthly_metrics": monthly,
        "per_window": per_window,
    }
    yaml_path = output / "monthly_metrics.yaml"
    yaml_path.write_text(yaml.safe_dump(summary, sort_keys=False))

    metric_names = sorted({key for item in per_window for key in item["metrics"]})
    window_rows = []
    for item in per_window:
        row = {
            "name": item["name"],
            "month": item["month"],
            "test_start_date": item["test_start_date"],
            "test_end_date": item["test_end_date"],
        }
        row.update(item["metrics"])
        window_rows.append(row)
    _write_csv(
        output / "per_window_metrics.csv",
        window_rows,
        ["name", "month", "test_start_date", "test_end_date", *metric_names],
    )

    monthly_rows = []
    for month_name, metrics in monthly.items():
        for metric, values in metrics.items():
            monthly_rows.append({"month": month_name, "metric": metric, **values})
    _write_csv(
        output / "monthly_metrics.csv",
        monthly_rows,
        ["month", "metric", "mean", "std", "n_windows"],
    )

    print(f"Monthly metrics: {yaml_path}", flush=True)
    for month_name, metrics in monthly.items():
        print(f"\n{month_name.capitalize()} mean metrics", flush=True)
        for metric, values in metrics.items():
            print(
                f"  {metric}: {values['mean']:.6g} "
                f"(std={values['std']:.6g}, n={values['n_windows']})",
                flush=True,
            )
    if missing:
        print(f"Missing windows: {', '.join(missing)}", file=sys.stderr)
    return yaml_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--script", help="Training script path or filename in scripts/.")
    parser.add_argument("--config", help="Base YAML path or filename in config/.")
    parser.add_argument("--year", type=int, help="Defaults to the year in test_start_date.")
    parser.add_argument(
        "--output-dir", type=Path,
        help="Batch directory to create; defaults to reports/monthly_windows/.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Write configs and print commands only.")
    parser.add_argument(
        "--aggregate-only", type=Path, metavar="BATCH_DIR",
        help="Rebuild summaries for a previously created batch without training.",
    )
    args = parser.parse_args()

    if args.aggregate_only:
        if args.script or args.config or args.year is not None or args.output_dir or args.dry_run:
            parser.error("--aggregate-only must be used alone.")
        try:
            aggregate_metrics(args.aggregate_only.resolve())
        except (ValueError, OSError, yaml.YAMLError) as exc:
            parser.error(str(exc))
        return 0

    if not args.script or not args.config:
        parser.error("--script and --config are required unless using --aggregate-only.")

    try:
        script = resolve_input(args.script, "scripts")
        source_config = resolve_input(args.config, "config")
        config = yaml.safe_load(source_config.read_text())
        if not isinstance(config, dict):
            raise ValueError("Config must contain a YAML mapping.")
        year = args.year
        if year is None:
            year = date.fromisoformat(str(config.get("test_start_date", ""))).year
        configured_years = config.get("years")
        if configured_years is not None:
            valid_years = configured_years if isinstance(configured_years, list) else [configured_years]
            if year not in [int(value) for value in valid_years]:
                raise ValueError(f"Window year {year} is not included in config years: {configured_years}")
        jobs = build_windows(config, year)
    except (ValueError, OSError, yaml.YAMLError) as exc:
        parser.error(f"{exc} (Use --year to specify the window year.)")

    default_output = (
        PROJECT_ROOT / "reports" / "monthly_windows" / script.stem
        / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    )
    output = (args.output_dir or default_output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    configs_dir = output / "configs"
    configs_dir.mkdir()

    manifest = []
    commands = []
    for job in jobs:
        window_config = job.pop("config")
        window_config["report_root"] = str(output / "runs")
        window_config["model_family"] = job["name"]
        config_path = configs_dir / f"{job['name']}.yaml"
        config_path.write_text(yaml.safe_dump(window_config, sort_keys=False))
        manifest.append(job)
        commands.append([sys.executable, str(script), "--config", str(config_path)])
    (output / "manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))

    print(f"Batch directory: {output}", flush=True)
    failures = []
    for index, (job, command) in enumerate(zip(manifest, commands), 1):
        print(f"[{index}/8] {shlex.join(command)}", flush=True)
        if args.dry_run:
            continue
        result = subprocess.run(command, cwd=PROJECT_ROOT)
        if result.returncode:
            failures.append({"name": job["name"], "exit_code": result.returncode})
            print(f"Failed: {job['name']} (exit {result.returncode}); continuing.", file=sys.stderr)

    if args.dry_run:
        return 0

    (output / "failures.yaml").write_text(yaml.safe_dump(failures, sort_keys=False))
    aggregate_metrics(output)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
