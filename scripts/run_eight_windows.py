"""Run a --config training script over four weeks each in May/November."""

import argparse
import copy
import math
from datetime import date, datetime
from pathlib import Path
import shlex
import subprocess
import statistics
import sys

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def resolve_input(value: str, folder: str) -> Path:
    path = Path(value).expanduser()
    for candidate in (path, PROJECT_ROOT / path, PROJECT_ROOT / folder / path):
        if candidate.is_file():
            return candidate.resolve()
    raise ValueError(f"File not found: {value}")


def build_windows(config: dict, year: int) -> list[tuple[str, dict]]:
    windows = []
    for month, label in ((5, "may"), (11, "nov")):
        for first in (1, 8, 15, 22):
            name = f"{year}_{label}_{first}_{first + 6}"
            cfg = copy.deepcopy(config)
            cfg.update(
                split_strategy="fixed_test_window",
                test_start_date=date(year, month, first).isoformat(),
                test_end_date=date(year, month, first + 6).isoformat(),
                run_name=name,
            )
            windows.append((name, cfg))
    return windows


def aggregate_metrics(output: Path) -> Path:
    """Summarize finite scalar metrics, weighting every window equally."""
    configs = sorted((output / "configs").glob("*.yaml"))
    if not configs:
        raise ValueError(f"No window configs found in {output / 'configs'}")
    values = {}
    per_window = {}
    missing = []
    for config_path in configs:
        name = config_path.stem
        paths = sorted((output / "runs" / name).rglob("metrics.yaml"))
        if not paths:
            missing.append(name)
            continue
        if len(paths) != 1:
            raise ValueError(f"Expected one metrics.yaml for {name}, found {len(paths)}")
        metrics = yaml.safe_load(paths[0].read_text())
        if not isinstance(metrics, dict):
            raise ValueError(f"Metrics must be a YAML mapping: {paths[0]}")
        per_window[name] = metrics
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
                values.setdefault(key, []).append(float(value))
    summary = {
        "aggregation": "Unweighted mean across windows; std is population standard deviation. Non-finite/non-numeric values excluded. RMSE is mean window RMSE, not pooled RMSE.",
        "expected_windows": len(configs),
        "windows_with_metrics": len(per_window),
        "missing_windows": missing,
        "metrics": {
            key: {"mean": statistics.mean(items), "std": statistics.pstdev(items), "n_windows": len(items)}
            for key, items in sorted(values.items())
        },
        "per_window": per_window,
    }
    destination = output / "aggregated_metrics.yaml"
    destination.write_text(yaml.safe_dump(summary, sort_keys=False))
    print(f"Aggregated metrics ({len(per_window)}/{len(configs)} windows): {destination}", flush=True)
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--script", help="Script path or filename in scripts/.")
    parser.add_argument("--config", help="YAML path or filename in config/.")
    parser.add_argument("--aggregate-only", type=Path, metavar="BATCH_DIR", help="Summarize an existing batch without training.")
    parser.add_argument("--year", type=int, help="Defaults to the year of test_start_date.")
    parser.add_argument("--output-dir", type=Path, help="New batch directory; must not already exist.")
    parser.add_argument("--dry-run", action="store_true", help="Write configs and print commands without training.")
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
        source = resolve_input(args.config, "config")
        config = yaml.safe_load(source.read_text())
        if not isinstance(config, dict):
            raise ValueError("Config must contain a YAML mapping.")
        year = args.year
        if year is None:
            year = date.fromisoformat(str(config.get("test_start_date", ""))).year
        years = config.get("years")
        if years is not None and year not in [int(y) for y in (years if isinstance(years, list) else [years])]:
            raise ValueError(f"Window year {year} is not included in config years: {years}")
        windows = build_windows(config, year)
    except (ValueError, OSError, yaml.YAMLError) as exc:
        parser.error(f"{exc} (Use --year to specify the window year.)")

    output = (args.output_dir or PROJECT_ROOT / "reports" / "eight_windows" / script.stem / datetime.now().strftime("%Y%m%d-%H%M%S-%f")).resolve()
    output.mkdir(parents=True, exist_ok=False)
    configs_dir = output / "configs"
    configs_dir.mkdir()
    jobs = []
    for name, cfg in windows:
        cfg["report_root"] = str(output / "runs")
        cfg["model_family"] = name
        path = configs_dir / f"{name}.yaml"
        path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        jobs.append((name, [sys.executable, str(script), "--config", str(path)]))

    print(f"Batch directory: {output}", flush=True)
    for index, (name, command) in enumerate(jobs, 1):
        print(f"[{index}/8] {shlex.join(command)}", flush=True)
        if not args.dry_run:
            # Inherit the terminal so training progress remains visible.
            result = subprocess.run(command, cwd=PROJECT_ROOT)
            if result.returncode:
                print(f"Failed: {name} (exit {result.returncode}); stopping batch.", file=sys.stderr)
                return result.returncode
    if not args.dry_run:
        aggregate_metrics(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
