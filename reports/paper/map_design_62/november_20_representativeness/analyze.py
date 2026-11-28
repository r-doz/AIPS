"""Recompute notebook 62's day-selection evidence from its saved seed-1 predictions."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml
from scipy.stats import trim_mean

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from src.models.metrics_lgcp import evaluate_metrics

SOURCE = ROOT / "reports/seeded_monthly_windows_hurdle/seed_1"
OUT = Path(__file__).resolve().parent
TARGET_DAY = "2025-11-20"
TARGET_WEEK = "2025_november_15_21"
KEYS = ["mae_obs", "rmse_obs", "wasserstein", "mae_daily", "rmse_daily", "f1"]
DAY_KEYS = ["mae_obs", "rmse_obs", "wasserstein", "mae_daily",
            "accuracy", "precision", "recall", "f1"]
METRIC_SETS = {
    "paired_count_errors": ["mae_obs", "rmse_obs", "mae_daily"],
    "count_errors": ["mae_obs", "rmse_obs", "wasserstein", "mae_daily"],
    "count_errors_and_activity": ["mae_obs", "rmse_obs", "wasserstein", "mae_daily", "f1"],
}


def across_days(daily, raw):
    """Descriptive sensitivity analysis; no metric weights fitted to the target."""
    aggregates, scores, joint = [], [], []
    for scope, days in [("all_56_days", daily),
                        ("observed_active_45_days", daily[daily.observed_total > 0])]:
        cells = raw[raw.date.isin(days.index)]
        pooled = evaluate_metrics(cells.observed.to_numpy(), cells.pred_lower.to_numpy(), cells.date)
        for key in DAY_KEYS:
            s = days[key]
            target = s.loc[TARGET_DAY]
            aggregates.append(dict(
                scope=scope, metric=key, n_days=len(days), target=target,
                mean_daily=s.mean(), median_daily=s.median(),
                trimmed_mean_10pct=trim_mean(s, .1), q25=s.quantile(.25), q75=s.quantile(.75),
                std_daily=s.std(), pooled=pooled[key],
                midrank_percentile=100 * (s.rank().loc[TARGET_DAY] - .5) / len(s),
                percent_from_mean=100 * (target / s.mean() - 1),
                percent_from_median=100 * (target / s.median() - 1),
                percent_from_pooled=100 * (target / pooled[key] - 1),
                standardized_distance_from_mean=(target - s.mean()) / s.std(),
            ))
        for label, keys in METRIC_SETS.items():
            values = days[keys]
            pct = (values.rank() - .5) / len(values)
            z = (values - values.mean()) / values.std()
            iqr_distance = (values - values.median()) / (values.quantile(.75) - values.quantile(.25))
            methods = {
                "mean_absolute_percentile_distance": (pct - .5).abs().mean(axis=1),
                "maximum_absolute_percentile_distance": (pct - .5).abs().max(axis=1),
                "rms_standardized_distance_from_mean": z.pow(2).mean(axis=1).pow(.5),
                "mean_absolute_IQR_scaled_distance_from_median": iqr_distance.abs().mean(axis=1),
            }
            for method, score in methods.items():
                # Equal theoretical scores can differ by floating-point roundoff.
                ranks = score.round(12).rank()
                for date in days.index:
                    scores.append(dict(scope=scope, metric_set=label, method=method,
                                       date=date, score=score.loc[date], centrality_rank=ranks.loc[date],
                                       n_days=len(days), is_target=date == TARGET_DAY))
            within = values.ge(values.quantile(.25)) & values.le(values.quantile(.75))
            for date in days.index:
                joint.append(dict(scope=scope, metric_set=label, date=date,
                                  all_within_middle_50pct=bool(within.loc[date].all()),
                                  outside_metrics=",".join(within.columns[~within.loc[date]])))
    pd.DataFrame(aggregates).to_csv(OUT / "across_days_aggregates.csv", index=False)
    pd.DataFrame(scores).to_csv(OUT / "multimetric_centrality.csv", index=False)
    pd.DataFrame(joint).to_csv(OUT / "joint_middle_50pct.csv", index=False)


def comparisons(frame, target, scope):
    rows = []
    for key in KEYS:
        values = frame[key]
        value, median = values.loc[target], values.median()
        rows.append(dict(
            scope=scope, metric=key, target_value=value, mean=values.mean(),
            median=median, percent_from_median=100 * (value / median - 1)
            if median else np.nan,
            ascending_rank=values.rank().loc[target], n=len(values),
            distance_from_median_rank=(values - median).abs().rank().loc[target],
            closest_to_median=values.sub(median).abs().idxmin(),
        ))
    return rows


def main():
    daily, weekly, sources, raw = [], [], [], []
    grid = pd.read_csv(ROOT / "notebooks/assets/map_design_61/retained_grid_centres.csv")
    manifest = yaml.safe_load((SOURCE / "manifest.yaml").read_text())
    for window in manifest:
        paths = list((SOURCE / "runs" / window["name"]).glob("*/hurdle_predictions.csv"))
        assert len(paths) == 1, (window["name"], paths)
        path = paths[0]
        params = yaml.safe_load((path.parent / "params.yaml").read_text())
        assert params["np_seed"] == params["data_seed"] == 1
        assert params["split_strategy"] == "fixed_test_window"
        assert (params.get("hurdle") or {}).get("primary_variant", "lower") == "lower"
        assert params["zero_gate"]["enabled"]
        assert params["zero_gate"]["mode"] == "confidence_redistribute"
        for key in ("test_start_date", "test_end_date"):
            assert pd.Timestamp(params[key]) == pd.Timestamp(window[key])
        data = pd.read_csv(path)
        raw.append(data)
        assert np.isfinite(data[["x", "y", "observed", "pred_lower"]]).all().all()
        assert (data[["observed", "pred_lower"]] >= 0).all().all()
        assert not data.duplicated(["date", "x", "y"]).any()
        assert data.groupby("date").size().eq(len(grid)).all()
        dates = pd.DatetimeIndex(pd.to_datetime(data.date.unique())).sort_values()
        assert dates.equals(pd.date_range(window["test_start_date"], window["test_end_date"]))
        metrics = evaluate_metrics(data.observed.to_numpy(), data.pred_lower.to_numpy(), data.date)
        saved = yaml.safe_load((path.parent / "metrics_lower.yaml").read_text())
        for key, value in metrics.items():
            np.testing.assert_allclose(value, saved[key], rtol=1e-6, atol=1e-7,
                                       err_msg=f"{window['name']}/{key}")
        weekly.append(dict(window=window["name"], month=window["month"], **metrics))
        sources.append(dict(window=window["name"], source_csv=str(path.relative_to(ROOT)),
                            prediction_column="pred_lower", seed=1))
        for date, group in data.groupby("date"):
            metrics = evaluate_metrics(group.observed.to_numpy(), group.pred_lower.to_numpy(), group.date)
            daily.append(dict(
                date=date, window=window["name"], observed_total=group.observed.sum(),
                predicted_total=group.pred_lower.sum(), active_cells=int((group.observed > 0).sum()),
                predicted_active_cells=int((group.pred_lower > 0).sum()), **metrics,
            ))
    d = pd.DataFrame(daily).sort_values("date").set_index("date")
    w = pd.DataFrame(weekly).set_index("window")
    assert len(d) == 56 and len(w) == 8 and d.index.is_unique
    across_days(d, pd.concat(raw, ignore_index=True))
    day_rows, week_rows = [], []
    for scope, frame in [
        ("May and November", d),
        ("November", d[d.index.str.startswith("2025-11")]),
        ("November 15–21", d[d.window == TARGET_WEEK]),
    ]:
        day_rows.extend(comparisons(frame, TARGET_DAY, scope))
        day_rows.extend(comparisons(frame[frame.observed_total > 0], TARGET_DAY,
                                    scope + "; observed-active days only"))
    for scope, frame in [("May and November", w), ("November", w[w.month == "november"])]:
        week_rows.extend(comparisons(frame, TARGET_WEEK, scope))
    d.drop(columns=["daily_delta_corr", "daily_direction_accuracy_moving"]).to_csv(OUT / "daily_metrics.csv")
    w.to_csv(OUT / "weekly_metrics.csv")
    pd.DataFrame(day_rows).to_csv(OUT / "day_centrality.csv", index=False)
    pd.DataFrame(week_rows).to_csv(OUT / "window_centrality.csv", index=False)
    pd.DataFrame(sources).to_csv(OUT / "sources.csv", index=False)
    print("PASS: 56 days, 49 cells per day; all eight pooled weekly reports reproduced.")
    print(d.loc[TARGET_DAY].to_string())
    print(pd.DataFrame(day_rows).query("metric == 'mae_obs'").to_string(index=False))
    print(pd.DataFrame(week_rows).to_string(index=False))


if __name__ == "__main__":
    main()
