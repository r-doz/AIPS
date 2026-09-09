import contextlib
import io
from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd
import yaml

from src.models.daily_count_experiment import run
from src.models.daily_allocation import load_daily_totals, allocate_daily_totals


class LightGBMDailyTests(unittest.TestCase):
    model_kind = "lightgbm"
    def test_artifacts_and_no_same_day_target_leakage(self):
        with tempfile.TemporaryDirectory() as tmp:
            dates = pd.date_range('2025-01-01', periods=60)
            frame = pd.DataFrame([
                dict(date=d, longitude=c, latitude=0, env=t % 7,
                     ais_vessels_count=(t % 7)+c)
                for t, d in enumerate(dates) for c in (1, 2)])
            source = Path(tmp) / 'data.parquet'
            cfg = dict(parquet_path=str(source), covariate_cols=['env', 'daily_total_lag_1'],
                       include_time=False, include_coords=False,
                       lag_features=dict(enabled=True, daily_total_lags=[1]),
                       split_strategy='fixed_test_window', test_start_date=str(dates[50].date()),
                       test_end_date=str(dates[-1].date()), report_root=tmp,
                       model=dict(objective='poisson', n_estimators=20, num_leaves=3,
                                  min_child_samples=5, verbosity=-1, n_jobs=1, random_state=0))
            if self.model_kind == 'catboost':
                cfg['model'] = dict(loss_function=getattr(self, 'loss', 'Poisson'), iterations=20, depth=3,
                                    verbose=False, thread_count=1, random_seed=0,
                                    allow_writing_files=False, has_time=True)
            cfg['training'] = {'half_life_days': 90}
            frame.to_parquet(source)
            with contextlib.redirect_stdout(io.StringIO()):
                first = run(cfg, self.model_kind)
            totals = load_daily_totals(first / 'daily_predictions.csv')
            self.assertEqual(len(totals), 10)
            self.assertTrue((totals > 0).all())
            allocated = allocate_daily_totals(np.ones(20), np.repeat(totals.index, 2), totals)
            np.testing.assert_allclose(allocated.reshape(10, 2).sum(axis=1), totals)
            self.assertEqual(len(list(first.glob('*.png'))), 1)
            metrics = yaml.safe_load((first / 'metrics.yaml').read_text())
            errors = pd.read_csv(first / 'daily_metrics.csv')
            self.assertAlmostEqual(metrics['mae_daily'], errors.absolute_error.mean())
            self.assertNotIn('mean_nb_ll_daily', metrics)
            # A changed target cannot affect its own or previous forecasts.
            frame.loc[frame.date >= dates[55], 'ais_vessels_count'] += 100
            frame.to_parquet(source)
            with contextlib.redirect_stdout(io.StringIO()):
                second = run(cfg, self.model_kind)
            changed = load_daily_totals(second / 'daily_predictions.csv')
            np.testing.assert_allclose(totals.iloc[:6], changed.iloc[:6])
            a, b = (joblib.load(p / 'model.joblib') for p in (first, second))
            if getattr(self, 'loss', None) == 'RMSE':
                from src.models.data_daily_count import build_daily_design
                X, _, _, _ = build_daily_design(frame, cfg)
                transformed = a['scaler'].transform(X)[:, a['active_features']]
                raw = np.maximum(a['model'].predict(transformed[50:56], prediction_type='RawFormulaVal'), 0)
                np.testing.assert_allclose(totals.iloc[:6], raw)
            np.testing.assert_array_equal(a['scaler'].mean_, b['scaler'].mean_)
            if self.model_kind == 'lightgbm':
                self.assertEqual(a['model'].booster_.model_to_string(), b['model'].booster_.model_to_string())
            else:
                np.testing.assert_array_equal(a['model'].get_leaf_values(), b['model'].get_leaf_values())


class CatBoostDailyTests(LightGBMDailyTests):
    model_kind = 'catboost'


class CatBoostRMSEDailyTests(CatBoostDailyTests):
    loss = 'RMSE'


class RecencyTests(unittest.TestCase):
    def test_half_life_and_normalization(self):
        from src.models.daily_count_weighting import recency_weights
        dates = pd.to_datetime(['2025-01-01', '2025-01-31', '2025-03-02'])
        w = recency_weights(dates, 30)
        np.testing.assert_allclose(w[1:]/w[:-1], [2, 2])
        self.assertAlmostEqual(w.mean(), 1)
        self.assertIsNone(recency_weights(dates, None))
        for invalid in [0, -1, float('nan')]:
            with self.assertRaises(ValueError):
                recency_weights(dates, invalid)


if __name__ == '__main__':
    unittest.main()
