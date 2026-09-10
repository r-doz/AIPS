import unittest

import numpy as np
import pandas as pd
from scipy.optimize import check_grad
from scipy.stats import nbinom

from src.models.daily_count import NBINGARCH
from src.models.data_daily_count import build_daily_design


class DailyCountTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(12)
        self.X = rng.normal(size=(100, 2))
        self.y = rng.poisson(np.exp(2 + 0.2*self.X[:, 0])).astype(float)

    def test_likelihood_and_analytic_gradient(self):
        model = NBINGARCH(ridge=0.2)
        model.initial_eta_ = 2.0
        params = np.array([0.7, 0.1, -0.2, -0.5, -1., 0.2, 1.5])
        X, y, lagged = self.X[7:], self.y[7:], model._design(self.y)
        error = check_grad(lambda p: model._objective(p, X, y, lagged)[0],
                           lambda p: model._objective(p, X, y, lagged)[1], params)
        self.assertLess(error, 1e-5)
        eta = model._filter(params, X, lagged)
        k = np.exp(params[-1])
        expected = -nbinom.logpmf(y, k, k/(k+np.exp(eta))).mean() + 0.1*np.sum(params[1:3]**2)
        self.assertAlmostEqual(model._objective(params, X, y, lagged)[0], expected)

    def test_forecasts_are_causal_and_positive(self):
        model = NBINGARCH().fit(self.X[:70], self.y[:70])
        self.assertTrue(model.fit_info_['converged'])
        before = model.predict_one_step(self.X, self.y)
        changed = self.y.copy()
        changed[80:] += 100
        after = model.predict_one_step(self.X, changed)
        np.testing.assert_allclose(before[:81], after[:81], equal_nan=True)
        self.assertTrue(np.isfinite(before[7:]).all())
        self.assertTrue((before[7:] > 0).all())
        self.assertGreater(np.max(np.abs(before[81:]-after[81:])), 0)

    def test_full_cell_inputs_order_and_target_isolation(self):
        frame = pd.DataFrame([
            dict(date=date, longitude=cell, latitude=0, env=cell+t,
                 ais_vessels_count=t+cell)
            for t, date in enumerate(pd.date_range('2025-01-01', periods=10))
            for cell in (1, 2)])
        cfg = dict(covariate_cols=['env'], include_coords=False, include_time=False)
        X, y, dates, names = build_daily_design(frame.sample(frac=1, random_state=2), cfg)
        np.testing.assert_equal(X[0], [1, 2])
        np.testing.assert_equal(y, 2*np.arange(10)+3)
        changed = frame.copy()
        changed.loc[changed.date >= dates[5], 'ais_vessels_count'] += 100
        np.testing.assert_equal(build_daily_design(changed, cfg)[0], X)
        self.assertEqual(len(names), 2)
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            build_daily_design(pd.concat([frame, frame.iloc[:1]]), cfg)
        with self.assertRaisesRegex(ValueError, 'consecutive'):
            build_daily_design(frame[frame.date != dates[5]], cfg)
        with self.assertRaisesRegex(ValueError, 'Unbalanced'):
            build_daily_design(frame.iloc[1:], cfg)
        with self.assertRaisesRegex(ValueError, 'target'):
            build_daily_design(frame, dict(cfg, covariate_cols=['ais_vessels_count']))

    def test_invalid_counts_and_lags(self):
        for lags in ([0], [1, 1], [1.5], []):
            with self.assertRaises(ValueError):
                NBINGARCH(observation_lags=lags)
        with self.assertRaisesRegex(ValueError, 'integer'):
            NBINGARCH().fit(self.X, self.y+0.1)
        with self.assertRaisesRegex(ValueError, 'finite'):
            NBINGARCH().fit(self.X*np.nan, self.y)


if __name__ == '__main__':
    unittest.main()
