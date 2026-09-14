import importlib.util
from pathlib import Path
import unittest

import numpy as np

from src.models.metrics_lgcp import evaluate_metrics, evaluate_wasserstein


class WassersteinMetricsTests(unittest.TestCase):
    def test_known_cdf_integral(self):
        # CDF difference is 1/3 on (0, 1) and 1/3 on (1, 3).
        self.assertAlmostEqual(evaluate_wasserstein([0, 0, 3], [0, 1, 1]), 1.0)

    def test_identical_distributions_ignore_order(self):
        self.assertEqual(evaluate_wasserstein([0, 2, 5], [5, 0, 2]), 0.)

    def test_shift_and_scale_in_count_units(self):
        self.assertAlmostEqual(evaluate_wasserstein([0, 1, 4], [2, 3, 6]), 2.)
        self.assertAlmostEqual(evaluate_wasserstein([0, 10, 40], [20, 30, 60]), 20.)

    def test_empty_and_nonfinite_inputs(self):
        for observed, predicted in [([], []), ([0], []), ([np.nan], [0]),
                                    ([0], [np.inf])]:
            with self.subTest(observed=observed, predicted=predicted):
                self.assertTrue(np.isnan(evaluate_wasserstein(observed, predicted)))

    def test_shared_and_global_cell_mean_use_unfloored_predictions(self):
        path = Path(__file__).resolve().parents[1] / 'scripts/12b_global_cell_mean.py'
        spec = importlib.util.spec_from_file_location('global_cell_mean', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        observed = np.array([0, 0, 3])
        predicted = np.array([0., 1., 1.])
        dates = ['2025-11-01', '2025-11-01', '2025-11-02']
        shared = evaluate_metrics(observed, predicted, dates)
        baseline, _ = module.poisson_metrics(observed, predicted, dates, eps=.1)
        for key in ('daily_delta_corr', 'daily_direction_accuracy_moving'):
            np.testing.assert_allclose(baseline[key], shared[key], equal_nan=True)
        for result in (shared, baseline):
            self.assertAlmostEqual(result['wasserstein'], 1.0)


if __name__ == '__main__':
    unittest.main()
