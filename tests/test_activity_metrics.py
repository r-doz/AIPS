import importlib.util
from pathlib import Path
import unittest

import numpy as np

from src.models.metrics_lgcp import evaluate_activity_metrics, evaluate_metrics


class ActivityMetricsTests(unittest.TestCase):
    def test_confusion_matrix(self):
        # TP=2, TN=1, FP=1, FN=2.
        result = evaluate_activity_metrics([1, 3, 0, 0, 2, 1], [2, .1, 0, 4, 0, 0])
        self.assertEqual(result['accuracy'], .5)
        self.assertAlmostEqual(result['precision'], 2 / 3)
        self.assertEqual(result['recall'], .5)

    def test_zero_denominators(self):
        self.assertEqual(evaluate_activity_metrics([0, 0], [0, 0]),
                         dict(accuracy=1., precision=0., recall=0.))
        self.assertEqual(evaluate_activity_metrics([1, 0], [0, 0]),
                         dict(accuracy=.5, precision=0., recall=0.))
        self.assertEqual(evaluate_activity_metrics([0, 0], [1, 1]),
                         dict(accuracy=0., precision=0., recall=0.))

    def test_strict_positive_threshold(self):
        self.assertEqual(evaluate_activity_metrics([0, 1], [1e-12, .1]),
                         dict(accuracy=.5, precision=.5, recall=1.))

    def test_empty_and_mismatched_inputs(self):
        self.assertTrue(all(np.isnan(x) for x in evaluate_activity_metrics([], []).values()))
        with self.assertRaises(ValueError):
            evaluate_activity_metrics([1, 0], [1])

    def test_shared_and_global_cell_mean_evaluators(self):
        path = Path(__file__).resolve().parents[1] / 'scripts/12b_global_cell_mean.py'
        spec = importlib.util.spec_from_file_location('global_cell_mean', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        observed = np.array([0, 2, 1, 0])
        predicted = np.array([0., .2, 0., 1.])
        dates = np.array(['2025-11-01', '2025-11-01', '2025-11-02', '2025-11-02'])
        expected = dict(accuracy=.5, precision=.5, recall=.5)
        shared = evaluate_metrics(observed, predicted, dates)
        baseline, daily = module.poisson_metrics(observed, predicted, dates)
        for result in (shared, baseline):
            for key, value in expected.items():
                self.assertEqual(result[key], value)
            self.assertIn('mae_obs', result)
            self.assertIn('mean_ll_daily', result)
        self.assertEqual(len(daily), 2)


if __name__ == '__main__':
    unittest.main()
