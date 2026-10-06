import unittest

import numpy as np

from src.models.metrics_lgcp import evaluate_first_three_days


class FirstThreeDayMetricsTests(unittest.TestCase):
    def test_pools_sorted_calendar_days_and_excludes_later_predictions(self):
        dates = ['2025-11-04', '2025-11-02', '2025-11-01 12:00',
                 '2025-11-03', '2025-11-01 18:00']
        result = evaluate_first_three_days(
            [999, 2, 0, 4, 1], [0, 2, 0, 4, 1], dates
        )
        self.assertEqual(result['dates'], ['2025-11-01', '2025-11-02', '2025-11-03'])
        self.assertEqual(result['num_observations'], 4)
        self.assertEqual(result['metrics']['mae_obs'], 0)
        self.assertEqual(result['metrics']['wasserstein'], 0)
        self.assertEqual(result['metrics']['accuracy'], 1)
        self.assertEqual(result['metrics']['f1'], 1)
        self.assertIsNone(result['metrics']['daily_delta_corr'])
        self.assertEqual(len(result['metrics']), 13)

    def test_short_windows_skip_supplementary_report(self):
        for n in range(4):
            self.assertIsNone(evaluate_first_three_days(
                np.ones(n), np.ones(n), [f'2025-11-0{i+1}' for i in range(n)]
            ))

    def test_custom_evaluator_receives_only_first_three_days(self):
        result = evaluate_first_three_days(
            [1, 2, 3, 999], [1, 2, 3, 0],
            ['2025-11-01', '2025-11-02', '2025-11-03', '2025-11-04'],
            evaluator=lambda y, rate, dates: {'total': float(y.sum())},
        )
        self.assertEqual(result['metrics'], {'total': 6.0})
