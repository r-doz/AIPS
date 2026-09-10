import unittest

import numpy as np

from src.models.zero_gate import apply_zero_gate


class FixedClassifier:
    # Reversed class order checks that the active class is looked up.
    classes_ = np.array([1, 0])

    def predict_proba(self, features):
        p = np.array([0.0, 0.25, 0.5, 1.0])
        return np.column_stack([p, 1 - p])


class ZeroGateTests(unittest.TestCase):
    def setUp(self):
        self.coords = np.zeros((4, 3))
        self.covs = np.zeros((4, 2))
        self.rates = np.array([2., 4., 6., 8.])

    def apply(self, rates=None, **kwargs):
        return apply_zero_gate(
            FixedClassifier(), self.coords, self.covs,
            self.rates if rates is None else rates, **kwargs,
        )

    def test_default_preserves_hard_threshold_and_input(self):
        np.testing.assert_array_equal(self.apply(), [0, 0, 6, 8])
        np.testing.assert_array_equal(self.rates, [2, 4, 6, 8])

    def test_soft_ignores_threshold_and_preserves_input(self):
        for threshold in [0.03, 0.5, 0.9]:
            np.testing.assert_allclose(
                self.apply(mode="soft", threshold=threshold), [0, 1, 3, 8]
            )
        np.testing.assert_array_equal(self.rates, [2, 4, 6, 8])

    def test_soft_column_rates_do_not_broadcast_across_rows(self):
        np.testing.assert_allclose(
            self.apply(self.rates[:, None], mode="soft"), [[0], [1], [3], [8]]
        )

    def test_invalid_mode(self):
        with self.assertRaisesRegex(ValueError, "zero_gate.mode"):
            self.apply(mode="invalid")

    def test_redistribution_preserves_each_day_and_relative_weights(self):
        dates = ['2025-05-01', '2025-05-02', '2025-05-01', '2025-05-02']
        result = self.apply(mode='hard_redistribute', dates=dates)
        np.testing.assert_allclose(result, [0, 0, 8, 12])
        result = self.apply(mode='hard_redistribute', dates=['2025-05-01'] * 4)
        np.testing.assert_allclose(result, [0, 0, 20 * 6 / 14, 20 * 8 / 14])
        np.testing.assert_array_equal(self.rates, [2, 4, 6, 8])

    def test_redistribution_empty_support_falls_back(self):
        with self.assertWarnsRegex(RuntimeWarning, 'keeping original'):
            result = self.apply(mode='hard_redistribute', dates=['2025-05-01'] * 2 + ['2025-05-02'] * 2)
        np.testing.assert_allclose(result, self.rates)

    def test_redistribution_zero_totals_and_column_shape(self):
        result = self.apply(np.zeros((4, 1)), mode='hard_redistribute', dates=['2025-05-01'] * 4)
        np.testing.assert_array_equal(result, np.zeros((4, 1)))
        result = self.apply(self.rates[:, None], mode='hard_redistribute', dates=['2025-05-01'] * 4)
        self.assertEqual(result.shape, (4, 1))
        self.assertAlmostEqual(result.sum(), self.rates.sum())

    def test_redistribution_requires_valid_dates_and_rates(self):
        for dates in (None, ['2025-05-01'], [None] * 4):
            with self.assertRaises(ValueError):
                self.apply(mode='hard_redistribute', dates=dates)
        for rates in (np.array([-1, 2, 3, 4]), np.array([np.nan, 2, 3, 4])):
            with self.assertRaises(ValueError):
                self.apply(rates, mode='hard_redistribute', dates=['2025-05-01'] * 4)


if __name__ == "__main__":
    unittest.main()
