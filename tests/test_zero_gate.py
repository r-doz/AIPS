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


if __name__ == "__main__":
    unittest.main()
