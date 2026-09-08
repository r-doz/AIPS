import unittest
import numpy as np
import pandas as pd
from src.models.daily_allocation import allocate_daily_totals


class AllocationTests(unittest.TestCase):
    def test_alignment_conservation_and_concentration(self):
        dates = ['2025-01-02', '2025-01-01', '2025-01-02', '2025-01-01']
        totals = pd.Series([12., 8.], index=pd.to_datetime(['2025-01-01', '2025-01-02']))
        rates = np.array([1., 1., 3., 1.])
        np.testing.assert_allclose(allocate_daily_totals(rates, dates, totals), [2, 6, 6, 6])
        np.testing.assert_allclose(allocate_daily_totals(rates, dates, totals, 2), [.8, 6, 7.2, 6])
        np.testing.assert_array_equal(rates, [1, 1, 3, 1])

    def test_zero_total_and_unallocatable_total(self):
        dates = ['2025-01-01'] * 2
        totals = pd.Series([0.], index=pd.to_datetime(['2025-01-01']))
        np.testing.assert_array_equal(allocate_daily_totals([0, 0], dates, totals), [0, 0])
        totals.iloc[0] = 1
        with self.assertRaisesRegex(ValueError, 'No positive'):
            allocate_daily_totals([0, 0], dates, totals)

    def test_missing_date_and_invalid_gamma(self):
        totals = pd.Series([1.], index=pd.to_datetime(['2025-01-01']))
        with self.assertRaisesRegex(ValueError, 'Missing'):
            allocate_daily_totals([1], ['2025-01-02'], totals)
        with self.assertRaisesRegex(ValueError, 'gamma'):
            allocate_daily_totals([1], ['2025-01-01'], totals, 0)

    def test_conflict_policies_only_affect_unallocatable_days(self):
        dates = ['2025-01-02', '2025-01-01', '2025-01-02', '2025-01-01']
        totals = pd.Series([12., 8.], index=pd.to_datetime(['2025-01-01', '2025-01-02']))
        rates = [0., 0., 0., 2.]
        with self.assertWarnsRegex(RuntimeWarning, 'conflict_policy=classifier'):
            result = allocate_daily_totals(rates, dates, totals, conflict_policy='classifier')
        np.testing.assert_allclose(result, [0, 0, 0, 12])
        with self.assertWarnsRegex(RuntimeWarning, 'conflict_policy=daily_total'):
            result = allocate_daily_totals(
                rates, dates, totals, gamma=2, conflict_policy='daily_total',
                fallback_rates=[1., 5., 3., 2.],
            )
        np.testing.assert_allclose(result, [.8, 0, 7.2, 12])
        np.testing.assert_array_equal(rates, [0, 0, 0, 2])

    def test_daily_total_uniform_fallback(self):
        totals = pd.Series([8.], index=pd.to_datetime(['2025-01-01']))
        with self.assertWarns(RuntimeWarning):
            result = allocate_daily_totals(
                [0, 0], ['2025-01-01'] * 2, totals,
                conflict_policy='daily_total', fallback_rates=[0, 0],
            )
        np.testing.assert_allclose(result, [4, 4])

    def test_invalid_policy_and_fallback(self):
        totals = pd.Series([8.], index=pd.to_datetime(['2025-01-01']))
        with self.assertRaisesRegex(ValueError, 'conflict_policy'):
            allocate_daily_totals([1], ['2025-01-01'], totals, conflict_policy='typo')
        for fallback in (None, [1, 2], [-1], [np.nan]):
            with self.subTest(fallback=fallback), self.assertRaisesRegex(ValueError, 'fallback_rates'):
                allocate_daily_totals(
                    [0], ['2025-01-01'], totals,
                    conflict_policy='daily_total', fallback_rates=fallback,
                )
