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
