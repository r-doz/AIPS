import unittest

import pandas as pd

from src.data.build_dataset_flaggear import add_grouped_ais_counts


class FlagGearDatasetTests(unittest.TestCase):
    def test_sums_vessel_counts_instead_of_counting_aggregate_rows(self):
        grid = pd.DataFrame(
            {
                "date": [pd.Timestamp("2024-01-02")],
                "latitude": [45.6],
                "longitude": [13.6],
                "chl": [1.2],
                "thetao": [12.4],
            }
        )
        ais_groups = pd.DataFrame(
            {
                "date": [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-02")],
                "latitude": [45.6, 45.6],
                "longitude": [13.6, 13.6],
                "flag": ["ESP", "ESP"],
                "gear_type": ["trawlers", "set_gillnets"],
                "vessel_count": [4, 3],
            }
        )

        result = add_grouped_ais_counts(grid, ais_groups)

        self.assertEqual(result.loc[0, "ais_vessels_count"], 7)
        self.assertEqual(result.loc[0, "TRAWLERS"], 4)
        self.assertEqual(result.loc[0, "SET_GILLNETS"], 3)


if __name__ == "__main__":
    unittest.main()
