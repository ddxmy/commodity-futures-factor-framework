import unittest

import pandas as pd

from src.data_alignment import align_factor_to_calendar


class DataAlignmentTest(unittest.TestCase):
    def setUp(self):
        self.calendar = pd.DataFrame(
            {
                "trade_date": pd.to_datetime(
                    [
                        "2024-01-05",
                        "2024-01-08",
                        "2024-01-09",
                        "2024-01-10",
                    ]
                )
            }
        )

    def test_publication_lag_uses_the_next_trading_day(self):
        data = pd.DataFrame(
            {
                "fut_code": ["RB"],
                "available_date": ["2024-01-05"],
                "observation_date": ["2024-01-04"],
                "raw_factor": [2.0],
            }
        )

        result = align_factor_to_calendar(
            data,
            self.calendar,
            publication_lag=1,
        )

        first_valid = result.loc[result["raw_factor"].notna()].iloc[0]
        self.assertEqual(first_valid["trade_date"], pd.Timestamp("2024-01-08"))
        self.assertEqual(first_valid["available_date"], pd.Timestamp("2024-01-05"))

    def test_weekend_publication_maps_to_monday(self):
        data = pd.DataFrame(
            {
                "fut_code": ["RB"],
                "available_date": ["2024-01-06"],
                "raw_factor": [1.0],
            }
        )

        result = align_factor_to_calendar(
            data,
            self.calendar,
            publication_lag=1,
        )

        first_valid = result.loc[result["raw_factor"].notna(), "trade_date"].min()
        self.assertEqual(first_valid, pd.Timestamp("2024-01-08"))

    def test_does_not_mix_commodities_and_expires_stale_values(self):
        data = pd.DataFrame(
            {
                "fut_code": ["RB", "CU"],
                "available_date": ["2024-01-05", "2024-01-08"],
                "raw_factor": [1.0, -2.0],
            }
        )

        result = align_factor_to_calendar(
            data,
            self.calendar,
            publication_lag=1,
            max_staleness=1,
        )

        rb = result[result["fut_code"] == "RB"].set_index("trade_date")
        cu = result[result["fut_code"] == "CU"].set_index("trade_date")
        self.assertEqual(rb.loc["2024-01-08", "raw_factor"], 1.0)
        self.assertTrue(pd.isna(rb.loc["2024-01-10", "raw_factor"]))
        self.assertTrue(pd.isna(cu.loc["2024-01-08", "raw_factor"]))
        self.assertEqual(cu.loc["2024-01-09", "raw_factor"], -2.0)

    def test_rejects_invalid_parameters_and_duplicate_publications(self):
        data = pd.DataFrame(
            {
                "fut_code": ["RB", "RB"],
                "available_date": ["2024-01-05", "2024-01-05"],
                "raw_factor": [1.0, 2.0],
            }
        )

        with self.assertRaises(ValueError):
            align_factor_to_calendar(data, self.calendar)
        with self.assertRaises(ValueError):
            align_factor_to_calendar(
                data.iloc[[0]], self.calendar, publication_lag=-1
            )


if __name__ == "__main__":
    unittest.main()
