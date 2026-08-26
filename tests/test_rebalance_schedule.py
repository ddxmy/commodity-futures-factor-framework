import unittest

import pandas as pd

from src.p04_rebalance_schedule import get_rebalance_dates


class RebalanceScheduleTest(unittest.TestCase):
    def setUp(self):
        self.dates = pd.bdate_range("2024-01-01", periods=10)

    def test_selects_every_nth_trading_day(self):
        result = get_rebalance_dates(self.dates, 5)
        self.assertEqual(result, {self.dates[0], self.dates[5]})

    def test_selects_last_actual_trading_day_of_each_friday_ending_week(self):
        dates = pd.to_datetime(
            [
                "2024-01-02",
                "2024-01-03",
                "2024-01-04",
                "2024-01-05",
                "2024-01-08",
                "2024-01-09",
                "2024-01-10",
                "2024-01-11",
            ]
        )

        result = get_rebalance_dates(dates, "W-FRI")

        self.assertEqual(
            result,
            {pd.Timestamp("2024-01-05"), pd.Timestamp("2024-01-11")},
        )

    def test_integer_schedule_is_anchored_on_first_sample_date(self):
        dates = pd.to_datetime(
            ["2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"]
        )

        result = get_rebalance_dates(dates, 2)

        self.assertEqual(
            result,
            {pd.Timestamp("2024-01-03"), pd.Timestamp("2024-01-05")},
        )

    def test_rejects_invalid_frequency(self):
        for freq in [0, -1, "W-SUN", "MONTHLY"]:
            with self.subTest(freq=freq):
                with self.assertRaises(ValueError):
                    get_rebalance_dates(self.dates, freq)


if __name__ == "__main__":
    unittest.main()
