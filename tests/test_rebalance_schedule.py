import unittest

import pandas as pd

from src.p04_rebalance_schedule import get_rebalance_dates


class RebalanceScheduleTest(unittest.TestCase):
    def setUp(self):
        self.dates = pd.bdate_range("2024-01-01", periods=10)

    def test_selects_every_nth_trading_day(self):
        result = get_rebalance_dates(self.dates, 5)
        self.assertEqual(result, {self.dates[0], self.dates[5]})

    def test_selects_requested_weekday(self):
        result = get_rebalance_dates(self.dates, "W-FRI")
        self.assertTrue(all(date.dayofweek == 4 for date in result))

    def test_rejects_invalid_frequency(self):
        for freq in [0, -1, "W-SUN", "MONTHLY"]:
            with self.subTest(freq=freq):
                with self.assertRaises(ValueError):
                    get_rebalance_dates(self.dates, freq)


if __name__ == "__main__":
    unittest.main()
