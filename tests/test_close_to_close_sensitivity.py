import importlib.util
import unittest
import warnings

import pandas as pd

from src.close_to_close_sensitivity import run_close_to_close_backtest


class CloseToCloseModuleTest(unittest.TestCase):
    def test_sensitivity_module_is_available(self):
        self.assertIsNotNone(
            importlib.util.find_spec("src.close_to_close_sensitivity")
        )


class CloseToCloseBacktestTest(unittest.TestCase):
    def setUp(self):
        self.dates = pd.to_datetime(
            ["2024-01-02", "2024-01-03", "2024-01-04"]
        )
        self.weights = pd.DataFrame(
            {
                "trade_date": self.dates,
                "fut_code": ["RB", "RB", "RB"],
                "weight": [1.0, 1.0, 1.0],
                "ts_code_A": ["RB_A", "RB_A", "RB_A"],
                "is_rebalance": [True, True, True],
            }
        )
        self.prices = pd.DataFrame(
            {
                "trade_date": self.dates,
                "ts_code": ["RB_A", "RB_A", "RB_A"],
                "close": [100.0, 110.0, 121.0],
                "prev_close": [pd.NA, 100.0, 110.0],
            }
        )

    def test_same_close_earns_the_following_close_return(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            result = run_close_to_close_backtest(
                self.weights,
                self.prices,
                execution_lag=0,
                cost_rate=0.001,
            )

        self.assertEqual(len(result), 3)
        for actual, expected in zip(
            result["gross_return"], [0.0, 0.1, 0.1], strict=True
        ):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(result["turnover"].tolist(), [1.0, 0.0, 0.0])
        self.assertEqual(result["cost"].tolist(), [0.001, 0.0, 0.0])

    def test_next_close_delays_the_first_earned_return_by_one_day(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            result = run_close_to_close_backtest(
                self.weights,
                self.prices,
                execution_lag=1,
                cost_rate=0.001,
            )

        self.assertEqual(len(result), 3)
        for actual, expected in zip(
            result["gross_return"], [0.0, 0.0, 0.1], strict=True
        ):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(result["turnover"].tolist(), [0.0, 1.0, 0.0])
        self.assertEqual(result["cost"].tolist(), [0.0, 0.001, 0.0])

    def test_contract_roll_counts_close_and_reopen_turnover(self):
        weights = self.weights.iloc[:2].copy()
        weights["weight"] = [0.5, 0.5]
        weights["ts_code_A"] = ["RB_A", "RB_B"]
        prices = pd.DataFrame(
            {
                "trade_date": [self.dates[0], self.dates[1], self.dates[1]],
                "ts_code": ["RB_A", "RB_A", "RB_B"],
                "close": [100.0, 101.0, 200.0],
                "prev_close": [pd.NA, 100.0, 198.0],
            }
        )

        result = run_close_to_close_backtest(
            weights,
            prices,
            execution_lag=0,
            cost_rate=0.001,
        )

        self.assertAlmostEqual(result.loc[1, "gross_return"], 0.005)
        self.assertEqual(result.loc[1, "turnover"], 1.0)
        self.assertEqual(result.loc[1, "cost"], 0.001)


if __name__ == "__main__":
    unittest.main()
