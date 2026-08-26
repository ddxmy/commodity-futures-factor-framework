import unittest

import pandas as pd


class DriftAwareBacktestTest(unittest.TestCase):
    def test_missing_open_carries_position_without_trading(self):
        from src.basis_momentum_weight_drift import run_drift_aware_backtest

        dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
        weights = pd.DataFrame(
            {
                "trade_date": dates,
                "fut_code": ["A"] * 3,
                "weight": [0.05, 0.06, 0.06],
                "is_rebalance": [True] * 3,
                "ts_code_A": ["A2405"] * 3,
            }
        )
        prices = pd.DataFrame(
            {
                "trade_date": dates,
                "ts_code": ["A2405"] * 3,
                "open": [100.0, 100.0, float("nan")],
                "close": [100.0, 100.0, 100.0],
            }
        )

        _, _, _, daily, positions = run_drift_aware_backtest(
            weights,
            prices,
            cost_rate=0.0,
            return_positions=True,
        )

        execution = positions.loc[
            positions["trade_date"].eq(pd.Timestamp("2024-01-04"))
        ].iloc[0]
        self.assertTrue(execution["blocked_trade"])
        self.assertAlmostEqual(execution["turnover"], 0.0)
        self.assertAlmostEqual(
            daily.loc[pd.Timestamp("2024-01-04"), "daily_return"],
            0.0,
        )

    def test_rebalances_from_drifted_open_weight(self):
        from src.basis_momentum_weight_drift import run_drift_aware_backtest

        dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
        weights = pd.DataFrame(
            {
                "trade_date": dates,
                "fut_code": ["A"] * 3,
                "weight": [0.05, 0.06, 0.06],
                "is_rebalance": [True] * 3,
                "ts_code_A": ["A2405"] * 3,
            }
        )
        prices = pd.DataFrame(
            {
                "trade_date": dates,
                "ts_code": ["A2405"] * 3,
                "open": [100.0, 100.0, 102.0],
                "close": [100.0, 100.0, 102.0],
            }
        )

        _, _, _, daily, positions = run_drift_aware_backtest(
            weights,
            prices,
            cost_rate=0.0,
            return_positions=True,
        )

        execution = positions.loc[
            positions["trade_date"].eq(pd.Timestamp("2024-01-04"))
        ].iloc[0]
        expected_pretrade_weight = 0.051 / 1.001
        expected_turnover = 0.06 - expected_pretrade_weight
        self.assertAlmostEqual(
            execution["pretrade_weight"],
            expected_pretrade_weight,
        )
        self.assertAlmostEqual(execution["turnover"], expected_turnover)
        self.assertAlmostEqual(
            daily.loc[pd.Timestamp("2024-01-04"), "turnover"],
            expected_turnover,
        )

    def test_roll_counts_close_and_reopen_notional(self):
        from src.basis_momentum_weight_drift import run_drift_aware_backtest

        dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
        weights = pd.DataFrame(
            {
                "trade_date": dates,
                "fut_code": ["A"] * 3,
                "weight": [0.05, 0.05, 0.05],
                "is_rebalance": [True] * 3,
                "ts_code_A": ["A2405", "A2409", "A2409"],
            }
        )
        prices = pd.DataFrame(
            {
                "trade_date": [
                    dates[0],
                    dates[1],
                    dates[1],
                    dates[2],
                    dates[2],
                ],
                "ts_code": ["A2405", "A2405", "A2409", "A2405", "A2409"],
                "open": [100.0, 100.0, 200.0, 100.0, 200.0],
                "close": [100.0, 100.0, 200.0, 100.0, 200.0],
            }
        )

        _, _, _, daily, _ = run_drift_aware_backtest(
            weights,
            prices,
            cost_rate=0.0,
            return_positions=True,
        )

        self.assertAlmostEqual(
            daily.loc[pd.Timestamp("2024-01-04"), "turnover"],
            0.10,
        )
        self.assertAlmostEqual(
            daily.loc[pd.Timestamp("2024-01-04"), "actual_roll_turnover"],
            0.10,
        )


if __name__ == "__main__":
    unittest.main()
