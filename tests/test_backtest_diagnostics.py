import unittest
from unittest.mock import patch

import pandas as pd

from src import p06_backtest_engine as backtest_engine


class BacktestDiagnosticsTest(unittest.TestCase):
    def test_tiny_target_weight_closes_position_to_exact_zero(self):
        targets = pd.DataFrame(
            {
                "trade_date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
                "fut_code": ["A", "A"],
                "desired_exec_weight": [0.04, -1e-12],
                "desired_trade_ts_code": ["A2405.DCE", "A2405.DCE"],
            }
        )
        prices = pd.DataFrame(
            {
                "trade_date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
                "ts_code": ["A2405.DCE", "A2405.DCE"],
                "open": [100.0, 101.0],
            }
        )

        result = backtest_engine.resolve_executed_positions(targets, prices)

        self.assertEqual(result.loc[1, "exec_weight"], 0.0)
        self.assertIsNone(result.loc[1, "trade_ts_code"])

    @patch("src.p06_backtest_engine.load_contract_prices")
    @patch("src.p06_backtest_engine.generate_weights")
    def test_run_backtest_does_not_trade_when_open_is_missing(
        self,
        mock_generate_weights,
        mock_load_contract_prices,
    ):
        trade_dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
        mock_generate_weights.return_value = pd.DataFrame(
            {
                "trade_date": trade_dates,
                "fut_code": ["A", "A", "A"],
                "weight": [0.04, 0.06, 0.06],
                "is_rebalance": [True, True, True],
                "ts_code_A": ["A2405.DCE", "A2405.DCE", "A2405.DCE"],
            }
        )
        mock_load_contract_prices.return_value = pd.DataFrame(
            {
                "trade_date": trade_dates,
                "ts_code": ["A2405.DCE", "A2405.DCE", "A2405.DCE"],
                "open": [99.0, 100.0, float("nan")],
                "close": [99.0, 101.0, 101.0],
                "prev_close": [98.0, 99.0, 101.0],
            }
        )

        _, _, _, diagnostics = backtest_engine.run_backtest(
            start_date="20240102",
            end_date="20240104",
            rebalance_freq=1,
            return_diagnostics=True,
        )

        blocked_day = diagnostics.loc[pd.Timestamp("2024-01-04")]
        self.assertAlmostEqual(blocked_day["turnover"], 0.0)
        self.assertAlmostEqual(blocked_day["daily_return"], 0.0)
        self.assertEqual(blocked_day["blocked_trade"], 1)

    def test_missing_open_carries_actual_position_until_trade_is_possible(self):
        resolve_positions = getattr(
            backtest_engine,
            "resolve_executed_positions",
            None,
        )
        self.assertIsNotNone(resolve_positions)

        targets = pd.DataFrame(
            {
                "trade_date": pd.to_datetime(
                    ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
                ),
                "fut_code": ["A", "A", "A", "A"],
                "desired_exec_weight": [0.04, 0.06, 0.03, 0.03],
                "desired_trade_ts_code": [
                    "A2405.DCE",
                    "A2405.DCE",
                    "A2409.DCE",
                    "A2409.DCE",
                ],
            }
        )
        prices = pd.DataFrame(
            {
                "trade_date": pd.to_datetime(
                    [
                        "2024-01-02",
                        "2024-01-03",
                        "2024-01-04",
                        "2024-01-04",
                        "2024-01-05",
                        "2024-01-05",
                    ]
                ),
                "ts_code": [
                    "A2405.DCE",
                    "A2405.DCE",
                    "A2405.DCE",
                    "A2409.DCE",
                    "A2405.DCE",
                    "A2409.DCE",
                ],
                "open": [100.0, float("nan"), 101.0, float("nan"), 102.0, 200.0],
            }
        )

        result = resolve_positions(targets, prices)

        self.assertEqual(result["exec_weight"].tolist(), [0.04, 0.04, 0.04, 0.03])
        self.assertEqual(
            result["trade_ts_code"].tolist(),
            ["A2405.DCE", "A2405.DCE", "A2405.DCE", "A2409.DCE"],
        )
        self.assertEqual(result["blocked_trade"].tolist(), [False, True, True, False])
        self.assertEqual(result["delayed_roll"].tolist(), [False, False, True, False])

    def test_aggregate_daily_diagnostics(self):
        aggregate = getattr(
            backtest_engine,
            "aggregate_daily_diagnostics",
            None,
        )
        self.assertIsNotNone(aggregate)

        position_results = pd.DataFrame(
            {
                "trade_date": pd.to_datetime(
                    ["2024-01-02", "2024-01-02", "2024-01-03"]
                ),
                "gross_pnl": [0.010, -0.004, 0.002],
                "turnover": [0.4, 0.6, 0.1],
                "cost": [0.00020, 0.00030, 0.00005],
            }
        )

        daily = aggregate(position_results)

        self.assertAlmostEqual(daily.loc["2024-01-02", "gross_return"], 0.006)
        self.assertAlmostEqual(daily.loc["2024-01-02", "turnover"], 1.0)
        self.assertAlmostEqual(daily.loc["2024-01-02", "cost"], 0.0005)
        self.assertAlmostEqual(daily.loc["2024-01-02", "daily_return"], 0.0055)
        self.assertAlmostEqual(daily.loc["2024-01-03", "daily_return"], 0.00195)

    def test_actual_turnover_categories_are_additive(self):
        position_results = pd.DataFrame(
            {
                "turnover": [0.01, 0.07, 0.05],
                "prev_trade_ts_code": [
                    "A2405.DCE",
                    "B2405.DCE",
                    "C2405.DCE",
                ],
                "trade_ts_code": [
                    "A2405.DCE",
                    "B2409.DCE",
                    None,
                ],
                "mandatory_exit_requested": [False, False, True],
            }
        )

        result = backtest_engine.classify_actual_turnover(
            position_results
        )

        self.assertAlmostEqual(
            result.loc[0, "actual_active_turnover"],
            0.01,
        )
        self.assertAlmostEqual(
            result.loc[1, "actual_roll_turnover"],
            0.07,
        )
        self.assertAlmostEqual(
            result.loc[2, "actual_mandatory_exit_turnover"],
            0.05,
        )

        category_total = result[
            [
                "actual_active_turnover",
                "actual_roll_turnover",
                "actual_mandatory_exit_turnover",
            ]
        ].sum(axis=1)
        pd.testing.assert_series_equal(
            category_total,
            result["actual_total_turnover"],
            check_names=False,
        )


class OptimizedExecutionPathTest(unittest.TestCase):
    def _build_signals(self, day_two_scores=(1.0, -1.0)):
        dates = pd.to_datetime(
            ["2024-01-02", "2024-01-03", "2024-01-04"]
        )

        return pd.DataFrame(
            {
                "trade_date": dates.repeat(2),
                "fut_code": ["A", "B"] * 3,
                "factor": [
                    1.0,
                    -1.0,
                    day_two_scores[0],
                    day_two_scores[1],
                    day_two_scores[0],
                    day_two_scores[1],
                ],
                "passes_liquidity": [True] * 6,
                "is_rebalance": [True] * 6,
                "ts_code_A": ["A2405.DCE", "B2405.DCE"] * 3,
            }
        )

    def _build_prices(self, day_two_open=100.0):
        dates = pd.to_datetime(
            ["2024-01-02", "2024-01-03", "2024-01-04"]
        )

        return pd.DataFrame(
            {
                "trade_date": dates.repeat(2),
                "ts_code": ["A2405.DCE", "B2405.DCE"] * 3,
                "open": [
                    100.0,
                    100.0,
                    day_two_open,
                    day_two_open,
                    100.0,
                    100.0,
                ],
            }
        )

    def test_signal_executes_at_next_open(self):
        build_path = getattr(
            backtest_engine,
            "build_optimized_execution_path",
            None,
        )
        self.assertIsNotNone(build_path)

        result, _ = build_path(
            self._build_signals(),
            self._build_prices(),
        )

        day_one = result[
            result["trade_date"] == pd.Timestamp("2024-01-02")
        ].set_index("fut_code")
        day_two = result[
            result["trade_date"] == pd.Timestamp("2024-01-03")
        ].set_index("fut_code")

        self.assertTrue(day_one["exec_weight"].eq(0.0).all())
        self.assertGreater(day_two.loc["A", "exec_weight"], 0.0)
        self.assertLess(day_two.loc["B", "exec_weight"], 0.0)

    def test_blocked_trade_feeds_actual_position_into_next_signal(self):
        build_path = getattr(
            backtest_engine,
            "build_optimized_execution_path",
            None,
        )
        self.assertIsNotNone(build_path)

        result, optimizer_diagnostics = build_path(
            self._build_signals(day_two_scores=(-1.0, 1.0)),
            self._build_prices(day_two_open=float("nan")),
        )

        day_two = result[
            result["trade_date"] == pd.Timestamp("2024-01-03")
        ].set_index("fut_code")
        day_three = result[
            result["trade_date"] == pd.Timestamp("2024-01-04")
        ].set_index("fut_code")

        self.assertTrue(day_two["exec_weight"].eq(0.0).all())
        self.assertTrue(day_two["blocked_trade"].all())
        self.assertLess(day_three.loc["A", "exec_weight"], 0.0)
        self.assertGreater(day_three.loc["B", "exec_weight"], 0.0)
        self.assertAlmostEqual(
            optimizer_diagnostics.loc[
                pd.Timestamp("2024-01-03"),
                "effective_turnover_limit",
            ],
            1.0,
        )

    @patch("src.p06_backtest_engine.load_contract_prices")
    @patch("src.p06_backtest_engine.generate_weights")
    def test_run_backtest_uses_optimizer_targets_on_next_day(
        self,
        mock_generate_weights,
        mock_load_contract_prices,
    ):
        dates = pd.to_datetime(
            ["2024-01-02", "2024-01-03", "2024-01-04"]
        )
        mock_generate_weights.return_value = pd.DataFrame(
            {
                "trade_date": dates.repeat(2),
                "fut_code": ["A", "B"] * 3,
                "factor": [1.0, -1.0] * 3,
                "weight": [0.5, -0.5] * 3,
                "passes_liquidity": [True] * 6,
                "is_rebalance": [True] * 6,
                "ts_code_A": ["A2405.DCE", "B2405.DCE"] * 3,
            }
        )
        mock_load_contract_prices.return_value = pd.DataFrame(
            {
                "trade_date": dates.repeat(2),
                "ts_code": ["A2405.DCE", "B2405.DCE"] * 3,
                "open": [100.0, 100.0, 100.0, 100.0, 101.0, 99.0],
                "close": [100.0, 100.0, 101.0, 99.0, 101.0, 99.0],
                "prev_close": [100.0, 100.0, 100.0, 100.0, 101.0, 99.0],
            }
        )

        _, metrics, _, diagnostics = backtest_engine.run_backtest(
            start_date="20240102",
            end_date="20240104",
            rebalance_freq=1,
            use_optimizer=True,
            return_diagnostics=True,
        )

        self.assertAlmostEqual(
            diagnostics.loc[pd.Timestamp("2024-01-02"), "turnover"],
            0.0,
        )
        self.assertGreater(
            diagnostics.loc[pd.Timestamp("2024-01-03"), "gross_return"],
            0.0,
        )
        self.assertGreater(
            diagnostics.loc[pd.Timestamp("2024-01-03"), "turnover"],
            0.0,
        )
        execution_day = diagnostics.loc[
            pd.Timestamp("2024-01-03")
        ]
        classified_turnover = (
            execution_day["actual_active_turnover"]
            + execution_day["actual_roll_turnover"]
            + execution_day["actual_mandatory_exit_turnover"]
        )
        self.assertAlmostEqual(
            classified_turnover,
            execution_day["actual_total_turnover"],
        )
        self.assertAlmostEqual(
            execution_day["cost"],
            execution_day["actual_total_turnover"] * 0.0005,
        )
        self.assertGreater(
            diagnostics.loc[
                pd.Timestamp("2024-01-02"),
                "optimizer_budget_turnover",
            ],
            0.0,
        )
        self.assertIn("annual_active_turnover", metrics)
        self.assertIn("optimizer_binding_rate", metrics)


if __name__ == "__main__":
    unittest.main()
