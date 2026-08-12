import unittest
from unittest.mock import patch

import pandas as pd

from src.p07_factor_evaluation import (
    assign_five_groups,
    attach_locked_forward_returns,
    build_execution_schedule,
    build_factor_test_panel,
    calculate_group_nav,
    calculate_ic_series,
    calculate_group_returns,
    summarize_ic_statistics,
    summarize_factor_periods,
)


class ExecutionScheduleTest(unittest.TestCase):
    def test_maps_signal_to_next_open_and_next_execution_open(self):
        trade_dates = pd.to_datetime(
            [
                "2024-01-05",
                "2024-01-08",
                "2024-01-09",
                "2024-01-12",
                "2024-01-15",
            ]
        )
        signal_dates = pd.to_datetime(
            ["2024-01-05", "2024-01-12"]
        )

        result = build_execution_schedule(
            trade_dates,
            signal_dates,
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(
            result.loc[0, "signal_date"],
            pd.Timestamp("2024-01-05"),
        )
        self.assertEqual(
            result.loc[0, "entry_date"],
            pd.Timestamp("2024-01-08"),
        )
        self.assertEqual(
            result.loc[0, "exit_date"],
            pd.Timestamp("2024-01-15"),
        )


class LockedForwardReturnTest(unittest.TestCase):
    def test_can_drop_an_observation_with_a_missing_execution_price(self):
        signals = pd.DataFrame(
            {
                "signal_date": pd.to_datetime(["2024-01-05", "2024-01-05"]),
                "fut_code": ["A", "B"],
                "trade_ts_code": ["A2405.DCE", "B2405.DCE"],
                "raw_factor": [2.0, -1.0],
            }
        )
        schedule = pd.DataFrame(
            {
                "signal_date": pd.to_datetime(["2024-01-05"]),
                "entry_date": pd.to_datetime(["2024-01-08"]),
                "exit_date": pd.to_datetime(["2024-01-15"]),
            }
        )
        prices = pd.DataFrame(
            {
                "trade_date": pd.to_datetime(
                    ["2024-01-08", "2024-01-15", "2024-01-08"]
                ),
                "ts_code": ["A2405.DCE", "A2405.DCE", "B2405.DCE"],
                "open": [100.0, 110.0, 200.0],
            }
        )

        result = attach_locked_forward_returns(
            signals,
            schedule,
            prices,
            drop_missing=True,
        )

        self.assertEqual(result["fut_code"].tolist(), ["A"])
        self.assertAlmostEqual(result.loc[0, "forward_return"], 0.10)

    def test_uses_same_locked_contract_for_entry_and_exit(self):
        signals = pd.DataFrame(
            {
                "signal_date": pd.to_datetime(
                    ["2024-01-05", "2024-01-05"]
                ),
                "fut_code": ["A", "B"],
                "trade_ts_code": ["A2405.DCE", "B2405.DCE"],
                "raw_factor": [2.0, -1.0],
            }
        )
        schedule = pd.DataFrame(
            {
                "signal_date": pd.to_datetime(["2024-01-05"]),
                "entry_date": pd.to_datetime(["2024-01-08"]),
                "exit_date": pd.to_datetime(["2024-01-15"]),
            }
        )
        prices = pd.DataFrame(
            {
                "trade_date": pd.to_datetime(
                    [
                        "2024-01-08",
                        "2024-01-15",
                        "2024-01-08",
                        "2024-01-15",
                    ]
                ),
                "ts_code": [
                    "A2405.DCE",
                    "A2405.DCE",
                    "B2405.DCE",
                    "B2405.DCE",
                ],
                "open": [100.0, 110.0, 200.0, 190.0],
            }
        )

        result = attach_locked_forward_returns(
            signals,
            schedule,
            prices,
        )

        entry_opens = result.set_index("fut_code")["entry_open"]
        self.assertAlmostEqual(entry_opens.loc["A"], 100.0)
        self.assertAlmostEqual(entry_opens.loc["B"], 200.0)

        exit_opens = result.set_index("fut_code")["exit_open"]
        self.assertAlmostEqual(exit_opens.loc["A"], 110.0)
        self.assertAlmostEqual(exit_opens.loc["B"], 190.0)

        returns = result.set_index("fut_code")["forward_return"]
        self.assertAlmostEqual(returns.loc["A"], 0.10)
        self.assertAlmostEqual(returns.loc["B"], -0.05)
        self.assertEqual(
            result["trade_ts_code"].tolist(),
            ["A2405.DCE", "B2405.DCE"],
        )

    def test_raises_when_an_execution_price_is_missing(self):
        signals = pd.DataFrame(
            {
                "signal_date": pd.to_datetime(["2024-01-05"]),
                "fut_code": ["A"],
                "trade_ts_code": ["A2405.DCE"],
                "raw_factor": [2.0],
            }
        )
        schedule = pd.DataFrame(
            {
                "signal_date": pd.to_datetime(["2024-01-05"]),
                "entry_date": pd.to_datetime(["2024-01-08"]),
                "exit_date": pd.to_datetime(["2024-01-15"]),
            }
        )
        prices = pd.DataFrame(
            {
                "trade_date": pd.to_datetime(["2024-01-08"]),
                "ts_code": ["A2405.DCE"],
                "open": [100.0],
            }
        )

        with self.assertRaises(ValueError):
            attach_locked_forward_returns(signals, schedule, prices)

    def test_raises_when_contract_price_is_duplicated(self):
        signals = pd.DataFrame(
            {
                "signal_date": pd.to_datetime(["2024-01-05"]),
                "fut_code": ["A"],
                "trade_ts_code": ["A2405.DCE"],
                "raw_factor": [2.0],
            }
        )
        schedule = pd.DataFrame(
            {
                "signal_date": pd.to_datetime(["2024-01-05"]),
                "entry_date": pd.to_datetime(["2024-01-08"]),
                "exit_date": pd.to_datetime(["2024-01-15"]),
            }
        )
        prices = pd.DataFrame(
            {
                "trade_date": pd.to_datetime(
                    ["2024-01-08", "2024-01-08", "2024-01-15"]
                ),
                "ts_code": ["A2405.DCE", "A2405.DCE", "A2405.DCE"],
                "open": [100.0, 101.0, 110.0],
            }
        )

        with self.assertRaises(pd.errors.MergeError):
            attach_locked_forward_returns(signals, schedule, prices)


class FactorTestPanelTest(unittest.TestCase):
    @patch("src.p07_factor_evaluation.load_contract_prices")
    @patch("src.p07_factor_evaluation.load_trade_calendar")
    @patch("src.p07_factor_evaluation.generate_weights")
    def test_recounts_assets_after_dropping_an_unexecutable_observation(
        self,
        mock_generate_weights,
        mock_calendar,
        mock_prices,
    ):
        signal_dates = pd.to_datetime(
            [
                "2024-01-05",
                "2024-01-05",
                "2024-01-05",
                "2024-01-08",
                "2024-01-08",
                "2024-01-08",
            ]
        )
        mock_generate_weights.return_value = pd.DataFrame(
            {
                "trade_date": signal_dates,
                "fut_code": ["A", "B", "C", "A", "B", "C"],
                "weight_factor": [3.0, 2.0, 1.0, 3.0, 2.0, 1.0],
                "ts_code_A": [
                    "A2405.DCE",
                    "B2405.DCE",
                    "C2405.DCE",
                    "A2405.DCE",
                    "B2405.DCE",
                    "C2405.DCE",
                ],
                "is_rebalance": [True] * 6,
            }
        )
        mock_calendar.return_value = pd.DataFrame(
            {
                "trade_date": pd.to_datetime(
                    ["2024-01-05", "2024-01-08", "2024-01-09"]
                )
            }
        )
        mock_prices.return_value = pd.DataFrame(
            {
                "trade_date": pd.to_datetime(
                    [
                        "2024-01-08",
                        "2024-01-09",
                        "2024-01-08",
                        "2024-01-09",
                        "2024-01-08",
                    ]
                ),
                "ts_code": [
                    "A2405.DCE",
                    "A2405.DCE",
                    "B2405.DCE",
                    "B2405.DCE",
                    "C2405.DCE",
                ],
                "open": [100.0, 110.0, 200.0, 190.0, 300.0],
            }
        )

        result = build_factor_test_panel(
            start_date="20240101",
            end_date="20240131",
            factor_type="AB",
            lookback=120,
            rebalance_freq=1,
            min_assets=2,
        )

        self.assertEqual(result["fut_code"].tolist(), ["A", "B"])
        self.assertTrue(result["asset_count"].eq(2).all())

    @patch("src.p07_factor_evaluation.load_contract_prices")
    @patch("src.p07_factor_evaluation.load_trade_calendar")
    @patch("src.p07_factor_evaluation.generate_weights")
    def test_uses_only_eligible_daily_factor_rows(
        self,
        mock_generate_weights,
        mock_calendar,
        mock_prices,
    ):
        mock_generate_weights.return_value = pd.DataFrame(
            {
                "trade_date": pd.to_datetime(
                    [
                        "2024-01-05",
                        "2024-01-05",
                        "2024-01-05",
                        "2024-01-08",
                        "2024-01-08",
                    ]
                ),
                "fut_code": ["B", "A", "C", "B", "A"],
                "weight_factor": [-1.0, 2.0, float("nan"), -2.0, 1.0],
                "ts_code_A": [
                    "B2405.DCE",
                    "A2405.DCE",
                    "C2405.DCE",
                    "B2405.DCE",
                    "A2405.DCE",
                ],
                "is_rebalance": [True, True, True, True, True],
            }
        )
        mock_calendar.return_value = pd.DataFrame(
            {
                "trade_date": pd.to_datetime(
                    ["2024-01-05", "2024-01-08", "2024-01-09"]
                )
            }
        )
        mock_prices.return_value = pd.DataFrame(
            {
                "trade_date": pd.to_datetime(
                    ["2024-01-08", "2024-01-09", "2024-01-08", "2024-01-09"]
                ),
                "ts_code": [
                    "A2405.DCE",
                    "A2405.DCE",
                    "B2405.DCE",
                    "B2405.DCE",
                ],
                "open": [100.0, 110.0, 200.0, 190.0],
                "close": [100.0, 110.0, 200.0, 190.0],
                "prev_close": [99.0, 100.0, 199.0, 200.0],
            }
        )

        result = build_factor_test_panel(
            start_date="20240101",
            end_date="20240131",
            factor_type="AB",
            lookback=120,
            rebalance_freq=1,
            min_assets=2,
        )

        self.assertTrue(result["asset_count"].eq(2).all())
        self.assertEqual(result["fut_code"].tolist(), ["A", "B"])
        self.assertEqual(result["raw_factor"].tolist(), [2.0, -1.0])
        self.assertAlmostEqual(result.loc[0, "forward_return"], 0.10)
        self.assertAlmostEqual(result.loc[1, "forward_return"], -0.05)
        mock_generate_weights.assert_called_once_with(
            start_date="20240101",
            end_date="20240131",
            factor_type="AB",
            lookback=120,
            normalize="rank",
            rebalance_freq=1,
        )

    def test_rejects_invalid_panel_parameters(self):
        for invalid_freq in [0, 21, 1.5]:
            with self.subTest(rebalance_freq=invalid_freq):
                with self.assertRaises(ValueError):
                    build_factor_test_panel(
                        start_date="20240101",
                        end_date="20240131",
                        factor_type="AB",
                        lookback=120,
                        rebalance_freq=invalid_freq,
                        min_assets=10,
                    )

        with self.assertRaises(ValueError):
            build_factor_test_panel(
                start_date="20240101",
                end_date="20240131",
                factor_type="AB",
                lookback=120,
                rebalance_freq=1,
                min_assets=1,
            )

    @patch("src.p07_factor_evaluation.load_contract_prices")
    @patch("src.p07_factor_evaluation.load_trade_calendar")
    @patch("src.p07_factor_evaluation.generate_weights")
    def test_rejects_duplicate_commodity_on_a_signal_date(
        self,
        mock_generate_weights,
        mock_calendar,
        mock_prices,
    ):
        mock_generate_weights.return_value = pd.DataFrame(
            {
                "trade_date": pd.to_datetime(
                    [
                        "2024-01-05",
                        "2024-01-05",
                        "2024-01-05",
                        "2024-01-08",
                        "2024-01-08",
                    ]
                ),
                "fut_code": ["A", "A", "B", "A", "B"],
                "weight_factor": [2.0, 2.0, -1.0, 1.0, -2.0],
                "ts_code_A": [
                    "A2405.DCE",
                    "A2405.DCE",
                    "B2405.DCE",
                    "A2405.DCE",
                    "B2405.DCE",
                ],
                "is_rebalance": [True, True, True, True, True],
            }
        )
        mock_calendar.return_value = pd.DataFrame(
            {
                "trade_date": pd.to_datetime(
                    ["2024-01-05", "2024-01-08", "2024-01-09"]
                )
            }
        )
        mock_prices.return_value = pd.DataFrame(
            {
                "trade_date": pd.to_datetime(
                    ["2024-01-08", "2024-01-09", "2024-01-08", "2024-01-09"]
                ),
                "ts_code": [
                    "A2405.DCE",
                    "A2405.DCE",
                    "B2405.DCE",
                    "B2405.DCE",
                ],
                "open": [100.0, 110.0, 200.0, 190.0],
                "close": [100.0, 110.0, 200.0, 190.0],
                "prev_close": [99.0, 100.0, 199.0, 200.0],
            }
        )

        with self.assertRaises(ValueError):
            build_factor_test_panel(
                start_date="20240101",
                end_date="20240131",
                factor_type="AB",
                lookback=120,
                rebalance_freq=1,
                min_assets=2,
            )


class ICSeriesTest(unittest.TestCase):
    def test_perfect_order_has_ic_and_rank_ic_of_one(self):
        panel = pd.DataFrame(
            {
                "signal_date": pd.to_datetime(["2024-01-05"] * 5),
                "fut_code": list("ABCDE"),
                "raw_factor": [5, 4, 3, 2, 1],
                "forward_return": [0.05, 0.04, 0.03, 0.02, 0.01],
            }
        )

        result = calculate_ic_series(panel)

        self.assertEqual(result.loc[0, "asset_count"], 5)
        self.assertAlmostEqual(result.loc[0, "ic"], 1.0)
        self.assertAlmostEqual(result.loc[0, "rank_ic"], 1.0)

    def test_opposite_order_has_ic_and_rank_ic_of_minus_one(self):
        panel = pd.DataFrame(
            {
                "signal_date": pd.to_datetime(["2024-01-05"] * 5),
                "fut_code": list("ABCDE"),
                "raw_factor": [5, 4, 3, 2, 1],
                "forward_return": [-0.05, -0.04, -0.03, -0.02, -0.01],
            }
        )

        result = calculate_ic_series(panel)

        self.assertAlmostEqual(result.loc[0, "ic"], -1.0)
        self.assertAlmostEqual(result.loc[0, "rank_ic"], -1.0)

    def test_constant_factor_returns_nan_correlations(self):
        panel = pd.DataFrame(
            {
                "signal_date": pd.to_datetime(["2024-01-05"] * 3),
                "fut_code": list("ABC"),
                "raw_factor": [1.0, 1.0, 1.0],
                "forward_return": [0.01, 0.02, 0.03],
            }
        )

        result = calculate_ic_series(panel)

        self.assertTrue(pd.isna(result.loc[0, "ic"]))
        self.assertTrue(pd.isna(result.loc[0, "rank_ic"]))


class ICStatisticsTest(unittest.TestCase):
    def test_summarizes_ic_and_rank_ic_with_significance_tests(self):
        ic_series = pd.DataFrame(
            {
                "signal_date": pd.to_datetime(
                    [
                        "2024-01-02",
                        "2024-01-03",
                        "2024-01-04",
                        "2024-01-05",
                    ]
                ),
                "ic": [0.01, 0.02, 0.03, 0.04],
                "rank_ic": [-0.04, -0.03, -0.02, -0.01],
            }
        )

        summary = summarize_ic_statistics(
            ic_series,
            nw_lags=1,
        ).set_index("metric")

        ic = summary.loc["ic"]
        rank_ic = summary.loc["rank_ic"]

        self.assertEqual(ic["observations"], 4)
        self.assertAlmostEqual(ic["mean"], 0.025)
        self.assertAlmostEqual(ic["std"], 0.012909944487358056)
        self.assertAlmostEqual(ic["icir_raw"], 1.9364916731037085)
        self.assertAlmostEqual(
            ic["icir_annualized"],
            1.9364916731037085 * (252 ** 0.5),
        )
        self.assertAlmostEqual(ic["positive_rate"], 1.0)
        self.assertAlmostEqual(ic["minimum"], 0.01)
        self.assertAlmostEqual(ic["maximum"], 0.04)
        self.assertAlmostEqual(ic["t_stat"], 3.872983346207417)
        self.assertAlmostEqual(ic["p_value"], 0.030466291662170984)
        self.assertAlmostEqual(ic["nw_t_stat"], 4.0)
        self.assertAlmostEqual(ic["nw_p_value"], 6.334248366623973e-05)
        self.assertEqual(ic["nw_lags"], 1)

        self.assertAlmostEqual(rank_ic["mean"], -0.025)
        self.assertAlmostEqual(rank_ic["positive_rate"], 0.0)
        self.assertAlmostEqual(rank_ic["t_stat"], -3.872983346207417)
        self.assertAlmostEqual(rank_ic["nw_t_stat"], -4.0)

    def test_constant_series_has_undefined_ratios_and_tests(self):
        ic_series = pd.DataFrame(
            {
                "signal_date": pd.to_datetime(
                    ["2024-01-02", "2024-01-03", "2024-01-04"]
                ),
                "ic": [0.02, 0.02, 0.02],
                "rank_ic": [-0.01, -0.01, -0.01],
            }
        )

        summary = summarize_ic_statistics(ic_series).set_index("metric")

        for metric in ["ic", "rank_ic"]:
            with self.subTest(metric=metric):
                row = summary.loc[metric]
                self.assertTrue(pd.isna(row["icir_raw"]))
                self.assertTrue(pd.isna(row["icir_annualized"]))
                self.assertTrue(pd.isna(row["t_stat"]))
                self.assertTrue(pd.isna(row["p_value"]))
                self.assertTrue(pd.isna(row["nw_t_stat"]))
                self.assertTrue(pd.isna(row["nw_p_value"]))

    def test_rejects_invalid_newey_west_lags(self):
        ic_series = pd.DataFrame(
            {
                "signal_date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
                "ic": [0.01, 0.02],
                "rank_ic": [0.02, 0.03],
            }
        )

        for invalid_lags in [-1, 1.5]:
            with self.subTest(nw_lags=invalid_lags):
                with self.assertRaises(ValueError):
                    summarize_ic_statistics(
                        ic_series,
                        nw_lags=invalid_lags,
                    )


class FiveGroupTest(unittest.TestCase):
    def test_assigns_highest_factor_to_g1_and_balances_counts(self):
        panel = pd.DataFrame(
            {
                "signal_date": pd.to_datetime(["2024-01-05"] * 11),
                "fut_code": [f"F{i:02d}" for i in range(11)],
                "raw_factor": [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
                "forward_return": [
                    0.10,
                    0.09,
                    0.08,
                    0.07,
                    0.06,
                    0.05,
                    0.04,
                    0.03,
                    0.02,
                    0.01,
                    0.00,
                ],
            }
        )

        grouped = assign_five_groups(panel)
        counts = grouped.groupby("group").size()

        highest_group = grouped.loc[
            grouped["raw_factor"].idxmax(),
            "group",
        ]
        lowest_group = grouped.loc[
            grouped["raw_factor"].idxmin(),
            "group",
        ]

        self.assertEqual(highest_group, 1)
        self.assertEqual(lowest_group, 5)
        self.assertLessEqual(counts.max() - counts.min(), 1)
        self.assertEqual(len(grouped), len(panel))

    def test_breaks_factor_ties_by_commodity_code(self):
        panel = pd.DataFrame(
            {
                "signal_date": pd.to_datetime(["2024-01-05"] * 5),
                "fut_code": ["B", "A", "E", "D", "C"],
                "raw_factor": [1.0, 1.0, -2.0, -1.0, 0.0],
                "forward_return": [0.01, 0.02, -0.02, -0.01, 0.0],
            }
        )

        grouped = assign_five_groups(panel)
        groups = grouped.set_index("fut_code")["group"]

        self.assertEqual(groups.loc["A"], 1)
        self.assertEqual(groups.loc["B"], 2)

    def test_rejects_dates_with_fewer_assets_than_groups(self):
        panel = pd.DataFrame(
            {
                "signal_date": pd.to_datetime(["2024-01-05"] * 4),
                "fut_code": list("ABCD"),
                "raw_factor": [4.0, 3.0, 2.0, 1.0],
                "forward_return": [0.04, 0.03, 0.02, 0.01],
            }
        )

        with self.assertRaises(ValueError):
            assign_five_groups(panel, group_count=5)

    def test_calculates_equal_weight_group_and_long_short_returns(self):
        grouped_panel = pd.DataFrame(
            {
                "signal_date": pd.to_datetime(["2024-01-05"] * 5),
                "fut_code": list("ABCDE"),
                "group": [1, 2, 3, 4, 5],
                "forward_return": [0.10, 0.05, 0.02, 0.00, -0.05],
            }
        )

        result = calculate_group_returns(grouped_panel)

        self.assertAlmostEqual(result.loc[0, "G1"], 0.10)
        self.assertAlmostEqual(result.loc[0, "G2"], 0.05)
        self.assertAlmostEqual(result.loc[0, "G3"], 0.02)
        self.assertAlmostEqual(result.loc[0, "G4"], 0.00)
        self.assertAlmostEqual(result.loc[0, "G5"], -0.05)
        self.assertAlmostEqual(result.loc[0, "spread_raw"], 0.15)
        self.assertAlmostEqual(result.loc[0, "long_g1"], 0.05)
        self.assertAlmostEqual(result.loc[0, "short_g5"], 0.025)
        self.assertAlmostEqual(result.loc[0, "long_short"], 0.075)


class FactorSummaryTest(unittest.TestCase):
    def test_group_nav_sorts_dates_and_compounds_returns(self):
        group_returns = pd.DataFrame(
            {
                "signal_date": pd.to_datetime(
                    ["2024-01-12", "2024-01-05"]
                ),
                "G1": [-0.10, 0.10],
                "G2": [0.0, 0.0],
                "G3": [0.0, 0.0],
                "G4": [0.0, 0.0],
                "G5": [0.0, 0.0],
                "spread_raw": [-0.10, 0.10],
                "long_g1": [-0.05, 0.05],
                "short_g5": [0.0, 0.0],
                "long_short": [-0.05, 0.05],
            }
        )

        nav = calculate_group_nav(group_returns)

        self.assertEqual(
            nav["signal_date"].tolist(),
            pd.to_datetime(["2024-01-05", "2024-01-12"]).tolist(),
        )
        self.assertAlmostEqual(nav.loc[0, "G1"], 1.10)
        self.assertAlmostEqual(nav.loc[1, "G1"], 0.99)
        self.assertAlmostEqual(nav.loc[0, "long_short"], 1.05)
        self.assertAlmostEqual(nav.loc[1, "long_short"], 0.9975)

    def test_summarizes_inclusive_named_periods(self):
        dates = pd.to_datetime(
            ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
        )
        ic_series = pd.DataFrame(
            {
                "signal_date": dates,
                "ic": [0.01, 0.02, 0.03, 0.04],
                "rank_ic": [0.04, 0.03, 0.02, 0.01],
            }
        )
        group_returns = pd.DataFrame(
            {
                "signal_date": dates,
                "G1": [0.10, 0.00, 0.20, -0.10],
                "G2": [0.05, 0.00, 0.10, -0.05],
                "G3": [0.00, 0.00, 0.00, 0.00],
                "G4": [-0.02, 0.00, -0.04, 0.02],
                "G5": [-0.05, 0.00, -0.10, 0.05],
                "spread_raw": [0.15, 0.00, 0.30, -0.15],
                "long_g1": [0.05, 0.00, 0.10, -0.05],
                "short_g5": [0.025, 0.00, 0.05, -0.025],
                "long_short": [0.075, 0.00, 0.15, -0.075],
            }
        )
        periods = {
            "前半段": ("2024-01-02", "2024-01-03"),
            "后半段": ("2024-01-04", "2024-01-05"),
        }

        ic_summary, performance_summary = summarize_factor_periods(
            ic_series,
            group_returns,
            periods,
            nw_lags=1,
        )

        early_ic = ic_summary[
            (ic_summary["period"] == "前半段")
            & (ic_summary["metric"] == "ic")
        ].iloc[0]
        late_g1 = performance_summary[
            (performance_summary["period"] == "后半段")
            & (performance_summary["portfolio"] == "G1")
        ].iloc[0]

        self.assertEqual(early_ic["observations"], 2)
        self.assertEqual(late_g1["observations"], 2)
        self.assertAlmostEqual(late_g1["total_return"], 0.08)
        self.assertAlmostEqual(
            late_g1["annual_volatility"],
            3.3674916480965478,
        )
        self.assertAlmostEqual(
            late_g1["sharpe"],
            3.7416573867739413,
        )
        self.assertAlmostEqual(late_g1["max_drawdown"], -0.10)


if __name__ == "__main__":
    unittest.main()
