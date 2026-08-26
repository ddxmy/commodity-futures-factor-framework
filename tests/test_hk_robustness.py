import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import pandas as pd

from config.research_config import ResearchConfig
from src.hk_robustness import (
    build_robustness_summary,
    calculate_hk_robustness,
    parse_arguments,
    run_hk_robustness,
    run_robustness_cell,
    save_robustness_results,
    validate_grid_values,
)


def small_details() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "K": 30,
                "H": 1,
                "net_sharpe": 0.1,
                "mean_rank_ic": 0.01,
                "annual_return": 0.02,
            },
            {
                "K": 30,
                "H": 5,
                "net_sharpe": 0.2,
                "mean_rank_ic": 0.02,
                "annual_return": 0.03,
            },
            {
                "K": 60,
                "H": 1,
                "net_sharpe": 0.3,
                "mean_rank_ic": 0.03,
                "annual_return": 0.04,
            },
            {
                "K": 60,
                "H": 5,
                "net_sharpe": 0.4,
                "mean_rank_ic": 0.04,
                "annual_return": 0.05,
            },
        ]
    )


class GridValidationTest(unittest.TestCase):
    def test_accepts_positive_unique_integer_values(self):
        self.assertEqual(
            validate_grid_values([30, 60, 90, 120], "K"),
            (30, 60, 90, 120),
        )

    def test_rejects_empty_duplicate_boolean_and_nonpositive_values(self):
        invalid = [[], [30, 30], [True, 30], [0, 30], [-1, 30], [30.0]]
        for values in invalid:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    validate_grid_values(values, "K")


class RobustnessCalculationTest(unittest.TestCase):
    @patch("src.hk_robustness.summarize_ic_statistics")
    @patch("src.hk_robustness.calculate_ic_series")
    @patch("src.hk_robustness.build_factor_test_panel")
    @patch("src.hk_robustness.run_backtest_from_weights")
    @patch("src.hk_robustness.build_target_weights")
    def test_cell_reuses_rank_weights_and_extracts_headline_metrics(
        self,
        mock_weights,
        mock_backtest,
        mock_panel,
        mock_ic_series,
        mock_ic_summary,
    ):
        dates = pd.to_datetime(["2022-01-04", "2022-01-05"])
        weights = pd.DataFrame(
            {
                "trade_date": dates,
                "fut_code": ["RB", "RB"],
                "weight": [0.5, 0.5],
            }
        )
        mock_weights.return_value = weights
        mock_backtest.return_value = (
            pd.Series([1.0, 1.1], index=dates),
            {
                "annual_return": 0.05,
                "sharpe": 0.8,
                "max_drawdown": -0.1,
                "annual_turnover": 2.0,
            },
            pd.DataFrame(),
        )
        mock_panel.return_value = pd.DataFrame(
            {
                "signal_date": [dates[0], dates[0]],
                "fut_code": ["A", "B"],
                "raw_factor": [1.0, -1.0],
                "forward_return": [0.02, -0.01],
            }
        )
        mock_ic_series.return_value = pd.DataFrame(
            {"ic": [0.01], "rank_ic": [0.03]}
        )
        mock_ic_summary.return_value = pd.DataFrame(
            {
                "metric": ["ic", "rank_ic"],
                "observations": [5, 5],
                "mean": [0.01, 0.03],
            }
        )

        result = run_robustness_cell(
            factor_data=pd.DataFrame(),
            contract_data=pd.DataFrame(),
            trade_calendar=pd.DataFrame(),
            prices=pd.DataFrame(),
            config=ResearchConfig(rebalance_freq=5),
            k=90,
            h=5,
        )

        self.assertEqual(result["K"], 90)
        self.assertEqual(result["H"], 5)
        self.assertAlmostEqual(result["total_return"], 0.10)
        self.assertAlmostEqual(result["net_sharpe"], 0.8)
        self.assertAlmostEqual(result["mean_rank_ic"], 0.03)
        self.assertIs(
            mock_panel.call_args.kwargs["prepared_weights"],
            weights,
        )
        self.assertEqual(
            mock_ic_summary.call_args.kwargs["annualization_periods"],
            252.0 / 5.0,
        )

    @patch("src.hk_robustness.load_factor_module")
    @patch("src.hk_robustness.run_robustness_cell")
    @patch("src.hk_robustness.prepare_contract_context")
    @patch("src.hk_robustness.calculate_factor")
    @patch("src.hk_robustness.load_contract_prices")
    @patch("src.hk_robustness.load_trade_calendar")
    def test_calculates_each_k_once_and_returns_twelve_cells(
        self,
        mock_calendar,
        mock_prices,
        mock_factor,
        mock_context,
        mock_cell,
        mock_module_loader,
    ):
        calendar = pd.DataFrame(
            {
                "trade_date": pd.to_datetime(
                    ["2022-01-04", "2022-01-05"]
                )
            }
        )
        mock_calendar.return_value = calendar
        mock_prices.return_value = pd.DataFrame()
        mock_factor.side_effect = lambda **kwargs: pd.DataFrame(
            {
                "trade_date": calendar["trade_date"],
                "fut_code": ["RB", "RB"],
                "raw_factor": [0.1, 0.2],
            }
        )
        mock_context.return_value = pd.DataFrame()
        mock_cell.side_effect = lambda **kwargs: {
            "K": kwargs["k"],
            "H": kwargs["h"],
            "trading_days": 100,
            "ic_observations": 20,
            "total_return": 0.1,
            "annual_return": 0.05,
            "net_sharpe": 0.8,
            "max_drawdown": -0.1,
            "annual_turnover": 2.0,
            "mean_ic": 0.01,
            "mean_rank_ic": 0.02,
        }
        module = Mock()
        module.DEFAULT_PARAMETERS = {"lookback": 90}
        mock_module_loader.return_value = module

        result = calculate_hk_robustness(
            "carry",
            "20220101",
            "20251231",
            [30, 60, 90, 120],
            [1, 5, 10],
            base_config=ResearchConfig(
                signal_min_days_to_maturity=5,
                trade_min_days_to_maturity=60,
            ),
        )

        self.assertEqual(len(result), 12)
        self.assertEqual(
            list(result[["K", "H"]].itertuples(index=False, name=None)),
            [
                (30, 1),
                (30, 5),
                (30, 10),
                (60, 1),
                (60, 5),
                (60, 10),
                (90, 1),
                (90, 5),
                (90, 10),
                (120, 1),
                (120, 5),
                (120, 10),
            ],
        )
        self.assertEqual(mock_factor.call_count, 4)
        self.assertEqual(mock_cell.call_count, 12)
        self.assertEqual(mock_calendar.call_count, 1)
        self.assertEqual(mock_prices.call_count, 1)
        self.assertEqual(
            [call.kwargs["parameters"] for call in mock_factor.call_args_list],
            [
                {"lookback": 30, "signal_min_days_to_maturity": 5},
                {"lookback": 60, "signal_min_days_to_maturity": 5},
                {"lookback": 90, "signal_min_days_to_maturity": 5},
                {"lookback": 120, "signal_min_days_to_maturity": 5},
            ],
        )
        self.assertEqual(mock_context.call_count, 4)
        self.assertTrue(
            all(
                call.kwargs["min_days_to_maturity"] == 60
                for call in mock_context.call_args_list
            )
        )


class RobustnessSummaryTest(unittest.TestCase):
    def test_builds_one_row_per_k_with_h_specific_metric_columns(self):
        summary = build_robustness_summary(small_details(), [1, 5])

        self.assertEqual(
            summary.columns.tolist(),
            [
                "K",
                "net_sharpe_H1",
                "net_sharpe_H5",
                "mean_rank_ic_H1",
                "mean_rank_ic_H5",
                "annual_return_H1",
                "annual_return_H5",
            ],
        )
        self.assertEqual(summary["K"].tolist(), [30, 60])
        self.assertAlmostEqual(summary.loc[1, "net_sharpe_H5"], 0.4)

    def test_rejects_incomplete_or_duplicate_grids(self):
        details = small_details()
        cases = [
            details.iloc[:-1].copy(),
            pd.concat([details, details.iloc[[0]]], ignore_index=True),
        ]
        for case in cases:
            with self.subTest(rows=len(case)):
                with self.assertRaises(ValueError):
                    build_robustness_summary(case, [1, 5])


class RobustnessOutputTest(unittest.TestCase):
    def test_saves_two_tables_and_three_nonempty_heatmaps(self):
        expected = {
            "robustness_details.csv",
            "robustness_summary.csv",
            "net_sharpe_heatmap.png",
            "rank_ic_heatmap.png",
            "annual_return_heatmap.png",
        }
        with TemporaryDirectory() as directory:
            output = Path(directory) / "report"

            result = save_robustness_results(
                small_details(),
                output,
                k_values=[30, 60],
                h_values=[1, 5],
            )

            self.assertEqual(result, output)
            self.assertEqual(
                {path.name for path in output.iterdir()},
                expected,
            )
            self.assertTrue(
                all((output / name).stat().st_size > 0 for name in expected)
            )

    def test_refuses_to_overwrite_any_existing_target(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "report"
            output.mkdir()
            existing = output / "robustness_details.csv"
            existing.write_text("original", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                save_robustness_results(
                    small_details(),
                    output,
                    k_values=[30, 60],
                    h_values=[1, 5],
                )

            self.assertEqual(
                {path.name for path in output.iterdir()},
                {"robustness_details.csv"},
            )
            self.assertEqual(existing.read_text(encoding="utf-8"), "original")


class RobustnessCliTest(unittest.TestCase):
    def test_defaults_match_the_approved_lightweight_carry_grid(self):
        args = parse_arguments([])

        self.assertEqual(args.factor, "carry")
        self.assertEqual(args.start, "20220101")
        self.assertEqual(args.end, "20251231")
        self.assertEqual(args.k_values, [30, 60, 90, 120])
        self.assertEqual(args.h_values, [1, 5, 10])
        self.assertEqual(args.signal_min_days_to_maturity, 0)
        self.assertEqual(args.trade_min_days_to_maturity, 45)

    def test_parses_separate_maturity_cutoffs(self):
        args = parse_arguments(
            [
                "--signal-min-days-to-maturity",
                "10",
                "--trade-min-days-to-maturity",
                "50",
            ]
        )

        self.assertEqual(args.signal_min_days_to_maturity, 10)
        self.assertEqual(args.trade_min_days_to_maturity, 50)

    @patch("src.hk_robustness.save_robustness_results")
    @patch("src.hk_robustness.calculate_hk_robustness")
    def test_runner_uses_dedicated_result_directory(
        self,
        mock_calculate,
        mock_save,
    ):
        mock_calculate.return_value = small_details()
        mock_save.side_effect = (
            lambda details, output_dir, **kwargs: Path(output_dir)
        )

        output = run_hk_robustness(
            factor_name="carry",
            start_date="20220101",
            end_date="20251231",
            k_values=[30, 60, 90, 120],
            h_values=[1, 5, 10],
            result_dir=Path("results"),
        )

        self.assertEqual(
            output,
            Path("results")
            / "carry"
            / (
                "carry-robustness-20220101-20251231"
                "-K30_60_90_120-H1_5_10"
            ),
        )

    @patch("src.hk_robustness.save_robustness_results")
    @patch("src.hk_robustness.calculate_hk_robustness")
    def test_runner_uses_distinct_directories_for_distinct_grids(
        self,
        mock_calculate,
        mock_save,
    ):
        mock_calculate.return_value = small_details()
        mock_save.side_effect = (
            lambda details, output_dir, **kwargs: Path(output_dir)
        )

        first = run_hk_robustness(
            factor_name="carry",
            start_date="20220101",
            end_date="20251231",
            k_values=[30, 60],
            h_values=[1, 5],
            result_dir=Path("results"),
        )
        second = run_hk_robustness(
            factor_name="carry",
            start_date="20220101",
            end_date="20251231",
            k_values=[5, 10, 20],
            h_values=[1, 5],
            result_dir=Path("results"),
        )

        self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
