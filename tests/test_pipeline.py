import argparse
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

from config.research_config import ResearchConfig
from main import (
    parse_arguments,
    parse_factor_parameters,
    parse_rebalance_frequency,
    validate_dates,
)
from src.research_pipeline import (
    build_factor_label,
    build_output_directory,
    evaluate_factor_from_data,
    run_factor_research,
)


class MainParameterTest(unittest.TestCase):
    def test_parses_typed_factor_parameters(self):
        result = parse_factor_parameters(
            ["lookback=252", "variant=AB", "lag=1.5", "enabled=true"]
        )
        self.assertEqual(
            result,
            {"lookback": 252, "variant": "AB", "lag": 1.5, "enabled": True},
        )

    def test_rejects_bad_parameters_and_dates(self):
        with self.assertRaises(ValueError):
            parse_factor_parameters(["lookback"])
        with self.assertRaises(ValueError):
            parse_factor_parameters(["x=1", "x=2"])
        with self.assertRaises(ValueError):
            validate_dates("20240102", "20240101")

    def test_parses_integer_and_weekly_rebalance_frequencies(self):
        self.assertEqual(parse_rebalance_frequency("1"), 1)
        self.assertEqual(parse_rebalance_frequency("5"), 5)
        self.assertEqual(parse_rebalance_frequency("w-fri"), "W-FRI")
        self.assertEqual(parse_arguments([]).rebalance_freq, 1)
        self.assertEqual(
            parse_arguments(["--rebalance-freq", "W-FRI"]).rebalance_freq,
            "W-FRI",
        )

    def test_parses_separate_signal_and_trade_maturity_cutoffs(self):
        defaults = parse_arguments([])
        selected = parse_arguments(
            [
                "--signal-min-days-to-maturity",
                "5",
                "--trade-min-days-to-maturity",
                "60",
            ]
        )

        self.assertEqual(defaults.signal_min_days_to_maturity, 0)
        self.assertEqual(defaults.trade_min_days_to_maturity, 45)
        self.assertEqual(selected.signal_min_days_to_maturity, 5)
        self.assertEqual(selected.trade_min_days_to_maturity, 60)

    def test_rejects_invalid_rebalance_frequency_text(self):
        for value in ["0", "-5", "W-SUN", "monthly"]:
            with self.subTest(value=value):
                with self.assertRaises(argparse.ArgumentTypeError):
                    parse_rebalance_frequency(value)


class OutputDirectoryTest(unittest.TestCase):
    def test_t_rank_label_includes_explicit_or_default_lookback(self):
        self.assertEqual(
            build_factor_label("t_rank", {"lookback": 20}),
            "t_rank_L20",
        )
        self.assertEqual(
            build_factor_label("t_rank", {}),
            "t_rank_L10",
        )

    def test_builds_frequency_specific_output_directories(self):
        root = Path("results")
        self.assertEqual(
            build_output_directory(
                root,
                "carry",
                "carry",
                "20220101",
                "20251231",
                1,
            ),
            root / "carry" / "carry-20220101-20251231" / "daily",
        )
        self.assertEqual(
            build_output_directory(
                root,
                "carry",
                "carry",
                "20220101",
                "20251231",
                "W-FRI",
            ),
            root
            / "carry"
            / "carry-20220101-20251231"
            / "weekly_last_trading_day",
        )
        self.assertEqual(
            build_output_directory(
                root,
                "carry",
                "carry",
                "20220101",
                "20251231",
                5,
            ),
            root
            / "carry"
            / "carry-20220101-20251231"
            / "every_5_trading_days",
        )
        self.assertEqual(
            build_output_directory(
                root,
                "basis_momentum",
                "AB_L252",
                "20190101",
                "20260710",
                1,
            ),
            root
            / "basis_momentum"
            / "AB_L252-20190101-20260710"
            / "daily",
        )


class ResearchPipelineTest(unittest.TestCase):
    @patch("src.research_pipeline.calculate_group_nav")
    @patch("src.research_pipeline.calculate_group_returns")
    @patch("src.research_pipeline.assign_five_groups")
    @patch("src.research_pipeline.summarize_ic_statistics")
    @patch("src.research_pipeline.calculate_ic_series")
    @patch("src.research_pipeline.build_factor_test_panel_from_data")
    def test_weekly_evaluation_passes_52_periods_to_ic_summary(
        self,
        mock_panel,
        mock_ic_series,
        mock_summary,
        mock_groups,
        mock_group_returns,
        mock_group_nav,
    ):
        mock_panel.return_value = pd.DataFrame(
            {"signal_date": [pd.Timestamp("2024-01-05")]}
        )
        mock_ic_series.return_value = pd.DataFrame(
            {
                "signal_date": [pd.Timestamp("2024-01-05")],
                "ic": [0.1],
                "rank_ic": [0.2],
            }
        )
        mock_summary.return_value = pd.DataFrame()
        mock_groups.return_value = pd.DataFrame()
        mock_group_returns.return_value = pd.DataFrame()
        mock_group_nav.return_value = pd.DataFrame()

        evaluate_factor_from_data(
            factor_data=pd.DataFrame(),
            contract_data=pd.DataFrame(),
            trade_calendar=pd.DataFrame(),
            prices=pd.DataFrame(),
            config=ResearchConfig(rebalance_freq="W-FRI"),
        )

        self.assertEqual(
            mock_summary.call_args.kwargs["annualization_periods"],
            52.0,
        )

    @patch("src.research_pipeline.plot_ic_history")
    @patch("src.research_pipeline.plot_group_nav")
    @patch("src.research_pipeline.plot_strategy_nav")
    @patch("src.research_pipeline.save_result_tables")
    @patch("src.research_pipeline.evaluate_factor_from_data")
    @patch("src.research_pipeline.run_strategy_comparison_from_data")
    @patch("src.research_pipeline.load_contract_prices")
    @patch("src.research_pipeline.load_trade_calendar")
    @patch("src.research_pipeline.calculate_factor")
    def test_calculates_factor_once_and_reuses_the_panel(
        self,
        mock_factor,
        mock_calendar,
        mock_prices,
        mock_strategy,
        mock_evaluation,
        mock_save,
        mock_plot_strategy,
        mock_plot_group,
        mock_plot_ic,
    ):
        dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
        factor_data = pd.DataFrame(
            {
                "trade_date": dates,
                "fut_code": ["RB", "RB"],
                "raw_factor": [1.0, 2.0],
                "ts_code_A": ["RB01", "RB01"],
                "avg_vol_A": [5000.0, 5000.0],
                "avg_oi_A": [30000.0, 30000.0],
                "avg_amount_A": [10000.0, 10000.0],
            }
        )
        mock_factor.return_value = factor_data
        mock_calendar.return_value = pd.DataFrame({"trade_date": dates})
        mock_prices.return_value = pd.DataFrame()
        strategy_metrics = pd.DataFrame({"name": ["demo_rank"]})
        strategy_nav = pd.DataFrame(
            {"demo_rank": [1.0, 1.01]}, index=dates
        )
        mock_strategy.return_value = (
            strategy_metrics,
            strategy_nav,
            pd.DataFrame(),
        )
        factor_results = {
            "ic_summary": pd.DataFrame(),
            "ic_series": pd.DataFrame(
                {"signal_date": dates, "ic": [0.1, 0.2], "rank_ic": [0.2, 0.1]}
            ),
            "group_returns": pd.DataFrame(),
            "group_nav": pd.DataFrame(),
        }
        mock_evaluation.return_value = factor_results

        with (
            patch(
                "src.research_pipeline.prepare_contract_context",
                return_value=factor_data,
            ) as mock_context,
            TemporaryDirectory() as directory,
        ):
            output = run_factor_research(
                "demo",
                "20240101",
                "20241231",
                {},
                ResearchConfig(
                    signal_min_days_to_maturity=5,
                    trade_min_days_to_maturity=60,
                ),
                Path(directory),
            )

        self.assertEqual(output.name, "daily")
        self.assertEqual(output.parent.name, "demo-20240101-20241231")
        self.assertEqual(output.parent.parent.name, "demo")
        mock_factor.assert_called_once_with(
            factor_name="demo",
            start_date="20240101",
            end_date="20241231",
            parameters={"signal_min_days_to_maturity": 5},
        )
        strategy_factor = mock_strategy.call_args.kwargs["factor_data"]
        evaluation_factor = mock_evaluation.call_args.kwargs["factor_data"]
        mock_context.assert_called_once()
        context_args = mock_context.call_args
        self.assertIs(context_args.args[0], strategy_factor)
        self.assertEqual(context_args.args[1:], ("20240101", "20241231"))
        self.assertEqual(
            context_args.kwargs,
            {"min_days_to_maturity": 60},
        )
        self.assertIs(strategy_factor, evaluation_factor)
        mock_save.assert_called_once()
        metadata = mock_save.call_args.args[1]
        self.assertEqual(metadata["signal_min_days_to_maturity"], 5)
        self.assertEqual(metadata["trade_min_days_to_maturity"], 60)
        mock_plot_strategy.assert_called_once()
        mock_plot_group.assert_called_once()
        mock_plot_ic.assert_called_once()


if __name__ == "__main__":
    unittest.main()
