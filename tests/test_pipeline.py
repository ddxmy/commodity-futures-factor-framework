import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

from config.research_config import ResearchConfig
from main import parse_factor_parameters, validate_dates
from src.research_pipeline import run_factor_research


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


class ResearchPipelineTest(unittest.TestCase):
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

        with TemporaryDirectory() as directory:
            output = run_factor_research(
                "demo",
                "20240101",
                "20241231",
                {},
                ResearchConfig(),
                Path(directory),
            )

        self.assertEqual(output.name, "demo-20240101-20241231")
        mock_factor.assert_called_once()
        strategy_factor = mock_strategy.call_args.kwargs["factor_data"]
        evaluation_factor = mock_evaluation.call_args.kwargs["factor_data"]
        self.assertIs(strategy_factor, evaluation_factor)
        mock_save.assert_called_once()
        mock_plot_strategy.assert_called_once()
        mock_plot_group.assert_called_once()
        mock_plot_ic.assert_called_once()


if __name__ == "__main__":
    unittest.main()
