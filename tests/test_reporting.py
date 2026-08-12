import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from src.p08_reporting import (
    calculate_rolling_ic,
    plot_group_nav,
    plot_ic_history,
    plot_strategy_nav,
    save_result_tables,
)


class ReportingTest(unittest.TestCase):
    def setUp(self):
        self.dates = pd.to_datetime(
            ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
        )
        self.strategy_metrics = pd.DataFrame(
            {"name": ["demo_rank", "demo_zscore"], "sharpe": [0.5, 0.8]}
        )
        self.strategy_nav = pd.DataFrame(
            {
                "demo_rank": [1.0, 1.01, 1.02, 1.03],
                "demo_zscore": [1.0, 1.02, 1.03, 1.05],
            },
            index=self.dates,
        )
        self.strategy_nav.index.name = "trade_date"

        groups = {
            "G1": [1.01, 1.02, 1.03, 1.04],
            "G2": [1.00, 1.01, 1.02, 1.03],
            "G3": [1.00, 1.00, 1.01, 1.02],
            "G4": [1.00, 0.99, 1.00, 1.01],
            "G5": [1.00, 0.98, 0.97, 0.96],
        }
        self.factor_results = {
            "ic_summary": pd.DataFrame(
                {"metric": ["ic", "rank_ic"], "mean": [0.1, 0.12]}
            ),
            "ic_series": pd.DataFrame(
                {
                    "signal_date": self.dates,
                    "asset_count": [20, 20, 21, 21],
                    "ic": [0.1, 0.2, 0.3, 0.4],
                    "rank_ic": [0.2, 0.1, 0.4, 0.3],
                }
            ),
            "group_returns": pd.DataFrame(
                {"signal_date": self.dates, **groups}
            ),
            "group_nav": pd.DataFrame(
                {"signal_date": self.dates, **groups}
            ),
        }

    def test_rolling_ic_requires_a_complete_window(self):
        result = calculate_rolling_ic(self.factor_results["ic_series"], 3)

        self.assertTrue(result.loc[:1, "ic_rolling"].isna().all())
        self.assertAlmostEqual(result.loc[2, "ic_rolling"], 0.2)

    def test_saves_the_standard_tables_and_figures(self):
        expected_files = {
            "run_config.csv",
            "strategy_metrics.csv",
            "strategy_nav.csv",
            "ic_summary.csv",
            "ic_series.csv",
            "group_returns.csv",
            "group_nav.csv",
            "strategy_nav.png",
            "five_group_nav.png",
            "ic_rankic_rolling20.png",
        }

        with TemporaryDirectory() as directory:
            output = Path(directory)
            save_result_tables(
                output,
                {"factor_name": "demo"},
                self.strategy_metrics,
                self.strategy_nav,
                self.factor_results,
            )
            plot_strategy_nav(self.strategy_nav, output / "strategy_nav.png")
            plot_group_nav(
                self.factor_results["group_nav"],
                output / "five_group_nav.png",
            )
            plot_ic_history(
                self.factor_results["ic_series"],
                20,
                output / "ic_rankic_rolling20.png",
            )

            actual_files = {
                path.name for path in output.iterdir() if path.is_file()
            }
            self.assertEqual(actual_files, expected_files)
            self.assertTrue(
                all((output / name).stat().st_size > 0 for name in expected_files)
            )


if __name__ == "__main__":
    unittest.main()
