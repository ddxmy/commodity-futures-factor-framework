import unittest
from importlib import import_module
from unittest.mock import patch
import warnings

import pandas as pd

from src.factor_loader import calculate_factor as calculate_loaded_factor


def load_t_rank_module():
    try:
        return import_module("src.factors.t_rank")
    except ModuleNotFoundError as error:
        raise AssertionError("src.factors.t_rank must exist") from error


class TRankFormulaTest(unittest.TestCase):
    def test_standardizes_daily_cross_sectional_ranks_and_averages_them(self):
        t_rank = load_t_rank_module()

        compute_t_rank = getattr(t_rank, "compute_t_rank", None)
        self.assertTrue(callable(compute_t_rank))

        dates = pd.to_datetime(
            ["2024-01-02", "2024-01-03", "2024-01-04"]
        )
        mapping = pd.DataFrame(
            {
                "trade_date": dates.repeat(3),
                "fut_code": ["AL", "CU", "RB"] * 3,
                "ts_code_A": ["AL_A", "CU_A", "RB_A"] * 3,
                "daily_return_A": [
                    0.01,
                    0.02,
                    0.03,
                    0.01,
                    0.02,
                    0.03,
                    0.03,
                    0.02,
                    0.01,
                ],
            }
        )
        calendar = pd.DataFrame({"trade_date": dates})

        result = compute_t_rank(
            contract_mapping=mapping,
            trade_calendar=calendar,
            lookback=2,
        )

        required_columns = {
            "participant_count",
            "return_rank",
            "rank_score",
            "t_rank",
        }
        self.assertTrue(required_columns.issubset(result.columns))

        panel = result.set_index(["trade_date", "fut_code"])
        score = 1.224744871391589

        self.assertEqual(
            panel.loc[(dates[0], "AL"), "participant_count"],
            3,
        )
        self.assertAlmostEqual(
            panel.loc[(dates[0], "AL"), "rank_score"],
            -score,
        )
        self.assertAlmostEqual(
            panel.loc[(dates[0], "CU"), "rank_score"],
            0.0,
        )
        self.assertAlmostEqual(
            panel.loc[(dates[0], "RB"), "rank_score"],
            score,
        )

        self.assertTrue(
            pd.isna(panel.loc[(dates[0], "RB"), "t_rank"])
        )
        self.assertAlmostEqual(
            panel.loc[(dates[1], "RB"), "t_rank"],
            score,
        )
        self.assertAlmostEqual(
            panel.loc[(dates[2], "RB"), "t_rank"],
            0.0,
        )

    def test_two_asset_cross_section_has_valid_standardized_scores(self):
        t_rank = load_t_rank_module()
        date = pd.Timestamp("2024-01-02")
        mapping = pd.DataFrame(
            {
                "trade_date": [date, date],
                "fut_code": ["CU", "RB"],
                "ts_code_A": ["CU_A", "RB_A"],
                "daily_return_A": [-0.01, 0.02],
            }
        )

        result = t_rank.compute_t_rank(
            contract_mapping=mapping,
            trade_calendar=pd.DataFrame({"trade_date": [date]}),
            lookback=1,
        ).set_index("fut_code")

        self.assertAlmostEqual(result.loc["CU", "rank_score"], -1.0)
        self.assertAlmostEqual(result.loc["RB", "rank_score"], 1.0)
        self.assertAlmostEqual(result.loc["CU", "t_rank"], -1.0)
        self.assertAlmostEqual(result.loc["RB", "t_rank"], 1.0)

    def test_empty_cross_section_produces_missing_scores_without_warning(self):
        t_rank = load_t_rank_module()
        dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
        mapping = pd.DataFrame(
            {
                "trade_date": [dates[0]],
                "fut_code": ["RB"],
                "ts_code_A": ["RB_A"],
                "daily_return_A": [0.01],
            }
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            result = t_rank.compute_t_rank(
                contract_mapping=mapping,
                trade_calendar=pd.DataFrame({"trade_date": dates}),
                lookback=1,
            )

        empty_day = result[result["trade_date"].eq(dates[1])]
        self.assertEqual(empty_day["participant_count"].tolist(), [0])
        self.assertTrue(empty_day["rank_score"].isna().all())
        self.assertTrue(empty_day["t_rank"].isna().all())


class TRankLoadingTest(unittest.TestCase):
    def test_loads_buffered_mapping_and_calendar_history(self):
        t_rank = load_t_rank_module()

        mapping = pd.DataFrame(
            {
                "trade_date": pd.to_datetime(["2023-12-29"]),
                "fut_code": ["RB"],
                "ts_code_A": ["RB_A"],
                "daily_return_A": [0.01],
            }
        )
        calendar = pd.DataFrame(
            {"trade_date": pd.to_datetime(["2023-12-29"])}
        )
        computed = mapping.assign(t_rank=0.25)

        with (
            patch.object(
                t_rank,
                "build_contract_mapping",
                return_value=mapping,
            ) as mock_mapping,
            patch.object(
                t_rank,
                "load_trade_calendar",
                return_value=calendar,
            ) as mock_calendar,
            patch.object(
                t_rank,
                "compute_t_rank",
                return_value=computed,
            ) as mock_compute,
        ):
            result = t_rank.load_t_rank(
                start_date="20240101",
                end_date="20241231",
                lookback=10,
                signal_min_days_to_maturity=0,
            )

        mock_mapping.assert_called_once_with(
            "20231112",
            "20241231",
            min_days_to_maturity=0,
        )
        mock_calendar.assert_called_once_with("20231112", "20241231")
        mock_compute.assert_called_once_with(
            mapping,
            calendar,
            10,
        )
        pd.testing.assert_frame_equal(result, computed)


class TRankPluginTest(unittest.TestCase):
    def test_dynamic_loader_returns_requested_raw_factor_panel(self):
        t_rank = load_t_rank_module()
        panel = pd.DataFrame(
            {
                "trade_date": pd.to_datetime(
                    ["2023-12-29", "2024-01-02", "2024-01-03"]
                ),
                "fut_code": ["RB", "RB", "RB"],
                "t_rank": [0.10, 0.20, 0.30],
            }
        )

        with patch.object(
            t_rank,
            "load_t_rank",
            return_value=panel,
        ) as mock_load:
            result = calculate_loaded_factor(
                factor_name="t_rank",
                start_date="20240101",
                end_date="20240103",
                parameters={"lookback": 10},
            )

        mock_load.assert_called_once_with(
            "20240101",
            "20240103",
            10,
            0,
        )
        self.assertEqual(
            result["trade_date"].tolist(),
            list(pd.to_datetime(["2024-01-02", "2024-01-03"])),
        )
        self.assertEqual(result["raw_factor"].tolist(), [0.20, 0.30])

    def test_rejects_nonpositive_lookback(self):
        t_rank = load_t_rank_module()
        with self.assertRaisesRegex(
            ValueError,
            "lookback must be a positive integer",
        ):
            t_rank.calculate_factor(
                "20240101",
                "20240103",
                {"lookback": 0},
            )


if __name__ == "__main__":
    unittest.main()
