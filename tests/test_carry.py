import unittest
from unittest.mock import patch

import pandas as pd

from src.factors import carry
from src.factors.carry import (
    calculate_factor,
    compute_main_sub_carry,
    load_main_sub_carry,
)
from src.factor_loader import calculate_factor as calculate_loaded_factor


class MainSubCarryFormulaTest(unittest.TestCase):
    def test_missing_contract_day_breaks_the_complete_rolling_window(self):
        calendar_dates = pd.to_datetime(
            ["2024-01-02", "2024-01-03", "2024-01-04"]
        )
        mapping = pd.DataFrame(
            {
                "trade_date": calendar_dates[[0, 2]],
                "fut_code": ["RB", "RB"],
                "ts_code_A": ["RB_A", "RB_A"],
                "ts_code_B": ["RB_B", "RB_B"],
                "close_A": [100.0, 100.0],
                "close_B": [101.0, 102.0],
                "d_AB": [90.0, 90.0],
            }
        )
        calendar = pd.DataFrame({"trade_date": calendar_dates})

        result = compute_main_sub_carry(
            contract_mapping=mapping,
            trade_calendar=calendar,
            lookback=2,
        )

        self.assertEqual(result["trade_date"].tolist(), list(calendar_dates))
        self.assertTrue(pd.isna(result.loc[1, "daily_carry"]))
        self.assertTrue(result["main_sub_carry"].isna().all())

    def test_rejects_duplicate_date_commodity_keys(self):
        dates = pd.to_datetime(["2024-01-02", "2024-01-02"])
        mapping = pd.DataFrame(
            {
                "trade_date": dates,
                "fut_code": ["RB", "RB"],
                "ts_code_A": ["RB_A", "RB_A"],
                "ts_code_B": ["RB_B", "RB_B"],
                "close_A": [100.0, 100.0],
                "close_B": [101.0, 101.0],
                "d_AB": [90.0, 90.0],
            }
        )
        calendar = pd.DataFrame(
            {"trade_date": pd.to_datetime(["2024-01-02"])}
        )

        with self.assertRaisesRegex(
            ValueError,
            "contract mapping contains duplicate date-commodity keys",
        ):
            compute_main_sub_carry(mapping, calendar, lookback=1)

    def test_rejects_missing_contract_columns(self):
        mapping = pd.DataFrame(
            {
                "trade_date": pd.to_datetime(["2024-01-02"]),
                "fut_code": ["RB"],
                "ts_code_A": ["RB_A"],
                "ts_code_B": ["RB_B"],
                "close_A": [100.0],
                "d_AB": [90.0],
            }
        )
        calendar = pd.DataFrame(
            {"trade_date": pd.to_datetime(["2024-01-02"])}
        )

        with self.assertRaisesRegex(
            ValueError,
            "contract_mapping is missing columns: close_B",
        ):
            compute_main_sub_carry(mapping, calendar, lookback=1)

    def test_rejects_nonpositive_lookback(self):
        mapping = pd.DataFrame(
            {
                "trade_date": pd.to_datetime(["2024-01-02"]),
                "fut_code": ["RB"],
                "ts_code_A": ["RB_A"],
                "ts_code_B": ["RB_B"],
                "close_A": [100.0],
                "close_B": [101.0],
                "d_AB": [90.0],
            }
        )
        calendar = pd.DataFrame(
            {"trade_date": pd.to_datetime(["2024-01-02"])}
        )

        with self.assertRaisesRegex(
            ValueError,
            "lookback must be a positive integer",
        ):
            compute_main_sub_carry(mapping, calendar, lookback=0)

    def test_uses_close_prices_and_signed_maturity_gap(self):
        dates = pd.to_datetime(
            ["2024-01-02", "2024-01-03", "2024-01-04"]
        )
        mapping = pd.DataFrame(
            {
                "trade_date": dates,
                "fut_code": ["RB", "RB", "RB"],
                "ts_code_A": ["RB_A", "RB_A", "RB_A"],
                "ts_code_B": ["RB_B", "RB_B", "RB_B"],
                "close_A": [100.0, 100.0, 100.0],
                "close_B": [110.0, 95.0, 101.0],
                "d_AB": [90.0, -90.0, 90.0],
            }
        )
        calendar = pd.DataFrame({"trade_date": dates})

        result = compute_main_sub_carry(
            contract_mapping=mapping,
            trade_calendar=calendar,
            lookback=2,
        )

        self.assertIsInstance(result, pd.DataFrame)

        expected_daily = [
            -0.40555555555555556,
            -0.20277777777777778,
            -0.04055555555555556,
        ]
        for actual, expected in zip(
            result["daily_carry"],
            expected_daily,
            strict=True,
        ):
            self.assertAlmostEqual(actual, expected)

        self.assertTrue(pd.isna(result.loc[0, "main_sub_carry"]))
        self.assertAlmostEqual(
            result.loc[1, "main_sub_carry"],
            -0.3041666666666667,
        )
        self.assertAlmostEqual(
            result.loc[2, "main_sub_carry"],
            -0.12166666666666667,
        )

    def test_zero_maturity_gap_produces_missing_carry(self):
        dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
        mapping = pd.DataFrame(
            {
                "trade_date": dates,
                "fut_code": ["RB", "RB"],
                "ts_code_A": ["RB_A", "RB_A"],
                "ts_code_B": ["RB_B", "RB_B"],
                "close_A": [100.0, 100.0],
                "close_B": [101.0, 102.0],
                "d_AB": [0.0, 90.0],
            }
        )
        calendar = pd.DataFrame({"trade_date": dates})

        result = compute_main_sub_carry(
            contract_mapping=mapping,
            trade_calendar=calendar,
            lookback=1,
        )

        self.assertTrue(pd.isna(result.loc[0, "daily_carry"]))
        self.assertTrue(pd.isna(result.loc[0, "main_sub_carry"]))
        self.assertAlmostEqual(
            result.loc[1, "daily_carry"],
            -0.0811111111111111,
        )


class MainSubCarryLoadingTest(unittest.TestCase):
    def test_loads_buffered_history_and_computes_the_panel(self):
        dates = pd.to_datetime(["2023-11-28", "2023-11-29"])
        mapping = pd.DataFrame(
            {
                "trade_date": dates,
                "fut_code": ["RB", "RB"],
                "ts_code_A": ["RB_A", "RB_A"],
                "ts_code_B": ["RB_B", "RB_B"],
                "close_A": [100.0, 100.0],
                "close_B": [101.0, 102.0],
                "d_AB": [90.0, 90.0],
            }
        )
        calendar = pd.DataFrame({"trade_date": dates})

        with (
            patch.object(
                carry,
                "build_contract_mapping",
                return_value=mapping,
                create=True,
            ) as mock_mapping,
            patch.object(
                carry,
                "load_trade_calendar",
                return_value=calendar,
                create=True,
            ) as mock_calendar,
        ):
            result = load_main_sub_carry(
                start_date="20240101",
                end_date="20241231",
                lookback=2,
                signal_min_days_to_maturity=0,
            )

        mock_mapping.assert_called_once_with(
            "20231128",
            "20241231",
            min_days_to_maturity=0,
        )
        mock_calendar.assert_called_once_with("20231128", "20241231")
        self.assertIsInstance(result, pd.DataFrame)
        self.assertAlmostEqual(
            result.loc[1, "main_sub_carry"],
            -0.06083333333333334,
        )


class CarryPluginTest(unittest.TestCase):
    def test_dynamic_loader_accepts_the_carry_plugin(self):
        carry_panel = pd.DataFrame(
            {
                "trade_date": pd.to_datetime(["2024-01-02"]),
                "fut_code": ["RB"],
                "main_sub_carry": [0.20],
            }
        )

        with patch.object(
            carry,
            "load_main_sub_carry",
            return_value=carry_panel,
        ) as mock_load:
            result = calculate_loaded_factor(
                factor_name="carry",
                start_date="20240101",
                end_date="20240103",
                parameters={"lookback": 20},
            )

        mock_load.assert_called_once_with("20240101", "20240103", 20, 0)
        self.assertEqual(
            list(result.columns),
            ["trade_date", "fut_code", "main_sub_carry", "raw_factor"],
        )
        self.assertEqual(result.loc[0, "raw_factor"], 0.20)

    def test_uses_default_lookback_and_returns_raw_factor_in_period(self):
        carry_panel = pd.DataFrame(
            {
                "trade_date": pd.to_datetime(
                    ["2023-12-29", "2024-01-02", "2024-01-03"]
                ),
                "fut_code": ["RB", "RB", "RB"],
                "main_sub_carry": [0.10, 0.20, 0.30],
                "daily_carry": [0.01, 0.02, 0.03],
            }
        )

        with patch.object(
            carry,
            "load_main_sub_carry",
            return_value=carry_panel,
        ) as mock_load:
            result = calculate_factor(
                start_date="20240101",
                end_date="20240103",
                parameters=None,
            )

        mock_load.assert_called_once_with("20240101", "20240103", 90, 0)
        self.assertEqual(
            result["trade_date"].tolist(),
            list(pd.to_datetime(["2024-01-02", "2024-01-03"])),
        )
        self.assertEqual(result["raw_factor"].tolist(), [0.20, 0.30])

    def test_rejects_invalid_signal_maturity_cutoffs(self):
        with patch.object(carry, "load_main_sub_carry") as loader:
            for value in [-1, True, 1.5]:
                with self.subTest(value=value), self.assertRaises(ValueError):
                    calculate_factor(
                        "20240101",
                        "20240103",
                        {
                            "lookback": 20,
                            "signal_min_days_to_maturity": value,
                        },
                    )

        loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
