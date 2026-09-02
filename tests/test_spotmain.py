from importlib import import_module
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src import p01_market_data
from src.factor_loader import calculate_factor as calculate_loaded_factor

try:
    spotmain = import_module("src.factors.spotmain")
except ModuleNotFoundError as error:
    if error.name != "src.factors.spotmain":
        raise
    spotmain = None


def build_contract_mapping(dates, close_values):
    return pd.DataFrame(
        {
            "trade_date": dates,
            "fut_code": ["RB"] * len(dates),
            "ts_code_A": ["RB2405.SHF"] * len(dates),
            "close_A": close_values,
            "d_A": [365.0] * len(dates),
        }
    )


class SpotPriceLoadingTest(unittest.TestCase):
    def test_load_spot_daily_filters_dates_and_parses_trade_date(self):
        load_spot_daily = getattr(p01_market_data, "load_spot_daily", None)
        self.assertIsNotNone(load_spot_daily)

        with tempfile.NamedTemporaryFile(suffix=".db") as database:
            with sqlite3.connect(database.name) as connection:
                connection.execute(
                    """
                    CREATE TABLE spot_price (
                        trade_date TEXT NOT NULL,
                        fut_code TEXT NOT NULL,
                        spot_price REAL NOT NULL,
                        source TEXT NOT NULL,
                        PRIMARY KEY (trade_date, fut_code, source)
                    )
                    """
                )
                connection.executemany(
                    """
                    INSERT INTO spot_price
                        (trade_date, fut_code, spot_price, source)
                    VALUES (?, ?, ?, ?)
                    """,
                    [
                        ("20240101", "RB", 3800.0, "test"),
                        ("20240102", "RB", 3810.0, "test"),
                        ("20240103", "CU", 69000.0, "test"),
                        ("20240104", "RB", 3820.0, "test"),
                    ],
                )

            result = load_spot_daily(
                "20240102",
                "20240103",
                db_path=database.name,
            )

        self.assertEqual(result["fut_code"].tolist(), ["RB", "CU"])
        self.assertEqual(result["spot_price"].tolist(), [3810.0, 69000.0])
        self.assertTrue(
            pd.api.types.is_datetime64_any_dtype(result["trade_date"])
        )
        self.assertEqual(result["source"].tolist(), ["test", "test"])


class SpotMainFormulaTest(unittest.TestCase):
    def compute(self, mapping, spots, calendar, lookback):
        self.assertIsNotNone(spotmain, "src.factors.spotmain must exist")
        compute_spot_main = getattr(spotmain, "compute_spot_main", None)
        self.assertTrue(callable(compute_spot_main))
        return compute_spot_main(mapping, spots, calendar, lookback)

    def test_uses_close_price_and_preserves_factor_direction(self):
        dates = pd.to_datetime(
            ["2024-01-02", "2024-01-03", "2024-01-04"]
        )
        mapping = build_contract_mapping(dates, [90.0, 110.0, 100.0])
        spots = pd.DataFrame(
            {
                "trade_date": dates,
                "fut_code": ["RB"] * 3,
                "spot_price": [100.0] * 3,
                "source": ["test"] * 3,
            }
        )
        calendar = pd.DataFrame({"trade_date": dates})

        result = self.compute(mapping, spots, calendar, lookback=2)

        expected_daily = [0.10, -0.10, 0.0]
        for actual, expected in zip(
            result["daily_spotmain"],
            expected_daily,
            strict=True,
        ):
            self.assertAlmostEqual(actual, expected)
        self.assertTrue(pd.isna(result.loc[0, "spotmain"]))
        self.assertAlmostEqual(result.loc[1, "spotmain"], 0.0)
        self.assertAlmostEqual(result.loc[2, "spotmain"], -0.05)

    def test_fills_only_one_missing_day_and_keeps_observation_date(self):
        dates = pd.to_datetime(
            ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
        )
        mapping = build_contract_mapping(dates, [110.0] * 4)
        spots = pd.DataFrame(
            {
                "trade_date": dates[[0, 3]],
                "fut_code": ["RB", "RB"],
                "spot_price": [100.0, 200.0],
                "source": ["test", "test"],
            }
        )
        calendar = pd.DataFrame({"trade_date": dates})

        result = self.compute(mapping, spots, calendar, lookback=1)

        self.assertEqual(result.loc[1, "spot_price"], 100.0)
        self.assertEqual(
            result.loc[1, "spot_observation_date"],
            pd.Timestamp("2024-01-02"),
        )
        self.assertTrue(pd.isna(result.loc[2, "spot_price"]))
        self.assertTrue(pd.isna(result.loc[2, "daily_spotmain"]))
        self.assertEqual(result.loc[3, "spot_price"], 200.0)
        self.assertEqual(
            result.loc[3, "spot_observation_date"],
            pd.Timestamp("2024-01-05"),
        )

    def test_unresolved_gap_restarts_the_complete_rolling_window(self):
        dates = pd.to_datetime(
            [
                "2024-01-02",
                "2024-01-03",
                "2024-01-04",
                "2024-01-05",
                "2024-01-08",
            ]
        )
        mapping = build_contract_mapping(dates, [90.0] * 5)
        spots = pd.DataFrame(
            {
                "trade_date": dates[[0, 3, 4]],
                "fut_code": ["RB"] * 3,
                "spot_price": [100.0] * 3,
                "source": ["test"] * 3,
            }
        )
        calendar = pd.DataFrame({"trade_date": dates})

        result = self.compute(mapping, spots, calendar, lookback=2)

        self.assertAlmostEqual(result.loc[1, "spotmain"], 0.10)
        self.assertTrue(pd.isna(result.loc[2, "spotmain"]))
        self.assertTrue(pd.isna(result.loc[3, "spotmain"]))
        self.assertAlmostEqual(result.loc[4, "spotmain"], 0.10)

    def test_product_without_spot_history_remains_in_the_panel(self):
        dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
        mapping = build_contract_mapping(dates, [100.0, 100.0])
        spots = pd.DataFrame(
            {
                "trade_date": pd.to_datetime([]),
                "fut_code": pd.Series(dtype="object"),
                "spot_price": pd.Series(dtype="float64"),
            }
        )
        calendar = pd.DataFrame({"trade_date": dates})

        result = self.compute(mapping, spots, calendar, lookback=1)

        self.assertEqual(result["fut_code"].tolist(), ["RB", "RB"])
        self.assertTrue(result["daily_spotmain"].isna().all())
        self.assertTrue(result["spotmain"].isna().all())

    def test_rejects_duplicate_date_commodity_keys(self):
        date = pd.to_datetime(["2024-01-02"])
        mapping = build_contract_mapping(date, [100.0])
        spots = pd.DataFrame(
            {
                "trade_date": [date[0], date[0]],
                "fut_code": ["RB", "RB"],
                "spot_price": [100.0, 101.0],
            }
        )
        calendar = pd.DataFrame({"trade_date": date})

        duplicate_mapping = pd.concat(
            [mapping, mapping],
            ignore_index=True,
        )
        with self.assertRaisesRegex(
            ValueError,
            "contract mapping contains duplicate date-commodity keys",
        ):
            self.compute(
                duplicate_mapping,
                spots.iloc[[0]],
                calendar,
                lookback=1,
            )

        with self.assertRaisesRegex(
            ValueError,
            "spot prices contain duplicate date-commodity keys",
        ):
            self.compute(mapping, spots, calendar, lookback=1)

    def test_rejects_missing_contract_columns_and_invalid_lookback(self):
        date = pd.to_datetime(["2024-01-02"])
        mapping = build_contract_mapping(date, [100.0]).drop(columns="d_A")
        spots = pd.DataFrame(
            {
                "trade_date": date,
                "fut_code": ["RB"],
                "spot_price": [100.0],
            }
        )
        calendar = pd.DataFrame({"trade_date": date})

        with self.assertRaisesRegex(
            ValueError,
            "contract_mapping is missing columns: d_A",
        ):
            self.compute(mapping, spots, calendar, lookback=1)
        with self.assertRaisesRegex(
            ValueError,
            "lookback must be a positive integer",
        ):
            self.compute(
                build_contract_mapping(date, [100.0]),
                spots,
                calendar,
                lookback=0,
            )

    def test_nonpositive_inputs_produce_missing_factor_without_infinity(self):
        dates = pd.to_datetime(
            ["2024-01-02", "2024-01-03", "2024-01-04"]
        )
        mapping = build_contract_mapping(dates, [0.0, 100.0, 100.0])
        mapping.loc[2, "d_A"] = 0.0
        spots = pd.DataFrame(
            {
                "trade_date": dates,
                "fut_code": ["RB"] * 3,
                "spot_price": [100.0, 0.0, 100.0],
            }
        )
        calendar = pd.DataFrame({"trade_date": dates})

        result = self.compute(mapping, spots, calendar, lookback=1)

        self.assertTrue(result["daily_spotmain"].isna().all())
        self.assertFalse(np.isinf(result["daily_spotmain"].dropna()).any())


class SpotMainLoadingTest(unittest.TestCase):
    def test_loads_buffered_history_for_all_three_inputs(self):
        load_spot_main = getattr(spotmain, "load_spot_main", None)
        self.assertTrue(callable(load_spot_main))

        dates = pd.to_datetime(["2023-11-28", "2023-11-29"])
        mapping = build_contract_mapping(dates, [90.0, 90.0])
        spots = pd.DataFrame(
            {
                "trade_date": dates,
                "fut_code": ["RB", "RB"],
                "spot_price": [100.0, 100.0],
                "source": ["test", "test"],
            }
        )
        calendar = pd.DataFrame({"trade_date": dates})

        with (
            patch.object(
                spotmain,
                "build_contract_mapping",
                return_value=mapping,
            ) as mock_mapping,
            patch.object(
                spotmain,
                "load_spot_daily",
                return_value=spots,
            ) as mock_spot,
            patch.object(
                spotmain,
                "load_trade_calendar",
                return_value=calendar,
            ) as mock_calendar,
        ):
            result = load_spot_main(
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
        for mocked_loader in [mock_spot, mock_calendar]:
            mocked_loader.assert_called_once_with("20231128", "20241231")
        self.assertAlmostEqual(result.loc[1, "spotmain"], 0.10)


class SpotMainPluginTest(unittest.TestCase):
    def test_uses_default_lookback_and_trims_requested_period(self):
        calculate_factor = getattr(spotmain, "calculate_factor", None)
        self.assertTrue(callable(calculate_factor))

        panel = pd.DataFrame(
            {
                "trade_date": pd.to_datetime(
                    ["2023-12-29", "2024-01-02", "2024-01-03"]
                ),
                "fut_code": ["RB", "RB", "RB"],
                "spotmain": [0.10, 0.20, 0.30],
                "daily_spotmain": [0.01, 0.02, 0.03],
            }
        )

        with patch.object(
            spotmain,
            "load_spot_main",
            return_value=panel,
        ) as mock_load:
            result = calculate_factor(
                "20240101",
                "20240103",
                parameters=None,
            )

        mock_load.assert_called_once_with("20240101", "20240103", 90, 0)
        self.assertEqual(
            result["trade_date"].tolist(),
            list(pd.to_datetime(["2024-01-02", "2024-01-03"])),
        )
        self.assertEqual(result["raw_factor"].tolist(), [0.20, 0.30])

    def test_rejects_invalid_signal_maturity_cutoffs(self):
        with patch.object(spotmain, "load_spot_main") as loader:
            for value in [-1, True, 1.5]:
                with self.subTest(value=value), self.assertRaises(ValueError):
                    spotmain.calculate_factor(
                        "20240101",
                        "20240103",
                        {
                            "lookback": 20,
                            "signal_min_days_to_maturity": value,
                        },
                    )

        loader.assert_not_called()

    def test_dynamic_loader_accepts_spotmain(self):
        load_spot_main = getattr(spotmain, "load_spot_main", None)
        self.assertTrue(callable(load_spot_main))

        panel = pd.DataFrame(
            {
                "trade_date": pd.to_datetime(["2024-01-02"]),
                "fut_code": ["RB"],
                "spotmain": [0.20],
            }
        )
        with patch.object(
            spotmain,
            "load_spot_main",
            return_value=panel,
        ):
            result = calculate_loaded_factor(
                factor_name="spotmain",
                start_date="20240101",
                end_date="20240103",
                parameters={"lookback": 20},
            )

        self.assertEqual(result.loc[0, "raw_factor"], 0.20)


if __name__ == "__main__":
    unittest.main()
