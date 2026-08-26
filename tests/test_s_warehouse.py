import sqlite3
import tempfile
import unittest
from importlib import import_module
from unittest.mock import patch

import pandas as pd

from src import p01_market_data


s_warehouse = import_module("src.factors.s_warehouse")


class WarehouseDailyLoadingTest(unittest.TestCase):
    def test_loads_requested_dates_and_preserves_quality_columns(self):
        load_warehouse_daily = getattr(
            p01_market_data,
            "load_warehouse_daily",
            None,
        )
        self.assertTrue(callable(load_warehouse_daily))

        with tempfile.NamedTemporaryFile(suffix=".db") as database:
            with sqlite3.connect(database.name) as connection:
                connection.execute(
                    """
                    CREATE TABLE warehouse_daily (
                        trade_date TEXT NOT NULL,
                        fut_code TEXT NOT NULL,
                        exchange TEXT NOT NULL,
                        warehouse_total REAL,
                        source_row_count INTEGER NOT NULL,
                        used_row_count INTEGER NOT NULL,
                        quality_status TEXT NOT NULL,
                        quality_note TEXT,
                        source TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        PRIMARY KEY (trade_date, fut_code)
                    )
                    """
                )
                connection.executemany(
                    """
                    INSERT INTO warehouse_daily VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                    """,
                    [
                        ("20240101", "RB", "SHFE", 100.0, 2, 2, "valid", None, "test", "now"),
                        ("20240102", "RB", "SHFE", 0.0, 1, 1, "zero", "zero", "test", "now"),
                        ("20240103", "CU", "SHFE", None, 1, 1, "all_null", "null", "test", "now"),
                        ("20240104", "RB", "SHFE", 90.0, 2, 2, "valid", None, "test", "now"),
                    ],
                )

            result = load_warehouse_daily(
                "20240102",
                "20240103",
                db_path=database.name,
            )

        self.assertEqual(result["fut_code"].tolist(), ["RB", "CU"])
        self.assertEqual(result["quality_status"].tolist(), ["zero", "all_null"])
        self.assertEqual(result["source_row_count"].tolist(), [1, 1])
        self.assertTrue(
            pd.api.types.is_datetime64_any_dtype(result["trade_date"])
        )


class SWarehouseValidationTest(unittest.TestCase):
    def setUp(self):
        self.warehouse = pd.DataFrame(
            {
                "trade_date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
                "fut_code": ["RB", "RB"],
                "warehouse_total": [100.0, 90.0],
                "quality_status": ["valid", "valid"],
            }
        )
        self.calendar = pd.DataFrame(
            {"trade_date": pd.to_datetime(["2024-01-02", "2024-01-03"])}
        )

    def compute(self, warehouse=None, calendar=None, **parameters):
        function = getattr(s_warehouse, "compute_s_warehouse", None)
        self.assertTrue(callable(function))
        return function(
            self.warehouse if warehouse is None else warehouse,
            self.calendar if calendar is None else calendar,
            lookback=parameters.get("lookback", 1),
            smooth_window=parameters.get("smooth_window", 2),
            min_observations=parameters.get("min_observations", 1),
        )

    def test_rejects_invalid_window_parameters(self):
        cases = [
            {"lookback": 0},
            {"smooth_window": True},
            {"min_observations": 0},
            {"smooth_window": 2, "min_observations": 3},
        ]
        for parameters in cases:
            with self.subTest(parameters=parameters):
                with self.assertRaises(ValueError):
                    self.compute(**parameters)

    def test_rejects_missing_columns_and_nonnumeric_totals(self):
        with self.assertRaisesRegex(
            ValueError,
            "warehouse_daily is missing columns: quality_status",
        ):
            self.compute(warehouse=self.warehouse.drop(columns="quality_status"))

        bad = self.warehouse.astype(
            {"warehouse_total": "object"}
        )
        bad.loc[0, "warehouse_total"] = "bad"
        with self.assertRaisesRegex(ValueError, "warehouse_total must be numeric"):
            self.compute(warehouse=bad)

        with self.assertRaisesRegex(
            ValueError,
            "trade_calendar is missing columns: trade_date",
        ):
            self.compute(calendar=self.calendar.drop(columns="trade_date"))

    def test_rejects_duplicate_keys_and_invalid_calendar_order(self):
        duplicate_warehouse = pd.concat(
            [self.warehouse, self.warehouse.iloc[[0]]],
            ignore_index=True,
        )
        with self.assertRaisesRegex(ValueError, "duplicate date-commodity keys"):
            self.compute(warehouse=duplicate_warehouse)

        duplicate_calendar = pd.concat(
            [self.calendar, self.calendar.iloc[[0]]],
            ignore_index=True,
        )
        with self.assertRaisesRegex(ValueError, "duplicate trade"):
            self.compute(calendar=duplicate_calendar)

        with self.assertRaisesRegex(ValueError, "sorted"):
            self.compute(calendar=self.calendar.iloc[::-1].reset_index(drop=True))


class SWarehousePanelTest(unittest.TestCase):
    def test_builds_complete_calendar_panel_and_invalidates_bad_observations(self):
        dates = pd.to_datetime(
            ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
        )
        warehouse = pd.DataFrame(
            {
                "trade_date": [dates[0], dates[1], dates[3], dates[0]],
                "fut_code": ["RB", "RB", "RB", "CU"],
                "warehouse_total": [100.0, 0.0, 80.0, 200.0],
                "quality_status": ["valid", "zero", "valid", "ambiguous"],
            }
        )
        calendar = pd.DataFrame({"trade_date": dates})

        result = s_warehouse.compute_s_warehouse(
            warehouse,
            calendar,
            lookback=1,
            smooth_window=2,
            min_observations=1,
        )

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 8)
        rb = result[result["fut_code"].eq("RB")].reset_index(drop=True)
        cu = result[result["fut_code"].eq("CU")].reset_index(drop=True)
        self.assertEqual(rb["trade_date"].tolist(), list(dates))
        self.assertEqual(rb.loc[0, "warehouse_value"], 100.0)
        self.assertTrue(pd.isna(rb.loc[1, "warehouse_value"]))
        self.assertEqual(rb.loc[1, "warehouse_total"], 0.0)
        self.assertTrue(pd.isna(rb.loc[2, "warehouse_value"]))
        self.assertEqual(rb.loc[3, "warehouse_value"], 80.0)
        self.assertTrue(cu["warehouse_value"].isna().all())
        self.assertEqual(
            rb["has_warehouse_observation"].tolist(),
            [True, True, False, True],
        )


class SWarehouseFormulaTest(unittest.TestCase):
    def test_uses_exact_market_day_lag_and_reverses_inventory_change(self):
        dates = pd.bdate_range("2024-01-02", periods=5)
        warehouse = pd.DataFrame(
            {
                "trade_date": dates,
                "fut_code": ["RB"] * 5,
                "warehouse_total": [100.0, 100.0, 80.0, 80.0, 60.0],
                "quality_status": ["valid"] * 5,
            }
        )

        result = s_warehouse.compute_s_warehouse(
            warehouse,
            pd.DataFrame({"trade_date": dates}),
            lookback=2,
            smooth_window=2,
            min_observations=2,
        )

        required = {
            "valid_observations",
            "smoothed_warehouse",
            "lagged_smoothed_warehouse",
            "lagged_valid_observations",
            "warehouse_change",
            "s_warehouse",
        }
        self.assertTrue(required.issubset(result.columns))
        self.assertTrue(pd.isna(result.loc[0, "smoothed_warehouse"]))
        self.assertEqual(result.loc[1, "smoothed_warehouse"], 100.0)
        self.assertEqual(result.loc[3, "lagged_smoothed_warehouse"], 100.0)
        self.assertAlmostEqual(result.loc[3, "warehouse_change"], -0.20)
        self.assertAlmostEqual(result.loc[3, "s_warehouse"], 0.20)
        self.assertAlmostEqual(result.loc[4, "smoothed_warehouse"], 70.0)
        self.assertAlmostEqual(result.loc[4, "lagged_smoothed_warehouse"], 90.0)
        self.assertAlmostEqual(result.loc[4, "s_warehouse"], 2.0 / 9.0)

    def test_requires_configured_valid_observation_count(self):
        dates = pd.bdate_range("2024-01-02", periods=20)
        rows = []
        for code, valid_count in [("RB", 18), ("CU", 17)]:
            for index, date in enumerate(dates):
                is_valid = index < valid_count
                rows.append(
                    {
                        "trade_date": date,
                        "fut_code": code,
                        "warehouse_total": 100.0 if is_valid else 0.0,
                        "quality_status": "valid" if is_valid else "zero",
                    }
                )

        result = s_warehouse.compute_s_warehouse(
            pd.DataFrame(rows),
            pd.DataFrame({"trade_date": dates}),
            lookback=1,
            smooth_window=20,
            min_observations=18,
        )

        self.assertIn("smoothed_warehouse", result.columns)
        last = result[result["trade_date"].eq(dates[-1])].set_index("fut_code")
        self.assertEqual(last.loc["RB", "valid_observations"], 18)
        self.assertEqual(last.loc["RB", "smoothed_warehouse"], 100.0)
        self.assertEqual(last.loc["CU", "valid_observations"], 17)
        self.assertTrue(pd.isna(last.loc["CU", "smoothed_warehouse"]))


class SWarehouseLoadingAndWrapperTest(unittest.TestCase):
    def test_loads_buffered_warehouse_and_calendar_history(self):
        function = getattr(s_warehouse, "load_s_warehouse", None)
        self.assertTrue(callable(function))

        dates = pd.bdate_range("2023-11-22", periods=6)
        warehouse = pd.DataFrame(
            {
                "trade_date": dates,
                "fut_code": ["RB"] * len(dates),
                "warehouse_total": [100.0] * len(dates),
                "quality_status": ["valid"] * len(dates),
            }
        )
        calendar = pd.DataFrame({"trade_date": dates})

        with (
            patch.object(
                s_warehouse,
                "load_warehouse_daily",
                return_value=warehouse,
            ) as mock_warehouse,
            patch.object(
                s_warehouse,
                "load_trade_calendar",
                return_value=calendar,
            ) as mock_calendar,
        ):
            result = function(
                start_date="20240101",
                end_date="20241231",
                lookback=2,
                smooth_window=3,
                min_observations=2,
            )

        mock_warehouse.assert_called_once_with("20231122", "20241231")
        mock_calendar.assert_called_once_with("20231122", "20241231")
        self.assertIn("s_warehouse", result.columns)

    def test_calculate_factor_applies_defaults_and_standard_output_contract(self):
        function = getattr(s_warehouse, "calculate_factor", None)
        self.assertTrue(callable(function))
        panel = pd.DataFrame(
            {
                "trade_date": pd.to_datetime(
                    ["2023-12-29", "2024-01-02", "2024-01-03", "2024-01-04"]
                ),
                "fut_code": ["RB"] * 4,
                "s_warehouse": [0.1, 0.2, 0.3, 0.4],
            }
        )

        with patch.object(
            s_warehouse,
            "load_s_warehouse",
            return_value=panel,
        ) as mock_load:
            result = function(
                "20240101",
                "20240103",
                {
                    "lookback": 2,
                    "smooth_window": 3,
                    "min_observations": 2,
                },
            )

        mock_load.assert_called_once_with(
            "20240101",
            "20240103",
            2,
            3,
            2,
        )
        self.assertEqual(
            result["trade_date"].tolist(),
            list(pd.to_datetime(["2024-01-02", "2024-01-03"])),
        )
        self.assertEqual(result["raw_factor"].tolist(), [0.2, 0.3])


if __name__ == "__main__":
    unittest.main()
