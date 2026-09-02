import sqlite3
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from src.p01_market_data import load_contract_daily


class ContractDailyMaturityFilterTest(unittest.TestCase):
    def _build_database(self, path: Path) -> None:
        with sqlite3.connect(path) as connection:
            connection.executescript(
                """
                CREATE TABLE tradable_universe (
                    trade_date TEXT,
                    exchange TEXT,
                    fut_code TEXT,
                    is_tradable INTEGER
                );
                CREATE TABLE fut_basic (
                    exchange TEXT,
                    fut_code TEXT,
                    ts_code TEXT,
                    list_date TEXT,
                    delist_date TEXT
                );
                CREATE TABLE fut_daily (
                    trade_date TEXT,
                    ts_code TEXT,
                    open REAL,
                    close REAL,
                    vol REAL,
                    oi REAL,
                    amount REAL
                );
                CREATE TABLE trade_cal (
                    cal_date TEXT,
                    is_open INTEGER
                );
                """
            )
            connection.execute(
                "INSERT INTO tradable_universe VALUES (?, ?, ?, ?)",
                ("20240102", "SHFE", "RB", 1),
            )
            connection.executemany(
                "INSERT INTO fut_basic VALUES (?, ?, ?, ?, ?)",
                [
                    ("SHFE", "RB", "RB_NEAR", "20230101", "20240122"),
                    ("SHFE", "RB", "RB_FAR", "20230101", "20240302"),
                ],
            )
            connection.executemany(
                "INSERT INTO fut_daily VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    ("20240102", "RB_NEAR", 100, 101, 5000, 30000, 10000),
                    ("20240102", "RB_FAR", 100, 101, 4000, 30000, 10000),
                ],
            )
            connection.execute(
                "INSERT INTO trade_cal VALUES (?, ?)",
                ("20240102", 1),
            )

    def test_explicit_cutoff_changes_the_selected_contract_universe(self):
        with TemporaryDirectory() as directory:
            database = Path(directory) / "market.db"
            self._build_database(database)

            unfiltered = load_contract_daily(
                "20240102",
                "20240102",
                db_path=str(database),
                min_days_to_maturity=0,
            )
            conservative = load_contract_daily(
                "20240102",
                "20240102",
                db_path=str(database),
                min_days_to_maturity=45,
            )

        self.assertEqual(unfiltered.iloc[0]["ts_code"], "RB_NEAR")
        self.assertEqual(unfiltered.iloc[0]["rank_by_vol"], 1)
        self.assertEqual(conservative["ts_code"].tolist(), ["RB_FAR"])

    def test_negative_cutoff_fails_before_database_access(self):
        with patch("src.p01_market_data.get_connection") as connection:
            with self.assertRaises(ValueError):
                load_contract_daily(
                    "20240102",
                    "20240102",
                    min_days_to_maturity=-1,
                )

        connection.assert_not_called()


if __name__ == "__main__":
    unittest.main()
