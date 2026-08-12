import unittest
from unittest.mock import patch

import pandas as pd

from src.p02_contract_selection import build_contract_mapping


class ContractSelectionTest(unittest.TestCase):
    @patch("src.p02_contract_selection.load_contract_daily")
    def test_builds_unique_abc_mapping_from_liquidity_ranks(self, mock_load):
        dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
        records = []
        for date_index, trade_date in enumerate(dates):
            for rank, code in enumerate(["A1", "A2", "A3", "A4"], start=1):
                records.append(
                    {
                        "trade_date": trade_date,
                        "fut_code": "A",
                        "ts_code": code,
                        "vol": 5000 - rank,
                        "oi": 10000 - rank,
                        "amount": 20000 - rank,
                        "close": 100 + date_index + rank,
                        "days_to_maturity": rank * 30,
                        "avg_vol": 4000 - rank,
                        "avg_oi": 9000 - rank,
                        "avg_amount": 18000 - rank,
                        "rank_by_vol": rank,
                    }
                )
        mock_load.return_value = pd.DataFrame(records).set_index("trade_date")

        result = build_contract_mapping("20240101", "20240131")

        self.assertFalse(result.duplicated(["trade_date", "fut_code"]).any())
        self.assertEqual(result["ts_code_A"].unique().tolist(), ["A1"])
        self.assertEqual(result["ts_code_B"].unique().tolist(), ["A2"])
        self.assertEqual(result["ts_code_C"].unique().tolist(), ["A3"])
        self.assertTrue(result["d_AB"].eq(30).all())
        self.assertTrue(result["d_BC"].eq(30).all())


if __name__ == "__main__":
    unittest.main()
