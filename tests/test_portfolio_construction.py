import unittest

import pandas as pd

from config.research_config import ResearchConfig
from src.p05_portfolio_construction import build_target_weights


class PortfolioConstructionTest(unittest.TestCase):
    def setUp(self):
        date = pd.Timestamp("2024-01-02")
        codes = [f"F{i:02d}" for i in range(10)]
        self.factor_data = pd.DataFrame(
            {
                "trade_date": [date] * 10,
                "fut_code": codes,
                "raw_factor": list(range(10)),
            }
        )
        self.contract_data = pd.DataFrame(
            {
                "trade_date": [date] * 10,
                "fut_code": codes,
                "ts_code_A": [f"{code}01" for code in codes],
                "avg_vol_A": [5000.0] * 10,
                "avg_oi_A": [30000.0] * 10,
                "avg_amount_A": [10000.0] * 10,
            }
        )

    def test_rank_weights_are_market_neutral_with_half_long_and_short(self):
        result = build_target_weights(
            self.factor_data,
            self.contract_data,
            ResearchConfig(),
            normalize="rank",
        )

        self.assertAlmostEqual(result.loc[result["weight"] > 0, "weight"].sum(), 0.5)
        self.assertAlmostEqual(result.loc[result["weight"] < 0, "weight"].sum(), -0.5)
        self.assertAlmostEqual(result["weight"].sum(), 0.0)

    def test_liquidity_filter_and_minimum_asset_count_apply_on_rebalance(self):
        context = self.contract_data.copy()
        context.loc[0, "avg_oi_A"] = 1.0
        result = build_target_weights(
            self.factor_data,
            context,
            ResearchConfig(min_assets=10),
        )

        self.assertTrue(result["weight_factor"].isna().all())
        self.assertTrue(result["weight"].eq(0).all())

    def test_zscore_clip_limits_factor_not_portfolio_direction(self):
        result = build_target_weights(
            self.factor_data,
            self.contract_data,
            ResearchConfig(),
            normalize="zscore",
            zscore_clip=1.0,
        )

        self.assertLessEqual(result["factor"].abs().max(), 1.0)
        self.assertAlmostEqual(result["weight"].sum(), 0.0)


if __name__ == "__main__":
    unittest.main()
