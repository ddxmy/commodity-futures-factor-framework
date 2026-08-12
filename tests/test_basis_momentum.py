import unittest

import numpy as np
import pandas as pd

from src.factors.basis_momentum import compute_basis_components


class BasisMomentumFormulaTest(unittest.TestCase):
    def test_computes_ab_and_bc_with_signed_maturity_gaps(self):
        dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
        mapping = pd.DataFrame(
            {
                "trade_date": dates,
                "fut_code": ["RB"] * 3,
                "ts_code_A": ["A"] * 3,
                "daily_return_A": [0.02, 0.03, 0.01],
                "daily_return_B": [0.01, 0.01, 0.02],
                "daily_return_C": [0.00, 0.02, 0.00],
                "d_AB": [30.0, 30.0, 30.0],
                "d_BC": [-60.0, -60.0, -60.0],
            }
        )
        calendar = pd.DataFrame({"trade_date": dates})

        result = compute_basis_components(mapping, calendar, lookback=2)

        expected_ab = np.mean([(0.02 - 0.01) / 30 * 365, (0.03 - 0.01) / 30 * 365])
        expected_bc = np.mean([(0.01 - 0.00) / -60 * 365, (0.01 - 0.02) / -60 * 365])
        self.assertAlmostEqual(result.loc[1, "factor_AB"], expected_ab)
        self.assertAlmostEqual(result.loc[1, "factor_BC"], expected_bc)

    def test_zero_maturity_gap_only_invalidates_that_component(self):
        dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
        mapping = pd.DataFrame(
            {
                "trade_date": dates,
                "fut_code": ["RB", "RB"],
                "ts_code_A": ["A", "A"],
                "daily_return_A": [0.02, 0.02],
                "daily_return_B": [0.01, 0.01],
                "daily_return_C": [0.00, 0.00],
                "d_AB": [0.0, 0.0],
                "d_BC": [30.0, 30.0],
            }
        )

        result = compute_basis_components(
            mapping,
            pd.DataFrame({"trade_date": dates}),
            lookback=2,
        )

        self.assertTrue(result["factor_AB"].isna().all())
        self.assertTrue(result["factor_BC"].iloc[-1] > 0)


if __name__ == "__main__":
    unittest.main()
