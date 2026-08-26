import unittest

import numpy as np
import pandas as pd
from unittest.mock import patch

from src.factors import basis_momentum
from src.factors.basis_momentum import (
    calculate_factor,
    compute_basis_components,
    load_basis_components,
)


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


class BasisMomentumMaturityParameterTest(unittest.TestCase):
    def test_loading_forwards_the_signal_maturity_cutoff(self):
        dates = pd.to_datetime(["2023-11-28", "2023-11-29"])
        mapping = pd.DataFrame(
            {
                "trade_date": dates,
                "fut_code": ["RB", "RB"],
                "ts_code_A": ["A", "A"],
                "daily_return_A": [0.02, 0.03],
                "daily_return_B": [0.01, 0.01],
                "daily_return_C": [0.00, 0.02],
                "d_AB": [30.0, 30.0],
                "d_BC": [60.0, 60.0],
            }
        )
        calendar = pd.DataFrame({"trade_date": dates})

        with (
            patch.object(
                basis_momentum,
                "build_contract_mapping",
                return_value=mapping,
            ) as mapping_loader,
            patch.object(
                basis_momentum,
                "load_trade_calendar",
                return_value=calendar,
            ),
        ):
            load_basis_components(
                "20240101",
                "20241231",
                lookback=2,
                signal_min_days_to_maturity=0,
            )

        mapping_loader.assert_called_once_with(
            "20231128",
            "20241231",
            min_days_to_maturity=0,
        )

    def test_factor_rejects_invalid_signal_maturity_cutoffs(self):
        with patch.object(basis_momentum, "load_basis_components") as loader:
            for value in [-1, True, 1.5]:
                with self.subTest(value=value), self.assertRaises(ValueError):
                    calculate_factor(
                        "20240101",
                        "20241231",
                        {
                            "lookback": 2,
                            "signal_min_days_to_maturity": value,
                        },
                    )

        loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
