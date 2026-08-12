import unittest

import numpy as np
import pandas as pd

from src.portfolio.turnover_optimizer import optimize_portfolio


class InitialPortfolioTest(unittest.TestCase):
    def test_builds_market_neutral_initial_portfolio(self):
        positive_codes = [f"P{i:02d}" for i in range(10)]
        negative_codes = [f"N{i:02d}" for i in range(10)]
        codes = positive_codes + negative_codes

        scores = pd.Series(
            np.concatenate(
                [
                    np.linspace(0.1, 1.0, 10),
                    np.linspace(-0.1, -1.0, 10),
                ]
            ),
            index=codes,
            dtype=float,
        )

        previous_weights = pd.Series(0.0, index=codes)
        eligible = pd.Series(True, index=codes)
        contract_changed = pd.Series(False, index=codes)

        weights, diagnostics = optimize_portfolio(
            scores=scores,
            previous_weights=previous_weights,
            eligible=eligible,
            contract_changed=contract_changed,
            is_initial=True,
        )

        self.assertAlmostEqual(weights.sum(), 0.0, places=6)
        self.assertAlmostEqual(weights.abs().sum(), 1.0, places=6)
        self.assertLessEqual(weights.abs().max(), 0.05 + 1e-6)
        self.assertGreater(weights.loc["P09"], 0.0)
        self.assertLess(weights.loc["N09"], 0.0)
        self.assertAlmostEqual(
            diagnostics["effective_turnover_limit"],
            1.0,
        )


class InputValidationTest(unittest.TestCase):
    def setUp(self):
        self.codes = ["A", "B"]
        self.scores = pd.Series([1.0, -1.0], index=self.codes)
        self.previous_weights = pd.Series(0.0, index=self.codes)
        self.eligible = pd.Series(True, index=self.codes)
        self.contract_changed = pd.Series(False, index=self.codes)

    def test_rejects_misaligned_commodity_indices(self):
        reversed_weights = self.previous_weights.iloc[::-1]

        with self.assertRaises(ValueError):
            optimize_portfolio(
                scores=self.scores,
                previous_weights=reversed_weights,
                eligible=self.eligible,
                contract_changed=self.contract_changed,
                is_initial=True,
            )

    def test_rejects_nonfinite_scores(self):
        invalid_scores = self.scores.copy()
        invalid_scores.loc["A"] = np.nan

        with self.assertRaises(ValueError):
            optimize_portfolio(
                scores=invalid_scores,
                previous_weights=self.previous_weights,
                eligible=self.eligible,
                contract_changed=self.contract_changed,
                is_initial=True,
            )

    def test_rejects_nonpositive_limits(self):
        for parameter in ["turnover_limit", "max_abs_weight"]:
            with self.subTest(parameter=parameter):
                arguments = {parameter: 0.0}

                with self.assertRaises(ValueError):
                    optimize_portfolio(
                        scores=self.scores,
                        previous_weights=self.previous_weights,
                        eligible=self.eligible,
                        contract_changed=self.contract_changed,
                        is_initial=True,
                        **arguments,
                    )


class TurnoverConstraintTest(unittest.TestCase):
    def setUp(self):
        positive_codes = [f"P{i:02d}" for i in range(10)]
        negative_codes = [f"N{i:02d}" for i in range(10)]
        self.codes = positive_codes + negative_codes

        self.scores = pd.Series(
            np.concatenate(
                [
                    np.linspace(0.1, 1.0, 10),
                    np.linspace(-0.1, -1.0, 10),
                ]
            ),
            index=self.codes,
            dtype=float,
        )

        self.previous_weights = pd.Series(
            [0.05] * 10 + [-0.05] * 10,
            index=self.codes,
            dtype=float,
        )

        self.eligible = pd.Series(True, index=self.codes)
        self.contract_changed = pd.Series(False, index=self.codes)

    def test_limits_ordinary_rebalance_turnover(self):
        weights, diagnostics = optimize_portfolio(
            scores=-self.scores,
            previous_weights=self.previous_weights,
            eligible=self.eligible,
            contract_changed=self.contract_changed,
            turnover_limit=0.15,
        )

        self.assertLessEqual(
            diagnostics["optimized_turnover"],
            0.15 + 1e-6,
        )
        self.assertTrue(diagnostics["constraint_binding"])
        self.assertAlmostEqual(weights.sum(), 0.0, places=6)

    def test_contract_roll_close_and_reopen_uses_turnover_budget(self):
        contract_changed = self.contract_changed.copy()
        contract_changed.loc["P09"] = True

        weights, diagnostics = optimize_portfolio(
            scores=self.scores,
            previous_weights=self.previous_weights,
            eligible=self.eligible,
            contract_changed=contract_changed,
            turnover_limit=0.15,
        )

        expected_roll_turnover = (
            abs(self.previous_weights.loc["P09"])
            + abs(weights.loc["P09"])
        )

        self.assertGreaterEqual(
            diagnostics["optimized_turnover"],
            expected_roll_turnover - 1e-6,
        )
        self.assertLessEqual(
            diagnostics["optimized_turnover"],
            0.15 + 1e-6,
        )

    def test_ineligible_position_is_closed_outside_active_budget(self):
        eligible = self.eligible.copy()
        eligible.loc["P09"] = False

        weights, diagnostics = optimize_portfolio(
            scores=self.scores,
            previous_weights=self.previous_weights,
            eligible=eligible,
            contract_changed=self.contract_changed,
            turnover_limit=0.15,
        )

        self.assertAlmostEqual(weights.loc["P09"], 0.0, places=8)
        self.assertAlmostEqual(
            diagnostics["mandatory_exit_turnover"],
            0.05,
            places=6,
        )
        self.assertLessEqual(
            diagnostics["optimized_turnover"],
            0.15 + 1e-6,
        )

    def test_relaxes_limit_to_minimum_feasible_roll_turnover(self):
        contract_changed = self.contract_changed.copy()
        contract_changed.loc[
            ["P08", "P09", "N08", "N09"]
        ] = True

        weights, diagnostics = optimize_portfolio(
            scores=self.scores,
            previous_weights=self.previous_weights,
            eligible=self.eligible,
            contract_changed=contract_changed,
            turnover_limit=0.15,
        )

        self.assertTrue(
            diagnostics["turnover_limit_relaxed"]
        )
        self.assertGreaterEqual(
            diagnostics["effective_turnover_limit"],
            0.20 - 1e-6,
        )
        self.assertLessEqual(
            diagnostics["optimized_turnover"],
            diagnostics["effective_turnover_limit"] + 1e-6,
        )
        self.assertAlmostEqual(weights.sum(), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
