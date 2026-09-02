import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from src.factor_loader import calculate_factor, load_factor_module
from src.factors import _factor_template
from src.p03_factor_processing import validate_factor_data


class FactorContractTest(unittest.TestCase):
    def setUp(self):
        self.valid = pd.DataFrame(
            {
                "trade_date": ["2024-01-03", "2024-01-02"],
                "fut_code": ["CU", "RB"],
                "raw_factor": [np.nan, 1.5],
            }
        )

    def test_accepts_missing_values_and_sorts_without_mutating_input(self):
        original = self.valid.copy(deep=True)
        result = validate_factor_data(self.valid)

        pd.testing.assert_frame_equal(self.valid, original)
        self.assertEqual(result["fut_code"].tolist(), ["RB", "CU"])

    def test_rejects_invalid_factor_panels(self):
        cases = []
        cases.append(self.valid.drop(columns="raw_factor"))
        cases.append(pd.concat([self.valid, self.valid.iloc[[0]]], ignore_index=True))
        cases.append(self.valid.assign(raw_factor=["x", "y"]))
        cases.append(self.valid.assign(raw_factor=[np.inf, 1.0]))

        for factor_data in cases:
            with self.subTest(columns=factor_data.columns.tolist()):
                with self.assertRaises((TypeError, ValueError)):
                    validate_factor_data(factor_data)

    def test_rejects_future_available_date_and_nontrading_date(self):
        future = self.valid.assign(
            available_date=["2024-01-04", "2024-01-02"]
        )
        with self.assertRaises(ValueError):
            validate_factor_data(future)

        calendar = pd.DataFrame({"trade_date": ["2024-01-02"]})
        with self.assertRaises(ValueError):
            validate_factor_data(self.valid, calendar)


class FactorLoaderTest(unittest.TestCase):
    def test_rejects_unsafe_or_unknown_names(self):
        for name in ["_private", "bad-name", "../factor"]:
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    load_factor_module(name)

        with self.assertRaises(ValueError):
            load_factor_module("factor_that_does_not_exist")

    @patch("src.factor_loader.load_factor_module")
    def test_merges_defaults_and_validates_result(self, mock_loader):
        module = Mock()
        module.DEFAULT_PARAMETERS = {"lookback": 120, "variant": "AB"}
        module.calculate_factor.return_value = pd.DataFrame(
            {
                "trade_date": ["2024-01-02"],
                "fut_code": ["RB"],
                "raw_factor": [1.0],
            }
        )
        mock_loader.return_value = module

        result = calculate_factor(
            "demo",
            "20240101",
            "20240131",
            {"lookback": 252},
        )

        self.assertEqual(result["raw_factor"].tolist(), [1.0])
        module.calculate_factor.assert_called_once_with(
            "20240101",
            "20240131",
            {"lookback": 252, "variant": "AB"},
        )


class FactorTemplateTest(unittest.TestCase):
    def test_standard_wrapper_applies_defaults_trims_dates_and_sets_raw_factor(self):
        panel = pd.DataFrame(
            {
                "trade_date": pd.to_datetime(
                    ["2023-12-29", "2024-01-02", "2024-01-03"]
                ),
                "fut_code": ["RB", "RB", "RB"],
                "factor_value": [0.10, 0.20, 0.30],
            }
        )
        calls = []

        def fake_load(
            start_date,
            end_date,
            lookback,
            signal_min_days_to_maturity,
        ):
            calls.append(
                (
                    start_date,
                    end_date,
                    lookback,
                    signal_min_days_to_maturity,
                )
            )
            return panel

        with patch.object(
            _factor_template,
            "load_factor_data",
            side_effect=fake_load,
            create=True,
        ):
            try:
                result = _factor_template.calculate_factor(
                    "20240101",
                    "20240103",
                )
            except NotImplementedError:
                self.fail("factor template must implement the standard wrapper")

        self.assertEqual(calls, [("20240101", "20240103", 90, 0)])
        self.assertEqual(
            result["trade_date"].tolist(),
            list(pd.to_datetime(["2024-01-02", "2024-01-03"])),
        )
        self.assertEqual(result["raw_factor"].tolist(), [0.20, 0.30])


if __name__ == "__main__":
    unittest.main()
