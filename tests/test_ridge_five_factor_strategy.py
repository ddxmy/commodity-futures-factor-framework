import unittest

import numpy as np
import pandas as pd


class RankMappingTest(unittest.TestCase):
    def test_equal_weight_signal_sums_the_five_mapped_scores(self):
        from src.ridge_five_factor_strategy import build_equal_weight_signal

        panel = pd.DataFrame(
            {
                "a_score": [-1.0, 0.5],
                "b_score": [0.0, 1.0],
                "c_score": [0.5, -0.5],
            }
        )

        result = build_equal_weight_signal(
            panel,
            ["a_score", "b_score", "c_score"],
        )

        np.testing.assert_allclose(result["equal_weight_signal"], [-0.5, 1.0])

    def test_maps_each_daily_cross_section_to_minus_one_and_one(self):
        from src.ridge_five_factor_strategy import map_factor_ranks

        panel = pd.DataFrame(
            {
                "trade_date": pd.to_datetime(["2020-01-02"] * 4),
                "fut_code": ["A", "B", "C", "D"],
                "carry": [10.0, 20.0, 20.0, 40.0],
            }
        )

        result = map_factor_ranks(panel, ["carry"])

        np.testing.assert_allclose(
            result["carry_score"],
            [-1.0, 0.0, 0.0, 1.0],
        )

    def test_missing_factor_values_map_to_zero_without_changing_valid_ranks(self):
        from src.ridge_five_factor_strategy import map_factor_ranks

        panel = pd.DataFrame(
            {
                "trade_date": pd.to_datetime(["2020-01-02"] * 4),
                "fut_code": ["A", "B", "C", "D"],
                "carry": [1.0, np.nan, 2.0, 3.0],
            }
        )

        result = map_factor_ranks(panel, ["carry"])

        np.testing.assert_allclose(
            result["carry_score"],
            [-1.0, 0.0, 0.0, 1.0],
        )


class ForwardLabelTest(unittest.TestCase):
    def test_uses_next_two_opens_of_the_signal_date_contract(self):
        from src.ridge_five_factor_strategy import build_next_open_labels

        signals = pd.DataFrame(
            {
                "trade_date": pd.to_datetime(
                    ["2020-01-02", "2020-01-03"]
                ),
                "fut_code": ["RB", "RB"],
                "ts_code_A": ["RB2005", "RB2010"],
            }
        )
        calendar = pd.DataFrame(
            {
                "trade_date": pd.to_datetime(
                    [
                        "2020-01-02",
                        "2020-01-03",
                        "2020-01-06",
                        "2020-01-07",
                    ]
                )
            }
        )
        prices = pd.DataFrame(
            {
                "trade_date": pd.to_datetime(
                    [
                        "2020-01-03",
                        "2020-01-06",
                        "2020-01-03",
                        "2020-01-06",
                        "2020-01-07",
                    ]
                ),
                "ts_code": [
                    "RB2005",
                    "RB2005",
                    "RB2010",
                    "RB2010",
                    "RB2010",
                ],
                "open": [100.0, 110.0, 200.0, 220.0, 242.0],
            }
        )

        result = build_next_open_labels(signals, calendar, prices)

        np.testing.assert_allclose(result["forward_return"], [0.10, 0.10])
        self.assertEqual(
            result["exit_date"].tolist(),
            [pd.Timestamp("2020-01-06"), pd.Timestamp("2020-01-07")],
        )


class RollingTrainingBoundaryTest(unittest.TestCase):
    def test_excludes_labels_realized_on_or_after_prediction_year(self):
        from src.ridge_five_factor_strategy import training_sample_for_year

        panel = pd.DataFrame(
            {
                "trade_date": pd.to_datetime(
                    ["2014-12-31", "2015-01-02", "2019-12-30", "2019-12-31"]
                ),
                "exit_date": pd.to_datetime(
                    ["2015-01-05", "2015-01-06", "2019-12-31", "2020-01-02"]
                ),
                "forward_return": [0.01, 0.02, 0.03, 0.04],
            }
        )

        result = training_sample_for_year(
            panel,
            prediction_year=2020,
            lookback_years=5,
        )

        self.assertEqual(
            result["trade_date"].tolist(),
            [pd.Timestamp("2015-01-02"), pd.Timestamp("2019-12-30")],
        )


class RollingRidgeTest(unittest.TestCase):
    def test_relative_coefficients_sum_to_one_within_each_year(self):
        from src.ridge_five_factor_strategy import add_relative_coefficients

        summary = pd.DataFrame(
            {
                "coef_a": [2.0],
                "coef_b": [-1.0],
            }
        )

        result = add_relative_coefficients(summary, ["coef_a", "coef_b"])

        np.testing.assert_allclose(
            result[["relative_coef_a", "relative_coef_b"]].iloc[0],
            [2.0 / 3.0, -1.0 / 3.0],
        )

    def test_unit_sample_penalty_scales_with_training_rows(self):
        from src.ridge_five_factor_strategy import ridge_alpha_from_penalty

        self.assertEqual(ridge_alpha_from_penalty(0.25, 400), 100.0)

    def test_forward_year_folds_never_train_on_validation_or_later_years(self):
        from src.ridge_five_factor_strategy import build_forward_year_folds

        panel = pd.DataFrame(
            {
                "trade_date": pd.to_datetime(
                    [f"{year}-01-02" for year in range(2015, 2020)]
                )
            }
        )

        folds = build_forward_year_folds(panel, min_train_years=2)

        self.assertEqual(
            [(fold["train_years"], fold["validation_year"]) for fold in folds],
            [
                ((2015, 2016), 2017),
                ((2015, 2016, 2017), 2018),
                ((2015, 2016, 2017, 2018), 2019),
            ],
        )

    def test_rolling_ridge_predicts_one_year_from_prior_five_years_only(self):
        from src.ridge_five_factor_strategy import fit_rolling_ridge_predictions

        rows = []
        for year in range(2015, 2021):
            for fut_code, score in (("A", -1.0), ("B", 1.0)):
                rows.append(
                    {
                        "trade_date": pd.Timestamp(year=year, month=1, day=2),
                        "exit_date": pd.Timestamp(year=year, month=1, day=3),
                        "fut_code": fut_code,
                        "carry_score": score,
                        "forward_return": (
                            score * 0.10 if year < 2020 else np.nan
                        ),
                    }
                )
        panel = pd.DataFrame(rows)

        predictions, summary, _ = fit_rolling_ridge_predictions(
            panel,
            feature_columns=["carry_score"],
            prediction_years=[2020],
            alphas=[1.0],
            lookback_years=5,
        )

        self.assertEqual(predictions["trade_date"].dt.year.unique().tolist(), [2020])
        predicted = predictions.set_index("fut_code")["ridge_prediction"]
        self.assertLess(predicted["A"], predicted["B"])
        self.assertEqual(summary.loc[0, "training_start"], pd.Timestamp("2015-01-02"))
        self.assertEqual(summary.loc[0, "training_end"], pd.Timestamp("2019-01-02"))
        self.assertEqual(summary.loc[0, "penalty_lambda"], 1.0)
        self.assertEqual(summary.loc[0, "alpha"], 10.0)


if __name__ == "__main__":
    unittest.main()
