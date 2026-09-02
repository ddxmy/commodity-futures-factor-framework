import unittest

import numpy as np
import pandas as pd


class TreeModelFactoryTest(unittest.TestCase):
    def test_builds_deterministic_lightgbm_and_xgboost_regressors(self):
        from src.tree_five_factor_strategy import build_tree_model

        features = np.array(
            [
                [-1.0, -0.5],
                [-0.5, 0.5],
                [0.5, -0.5],
                [1.0, 0.5],
            ]
            * 30
        )
        target = features[:, 0] * features[:, 1]

        for model_name in ("lightgbm", "xgboost"):
            first = build_tree_model(
                model_name,
                {"n_estimators": 20, "max_depth": 2},
            )
            second = build_tree_model(
                model_name,
                {"n_estimators": 20, "max_depth": 2},
            )
            first.fit(features, target)
            second.fit(features, target)

            np.testing.assert_allclose(
                first.predict(features),
                second.predict(features),
            )

    def test_rejects_unknown_model_name(self):
        from src.tree_five_factor_strategy import build_tree_model

        with self.assertRaisesRegex(ValueError, "unsupported tree model"):
            build_tree_model("random_forest", {})


class RollingTreePredictionTest(unittest.TestCase):
    @staticmethod
    def _panel():
        rows = []
        for year in range(2015, 2021):
            for month in (1, 4, 7, 10):
                date = pd.Timestamp(year=year, month=month, day=2)
                for fut_code, score in (("A", -1.0), ("B", 0.0), ("C", 1.0)):
                    rows.append(
                        {
                            "trade_date": date,
                            "exit_date": date + pd.Timedelta(days=1),
                            "fut_code": fut_code,
                            "factor_a": score,
                            "factor_b": score**2,
                            "forward_return": (
                                score * 0.02 if year < 2020 else np.nan
                            ),
                        }
                    )
        return pd.DataFrame(rows)

    def test_rolling_predictions_use_prior_five_years_and_record_parameters(self):
        from src.tree_five_factor_strategy import fit_rolling_tree_predictions

        predictions, summary, diagnostics = fit_rolling_tree_predictions(
            self._panel(),
            feature_columns=["factor_a", "factor_b"],
            prediction_years=[2020],
            model_name="lightgbm",
            parameter_grid=[{"n_estimators": 20, "max_depth": 2}],
            lookback_years=5,
        )

        self.assertEqual(predictions["trade_date"].dt.year.unique().tolist(), [2020])
        self.assertIn("lightgbm_prediction", predictions.columns)
        self.assertEqual(summary.loc[0, "training_start"], pd.Timestamp("2015-01-02"))
        self.assertEqual(summary.loc[0, "training_end"], pd.Timestamp("2019-10-02"))
        self.assertEqual(summary.loc[0, "selected_candidate"], 0)
        self.assertTrue((diagnostics["prediction_year"] == 2020).all())


class PairedRankICTest(unittest.TestCase):
    def test_reports_positive_difference_for_better_predictions(self):
        from src.tree_five_factor_strategy import paired_rank_ic_test

        rows = []
        for day in pd.date_range("2020-01-01", periods=30, freq="D"):
            for rank in range(5):
                realized = float(rank)
                rows.append(
                    {
                        "trade_date": day,
                        "forward_return": realized,
                        "better": realized,
                        "baseline": -realized,
                    }
                )
        panel = pd.DataFrame(rows)

        result = paired_rank_ic_test(
            panel,
            model_column="better",
            baseline_column="baseline",
            max_lags=3,
        )

        self.assertAlmostEqual(result["mean_rank_ic_difference"], 2.0)
        self.assertGreater(result["t_stat"], 0)
        self.assertEqual(result["observations"], 30)


class PredictionMergeTest(unittest.TestCase):
    def test_merges_models_only_on_identical_unique_prediction_keys(self):
        from src.tree_five_factor_strategy import merge_model_predictions

        ridge = pd.DataFrame(
            {
                "trade_date": pd.to_datetime(["2020-01-02", "2020-01-02"]),
                "fut_code": ["A", "B"],
                "forward_return": [0.01, -0.01],
                "ridge_prediction": [0.02, -0.02],
                "equal_weight_signal": [1.0, -1.0],
            }
        )
        lightgbm = ridge[["trade_date", "fut_code"]].copy()
        lightgbm["lightgbm_prediction"] = [0.03, -0.03]
        xgboost = ridge[["trade_date", "fut_code"]].copy()
        xgboost["xgboost_prediction"] = [0.04, -0.04]

        result = merge_model_predictions(ridge, lightgbm, xgboost)

        self.assertEqual(len(result), 2)
        self.assertEqual(
            result.columns.tolist(),
            [
                "trade_date",
                "fut_code",
                "forward_return",
                "ridge_prediction",
                "equal_weight_signal",
                "lightgbm_prediction",
                "xgboost_prediction",
            ],
        )

    def test_rejects_missing_model_prediction_rows(self):
        from src.tree_five_factor_strategy import merge_model_predictions

        ridge = pd.DataFrame(
            {
                "trade_date": pd.to_datetime(["2020-01-02", "2020-01-02"]),
                "fut_code": ["A", "B"],
                "forward_return": [0.01, -0.01],
                "ridge_prediction": [0.02, -0.02],
                "equal_weight_signal": [1.0, -1.0],
            }
        )
        lightgbm = ridge.loc[[0], ["trade_date", "fut_code"]].copy()
        lightgbm["lightgbm_prediction"] = [0.03]
        xgboost = ridge[["trade_date", "fut_code"]].copy()
        xgboost["xgboost_prediction"] = [0.04, -0.04]

        with self.assertRaisesRegex(ValueError, "prediction keys do not match"):
            merge_model_predictions(ridge, lightgbm, xgboost)


if __name__ == "__main__":
    unittest.main()
