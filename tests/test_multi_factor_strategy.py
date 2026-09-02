import unittest
from contextlib import closing
from importlib import import_module
from pathlib import Path
import sqlite3
import tempfile
from unittest.mock import patch
import warnings

import numpy as np
import pandas as pd

from config.settings import RESULT_DIR


def load_strategy_module():
    try:
        return import_module("src.multi_factor_strategy")
    except ModuleNotFoundError as error:
        raise AssertionError(
            "src.multi_factor_strategy must exist"
        ) from error


def require_callable(module, name):
    function = getattr(module, name, None)
    if not callable(function):
        raise AssertionError(f"{name} must be callable")
    return function


class ModuleContractTest(unittest.TestCase):
    def test_default_settings_reproduce_the_approved_research_design(self):
        strategy = load_strategy_module()

        settings = strategy.StrategySettings()

        self.assertEqual(settings.start_date, "20200101")
        self.assertEqual(settings.end_date, "20260701")
        self.assertEqual(settings.liquidity_lookback, 120)
        self.assertEqual(settings.min_amount_observations, 96)
        self.assertEqual(settings.pool_size, 40)
        self.assertEqual(settings.volatility_lookback, 20)
        self.assertEqual(settings.min_assets, 10)
        self.assertEqual(settings.trade_min_days_to_maturity, 45)
        self.assertEqual(settings.cost_rate, 0.0005)
        self.assertEqual(
            strategy.PORTFOLIO_METHODS,
            ("full_pool_invvol", "tail10_invvol"),
        )

    def test_factor_specs_fix_the_five_existing_factor_definitions(self):
        strategy = load_strategy_module()

        self.assertEqual(
            strategy.FACTOR_SPECS,
            {
                "basis_momentum": {
                    "variant": "AB",
                    "lookback": 252,
                },
                "carry": {"lookback": 90},
                "spotmain": {"lookback": 90},
                "s_warehouse": {
                    "lookback": 90,
                    "smooth_window": 20,
                    "min_observations": 18,
                },
                "t_rank": {"lookback": 20},
            },
        )


class ArgumentValidationTest(unittest.TestCase):
    def test_default_cli_arguments_match_the_research_settings(self):
        strategy = load_strategy_module()
        parse_arguments = require_callable(strategy, "parse_arguments")

        arguments = parse_arguments([])

        self.assertEqual(arguments.start, "20200101")
        self.assertEqual(arguments.end, "20260701")
        self.assertEqual(arguments.liquidity_lookback, 120)
        self.assertEqual(arguments.min_amount_observations, 96)
        self.assertEqual(arguments.pool_size, 40)
        self.assertEqual(arguments.volatility_lookback, 20)
        self.assertEqual(arguments.min_assets, 10)
        self.assertEqual(arguments.trade_min_days_to_maturity, 45)
        self.assertEqual(arguments.cost_rate, 0.0005)
        self.assertEqual(arguments.result_dir, Path(RESULT_DIR))
        self.assertFalse(arguments.overwrite)

    def test_cli_accepts_explicit_values_and_overwrite(self):
        strategy = load_strategy_module()
        parse_arguments = require_callable(strategy, "parse_arguments")

        arguments = parse_arguments(
            [
                "--start",
                "20210101",
                "--end",
                "20220101",
                "--liquidity-lookback",
                "100",
                "--min-amount-observations",
                "80",
                "--pool-size",
                "35",
                "--volatility-lookback",
                "30",
                "--min-assets",
                "12",
                "--trade-min-days-to-maturity",
                "60",
                "--cost-rate",
                "0.0003",
                "--result-dir",
                "custom-results",
                "--overwrite",
            ]
        )

        self.assertEqual(arguments.start, "20210101")
        self.assertEqual(arguments.end, "20220101")
        self.assertEqual(arguments.liquidity_lookback, 100)
        self.assertEqual(arguments.min_amount_observations, 80)
        self.assertEqual(arguments.pool_size, 35)
        self.assertEqual(arguments.volatility_lookback, 30)
        self.assertEqual(arguments.min_assets, 12)
        self.assertEqual(arguments.trade_min_days_to_maturity, 60)
        self.assertEqual(arguments.cost_rate, 0.0003)
        self.assertEqual(arguments.result_dir, Path("custom-results"))
        self.assertTrue(arguments.overwrite)

    def test_validate_dates_returns_compact_ordered_timestamps(self):
        strategy = load_strategy_module()
        validate_dates = require_callable(strategy, "validate_dates")

        start, end = validate_dates("20200101", "20260701")

        self.assertEqual(start, pd.Timestamp("2020-01-01"))
        self.assertEqual(end, pd.Timestamp("2026-07-01"))

    def test_validate_dates_rejects_bad_format_and_order(self):
        strategy = load_strategy_module()
        validate_dates = require_callable(strategy, "validate_dates")

        cases = [
            ("2020-01-01", "20260701"),
            ("202001", "20260701"),
            ("20200101", "20200101"),
            ("20200102", "20200101"),
        ]
        for start, end in cases:
            with self.subTest(start=start, end=end):
                with self.assertRaises(ValueError):
                    validate_dates(start, end)

    def test_settings_reject_invalid_numeric_parameters(self):
        strategy = load_strategy_module()

        cases = [
            {"liquidity_lookback": True},
            {"liquidity_lookback": 0},
            {"min_amount_observations": 121},
            {"pool_size": 0},
            {"volatility_lookback": -1},
            {"min_assets": 1},
            {"trade_min_days_to_maturity": -1},
            {"cost_rate": -0.0001},
        ]
        for parameters in cases:
            with self.subTest(parameters=parameters):
                with self.assertRaises(ValueError):
                    strategy.StrategySettings(**parameters)


class ProductAmountLoadingTest(unittest.TestCase):
    def test_aggregates_all_contract_amounts_to_unique_product_days(self):
        strategy = load_strategy_module()
        load_product_amounts = require_callable(
            strategy,
            "load_product_amounts",
        )

        with tempfile.NamedTemporaryFile(suffix=".db") as database:
            with closing(sqlite3.connect(database.name)) as connection:
                connection.executescript(
                    """
                    CREATE TABLE fut_basic (
                        ts_code TEXT PRIMARY KEY,
                        exchange TEXT NOT NULL,
                        fut_code TEXT NOT NULL
                    );
                    CREATE TABLE fut_daily (
                        ts_code TEXT NOT NULL,
                        trade_date TEXT NOT NULL,
                        amount REAL
                    );
                    """
                )
                connection.executemany(
                    "INSERT INTO fut_basic VALUES (?, ?, ?)",
                    [
                        ("RB2405.SHF", "SHFE", "RB"),
                        ("RB2410.SHF", "SHFE", "RB"),
                        ("CU2405.SHF", "SHFE", "CU"),
                        ("IF2403.CFX", "CFFEX", "IF"),
                    ],
                )
                connection.executemany(
                    "INSERT INTO fut_daily VALUES (?, ?, ?)",
                    [
                        ("RB2405.SHF", "20240102", 100.0),
                        ("RB2410.SHF", "20240102", 200.0),
                        ("CU2405.SHF", "20240102", 50.0),
                        ("IF2403.CFX", "20240102", 9999.0),
                        ("RB2405.SHF", "20240103", 110.0),
                        ("RB2410.SHF", "20240103", None),
                        ("CU2405.SHF", "20240103", -50.0),
                        ("RB2405.SHF", "20240104", 120.0),
                    ],
                )
                connection.commit()

            result = load_product_amounts(
                "20240102",
                "20240103",
                db_path=database.name,
            )

        expected = pd.DataFrame(
            {
                "trade_date": pd.to_datetime(
                    ["2024-01-02", "2024-01-02", "2024-01-03"]
                ),
                "fut_code": ["CU", "RB", "RB"],
                "product_amount": [50.0, 300.0, 110.0],
            }
        )
        pd.testing.assert_frame_equal(result, expected)
        self.assertFalse(
            result.duplicated(["trade_date", "fut_code"]).any()
        )


class SemiannualUniverseTest(unittest.TestCase):
    def test_excludes_selection_date_and_enforces_observation_boundary(self):
        strategy = load_strategy_module()
        build_universe = require_callable(
            strategy,
            "build_semiannual_universe",
        )

        selection_date = pd.Timestamp("2024-01-02")
        history_dates = pd.bdate_range(
            end="2023-12-29",
            periods=120,
        )
        report_dates = pd.bdate_range(selection_date, periods=7)
        calendar = pd.DataFrame(
            {
                "trade_date": history_dates.append(report_dates),
            }
        )

        rows = []
        for date in history_dates:
            rows.append(
                {
                    "trade_date": date,
                    "fut_code": "A",
                    "product_amount": 100.0,
                }
            )
        for date in history_dates[-96:]:
            rows.append(
                {
                    "trade_date": date,
                    "fut_code": "B",
                    "product_amount": 200.0,
                }
            )
        for date in history_dates[-95:]:
            rows.append(
                {
                    "trade_date": date,
                    "fut_code": "C",
                    "product_amount": 300.0,
                }
            )
        rows.append(
            {
                "trade_date": selection_date,
                "fut_code": "C",
                "product_amount": 1_000_000.0,
            }
        )

        result = build_universe(
            product_amounts=pd.DataFrame(rows),
            trade_calendar=calendar,
            start_date="20240101",
            end_date="20240110",
            liquidity_lookback=120,
            min_observations=96,
            pool_size=2,
        )

        ranking = result.ranking.set_index("fut_code")
        self.assertEqual(ranking.loc["A", "observation_count"], 120)
        self.assertEqual(ranking.loc["B", "observation_count"], 96)
        self.assertEqual(ranking.loc["C", "observation_count"], 95)
        self.assertEqual(ranking.loc["A", "rolling_amount"], 100.0)
        self.assertEqual(ranking.loc["B", "rolling_amount"], 200.0)
        self.assertEqual(ranking.loc["C", "rolling_amount"], 300.0)
        self.assertTrue(ranking.loc["A", "is_eligible"])
        self.assertTrue(ranking.loc["B", "is_eligible"])
        self.assertFalse(ranking.loc["C", "is_eligible"])
        self.assertEqual(
            result.members["fut_code"].tolist(),
            ["B", "A"],
        )
        self.assertTrue(result.members["is_selected"].all())

    def test_records_semiannual_membership_and_member_changes(self):
        strategy = load_strategy_module()
        build_universe = require_callable(
            strategy,
            "build_semiannual_universe",
        )

        calendar_dates = pd.bdate_range(
            "2023-12-20",
            "2024-07-05",
        ).difference(pd.DatetimeIndex(["2024-01-01"]))
        calendar = pd.DataFrame({"trade_date": calendar_dates})
        january_selection = pd.Timestamp("2024-01-02")
        july_selection = pd.Timestamp("2024-07-01")
        january_history = calendar_dates[calendar_dates < january_selection][
            -4:
        ]
        july_history = calendar_dates[calendar_dates < july_selection][-4:]

        rows = []
        for date in january_history:
            for code, amount in {"A": 300.0, "B": 200.0, "C": 100.0}.items():
                rows.append(
                    {
                        "trade_date": date,
                        "fut_code": code,
                        "product_amount": amount,
                    }
                )
        for date in july_history:
            for code, amount in {"A": 100.0, "B": 300.0, "C": 200.0}.items():
                rows.append(
                    {
                        "trade_date": date,
                        "fut_code": code,
                        "product_amount": amount,
                    }
                )

        result = build_universe(
            product_amounts=pd.DataFrame(rows),
            trade_calendar=calendar,
            start_date="20240101",
            end_date="20240705",
            liquidity_lookback=4,
            min_observations=4,
            pool_size=2,
        )

        rankings = result.ranking.set_index(
            ["selection_date", "fut_code"]
        )
        self.assertEqual(
            rankings.loc[(january_selection, "A"), "change_status"],
            "entered",
        )
        self.assertEqual(
            rankings.loc[(july_selection, "A"), "change_status"],
            "exited",
        )
        self.assertEqual(
            rankings.loc[(july_selection, "B"), "change_status"],
            "retained",
        )
        self.assertEqual(
            rankings.loc[(july_selection, "C"), "change_status"],
            "entered",
        )

        january_members = result.members.loc[
            result.members["selection_date"].eq(january_selection),
            "fut_code",
        ].tolist()
        july_members = result.members.loc[
            result.members["selection_date"].eq(july_selection),
            "fut_code",
        ].tolist()
        self.assertEqual(january_members, ["A", "B"])
        self.assertEqual(july_members, ["B", "C"])

        daily = result.daily_membership
        before_july = daily.loc[
            daily["trade_date"].eq(pd.Timestamp("2024-06-28")),
            "fut_code",
        ].tolist()
        from_july = daily.loc[
            daily["trade_date"].eq(july_selection),
            "fut_code",
        ].tolist()
        self.assertEqual(before_july, ["A", "B"])
        self.assertEqual(from_july, ["B", "C"])

        changes = result.changes.set_index(
            ["selection_date", "fut_code"]
        )["change_status"]
        self.assertEqual(changes.loc[(july_selection, "A")], "exited")
        self.assertEqual(changes.loc[(july_selection, "C")], "entered")

    def test_selects_exact_top_forty_with_stable_tie_breaking(self):
        strategy = load_strategy_module()
        build_universe = require_callable(
            strategy,
            "build_semiannual_universe",
        )

        selection_date = pd.Timestamp("2024-01-02")
        history_dates = pd.bdate_range("2023-12-26", periods=4)
        report_dates = pd.bdate_range(selection_date, periods=2)
        calendar = pd.DataFrame(
            {"trade_date": history_dates.append(report_dates)}
        )
        rows = [
            {
                "trade_date": date,
                "fut_code": f"F{number:02d}",
                "product_amount": 100.0,
            }
            for date in history_dates
            for number in range(45)
        ]

        result = build_universe(
            product_amounts=pd.DataFrame(rows),
            trade_calendar=calendar,
            start_date="20240101",
            end_date="20240103",
            liquidity_lookback=4,
            min_observations=4,
            pool_size=40,
        )

        self.assertEqual(len(result.members), 40)
        self.assertEqual(
            result.members["fut_code"].tolist(),
            [f"F{number:02d}" for number in range(40)],
        )
        self.assertEqual(
            result.members["liquidity_rank"].tolist(),
            list(range(1, 41)),
        )


class ContractVolatilityTest(unittest.TestCase):
    def test_loader_includes_ninety_calendar_days_of_history(self):
        strategy = load_strategy_module()
        load_closes = require_callable(
            strategy,
            "load_fixed_contract_closes",
        )

        with tempfile.NamedTemporaryFile(suffix=".db") as database:
            with closing(sqlite3.connect(database.name)) as connection:
                connection.execute(
                    """
                    CREATE TABLE fut_daily (
                        ts_code TEXT NOT NULL,
                        trade_date TEXT NOT NULL,
                        close REAL
                    )
                    """
                )
                connection.executemany(
                    "INSERT INTO fut_daily VALUES (?, ?, ?)",
                    [
                        ("A2405.DCE", "20231002", 99.0),
                        ("A2405.DCE", "20231003", 100.0),
                        ("A2405.DCE", "20240131", 110.0),
                        ("A2405.DCE", "20240201", 120.0),
                    ],
                )
                connection.commit()

            result = load_closes(
                "20240101",
                "20240131",
                db_path=database.name,
            )

        self.assertEqual(
            result["trade_date"].tolist(),
            [
                pd.Timestamp("2023-10-03"),
                pd.Timestamp("2024-01-31"),
            ],
        )

    def test_returns_never_cross_fixed_contract_boundaries(self):
        strategy = load_strategy_module()
        compute_volatility = require_callable(
            strategy,
            "compute_contract_volatility",
        )
        prices = pd.DataFrame(
            {
                "trade_date": pd.to_datetime(
                    [
                        "2024-01-03",
                        "2024-01-02",
                        "2024-01-03",
                        "2024-01-02",
                    ]
                ),
                "ts_code": ["A", "A", "B", "B"],
                "close": [110.0, 100.0, 900.0, 1000.0],
            }
        )

        result = compute_volatility(prices, lookback=2)
        indexed = result.set_index(["ts_code", "trade_date"])

        self.assertTrue(
            pd.isna(
                indexed.loc[("A", pd.Timestamp("2024-01-02")),
                            "contract_return"]
            )
        )
        self.assertTrue(
            pd.isna(
                indexed.loc[("B", pd.Timestamp("2024-01-02")),
                            "contract_return"]
            )
        )
        self.assertAlmostEqual(
            indexed.loc[("A", pd.Timestamp("2024-01-03")),
                        "contract_return"],
            0.10,
        )
        self.assertAlmostEqual(
            indexed.loc[("B", pd.Timestamp("2024-01-03")),
                        "contract_return"],
            -0.10,
        )

    def test_requires_twenty_complete_returns_and_uses_sample_std(self):
        strategy = load_strategy_module()
        compute_volatility = require_callable(
            strategy,
            "compute_contract_volatility",
        )
        returns = np.array(
            [0.01, -0.02, 0.03, -0.01, 0.02] * 4,
            dtype=float,
        )
        closes = 100.0 * np.cumprod(
            np.concatenate(([1.0], 1.0 + returns))
        )
        prices = pd.DataFrame(
            {
                "trade_date": pd.bdate_range("2024-01-02", periods=21),
                "ts_code": "A",
                "close": closes,
            }
        )

        result = compute_volatility(prices, lookback=20)
        expected = pd.Series(returns).std(ddof=1)

        self.assertTrue(pd.isna(result.loc[19, "volatility_20"]))
        self.assertAlmostEqual(
            result.loc[20, "volatility_20"],
            expected,
        )
        self.assertAlmostEqual(
            result.loc[20, "annualized_volatility_20"],
            expected * np.sqrt(252.0),
        )

    def test_validates_keys_window_and_invalid_close_values(self):
        strategy = load_strategy_module()
        compute_volatility = require_callable(
            strategy,
            "compute_contract_volatility",
        )
        base = pd.DataFrame(
            {
                "trade_date": pd.bdate_range("2024-01-02", periods=5),
                "ts_code": "A",
                "close": ["100", 0, np.inf, "bad", 110.0],
            }
        )

        with self.assertRaisesRegex(ValueError, "positive integer"):
            compute_volatility(base, lookback=0)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            compute_volatility(
                pd.concat([base.iloc[[0]], base.iloc[[0]]]),
                lookback=2,
            )
        with self.assertRaisesRegex(ValueError, "missing columns"):
            compute_volatility(
                base.drop(columns="close"),
                lookback=2,
            )

        result = compute_volatility(base, lookback=2)
        self.assertTrue(result.loc[1:3, "close"].isna().all())
        self.assertTrue(
            np.isfinite(result["contract_return"].dropna()).all()
        )


class InverseVolatilityWeightTest(unittest.TestCase):
    @staticmethod
    def make_inputs(
        count,
        raw_factors=None,
        volatilities=None,
    ):
        date = pd.Timestamp("2024-01-02")
        codes = [f"F{number:02d}" for number in range(count)]
        contracts = [f"{code}2405.DCE" for code in codes]
        if raw_factors is None:
            raw_factors = np.arange(count, dtype=float)
        if volatilities is None:
            volatilities = np.full(count, 0.02)

        factor_data = pd.DataFrame(
            {
                "trade_date": date,
                "fut_code": codes,
                "raw_factor": raw_factors,
            }
        )
        contract_context = pd.DataFrame(
            {
                "trade_date": date,
                "fut_code": codes,
                "ts_code_A": contracts,
            }
        )
        contract_volatility = pd.DataFrame(
            {
                "trade_date": date,
                "ts_code": contracts,
                "volatility_20": volatilities,
            }
        )
        daily_membership = pd.DataFrame(
            {
                "trade_date": date,
                "fut_code": codes,
            }
        )
        return (
            factor_data,
            contract_context,
            contract_volatility,
            daily_membership,
        )

    def build_weights(self, count, method, **input_kwargs):
        strategy = load_strategy_module()
        build = require_callable(
            strategy,
            "build_inverse_volatility_weights",
        )
        return build(
            *self.make_inputs(count, **input_kwargs),
            method=method,
            min_assets=10,
        )

    def test_maps_five_average_ranks_linearly_to_unit_interval(self):
        strategy = load_strategy_module()
        build = require_callable(
            strategy,
            "build_inverse_volatility_weights",
        )
        result = build(
            *self.make_inputs(5),
            method="full_pool_invvol",
            min_assets=2,
        ).sort_values("raw_factor")

        np.testing.assert_allclose(
            result["factor_score"].to_numpy(),
            [-1.0, -0.5, 0.0, 0.5, 1.0],
        )

    def test_full_pool_uses_score_over_volatility_and_is_neutral(self):
        volatilities = np.linspace(0.01, 0.10, 10)
        result = self.build_weights(
            10,
            "full_pool_invvol",
            volatilities=volatilities,
        ).sort_values("fut_code")
        scores = np.linspace(-1.0, 1.0, 10)
        risk_scores = scores / volatilities
        expected = np.zeros(10)
        expected[risk_scores > 0] = (
            0.5
            * risk_scores[risk_scores > 0]
            / risk_scores[risk_scores > 0].sum()
        )
        expected[risk_scores < 0] = (
            0.5
            * risk_scores[risk_scores < 0]
            / np.abs(risk_scores[risk_scores < 0]).sum()
        )

        np.testing.assert_allclose(result["risk_score"], risk_scores)
        np.testing.assert_allclose(result["weight"], expected)
        self.assertAlmostEqual(result.loc[result["weight"] > 0, "weight"].sum(), 0.5)
        self.assertAlmostEqual(result.loc[result["weight"] < 0, "weight"].sum(), -0.5)
        self.assertAlmostEqual(result["weight"].sum(), 0.0)
        self.assertAlmostEqual(result["weight"].abs().sum(), 1.0)

    def test_tail_method_uses_ceiling_counts(self):
        for count, expected_side_count in (
            (40, 4),
            (37, 4),
            (30, 3),
            (24, 3),
            (18, 2),
        ):
            with self.subTest(count=count):
                result = self.build_weights(count, "tail10_invvol")
                self.assertEqual(
                    result["long_count"].iloc[0],
                    expected_side_count,
                )
                self.assertEqual(
                    result["short_count"].iloc[0],
                    expected_side_count,
                )
                self.assertEqual(
                    result["weight"].gt(0).sum(),
                    expected_side_count,
                )
                self.assertEqual(
                    result["weight"].lt(0).sum(),
                    expected_side_count,
                )

    def test_fewer_than_minimum_assets_produces_zero_targets(self):
        result = self.build_weights(9, "full_pool_invvol")

        self.assertTrue(result["weight"].eq(0.0).all())
        self.assertEqual(result["eligible_count"].iloc[0], 9)
        self.assertEqual(result["long_count"].iloc[0], 0)
        self.assertEqual(result["short_count"].iloc[0], 0)

    def test_tail_ties_use_fut_code_as_stable_boundary(self):
        result = self.build_weights(
            10,
            "tail10_invvol",
            raw_factors=[0, 0, 1, 2, 3, 4, 5, 6, 7, 7],
        )
        selected = result.loc[result["weight"].ne(0), "fut_code"].tolist()

        self.assertEqual(selected, ["F00", "F08"])

    def test_rejects_unknown_method_and_duplicate_input_keys(self):
        strategy = load_strategy_module()
        build = require_callable(
            strategy,
            "build_inverse_volatility_weights",
        )
        inputs = self.make_inputs(10)
        with self.assertRaisesRegex(ValueError, "method"):
            build(*inputs, method="unknown", min_assets=10)

        duplicated_factors = pd.concat(
            [inputs[0], inputs[0].iloc[[0]]],
            ignore_index=True,
        )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            build(
                duplicated_factors,
                *inputs[1:],
                method="full_pool_invvol",
                min_assets=10,
            )

    def test_pool_merge_with_nonmembers_emits_no_future_warning(self):
        strategy = load_strategy_module()
        build = require_callable(
            strategy,
            "build_inverse_volatility_weights",
        )
        inputs = list(self.make_inputs(10))
        inputs[3] = inputs[3].iloc[:-1].copy()

        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            result = build(
                *inputs,
                method="full_pool_invvol",
                min_assets=10,
            )

        self.assertFalse(result.loc[result["fut_code"].eq("F09"), "passes_liquidity"].iloc[0])


class CombinedDiagnosticsTest(unittest.TestCase):
    def test_structurally_empty_optimizer_columns_remain_optional(self):
        strategy = load_strategy_module()
        combine = require_callable(
            strategy,
            "_combine_daily_diagnostics",
        )
        dates = pd.bdate_range("2024-01-02", periods=2)
        factor_results = {}
        for position, factor_name in enumerate(strategy.FACTOR_SPECS):
            factor_results[factor_name] = {
                "daily_diagnostics": pd.DataFrame(
                    {
                        "turnover": [0.1 + position * 0.01] * 2,
                        "cost": [0.001] * 2,
                        "optimizer_budget_turnover": [np.nan, np.nan],
                    },
                    index=dates,
                )
            }
        combined_return = pd.Series([0.01, -0.01], index=dates)

        result = combine(factor_results, combined_return)

        self.assertTrue(result["optimizer_budget_turnover"].isna().all())
        self.assertAlmostEqual(result.loc[dates[0], "turnover"], 0.12)


class FactorReturnCombinationTest(unittest.TestCase):
    def setUp(self):
        self.dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
        self.factor_returns = {
            "basis_momentum": pd.Series(
                [0.01, -0.01],
                index=self.dates,
            ),
            "carry": pd.Series(
                [0.02, -0.02],
                index=self.dates,
            ),
            "spotmain": pd.Series(
                [0.03, -0.03],
                index=self.dates,
            ),
            "s_warehouse": pd.Series(
                [0.04, -0.04],
                index=self.dates,
            ),
            "t_rank": pd.Series(
                [0.0, 0.0],
                index=self.dates,
            ),
        }

    def test_combines_daily_returns_with_constant_twenty_percent_sleeves(self):
        strategy = load_strategy_module()
        combine = require_callable(
            strategy,
            "combine_factor_returns",
        )

        result = combine(self.factor_returns)
        expected = pd.concat(
            self.factor_returns.values(),
            axis=1,
        ).mean(axis=1)

        self.assertEqual(
            result.columns.tolist(),
            [*self.factor_returns, "combined_return"],
        )
        pd.testing.assert_series_equal(
            result["combined_return"],
            expected.rename("combined_return"),
        )
        self.assertAlmostEqual(result.iloc[0]["combined_return"], 0.02)

    def test_rejects_missing_or_extra_factor_names(self):
        strategy = load_strategy_module()
        combine = require_callable(
            strategy,
            "combine_factor_returns",
        )
        missing = dict(self.factor_returns)
        missing.pop("carry")
        with self.assertRaisesRegex(ValueError, "factor names"):
            combine(missing)

        extra = dict(self.factor_returns)
        extra["extra_factor"] = extra["carry"]
        with self.assertRaisesRegex(ValueError, "factor names"):
            combine(extra)

    def test_rejects_duplicate_or_mismatched_calendars(self):
        strategy = load_strategy_module()
        combine = require_callable(
            strategy,
            "combine_factor_returns",
        )
        duplicated = dict(self.factor_returns)
        duplicated["carry"] = pd.Series(
            [0.01, 0.02],
            index=pd.to_datetime(["2024-01-02", "2024-01-02"]),
        )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            combine(duplicated)

        mismatched = dict(self.factor_returns)
        mismatched["carry"] = pd.Series(
            [0.01, 0.02],
            index=pd.to_datetime(["2024-01-02", "2024-01-04"]),
        )
        with self.assertRaisesRegex(ValueError, "calendar"):
            combine(mismatched)

    def test_rejects_missing_infinite_and_nonnumeric_returns(self):
        strategy = load_strategy_module()
        combine = require_callable(
            strategy,
            "combine_factor_returns",
        )
        for invalid_value in (np.nan, np.inf, "bad"):
            with self.subTest(invalid_value=invalid_value):
                invalid = dict(self.factor_returns)
                invalid["carry"] = pd.Series(
                    [0.01, invalid_value],
                    index=self.dates,
                )
                with self.assertRaisesRegex(ValueError, "finite"):
                    combine(invalid)


class ExecutionAdapterTest(unittest.TestCase):
    def test_shifts_targets_once_and_locks_the_signal_date_contract(self):
        strategy = load_strategy_module()
        build_audit = require_callable(
            strategy,
            "build_execution_audit",
        )
        dates = pd.bdate_range("2024-01-02", periods=4)
        weights = pd.DataFrame(
            {
                "trade_date": dates,
                "fut_code": "A",
                "weight": [0.05, 0.06, 0.07, 0.07],
                "is_rebalance": True,
                "ts_code_A": ["A1", "A2", "A2", "A2"],
            }
        )
        prices = pd.DataFrame(
            {
                "trade_date": dates.repeat(2),
                "ts_code": ["A1", "A2"] * 4,
                "open": [
                    100.0,
                    200.0,
                    101.0,
                    201.0,
                    102.0,
                    np.nan,
                    103.0,
                    203.0,
                ],
            }
        )

        result = build_audit(weights, prices).set_index("trade_date")

        self.assertEqual(result.loc[dates[0], "exec_weight"], 0.0)
        self.assertEqual(result.loc[dates[1], "exec_weight"], 0.05)
        self.assertEqual(result.loc[dates[1], "trade_ts_code"], "A1")
        self.assertEqual(
            result.loc[dates[2], "desired_trade_ts_code"],
            "A2",
        )
        self.assertTrue(result.loc[dates[2], "blocked_trade"])
        self.assertTrue(result.loc[dates[2], "delayed_roll"])
        self.assertEqual(result.loc[dates[2], "trade_ts_code"], "A1")
        self.assertEqual(result.loc[dates[3], "trade_ts_code"], "A2")

    def test_missing_open_blocks_entry_and_exit(self):
        strategy = load_strategy_module()
        build_audit = require_callable(
            strategy,
            "build_execution_audit",
        )
        dates = pd.bdate_range("2024-01-02", periods=4)
        weights = pd.DataFrame(
            {
                "trade_date": dates,
                "fut_code": "A",
                "weight": [0.05, 0.05, 0.0, 0.0],
                "is_rebalance": True,
                "ts_code_A": "A1",
            }
        )
        prices = pd.DataFrame(
            {
                "trade_date": dates,
                "ts_code": "A1",
                "open": [100.0, np.nan, 102.0, np.nan],
            }
        )

        result = build_audit(weights, prices).set_index("trade_date")

        self.assertTrue(result.loc[dates[1], "blocked_trade"])
        self.assertEqual(result.loc[dates[1], "exec_weight"], 0.0)
        self.assertEqual(result.loc[dates[2], "exec_weight"], 0.05)
        self.assertTrue(result.loc[dates[3], "blocked_trade"])
        self.assertEqual(result.loc[dates[3], "exec_weight"], 0.05)

    def test_factor_method_returns_authoritative_engine_outputs(self):
        strategy = load_strategy_module()
        run_method = require_callable(
            strategy,
            "run_factor_method",
        )
        dates = pd.bdate_range("2024-01-02", periods=3)
        weights = pd.DataFrame(
            {
                "trade_date": dates,
                "fut_code": "A",
                "weight": [0.05, 0.05, 0.05],
                "is_rebalance": True,
                "ts_code_A": "A1",
            }
        )
        prices = pd.DataFrame(
            {
                "trade_date": dates,
                "ts_code": "A1",
                "open": [100.0, 101.0, 102.0],
                "close": [100.5, 101.5, 102.5],
                "prev_close": [99.5, 100.5, 101.5],
            }
        )

        result = run_method(weights, prices, cost_rate=0.0005)

        self.assertEqual(
            set(result),
            {
                "nav",
                "metrics",
                "daily_return",
                "daily_diagnostics",
                "execution_weights",
            },
        )
        self.assertEqual(result["nav"].index.tolist(), dates.tolist())
        self.assertEqual(
            result["execution_weights"].loc[0, "exec_weight"],
            0.0,
        )


class AnnualStatisticsTest(unittest.TestCase):
    def setUp(self):
        self.dates = pd.to_datetime(
            [
                "2020-01-02",
                "2020-12-31",
                "2021-01-04",
                "2021-12-31",
                "2026-01-05",
                "2026-07-01",
            ]
        )
        self.returns = pd.Series(
            [-0.10, 0.20, 0.05, -0.02, 0.03, 0.04],
            index=self.dates,
            name="daily_return",
        )
        self.diagnostics = pd.DataFrame(
            {
                "turnover": [0.1] * len(self.dates),
                "cost": [0.00005] * len(self.dates),
            },
            index=self.dates,
        )

    def test_annual_metrics_use_actual_period_returns_and_fresh_drawdowns(self):
        strategy = load_strategy_module()
        calculate = require_callable(
            strategy,
            "calculate_annual_metrics",
        )

        result = calculate(
            {"strategy": self.returns},
            {"strategy": self.diagnostics},
        ).set_index("year")

        self.assertAlmostEqual(
            result.loc[2020, "period_return"],
            (1.0 - 0.10) * (1.0 + 0.20) - 1.0,
        )
        self.assertAlmostEqual(result.loc[2020, "max_drawdown"], -0.10)
        self.assertFalse(result.loc[2020, "is_partial_year"])
        self.assertFalse(result.loc[2021, "is_partial_year"])
        self.assertTrue(result.loc[2026, "is_partial_year"])
        self.assertEqual(result.loc[2026, "trading_days"], 2)

    def test_annual_return_table_contains_every_strategy(self):
        strategy = load_strategy_module()
        calculate = require_callable(
            strategy,
            "calculate_annual_returns",
        )
        strategy_returns = {
            name: self.returns * (position + 1)
            for position, name in enumerate(
                [*strategy.FACTOR_SPECS, "combined"]
            )
        }

        result = calculate(strategy_returns)

        self.assertEqual(
            result.columns.tolist(),
            ["year", *strategy_returns],
        )
        self.assertEqual(result["year"].tolist(), [2020, 2021, 2026])

    def test_annual_ic_reports_both_linear_and_rank_statistics(self):
        strategy = load_strategy_module()
        calculate = require_callable(
            strategy,
            "calculate_annual_ic",
        )
        ic_series = pd.DataFrame(
            {
                "signal_date": pd.to_datetime(
                    [
                        "2020-01-02",
                        "2020-01-03",
                        "2020-01-06",
                        "2021-01-04",
                        "2021-01-05",
                    ]
                ),
                "ic": [0.10, -0.05, 0.20, 0.02, 0.04],
                "rank_ic": [0.20, 0.10, -0.10, -0.02, 0.06],
            }
        )

        result = calculate(ic_series, "carry").set_index("year")

        self.assertEqual(result.loc[2020, "factor_name"], "carry")
        self.assertAlmostEqual(
            result.loc[2020, "mean_ic"],
            np.mean([0.10, -0.05, 0.20]),
        )
        self.assertAlmostEqual(result.loc[2020, "positive_rate_ic"], 2 / 3)
        self.assertEqual(result.loc[2020, "valid_count_ic"], 3)
        self.assertEqual(result.loc[2021, "valid_count_rank_ic"], 2)
        self.assertTrue(np.isfinite(result.loc[2020, "t_stat_ic"]))


class OutputContractTest(unittest.TestCase):
    def test_builds_stable_daily_result_path(self):
        strategy = load_strategy_module()
        build_path = require_callable(
            strategy,
            "build_output_directory",
        )

        result = build_path(
            "/tmp/research-results",
            strategy.StrategySettings(),
        )

        self.assertEqual(
            result,
            Path(
                "/tmp/research-results/multi_factor_strategy/"
                "top40_amount120_vol20-20200101-20260701/daily"
            ),
        )

    def test_refuses_known_collisions_without_deleting_unknown_files(self):
        strategy = load_strategy_module()
        ensure = require_callable(
            strategy,
            "ensure_output_directory",
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "daily"
            output.mkdir(parents=True)
            collision = output / "run_config.csv"
            collision.write_text("existing", encoding="utf-8")
            unknown = output / "research_notes.txt"
            unknown.write_text("preserve me", encoding="utf-8")

            with self.assertRaisesRegex(FileExistsError, "run_config.csv"):
                ensure(output, overwrite=False)

            ensure(output, overwrite=True)
            self.assertEqual(
                unknown.read_text(encoding="utf-8"),
                "preserve me",
            )
            self.assertEqual(
                collision.read_text(encoding="utf-8"),
                "existing",
            )

    def test_nav_plot_is_created_and_nonempty(self):
        strategy = load_strategy_module()
        plot = require_callable(
            strategy,
            "plot_nav_comparison",
        )
        nav = pd.DataFrame(
            {
                "carry": [1.0, 1.02, 1.01],
                "combined": [1.0, 1.01, 1.03],
            },
            index=pd.bdate_range("2024-01-02", periods=3),
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nav.png"
            plot(nav, output, title="NAV comparison")

            self.assertTrue(output.exists())
            self.assertGreater(output.stat().st_size, 0)


class MultiFactorSmokeTest(unittest.TestCase):
    @patch("src.multi_factor_strategy.save_strategy_results")
    @patch("src.multi_factor_strategy.calculate_annual_ic")
    @patch("src.multi_factor_strategy.summarize_ic_statistics")
    @patch("src.multi_factor_strategy.calculate_ic_series")
    @patch("src.multi_factor_strategy.build_factor_test_panel")
    @patch("src.multi_factor_strategy.run_factor_method")
    @patch("src.multi_factor_strategy.build_inverse_volatility_weights")
    @patch("src.multi_factor_strategy.calculate_factor")
    @patch("src.multi_factor_strategy.load_contract_prices")
    @patch("src.multi_factor_strategy.prepare_contract_context")
    @patch("src.multi_factor_strategy.compute_contract_volatility")
    @patch("src.multi_factor_strategy.load_fixed_contract_closes")
    @patch("src.multi_factor_strategy.build_semiannual_universe")
    @patch("src.multi_factor_strategy.load_product_amounts")
    @patch("src.multi_factor_strategy.load_trade_calendar")
    def test_runs_each_factor_once_and_both_methods(
        self,
        mock_calendar,
        mock_amounts,
        mock_universe,
        mock_load_closes,
        mock_compute_volatility,
        mock_context,
        mock_prices,
        mock_factor,
        mock_weights,
        mock_run_method,
        mock_test_panel,
        mock_ic_series,
        mock_ic_summary,
        mock_annual_ic,
        mock_save,
    ):
        strategy = load_strategy_module()
        run = require_callable(
            strategy,
            "run_multi_factor_strategy",
        )
        dates = pd.bdate_range("2024-01-02", periods=3)
        calendar = pd.DataFrame({"trade_date": dates})
        mock_calendar.return_value = calendar
        mock_amounts.return_value = pd.DataFrame(
            columns=["trade_date", "fut_code", "product_amount"]
        )
        mock_universe.return_value = strategy.UniverseResult(
            ranking=pd.DataFrame(),
            members=pd.DataFrame(),
            changes=pd.DataFrame(),
            daily_membership=pd.DataFrame(
                {"trade_date": dates, "fut_code": "A"}
            ),
        )
        mock_load_closes.return_value = pd.DataFrame()
        mock_compute_volatility.return_value = pd.DataFrame(
            {
                "trade_date": dates,
                "ts_code": "A1",
                "volatility_20": 0.02,
            }
        )
        mock_context.return_value = pd.DataFrame(
            {
                "trade_date": dates,
                "fut_code": "A",
                "ts_code_A": "A1",
            }
        )
        prices = pd.DataFrame(
            {
                "trade_date": dates,
                "ts_code": "A1",
                "open": 100.0,
                "close": 101.0,
                "prev_close": 100.0,
            }
        )
        mock_prices.return_value = prices
        factor_frame = pd.DataFrame(
            {
                "trade_date": dates,
                "fut_code": "A",
                "raw_factor": [1.0, 2.0, 3.0],
            }
        )
        mock_factor.return_value = factor_frame
        weight_frame = factor_frame.assign(
            weight=0.0,
            is_rebalance=True,
            is_eligible=True,
            ts_code_A="A1",
        )
        mock_weights.return_value = weight_frame

        def method_result(*args, **kwargs):
            daily_return = pd.Series(
                [0.0, 0.01, -0.005],
                index=dates,
                name="daily_return",
            )
            diagnostics = pd.DataFrame(
                {"turnover": 0.0, "cost": 0.0},
                index=dates,
            )
            return {
                "nav": (1.0 + daily_return).cumprod().rename("nav"),
                "metrics": {"sharpe": 1.0},
                "daily_return": daily_return,
                "daily_diagnostics": diagnostics,
                "execution_weights": weight_frame,
            }

        mock_run_method.side_effect = method_result
        mock_test_panel.return_value = pd.DataFrame()
        mock_ic_series.return_value = pd.DataFrame(
            columns=["signal_date", "ic", "rank_ic"]
        )
        mock_ic_summary.return_value = pd.DataFrame()
        mock_annual_ic.return_value = pd.DataFrame()

        with tempfile.TemporaryDirectory() as directory:
            expected = strategy.build_output_directory(
                directory,
                strategy.StrategySettings(
                    start_date="20240102",
                    end_date="20240104",
                ),
            )
            mock_save.return_value = expected
            result = run(
                strategy.StrategySettings(
                    start_date="20240102",
                    end_date="20240104",
                ),
                result_dir=directory,
                overwrite=False,
            )

        self.assertEqual(result, expected)
        self.assertEqual(mock_factor.call_count, 5)
        self.assertEqual(mock_weights.call_count, 10)
        self.assertEqual(mock_run_method.call_count, 10)
        called_parameters = {
            call.kwargs["factor_name"]: call.kwargs["parameters"]
            for call in mock_factor.call_args_list
        }
        self.assertEqual(called_parameters["carry"]["lookback"], 90)
        self.assertEqual(called_parameters["t_rank"]["lookback"], 20)
        self.assertEqual(
            called_parameters["basis_momentum"]["variant"],
            "AB",
        )


if __name__ == "__main__":
    unittest.main()
