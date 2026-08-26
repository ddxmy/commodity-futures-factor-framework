"""Run the approved five-factor inverse-volatility futures strategy.

The runner selects a lagged semiannual Top 40 commodity pool, builds daily
full-pool and tail portfolios from existing factor plugins, executes targets
at the next open, and saves factor, combined, annual, and audit results.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from contextlib import closing
from dataclasses import asdict, dataclass
import math
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd

from config.settings import COST_RATE, DB_PATH, RESULT_DIR
from src.factor_loader import calculate_factor
from src.p01_market_data import load_contract_prices, load_trade_calendar
from src.p06_backtest_engine import (
    compute_metrics,
    resolve_executed_positions,
    run_backtest_from_weights,
)
from src.p07_factor_evaluation import (
    build_factor_test_panel,
    calculate_ic_series,
    summarize_ic_statistics,
)
from src.research_pipeline import prepare_contract_context


DEFAULT_START_DATE = "20200101"
DEFAULT_END_DATE = "20260701"
DEFAULT_LIQUIDITY_LOOKBACK = 120
DEFAULT_MIN_AMOUNT_OBSERVATIONS = 96
DEFAULT_POOL_SIZE = 40
DEFAULT_VOLATILITY_LOOKBACK = 20
DEFAULT_MIN_ASSETS = 10
DEFAULT_TRADE_MIN_DAYS_TO_MATURITY = 45
DEFAULT_COST_RATE = COST_RATE

PORTFOLIO_METHODS = (
    "full_pool_invvol",
    "tail10_invvol",
)

RUN_OUTPUT_FILENAMES = (
    "run_config.csv",
    "universe_ranking.csv",
    "universe_members.csv",
    "universe_changes.csv",
    "strategy_metrics.csv",
    "factor_ic_summary.csv",
    "factor_ic_series.csv",
    "factor_ic_annual.csv",
    "combined_nav_comparison.png",
)

METHOD_OUTPUT_FILENAMES = (
    "daily_returns.csv",
    "nav.csv",
    "daily_diagnostics.csv",
    "weights.csv",
    "annual_metrics.csv",
    "annual_returns.csv",
    "nav_summary.png",
)

FACTOR_SPECS: dict[str, dict[str, object]] = {
    "basis_momentum": {
        "variant": "AB",
        "lookback": 252,
    },
    "carry": {
        "lookback": 90,
    },
    "spotmain": {
        "lookback": 90,
    },
    "s_warehouse": {
        "lookback": 90,
        "smooth_window": 20,
        "min_observations": 18,
    },
    "t_rank": {
        "lookback": 20,
    },
}


@dataclass(frozen=True)
class StrategySettings:
    """Fixed and configurable inputs for one multi-factor strategy run."""

    start_date: str = DEFAULT_START_DATE
    end_date: str = DEFAULT_END_DATE
    liquidity_lookback: int = DEFAULT_LIQUIDITY_LOOKBACK
    min_amount_observations: int = DEFAULT_MIN_AMOUNT_OBSERVATIONS
    pool_size: int = DEFAULT_POOL_SIZE
    volatility_lookback: int = DEFAULT_VOLATILITY_LOOKBACK
    min_assets: int = DEFAULT_MIN_ASSETS
    trade_min_days_to_maturity: int = (
        DEFAULT_TRADE_MIN_DAYS_TO_MATURITY
    )
    cost_rate: float = DEFAULT_COST_RATE

    def __post_init__(self) -> None:
        """Reject invalid settings before any database work begins."""
        validate_dates(self.start_date, self.end_date)

        positive_integers = {
            "liquidity_lookback": self.liquidity_lookback,
            "min_amount_observations": self.min_amount_observations,
            "pool_size": self.pool_size,
            "volatility_lookback": self.volatility_lookback,
        }
        for name, value in positive_integers.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer")

        if self.min_amount_observations > self.liquidity_lookback:
            raise ValueError(
                "min_amount_observations cannot exceed "
                "liquidity_lookback"
            )

        if (
            isinstance(self.min_assets, bool)
            or not isinstance(self.min_assets, int)
            or self.min_assets < 2
        ):
            raise ValueError("min_assets must be an integer of at least 2")

        if (
            isinstance(self.trade_min_days_to_maturity, bool)
            or not isinstance(self.trade_min_days_to_maturity, int)
            or self.trade_min_days_to_maturity < 0
        ):
            raise ValueError(
                "trade_min_days_to_maturity must be a "
                "non-negative integer"
            )

        if (
            isinstance(self.cost_rate, bool)
            or not isinstance(self.cost_rate, (int, float))
            or not math.isfinite(self.cost_rate)
            or self.cost_rate < 0
        ):
            raise ValueError("cost_rate must be a non-negative finite number")


@dataclass(frozen=True)
class UniverseResult:
    """Auditable tables produced by one semiannual selection pass."""

    ranking: pd.DataFrame
    members: pd.DataFrame
    changes: pd.DataFrame
    daily_membership: pd.DataFrame


def validate_dates(
    start_date: str,
    end_date: str,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return ordered compact dates after strict format validation."""
    parsed_dates = []
    for name, value in (
        ("start", start_date),
        ("end", end_date),
    ):
        if (
            not isinstance(value, str)
            or len(value) != 8
            or not value.isdigit()
        ):
            raise ValueError(f"{name} must use YYYYMMDD format")
        try:
            parsed = pd.to_datetime(
                value,
                format="%Y%m%d",
                errors="raise",
            )
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} must use YYYYMMDD format") from error
        parsed_dates.append(parsed)

    start, end = parsed_dates
    if start >= end:
        raise ValueError("start must be earlier than end")
    return start, end


def parse_arguments(
    argv: list[str] | None = None,
) -> argparse.Namespace:
    """Parse the standalone multi-factor strategy command line."""
    parser = argparse.ArgumentParser(
        description=(
            "Run the daily five-factor inverse-volatility strategy"
        ),
    )
    parser.add_argument("--start", default=DEFAULT_START_DATE)
    parser.add_argument("--end", default=DEFAULT_END_DATE)
    parser.add_argument(
        "--liquidity-lookback",
        type=int,
        default=DEFAULT_LIQUIDITY_LOOKBACK,
    )
    parser.add_argument(
        "--min-amount-observations",
        type=int,
        default=DEFAULT_MIN_AMOUNT_OBSERVATIONS,
    )
    parser.add_argument(
        "--pool-size",
        type=int,
        default=DEFAULT_POOL_SIZE,
    )
    parser.add_argument(
        "--volatility-lookback",
        type=int,
        default=DEFAULT_VOLATILITY_LOOKBACK,
    )
    parser.add_argument(
        "--min-assets",
        type=int,
        default=DEFAULT_MIN_ASSETS,
    )
    parser.add_argument(
        "--trade-min-days-to-maturity",
        type=int,
        default=DEFAULT_TRADE_MIN_DAYS_TO_MATURITY,
    )
    parser.add_argument(
        "--cost-rate",
        type=float,
        default=DEFAULT_COST_RATE,
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=Path(RESULT_DIR),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def load_product_amounts(
    start_date: str,
    end_date: str,
    db_path: str = DB_PATH,
) -> pd.DataFrame:
    """Return positive total contract amount by market date and product."""
    validate_dates(start_date, end_date)
    query = """
    SELECT
        d.trade_date,
        b.fut_code,
        SUM(d.amount) AS product_amount
    FROM fut_daily d
    JOIN fut_basic b
      ON b.ts_code = d.ts_code
    WHERE d.trade_date BETWEEN ? AND ?
      AND b.exchange IN ('DCE', 'CZCE', 'SHFE', 'INE', 'GFEX')
    GROUP BY d.trade_date, b.fut_code
    ORDER BY d.trade_date, b.fut_code
    """

    with closing(sqlite3.connect(db_path)) as connection:
        amounts = pd.read_sql_query(
            query,
            connection,
            params=(start_date, end_date),
        )

    amounts["trade_date"] = pd.to_datetime(
        amounts["trade_date"],
        errors="raise",
    )
    amounts["product_amount"] = pd.to_numeric(
        amounts["product_amount"],
        errors="coerce",
    )
    valid_amount = (
        np.isfinite(amounts["product_amount"])
        & amounts["product_amount"].gt(0)
    )
    amounts = amounts.loc[valid_amount].copy()

    if amounts.duplicated(["trade_date", "fut_code"]).any():
        raise ValueError("product amounts contain duplicate date-product keys")

    return amounts.sort_values(
        ["trade_date", "fut_code"]
    ).reset_index(drop=True)


def build_semiannual_universe(
    product_amounts: pd.DataFrame,
    trade_calendar: pd.DataFrame,
    start_date: str,
    end_date: str,
    liquidity_lookback: int = DEFAULT_LIQUIDITY_LOOKBACK,
    min_observations: int = DEFAULT_MIN_AMOUNT_OBSERVATIONS,
    pool_size: int = DEFAULT_POOL_SIZE,
) -> UniverseResult:
    """Select and audit lagged January/July Top-N amount pools."""
    requested_start, requested_end = validate_dates(start_date, end_date)

    integer_parameters = {
        "liquidity_lookback": liquidity_lookback,
        "min_observations": min_observations,
        "pool_size": pool_size,
    }
    for name, value in integer_parameters.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
        ):
            raise ValueError(f"{name} must be a positive integer")
    if min_observations > liquidity_lookback:
        raise ValueError(
            "min_observations cannot exceed liquidity_lookback"
        )

    required_amount_columns = {
        "trade_date",
        "fut_code",
        "product_amount",
    }
    missing_amount_columns = required_amount_columns - set(
        product_amounts.columns
    )
    if missing_amount_columns:
        raise ValueError(
            "product_amounts is missing columns: "
            + ", ".join(sorted(missing_amount_columns))
        )
    if "trade_date" not in trade_calendar.columns:
        raise ValueError("trade_calendar is missing column: trade_date")

    amounts = product_amounts.copy()
    amounts["trade_date"] = pd.to_datetime(
        amounts["trade_date"],
        errors="raise",
    )
    if amounts.duplicated(["trade_date", "fut_code"]).any():
        raise ValueError(
            "product_amounts contains duplicate date-product keys"
        )
    amounts["product_amount"] = pd.to_numeric(
        amounts["product_amount"],
        errors="raise",
    )
    valid_amount = (
        np.isfinite(amounts["product_amount"])
        & amounts["product_amount"].gt(0)
    )
    amounts["product_amount"] = amounts["product_amount"].where(
        valid_amount
    )

    calendar_dates = pd.DatetimeIndex(
        pd.to_datetime(
            trade_calendar["trade_date"],
            errors="raise",
        )
    )
    if calendar_dates.duplicated().any():
        raise ValueError("trade_calendar contains duplicate trade dates")
    if not calendar_dates.is_monotonic_increasing:
        raise ValueError("trade_calendar is not sorted by trade_date")

    report_dates = calendar_dates[
        (calendar_dates >= requested_start)
        & (calendar_dates <= requested_end)
    ]

    if report_dates.empty:
        raise ValueError(
            'trade_calendar has no trade dates in the requested range'
        )

    selection_frame = pd.DataFrame(
        {
            'selection_date': report_dates,
        }
    )
    selection_frame['year'] = (
        selection_frame['selection_date'].dt.year
    )
    selection_frame['month'] = (
        selection_frame['selection_date'].dt.month
    )

    selection_dates = (
        selection_frame.loc[
            selection_frame['month'].isin([1, 7])
        ]
        .groupby(
            ['year', 'month'],
            sort=True,
        )['selection_date']
        .min()
        .tolist()
    )

    if not selection_dates:
        raise ValueError(
            'trade_calendar has no trade dates in the requested range'
        )

    ranking_parts = []

    for selection_date in selection_dates:
        window_dates = calendar_dates[
            calendar_dates < selection_date
        ][-liquidity_lookback:]

        candidate_codes = sorted(
            amounts.loc[
                amounts['trade_date'] < selection_date,
                'fut_code',
            ]
            .dropna()
            .unique()
        )

        window_panel = (
            amounts.loc[
                amounts['trade_date'].isin(window_dates)
                & amounts['fut_code'].isin(candidate_codes)
            ]
            .pivot(
                index='trade_date',
                columns='fut_code',
                values='product_amount',
            )
            .reindex(
                index=window_dates,
                columns=candidate_codes,
            )
        )

        selection_stats = pd.DataFrame(
            {
                'fut_code': candidate_codes,
                'rolling_amount': window_panel.mean(
                    axis=0,
                    skipna=True,
                ).to_numpy(),
                'observation_count': window_panel.count(
                    axis=0,
                ).to_numpy(),
            }
        )

        selection_stats.insert(
            0,
            'selection_date',
            selection_date,
        )

        selection_stats['is_eligible'] = (
            selection_stats['observation_count']
            >= min_observations
        )

        selection_stats['liquidity_rank'] = pd.Series(
            pd.NA,
            index=selection_stats.index,
            dtype='Int64',
        )

        eligible_order = (
            selection_stats.loc[
                selection_stats['is_eligible']
            ]
            .sort_values(
                ['rolling_amount', 'fut_code'],
                ascending=[False, True],
            )
        )

        selection_stats.loc[
            eligible_order.index,
            'liquidity_rank',
        ] = np.arange(1, len(eligible_order) + 1)

        selection_stats['is_selected'] = (
            selection_stats['liquidity_rank']
            .le(pool_size)
            .fillna(False)
        )

        ranking_parts.append(selection_stats)

    ranking = pd.concat(
        ranking_parts,
        ignore_index=True,
    )

    ranking['effective_start'] = pd.NaT
    ranking['effective_end'] = pd.NaT
    ranking['change_status'] = 'not_selected'

    previous_selected = set()

    for position, selection_date in enumerate(selection_dates):
        if position + 1 < len(selection_dates):
            next_selection_date = selection_dates[position + 1]
            period_dates = report_dates[
                (report_dates >= selection_date)
                & (report_dates < next_selection_date)
            ]
        else:
            period_dates = report_dates[
                report_dates >= selection_date
            ]

        effective_end = period_dates[-1]

        selection_mask = ranking['selection_date'].eq(
            selection_date
        )

        ranking.loc[
            selection_mask,
            'effective_start',
        ] = selection_date

        ranking.loc[
            selection_mask,
            'effective_end',
        ] = effective_end

        current_selected = set(
            ranking.loc[
                selection_mask & ranking['is_selected'],
                'fut_code',
            ]
        )

        entered = current_selected - previous_selected
        retained = current_selected & previous_selected
        exited = previous_selected - current_selected

        ranking.loc[
            selection_mask &
            ranking['fut_code'].isin(entered),
            'change_status',
        ] = 'entered'

        ranking.loc[
            selection_mask &
            ranking['fut_code'].isin(retained),
            'change_status',
        ] = 'retained'

        ranking.loc[
            selection_mask &
            ranking['fut_code'].isin(exited),
            'change_status',
        ] = 'exited'

        previous_selected = current_selected

    ranking = ranking.sort_values(
        [
            'selection_date',
            'is_selected',
            'liquidity_rank',
            'fut_code',
        ],
        ascending=[True, False, True, True],
        na_position='last',
    ).reset_index(drop=True)

    members = (
        ranking.loc[ranking['is_selected']]
        .copy()
        .reset_index(drop=True)
    )

    changes = (
        ranking.loc[
            ranking['change_status'].isin(
                ['entered', 'exited']
            )
        ]
        .copy()
        .reset_index(drop=True)
    )

    daily_parts = []

    for selection_date in selection_dates:
        period_members = members.loc[
            members['selection_date'].eq(selection_date)
        ]

        if period_members.empty:
            continue

        effective_start = period_members[
            'effective_start'
        ].iloc[0]

        effective_end = period_members[
            'effective_end'
        ].iloc[0]

        active_dates = report_dates[
            (report_dates >= effective_start)
            & (report_dates <= effective_end)
        ]

        daily_period = pd.MultiIndex.from_product(
            [
                active_dates,
                period_members['fut_code'].to_list(),
            ],
            names=['trade_date', 'fut_code'],
        ).to_frame(index=False)

        daily_period.insert(
            1,
            'selection_date',
            selection_date,
        )

        daily_period = daily_period.merge(
            period_members[
                [
                    'selection_date',
                    'fut_code',
                    'liquidity_rank',
                    'rolling_amount',
                    'observation_count',
                ]
            ],
            on=['selection_date', 'fut_code'],
            how='left',
            validate='many_to_one'
        )

        daily_parts.append(daily_period)

    if daily_parts:
        daily_membership = (
            pd.concat(
                daily_parts,
                ignore_index=True,
            )
            .sort_values(
                [
                    'trade_date',
                    'liquidity_rank',
                    'fut_code',
                ]
            )
            .reset_index(drop=True)
        )
    else:
        daily_membership = pd.DataFrame(
            columns=[
                'trade_date',
                'selection_date',
                'fut_code',
                'liquidity_rank',
                'rolling_amount',
                'observation_count',
            ]
        )

    return UniverseResult(
        ranking=ranking,
        members=members,
        changes=changes,
        daily_membership=daily_membership,
    )


def load_fixed_contract_closes(
    start_date: str,
    end_date: str,
    db_path: str = DB_PATH,
) -> pd.DataFrame:
    """Load fixed-contract closes with a ninety-calendar-day warm-up."""
    requested_start, requested_end = validate_dates(start_date, end_date)
    history_start = requested_start - pd.Timedelta(days=90)
    query = """
    SELECT
        trade_date,
        ts_code,
        close
    FROM fut_daily
    WHERE trade_date BETWEEN ? AND ?
    ORDER BY ts_code, trade_date
    """

    with closing(sqlite3.connect(db_path)) as connection:
        prices = pd.read_sql_query(
            query,
            connection,
            params=(
                history_start.strftime("%Y%m%d"),
                requested_end.strftime("%Y%m%d"),
            ),
        )

    prices["trade_date"] = pd.to_datetime(
        prices["trade_date"],
        errors="raise",
    )
    prices["close"] = pd.to_numeric(
        prices["close"],
        errors="coerce",
    )
    if prices.duplicated(["trade_date", "ts_code"]).any():
        raise ValueError("fixed-contract closes contain duplicate keys")

    return prices.sort_values(
        ["ts_code", "trade_date"]
    ).reset_index(drop=True)


def compute_contract_volatility(
    contract_prices: pd.DataFrame,
    lookback: int = DEFAULT_VOLATILITY_LOOKBACK,
) -> pd.DataFrame:
    """Compute complete-window sample volatility within each contract."""
    if (
        isinstance(lookback, bool)
        or not isinstance(lookback, int)
        or lookback <= 0
    ):
        raise ValueError("lookback must be a positive integer")

    required_columns = {"trade_date", "ts_code", "close"}
    missing_columns = required_columns - set(contract_prices.columns)
    if missing_columns:
        raise ValueError(
            "contract_prices is missing columns: "
            + ", ".join(sorted(missing_columns))
        )

    prices = contract_prices.copy()
    prices["trade_date"] = pd.to_datetime(
        prices["trade_date"],
        errors="raise",
    )
    if prices.duplicated(["trade_date", "ts_code"]).any():
        raise ValueError("contract_prices contains duplicate keys")

    prices["close"] = pd.to_numeric(
        prices["close"],
        errors="coerce",
    )
    valid_close = np.isfinite(prices["close"]) & prices["close"].gt(0)
    prices["close"] = prices["close"].where(valid_close)
    prices = prices.sort_values(
        ["ts_code", "trade_date"]
    ).reset_index(drop=True)

    prices['contract_return'] = (
        prices.groupby(
            'ts_code',
            sort=False,
        )['close']
        .pct_change(fill_method=None)
    )

    prices['volatility_20'] = (
        prices.groupby(
            'ts_code',
            sort=False,
        )['contract_return']
        .transform(
            lambda values: values.rolling(
                window=lookback,
                min_periods=lookback,
            ).std(ddof=1)
        )
    )

    prices['annualized_volatility_20'] = (
        prices['volatility_20'] * np.sqrt(252)
    )

    return prices


def build_inverse_volatility_weights(
    factor_data: pd.DataFrame,
    contract_context: pd.DataFrame,
    contract_volatility: pd.DataFrame,
    daily_membership: pd.DataFrame,
    method: str,
    min_assets: int = DEFAULT_MIN_ASSETS,
) -> pd.DataFrame:
    """Build one daily market-neutral inverse-volatility target panel."""
    if method not in PORTFOLIO_METHODS:
        raise ValueError(
            "method must be one of: " + ", ".join(PORTFOLIO_METHODS)
        )
    if (
        isinstance(min_assets, bool)
        or not isinstance(min_assets, int)
        or min_assets < 2
    ):
        raise ValueError("min_assets must be an integer of at least 2")

    frame_specs = (
        (
            "factor_data",
            factor_data,
            {"trade_date", "fut_code", "raw_factor"},
            ["trade_date", "fut_code"],
        ),
        (
            "contract_context",
            contract_context,
            {"trade_date", "fut_code", "ts_code_A"},
            ["trade_date", "fut_code"],
        ),
        (
            "contract_volatility",
            contract_volatility,
            {"trade_date", "ts_code", "volatility_20"},
            ["trade_date", "ts_code"],
        ),
        (
            "daily_membership",
            daily_membership,
            {"trade_date", "fut_code"},
            ["trade_date", "fut_code"],
        ),
    )
    prepared_frames = {}
    for name, frame, required_columns, key_columns in frame_specs:
        missing_columns = required_columns - set(frame.columns)
        if missing_columns:
            raise ValueError(
                f"{name} is missing columns: "
                + ", ".join(sorted(missing_columns))
            )
        prepared = frame.copy()
        prepared["trade_date"] = pd.to_datetime(
            prepared["trade_date"],
            errors="raise",
        )
        if prepared.duplicated(key_columns).any():
            raise ValueError(f"{name} contains duplicate keys")
        prepared_frames[name] = prepared

    factors = prepared_frames["factor_data"][
        ["trade_date", "fut_code", "raw_factor"]
    ].copy()
    factors["raw_factor"] = pd.to_numeric(
        factors["raw_factor"],
        errors="coerce",
    )
    factors["raw_factor"] = factors["raw_factor"].where(
        np.isfinite(factors["raw_factor"])
    )

    context = prepared_frames["contract_context"][
        ["trade_date", "fut_code", "ts_code_A"]
    ].copy()
    volatility = prepared_frames["contract_volatility"][
        ["trade_date", "ts_code", "volatility_20"]
    ].rename(columns={"ts_code": "ts_code_A"})
    volatility["volatility_20"] = pd.to_numeric(
        volatility["volatility_20"],
        errors="coerce",
    )
    valid_volatility = (
        np.isfinite(volatility["volatility_20"])
        & volatility["volatility_20"].gt(0)
    )
    volatility["volatility_20"] = volatility[
        "volatility_20"
    ].where(valid_volatility)

    membership = prepared_frames["daily_membership"].copy()
    if "is_selected" in membership.columns:
        membership["_in_pool"] = membership["is_selected"].fillna(False)
    else:
        membership["_in_pool"] = True
    membership = membership[
        ["trade_date", "fut_code", "_in_pool"]
    ]

    panel = factors.merge(
        context,
        on=["trade_date", "fut_code"],
        how="left",
        validate="one_to_one",
    )
    panel = panel.merge(
        membership,
        on=["trade_date", "fut_code"],
        how="left",
        validate="one_to_one",
    )
    panel = panel.merge(
        volatility,
        on=["trade_date", "ts_code_A"],
        how="left",
        validate="many_to_one",
    )
    panel = panel.sort_values(
        ["trade_date", "fut_code"]
    ).reset_index(drop=True)

    panel["is_rebalance"] = True
    panel["passes_liquidity"] = panel["_in_pool"].eq(True)
    panel["is_eligible"] = (
        panel["passes_liquidity"]
        & panel["raw_factor"].notna()
        & panel["ts_code_A"].notna()
        & panel["volatility_20"].notna()
    )
    panel["eligible_count"] = panel.groupby(
        "trade_date"
    )["is_eligible"].transform("sum")
    panel["factor_score"] = np.nan
    panel["risk_score"] = np.nan
    panel["weight"] = 0.0
    panel["long_count"] = 0
    panel["short_count"] = 0
    panel["max_abs_weight"] = 0.0

    for trade_date, group in panel.groupby('trade_date', sort=True,):
        eligible_index = group.index[
            group['is_eligible']
        ]

        eligible_count = len(eligible_index)

        if eligible_count < min_assets:
            continue

        factor_rank = panel.loc[
            eligible_index,
            'raw_factor',
        ].rank(
            method='average',
            ascending=True,
        )

        factor_score = (
            2.0
            * (factor_rank - 1.0)
            / (eligible_count - 1.0)
            - 1.0
        )

        panel.loc[
            eligible_index,
            'factor_score',
        ] = factor_score

        panel.loc[
            eligible_index,
            'risk_score',
        ] = (
            factor_score
            / panel.loc[
                eligible_index,
                'volatility_20'
            ]
        )

    for trade_date, group in panel.groupby(
        'trade_date',
        sort=True,
    ):
        eligible_group = group.loc[
            group['is_eligible']
        ]

        eligible_count = len(eligible_group)

        if eligible_count < min_assets:
            continue

        if method == 'full_pool_invvol':
            long_index = eligible_group.index[
                eligible_group['risk_score'].gt(0)
            ]
            short_index = eligible_group.index[
                eligible_group['risk_score'].lt(0)
            ]

        else:
            tail_count = math.ceil(
                0.10 * eligible_count
            )

            long_index = (
                eligible_group.loc[
                    eligible_group['risk_score'].gt(0)
                ]
                .sort_values(
                    ['factor_score', 'fut_code'],
                    ascending=[False, True],
                )
                .head(tail_count)
                .index
            )

            short_index = (
                eligible_group.loc[
                    eligible_group['risk_score'].lt(0)
                ]
                .sort_values(
                    ['factor_score', 'fut_code'],
                    ascending=[True, True],
                )
                .head(tail_count)
                .index
            )

        long_total = panel.loc[
            long_index,
            'risk_score',
        ].sum()

        short_total = panel.loc[
            short_index,
            'risk_score',
        ].abs().sum()

        if long_total <= 0 or short_total <= 0:
            continue

        panel.loc[
            long_index,
            'weight',
        ] = (
            0.5
            * panel.loc[
                long_index,
                'risk_score',
            ]
            / long_total
        )

        panel.loc[
            short_index,
            'weight',
        ] = (
            0.5
            * panel.loc[
                short_index,
                'risk_score',
            ]
            / short_total
        )

        panel.loc[
            group.index,
            'long_count',
        ] = len(long_index)

        panel.loc[
            group.index,
            'short_count',
        ] = len(short_index)

        panel.loc[
            group.index,
            'max_abs_weight',
        ] = (
            panel.loc[
                group.index,
                'weight',
            ]
            .abs()
            .max()
        )

    return panel.drop(
        columns='_in_pool'
    )


def combine_factor_returns(
    factor_returns: Mapping[str, pd.Series],
) -> pd.DataFrame:
    """Combine five aligned factor sleeves at constant equal weights."""
    expected_names = list(FACTOR_SPECS)
    if (
        not isinstance(factor_returns, Mapping)
        or set(factor_returns) != set(expected_names)
    ):
        raise ValueError(
            "factor names must match: " + ", ".join(expected_names)
        )

    prepared_returns = {}
    reference_calendar = None
    for factor_name in expected_names:
        factor_series = factor_returns[factor_name]
        if not isinstance(factor_series, pd.Series):
            raise ValueError(
                f"{factor_name} returns must be a pandas Series"
            )
        if factor_series.empty:
            raise ValueError("factor return calendar cannot be empty")

        calendar = pd.DatetimeIndex(
            pd.to_datetime(
                factor_series.index,
                errors="raise",
            )
        )
        if calendar.duplicated().any():
            raise ValueError(
                f"{factor_name} contains duplicate return dates"
            )

        numeric_returns = pd.to_numeric(
            factor_series,
            errors="coerce",
        )
        if (
            numeric_returns.isna().any()
            or not np.isfinite(numeric_returns).all()
        ):
            raise ValueError(
                f"{factor_name} returns must all be finite"
            )

        prepared = pd.Series(
            numeric_returns.to_numpy(dtype=float),
            index=calendar,
            name=factor_name,
        ).sort_index()
        if reference_calendar is None:
            reference_calendar = prepared.index
        elif not prepared.index.equals(reference_calendar):
            raise ValueError("factor return calendars must match exactly")

        prepared_returns[factor_name] = prepared

    combined = pd.concat(
        [
            prepared_returns[factor_name]
            for factor_name in expected_names
        ],
        axis=1
    )

    combined['combined_return'] = combined[
        expected_names
    ].mean(axis=1)

    return combined


def build_execution_audit(
    weights: pd.DataFrame,
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """Shift close targets once and resolve their next-open execution."""
    required_weight_columns = {
        "trade_date",
        "fut_code",
        "weight",
        "is_rebalance",
        "ts_code_A",
    }
    missing_weight_columns = required_weight_columns - set(weights.columns)
    if missing_weight_columns:
        raise ValueError(
            "weights is missing columns: "
            + ", ".join(sorted(missing_weight_columns))
        )
    required_price_columns = {"trade_date", "ts_code", "open"}
    missing_price_columns = required_price_columns - set(prices.columns)
    if missing_price_columns:
        raise ValueError(
            "prices is missing columns: "
            + ", ".join(sorted(missing_price_columns))
        )

    audit = weights.copy()
    audit["trade_date"] = pd.to_datetime(
        audit["trade_date"],
        errors="raise",
    )
    if audit.duplicated(["trade_date", "fut_code"]).any():
        raise ValueError("weights contains duplicate date-product keys")
    if audit["is_rebalance"].isna().any():
        raise ValueError("is_rebalance cannot contain missing values")
    audit = audit.sort_values(
        ["fut_code", "trade_date"]
    ).reset_index(drop=True)

    execution_prices = prices.copy()
    execution_prices["trade_date"] = pd.to_datetime(
        execution_prices["trade_date"],
        errors="raise",
    )
    if execution_prices.duplicated(["trade_date", "ts_code"]).any():
        raise ValueError("prices contains duplicate date-contract keys")

    audit["target_weight"] = audit["weight"].where(
        audit["is_rebalance"],
        np.nan,
    )
    shifted_weight = audit.groupby(
        "fut_code",
        sort=False,
    )["target_weight"].shift(1)
    audit["desired_exec_weight"] = (
        shifted_weight.groupby(
            audit["fut_code"],
            sort=False,
        )
        .ffill()
        .fillna(0.0)
    )

    audit["target_ts_code"] = audit["ts_code_A"].where(
        audit["is_rebalance"],
        np.nan,
    )
    shifted_contract = audit.groupby(
        "fut_code",
        sort=False,
    )["target_ts_code"].shift(1)
    audit["desired_trade_ts_code"] = shifted_contract.groupby(
        audit["fut_code"],
        sort=False,
    ).ffill()

    return resolve_executed_positions(
        audit,
        execution_prices,
    )


def run_factor_method(
    weights: pd.DataFrame,
    prices: pd.DataFrame,
    cost_rate: float,
) -> dict[str, object]:
    """Run one factor-method sleeve through the authoritative engine."""
    if (
        isinstance(cost_rate, bool)
        or not isinstance(cost_rate, (int, float))
        or not math.isfinite(cost_rate)
        or cost_rate < 0
    ):
        raise ValueError("cost_rate must be a non-negative finite number")

    execution_weights = build_execution_audit(weights, prices)
    nav, metrics, daily_return, daily_diagnostics = (
        run_backtest_from_weights(
            weights=weights,
            prices=prices,
            cost_rate=float(cost_rate),
            return_diagnostics=True,
        )
    )
    return {
        "nav": nav,
        "metrics": metrics,
        "daily_return": daily_return,
        "daily_diagnostics": daily_diagnostics,
        "execution_weights": execution_weights,
    }


def _prepare_return_series(
    daily_return: pd.Series,
    name: str,
) -> pd.Series:
    """Validate and sort one finite, uniquely dated return series."""
    if not isinstance(daily_return, pd.Series) or daily_return.empty:
        raise ValueError(f"{name} daily_return must be a nonempty Series")
    dates = pd.DatetimeIndex(
        pd.to_datetime(daily_return.index, errors="raise")
    )
    if dates.duplicated().any():
        raise ValueError(f"{name} daily_return contains duplicate dates")
    values = pd.to_numeric(daily_return, errors="coerce")
    if values.isna().any() or not np.isfinite(values).all():
        raise ValueError(f"{name} daily_return must be finite")
    return pd.Series(
        values.to_numpy(dtype=float),
        index=dates,
        name="daily_return",
    ).sort_index()


def calculate_strategy_metrics(
    name: str,
    daily_return: pd.Series,
    daily_diagnostics: pd.DataFrame,
) -> dict[str, object]:
    """Calculate exact-period and annualized metrics for one sleeve."""
    returns = _prepare_return_series(daily_return, name)
    diagnostics = daily_diagnostics.copy()
    if "trade_date" in diagnostics.columns:
        diagnostics["trade_date"] = pd.to_datetime(
            diagnostics["trade_date"],
            errors="raise",
        )
        diagnostics = diagnostics.set_index("trade_date")
    diagnostics.index = pd.to_datetime(diagnostics.index, errors="raise")
    if diagnostics.index.duplicated().any():
        raise ValueError(f"{name} diagnostics contains duplicate dates")
    diagnostics = diagnostics.sort_index()
    if not diagnostics.index.equals(returns.index):
        raise ValueError(f"{name} diagnostics calendar must match returns")

    metrics = compute_metrics(returns)
    cumulative_values = np.concatenate(
        ([1.0], (1.0 + returns).cumprod().to_numpy())
    )
    drawdowns = (
        cumulative_values / np.maximum.accumulate(cumulative_values) - 1.0
    )
    max_drawdown = float(drawdowns.min())
    metrics["max_drawdown"] = max_drawdown
    metrics["calmar"] = (
        metrics["annual_return"] / abs(max_drawdown)
        if max_drawdown < 0
        else np.nan
    )
    metrics.update(
        {
            "strategy_name": name,
            "period_start": returns.index.min(),
            "period_end": returns.index.max(),
            "trading_days": len(returns),
            "period_return": float((1.0 + returns).prod() - 1.0),
            "turnover_sum": float(
                diagnostics.get(
                    "turnover",
                    pd.Series(0.0, index=diagnostics.index),
                ).sum()
            ),
            "cost_sum": float(
                diagnostics.get(
                    "cost",
                    pd.Series(0.0, index=diagnostics.index),
                ).sum()
            ),
        }
    )
    return metrics


def calculate_annual_metrics(
    strategy_returns: Mapping[str, pd.Series],
    strategy_diagnostics: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    """Return one explicitly labelled metric row per strategy and year."""
    if set(strategy_returns) != set(strategy_diagnostics):
        raise ValueError("strategy return and diagnostic names must match")
    records = []
    for strategy_name, daily_return in strategy_returns.items():
        returns = _prepare_return_series(daily_return, strategy_name)
        diagnostics = strategy_diagnostics[strategy_name].copy()
        if "trade_date" in diagnostics.columns:
            diagnostics["trade_date"] = pd.to_datetime(
                diagnostics["trade_date"],
                errors="raise",
            )
            diagnostics = diagnostics.set_index("trade_date")
        diagnostics.index = pd.to_datetime(diagnostics.index, errors="raise")

        for year, annual_return in returns.groupby(returns.index.year):
            annual_diagnostics = diagnostics.loc[
                diagnostics.index.year == year
            ]
            record = calculate_strategy_metrics(
                strategy_name,
                annual_return,
                annual_diagnostics,
            )
            period_start = record["period_start"]
            period_end = record["period_end"]
            record["year"] = int(year)
            record["is_partial_year"] = bool(
                period_start.month != 1
                or period_start.day > 7
                or period_end.month != 12
                or period_end.day < 24
            )
            records.append(record)

    return pd.DataFrame(records).sort_values(
        ["strategy_name", "year"]
    ).reset_index(drop=True)


def calculate_annual_returns(
    strategy_returns: Mapping[str, pd.Series],
) -> pd.DataFrame:
    """Pivot exact compounded calendar-year returns by strategy."""
    annual_series = {}
    for strategy_name, daily_return in strategy_returns.items():
        returns = _prepare_return_series(daily_return, strategy_name)
        annual_series[strategy_name] = (
            (1.0 + returns)
            .groupby(returns.index.year)
            .prod()
            .sub(1.0)
        )
    result = pd.concat(annual_series, axis=1).sort_index()
    result.index.name = "year"
    return result.reset_index()


def calculate_annual_ic(
    ic_series: pd.DataFrame,
    factor_name: str,
) -> pd.DataFrame:
    """Summarize daily IC and Rank IC separately by calendar year."""
    required_columns = {"signal_date", "ic", "rank_ic"}
    missing_columns = required_columns - set(ic_series.columns)
    if missing_columns:
        raise ValueError(
            "ic_series is missing columns: "
            + ", ".join(sorted(missing_columns))
        )
    data = ic_series.copy()
    data["signal_date"] = pd.to_datetime(
        data["signal_date"],
        errors="raise",
    )
    if data["signal_date"].duplicated().any():
        raise ValueError("ic_series contains duplicate signal dates")
    for metric in ("ic", "rank_ic"):
        data[metric] = pd.to_numeric(data[metric], errors="coerce")
        data[metric] = data[metric].where(np.isfinite(data[metric]))
    data["year"] = data["signal_date"].dt.year

    records = []
    for year, group in data.groupby("year", sort=True):
        record = {"factor_name": factor_name, "year": int(year)}
        for metric in ("ic", "rank_ic"):
            values = group[metric].dropna()
            count = len(values)
            mean_value = values.mean()
            std_value = values.std(ddof=1)
            t_stat = (
                mean_value / (std_value / np.sqrt(count))
                if count >= 2 and std_value > 0
                else np.nan
            )
            record.update(
                {
                    f"mean_{metric}": mean_value,
                    f"std_{metric}": std_value,
                    f"positive_rate_{metric}": (
                        values.gt(0).mean() if count else np.nan
                    ),
                    f"t_stat_{metric}": t_stat,
                    f"valid_count_{metric}": count,
                }
            )
        records.append(record)
    return pd.DataFrame(records)


def build_output_directory(
    result_dir: str | Path,
    settings: StrategySettings,
) -> Path:
    """Return the stable daily directory for one approved strategy run."""
    run_label = (
        f"top{settings.pool_size}_amount{settings.liquidity_lookback}"
        f"_vol{settings.volatility_lookback}-"
        f"{settings.start_date}-{settings.end_date}"
    )
    return (
        Path(result_dir)
        / "multi_factor_strategy"
        / run_label
        / "daily"
    )


def _known_output_paths(path: Path) -> list[Path]:
    """List only artifacts owned by this runner, without filesystem globbing."""
    owned = [path / filename for filename in RUN_OUTPUT_FILENAMES]
    owned.extend(
        path / f"{factor_name}_nav_comparison.png"
        for factor_name in FACTOR_SPECS
    )
    for method in PORTFOLIO_METHODS:
        owned.extend(
            path / method / filename
            for filename in METHOD_OUTPUT_FILENAMES
        )
    return owned


def ensure_output_directory(
    path: Path,
    overwrite: bool,
) -> None:
    """Create owned folders and reject known artifact collisions by default."""
    output_path = Path(path)
    collisions = [
        candidate.relative_to(output_path).as_posix()
        for candidate in _known_output_paths(output_path)
        if candidate.exists()
    ]
    if collisions and not overwrite:
        raise FileExistsError(
            "refusing to overwrite strategy outputs: "
            + ", ".join(collisions)
        )
    output_path.mkdir(parents=True, exist_ok=True)
    for method in PORTFOLIO_METHODS:
        (output_path / method).mkdir(parents=True, exist_ok=True)


def plot_nav_comparison(
    nav: pd.DataFrame,
    output_path: str | Path,
    title: str,
) -> None:
    """Plot nonempty NAV series normalized at one common first date."""
    if not isinstance(nav, pd.DataFrame) or nav.empty:
        raise ValueError("nav must be a nonempty DataFrame")
    plot_data = nav.copy()
    plot_data.index = pd.to_datetime(plot_data.index, errors="raise")
    plot_data = plot_data.sort_index().dropna(how="any")
    if plot_data.empty:
        raise ValueError("nav series have no common valid dates")
    plot_data = plot_data.apply(pd.to_numeric, errors="coerce")
    if (
        plot_data.isna().any().any()
        or not np.isfinite(plot_data.to_numpy()).all()
        or plot_data.iloc[0].eq(0).any()
    ):
        raise ValueError("nav series must contain finite nonzero values")
    normalized = plot_data.divide(plot_data.iloc[0], axis=1)

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(12, 6))
    normalized.plot(ax=axis, linewidth=1.6)
    axis.axhline(1.0, color="#777777", linestyle="--", linewidth=0.9)
    axis.set_title(title)
    axis.set_xlabel("Trade date")
    axis.set_ylabel("NAV (common start = 1)")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    figure.tight_layout()
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_strategy_results(
    output_dir: str | Path,
    settings: StrategySettings,
    universe: UniverseResult,
    method_results: Mapping[str, Mapping[str, Mapping[str, object]]],
    ic_summary: pd.DataFrame,
    ic_series: pd.DataFrame,
    ic_annual: pd.DataFrame,
) -> Path:
    """Write all run, method, annual, weight, IC, and chart artifacts."""
    output_path = Path(output_dir)
    pd.DataFrame([asdict(settings)]).to_csv(
        output_path / "run_config.csv",
        index=False,
    )
    universe.ranking.to_csv(output_path / "universe_ranking.csv", index=False)
    universe.members.to_csv(output_path / "universe_members.csv", index=False)
    universe.changes.to_csv(output_path / "universe_changes.csv", index=False)
    ic_summary.to_csv(output_path / "factor_ic_summary.csv", index=False)
    ic_series.to_csv(output_path / "factor_ic_series.csv", index=False)
    ic_annual.to_csv(output_path / "factor_ic_annual.csv", index=False)

    metric_records = []
    for method in PORTFOLIO_METHODS:
        if method not in method_results:
            raise ValueError(f"method_results is missing method: {method}")
        results = method_results[method]
        method_path = output_path / method
        return_series = {}
        nav_series = {}
        diagnostic_frames = []
        weight_frames = []

        for strategy_name, result in results.items():
            daily_return = result["daily_return"].copy()
            daily_return.name = strategy_name
            return_series[strategy_name] = daily_return
            nav = result["nav"].copy()
            nav.name = strategy_name
            nav_series[strategy_name] = nav

            diagnostics = result["daily_diagnostics"].copy()
            diagnostics.index.name = "trade_date"
            diagnostics = diagnostics.reset_index()
            diagnostics.insert(0, "strategy_name", strategy_name)
            diagnostic_frames.append(diagnostics)

            metrics = dict(result["metrics"])
            metric_records.append(
                {
                    "method": method,
                    "strategy_name": strategy_name,
                    **metrics,
                }
            )
            if "execution_weights" in result:
                details = result["execution_weights"].copy()
                details.insert(0, "factor_name", strategy_name)
                weight_frames.append(details)

        returns_frame = pd.concat(return_series, axis=1)
        returns_frame.index.name = "trade_date"
        nav_frame = pd.concat(nav_series, axis=1)
        nav_frame.index.name = "trade_date"
        diagnostics_frame = pd.concat(diagnostic_frames, ignore_index=True)
        weights_frame = (
            pd.concat(weight_frames, ignore_index=True)
            if weight_frames
            else pd.DataFrame()
        )
        annual_metrics = calculate_annual_metrics(
            return_series,
            {
                name: result["daily_diagnostics"]
                for name, result in results.items()
            },
        )
        annual_returns = calculate_annual_returns(return_series)

        returns_frame.to_csv(method_path / "daily_returns.csv")
        nav_frame.to_csv(method_path / "nav.csv")
        diagnostics_frame.to_csv(
            method_path / "daily_diagnostics.csv",
            index=False,
        )
        weights_frame.to_csv(method_path / "weights.csv", index=False)
        annual_metrics.to_csv(method_path / "annual_metrics.csv", index=False)
        annual_returns.to_csv(method_path / "annual_returns.csv", index=False)
        plot_nav_comparison(
            nav_frame,
            method_path / "nav_summary.png",
            title=f"{method}: five factors and combined NAV",
        )

    pd.DataFrame(metric_records).to_csv(
        output_path / "strategy_metrics.csv",
        index=False,
    )
    for factor_name in FACTOR_SPECS:
        comparison = pd.concat(
            {
                method: method_results[method][factor_name]["nav"]
                for method in PORTFOLIO_METHODS
            },
            axis=1,
        )
        plot_nav_comparison(
            comparison,
            output_path / f"{factor_name}_nav_comparison.png",
            title=f"{factor_name}: portfolio-method comparison",
        )
    combined_comparison = pd.concat(
        {
            method: method_results[method]["combined"]["nav"]
            for method in PORTFOLIO_METHODS
        },
        axis=1,
    )
    plot_nav_comparison(
        combined_comparison,
        output_path / "combined_nav_comparison.png",
        title="Combined five-factor NAV: portfolio-method comparison",
    )
    return output_path


def _combine_daily_diagnostics(
    factor_results: Mapping[str, Mapping[str, object]],
    combined_return: pd.Series,
) -> pd.DataFrame:
    """Combine sleeve diagnostics using the same constant 20% allocation."""
    combined = pd.DataFrame(index=combined_return.index)
    numeric_columns = sorted(
        set.intersection(
            *(
                set(
                    result["daily_diagnostics"].select_dtypes(
                        include=[np.number, "bool"]
                    ).columns
                )
                for result in factor_results.values()
            )
        )
    )
    for column in numeric_columns:
        aligned = pd.concat(
            [
                result["daily_diagnostics"][column].reindex(
                    combined_return.index
                )
                for result in factor_results.values()
            ],
            axis=1,
        )
        if aligned.isna().all().all():
            combined[column] = np.nan
            continue
        if aligned.isna().any().any():
            raise ValueError(
                f"factor diagnostic calendars do not align for {column}"
            )
        combined[column] = aligned.mean(axis=1)
    combined["daily_return"] = combined_return
    for column in ("turnover", "cost"):
        if column not in combined.columns:
            combined[column] = 0.0
    return combined


def run_multi_factor_strategy(
    settings: StrategySettings,
    result_dir: str | Path = RESULT_DIR,
    overwrite: bool = False,
) -> Path:
    """Run the approved five-factor strategy with shared data preparation."""
    if not isinstance(settings, StrategySettings):
        raise TypeError("settings must be a StrategySettings instance")
    output_dir = build_output_directory(result_dir, settings)
    ensure_output_directory(output_dir, overwrite=overwrite)

    requested_start, _ = validate_dates(
        settings.start_date,
        settings.end_date,
    )
    history_start = (
        requested_start - pd.Timedelta(days=365)
    ).strftime("%Y%m%d")
    trade_calendar = load_trade_calendar(
        history_start,
        settings.end_date,
    )
    product_amounts = load_product_amounts(
        history_start,
        settings.end_date,
    )
    universe = build_semiannual_universe(
        product_amounts=product_amounts,
        trade_calendar=trade_calendar,
        start_date=settings.start_date,
        end_date=settings.end_date,
        liquidity_lookback=settings.liquidity_lookback,
        min_observations=settings.min_amount_observations,
        pool_size=settings.pool_size,
    )

    fixed_closes = load_fixed_contract_closes(
        settings.start_date,
        settings.end_date,
    )
    contract_volatility = compute_contract_volatility(
        fixed_closes,
        lookback=settings.volatility_lookback,
    )
    contract_context = prepare_contract_context(
        factor_data=pd.DataFrame(),
        start_date=settings.start_date,
        end_date=settings.end_date,
        min_days_to_maturity=settings.trade_min_days_to_maturity,
    )
    prices = load_contract_prices(
        settings.start_date,
        settings.end_date,
    )
    report_calendar = trade_calendar.loc[
        trade_calendar["trade_date"].between(
            requested_start,
            pd.to_datetime(settings.end_date),
        )
    ].copy()

    method_results: dict[str, dict[str, dict[str, object]]] = {
        method: {} for method in PORTFOLIO_METHODS
    }
    factor_weights: dict[str, dict[str, pd.DataFrame]] = {
        method: {} for method in PORTFOLIO_METHODS
    }
    ic_summary_parts = []
    ic_series_parts = []
    ic_annual_parts = []

    for factor_name, approved_parameters in FACTOR_SPECS.items():
        parameters = dict(approved_parameters)
        parameters["signal_min_days_to_maturity"] = 0
        factor_data = calculate_factor(
            factor_name=factor_name,
            start_date=settings.start_date,
            end_date=settings.end_date,
            parameters=parameters,
        )

        for method in PORTFOLIO_METHODS:
            weights = build_inverse_volatility_weights(
                factor_data=factor_data,
                contract_context=contract_context,
                contract_volatility=contract_volatility,
                daily_membership=universe.daily_membership,
                method=method,
                min_assets=settings.min_assets,
            )
            factor_weights[method][factor_name] = weights
            method_results[method][factor_name] = run_factor_method(
                weights=weights,
                prices=prices,
                cost_rate=settings.cost_rate,
            )

        evaluation_weights = factor_weights[
            "full_pool_invvol"
        ][factor_name].copy()
        evaluation_weights["weight_factor"] = evaluation_weights[
            "raw_factor"
        ].where(evaluation_weights["is_eligible"])
        factor_test_panel = build_factor_test_panel(
            start_date=settings.start_date,
            end_date=settings.end_date,
            factor_type="prepared",
            lookback=1,
            rebalance_freq=1,
            min_assets=settings.min_assets,
            prepared_weights=evaluation_weights,
            prepared_calendar=report_calendar,
            prepared_prices=prices,
        )
        if factor_test_panel.empty:
            one_ic_series = pd.DataFrame(
                columns=["signal_date", "asset_count", "ic", "rank_ic"]
            )
            one_ic_summary = pd.DataFrame()
        else:
            one_ic_series = calculate_ic_series(factor_test_panel)
            one_ic_summary = summarize_ic_statistics(
                one_ic_series,
                nw_lags=5,
                annualization_periods=252.0,
            )
        one_ic_series.insert(0, "factor_name", factor_name)
        if not one_ic_summary.empty:
            one_ic_summary.insert(0, "factor_name", factor_name)
        one_ic_annual = calculate_annual_ic(
            one_ic_series.drop(columns="factor_name"),
            factor_name,
        )
        ic_series_parts.append(one_ic_series)
        ic_summary_parts.append(one_ic_summary)
        ic_annual_parts.append(one_ic_annual)

    for method in PORTFOLIO_METHODS:
        factor_returns = {
            factor_name: method_results[method][factor_name]["daily_return"]
            for factor_name in FACTOR_SPECS
        }
        combined_frame = combine_factor_returns(factor_returns)
        combined_return = combined_frame["combined_return"]
        factor_only_results = {
            factor_name: method_results[method][factor_name]
            for factor_name in FACTOR_SPECS
        }
        combined_diagnostics = _combine_daily_diagnostics(
            factor_only_results,
            combined_return,
        )
        combined_metrics = calculate_strategy_metrics(
            "combined",
            combined_return,
            combined_diagnostics,
        )
        method_results[method]["combined"] = {
            "nav": (1.0 + combined_return).cumprod().rename("nav"),
            "metrics": combined_metrics,
            "daily_return": combined_return,
            "daily_diagnostics": combined_diagnostics,
        }

    ic_summary = (
        pd.concat(ic_summary_parts, ignore_index=True)
        if any(not frame.empty for frame in ic_summary_parts)
        else pd.DataFrame()
    )
    ic_series = pd.concat(ic_series_parts, ignore_index=True)
    ic_annual = (
        pd.concat(ic_annual_parts, ignore_index=True)
        if any(not frame.empty for frame in ic_annual_parts)
        else pd.DataFrame()
    )
    return save_strategy_results(
        output_dir=output_dir,
        settings=settings,
        universe=universe,
        method_results=method_results,
        ic_summary=ic_summary,
        ic_series=ic_series,
        ic_annual=ic_annual,
    )


def main(argv: list[str] | None = None) -> Path:
    """Parse CLI arguments, run the strategy, and return its output path."""
    arguments = parse_arguments(argv)
    settings = StrategySettings(
        start_date=arguments.start,
        end_date=arguments.end,
        liquidity_lookback=arguments.liquidity_lookback,
        min_amount_observations=arguments.min_amount_observations,
        pool_size=arguments.pool_size,
        volatility_lookback=arguments.volatility_lookback,
        min_assets=arguments.min_assets,
        trade_min_days_to_maturity=arguments.trade_min_days_to_maturity,
        cost_rate=arguments.cost_rate,
    )
    output_path = run_multi_factor_strategy(
        settings=settings,
        result_dir=arguments.result_dir,
        overwrite=arguments.overwrite,
    )
    print(f"Results saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    main()
