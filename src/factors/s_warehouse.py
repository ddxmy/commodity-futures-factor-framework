"""Warehouse-receipt change factor.

The signal measures the percentage change in a commodity's smoothed
warehouse-receipt level relative to an exact market-trading-day lag.
The sign is reversed so that a faster inventory decline produces a
larger factor value.

Warehouse data dated T is used to form the signal after the close on T.
Portfolio execution remains subject to the framework's next-day trading
and contract-maturity rules.
"""

from collections.abc import Mapping

import numpy as np
import pandas as pd

from src.p01_market_data import (
    load_trade_calendar,
    load_warehouse_daily,
)


FACTOR_NAME = "s_warehouse"

DEFAULT_PARAMETERS: dict[str, object] = {
    "lookback": 90,
    "smooth_window": 20,
    "min_observations": 18,
}


def compute_s_warehouse(
    warehouse_daily: pd.DataFrame,
    trade_calendar: pd.DataFrame,
    lookback: int,
    smooth_window: int,
    min_observations: int,
) -> pd.DataFrame:
    """Compute the smoothed warehouse-receipt change factor."""

    parameters = {
        "lookback": lookback,
        "smooth_window": smooth_window,
        "min_observations": min_observations,
    }

    for name, value in parameters.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
        ):
            raise ValueError(
                f'{name} must be a positive integer'
            )

    if min_observations > smooth_window:
        raise ValueError(
            'min_observations cannot exceed smooth_window'
        )

    required_warehouse_columns = {
        'trade_date',
        'fut_code',
        'warehouse_total',
        'quality_status',
    }

    missing_warehouse_columns = (
        required_warehouse_columns
        - set(warehouse_daily.columns)
    )
    if missing_warehouse_columns:
        raise ValueError(
            'warehouse_daily is missing columns: '
            + ', '.join(
                sorted(missing_warehouse_columns)
            )
        )

    required_trade_calendar_columns = {
        'trade_date',
    }

    missing_trade_calendar_columns = (
        required_trade_calendar_columns
        - set(trade_calendar.columns)
    )
    if missing_trade_calendar_columns:
        raise ValueError(
            'trade_calendar is missing columns: '
            + ', '.join(
                sorted(missing_trade_calendar_columns)
            )
        )

    warehouse = warehouse_daily.copy()
    warehouse['trade_date'] = pd.to_datetime(
        warehouse['trade_date'],
        errors='raise',
    )

    if warehouse.duplicated(
        ['trade_date', 'fut_code']
    ).any():
        raise ValueError(
            'warehouse_daily contains duplicate '
            'date-commodity keys'
        )

    try:
        warehouse['warehouse_total'] = pd.to_numeric(
            warehouse['warehouse_total'],
            errors='raise',
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            'warehouse_total must be numeric'
        ) from error

    if np.isinf(
        warehouse['warehouse_total'].dropna()
    ).any():
        raise ValueError(
            'warehouse_total must be numeric and finite'
        )

    calendar_dates = pd.to_datetime(
        trade_calendar['trade_date'],
        errors='raise',
    )

    if calendar_dates.duplicated().any():
        raise ValueError(
            'trade_calendar contains duplicate trade dates'
        )

    if not calendar_dates.is_monotonic_increasing:
        raise ValueError(
            'trade_calendar is not sorted by trade_date'
        )

    products = sorted(
        warehouse['fut_code']
        .dropna()
        .unique()
    )

    panel = pd.MultiIndex.from_product(
        [
            calendar_dates,
            products,
        ],
        names=[
            'trade_date',
            'fut_code',
        ],
    ).to_frame(index=False)

    result = panel.merge(
        warehouse,
        on=[
            'trade_date',
            'fut_code',
        ],
        how='left',
        validate='one_to_one',
    )

    result['has_warehouse_observation'] = (
        result['quality_status'].notna()
    )

    valid_warehouse = (
        result['quality_status'].eq('valid')
        & result['warehouse_total'].gt(0)
    )

    result['warehouse_value'] = (
        result['warehouse_total'].where(
            valid_warehouse,
        )
    )

    result = result.sort_values(
        [
            'fut_code',
            'trade_date',
        ]
    ).reset_index(
        drop=True
    )

    result['valid_observations'] = (
        result.groupby(
            'fut_code',
            sort=False,
        )['warehouse_value']
        .transform(
            lambda values: values.rolling(
                window=smooth_window,
                min_periods=1,
            ).count()
        )
    )

    result['smoothed_warehouse'] = (
        result.groupby(
            'fut_code',
            sort=False,
        )['warehouse_value']
        .transform(
            lambda values: values.rolling(
                window=smooth_window,
                min_periods=min_observations,
            ).mean()
        )
    )

    result['lagged_smoothed_warehouse'] = (
        result.groupby(
            'fut_code',
            sort=False,
        )['smoothed_warehouse']
        .shift(lookback)
    )

    result['lagged_valid_observations'] = (
        result.groupby(
            'fut_code',
            sort=False,
        )['valid_observations']
        .shift(lookback)
    )

    valid_comparison = (
        result['smoothed_warehouse'].gt(0)
        & result['lagged_smoothed_warehouse'].gt(0)
    )

    result['warehouse_change'] = np.nan

    result.loc[
        valid_comparison,
        'warehouse_change',
    ] = (
        result.loc[
            valid_comparison,
            'smoothed_warehouse',
        ]
        / result.loc[
            valid_comparison,
            'lagged_smoothed_warehouse',
        ]
        - 1.0
    )

    result['s_warehouse'] = (
        -result['warehouse_change']
    )

    return result


def load_s_warehouse(
    start_date: str,
    end_date: str,
    lookback: int,
    smooth_window: int,
    min_observations: int,
) -> pd.DataFrame:
    """Load buffered warehouse history and compute the daily panel."""
    requested_start = pd.to_datetime(
        start_date,
        errors="raise",
    )
    buffer_days = (
        lookback + smooth_window
    ) * 2 + 30
    load_start = (
        requested_start
        - pd.Timedelta(days=buffer_days)
    ).strftime("%Y%m%d")

    warehouse = load_warehouse_daily(
        load_start,
        end_date,
    )
    calendar = load_trade_calendar(
        load_start,
        end_date,
    )

    return compute_s_warehouse(
        warehouse,
        calendar,
        lookback,
        smooth_window,
        min_observations,
    )


def calculate_factor(
    start_date: str,
    end_date: str,
    parameters: Mapping[str, object] | None = None,
) -> pd.DataFrame:
    """Return S_Warehouse as the standard ``raw_factor`` panel."""
    settings = dict(DEFAULT_PARAMETERS)
    settings.update(
        dict(parameters or {})
    )

    lookback = settings["lookback"]
    smooth_window = settings["smooth_window"]
    min_observations = settings["min_observations"]

    window_parameters = {
        "lookback": lookback,
        "smooth_window": smooth_window,
        "min_observations": min_observations,
    }
    for name, value in window_parameters.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
        ):
            raise ValueError(
                f"{name} must be a positive integer"
            )

    if min_observations > smooth_window:
        raise ValueError(
            "min_observations cannot exceed smooth_window"
        )

    panel = load_s_warehouse(
        start_date,
        end_date,
        lookback,
        smooth_window,
        min_observations,
    )
    if "s_warehouse" not in panel.columns:
        raise ValueError(
            "factor panel is missing column: s_warehouse"
        )

    result = panel.copy()
    result["raw_factor"] = result["s_warehouse"]

    requested_start = pd.to_datetime(
        start_date,
        errors="raise",
    )
    requested_end = pd.to_datetime(
        end_date,
        errors="raise",
    )
    return result.loc[
        result["trade_date"].between(
            requested_start,
            requested_end,
        )
    ].reset_index(drop=True)
