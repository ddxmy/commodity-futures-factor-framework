"""Spot-to-main annualized basis factor using futures close prices.

The signal compares the volume-ranked main futures contract with the
commodity spot price, annualizes the basis by the main contract's calendar
days to maturity, and averages it over a complete rolling window. A missing
spot quote may be carried forward for one trading day while its original
observation date remains available for audit.
"""

from collections.abc import Mapping

import numpy as np
import pandas as pd

from src.p01_market_data import (
    load_spot_daily,
    load_trade_calendar,
    validate_min_days_to_maturity,
)
from src.p02_contract_selection import build_contract_mapping


FACTOR_NAME = "spotmain"

DEFAULT_PARAMETERS: dict[str, object] = {
    "lookback": 90,
    "signal_min_days_to_maturity": 0,
}


def compute_spot_main(
    contract_mapping: pd.DataFrame,
    spot_prices: pd.DataFrame,
    trade_calendar: pd.DataFrame,
    lookback: int,
) -> pd.DataFrame:
    """Compute daily spot-main carry and its complete rolling mean."""
    if not isinstance(lookback, int) or lookback <= 0:
        raise ValueError(
            "lookback must be a positive integer"
        )

    required_contract_columns = {
        "trade_date",
        "fut_code",
        "ts_code_A",
        "close_A",
        "d_A",
    }
    missing_contract_columns = (
        required_contract_columns
        - set(contract_mapping.columns)
    )
    if missing_contract_columns:
        raise ValueError(
            "contract_mapping is missing columns: "
            + ", ".join(
                sorted(missing_contract_columns)
            )
        )

    required_spot_columns = {
        "trade_date",
        "fut_code",
        "spot_price",
    }
    missing_spot_columns = (
        required_spot_columns
        - set(spot_prices.columns)
    )
    if missing_spot_columns:
        raise ValueError(
            "spot_prices is missing columns: "
            + ", ".join(
                sorted(missing_spot_columns)
            )
        )

    if "trade_date" not in trade_calendar.columns:
        raise ValueError(
            "trade_calendar is missing trade_date"
        )

    mapping = contract_mapping.copy()
    mapping["trade_date"] = pd.to_datetime(
        mapping["trade_date"]
    )

    if mapping.duplicated(
        ["trade_date", "fut_code"]
    ).any():
        raise ValueError(
            "contract mapping contains duplicate "
            "date-commodity keys"
        )

    spots = spot_prices.copy()
    spots["trade_date"] = pd.to_datetime(
        spots["trade_date"]
    )

    if spots.duplicated(
        ["trade_date", "fut_code"]
    ).any():
        raise ValueError(
            "spot prices contain duplicate "
            "date-commodity keys"
        )

    if "source" not in spots.columns:
        spots["source"] = pd.NA

    spots['_has_spot_row'] = True

    valid_raw_spot = spots['spot_price'].gt(0)
    spots['spot_observation_date'] = (
        spots['trade_date'].where(
            valid_raw_spot
        )
    )
    spots.loc[
        ~valid_raw_spot,
        'spot_price',
    ] = np.nan

    calendar_dates = (
        pd.to_datetime(
            trade_calendar['trade_date']
        )
        .drop_duplicates()
        .sort_values()
    )
    products = sorted(
        mapping['fut_code']
        .dropna()
        .unique()
    )

    panel = pd.MultiIndex.from_product(
        [calendar_dates, products],
        names=['trade_date', 'fut_code'],
    ).to_frame(
        index=False
    )

    result = panel.merge(
        mapping,
        on=[
            'trade_date',
            'fut_code',
        ],
        how='left',
        validate='one_to_one',
    )

    result = result.merge(
        spots,
        on=[
            'trade_date',
            'fut_code',
        ],
        how='left',
        validate='one_to_one',
    )

    result['has_contract_mapping'] = (
        result['ts_code_A'].notna()
    )
    result = result.sort_values(
        [
            'fut_code',
            'trade_date'
        ]
    ).reset_index(drop=True)

    missing_spot_row = (
        result['_has_spot_row'].isna()
    )

    fill_columns = [
        'spot_price',
        'spot_observation_date',
        'source',
    ]

    for column in fill_columns:
        one_day_carried = (
            result.groupby('fut_code')[column]
            .ffill(limit=1)
        )
        result.loc[
            missing_spot_row,
            column,
        ] = one_day_carried.loc[
            missing_spot_row
        ]

    mask = (
        result['close_A'].gt(0)
        & result['spot_price'].gt(0)
        & result['d_A'].gt(0)
    )

    result['daily_spotmain'] = np.nan
    result.loc[mask, 'daily_spotmain'] = (
        -(result['close_A'] - result['spot_price'])
        / result['spot_price']
        * 365
        / result['d_A']
    )

    result['spotmain'] = (
        result.groupby('fut_code')['daily_spotmain']
        .transform(
            lambda x: x.rolling(
                window=lookback,
                min_periods=lookback,
            ).mean()
        )
    )

    result = result.drop(
        columns=['_has_spot_row']
    )

    return result


def load_spot_main(
    start_date: str,
    end_date: str,
    lookback: int,
    signal_min_days_to_maturity: int,
) -> pd.DataFrame:
    """Load buffered futures, spot, and calendar data for SpotMain."""
    requested_start = pd.to_datetime(
        start_date
    )
    buffer_days = lookback * 2 + 30
    load_start = (
        requested_start
        - pd.Timedelta(days=buffer_days)
    ).strftime("%Y%m%d")

    mapping = build_contract_mapping(
        load_start,
        end_date,
        min_days_to_maturity=signal_min_days_to_maturity,
    )
    spots = load_spot_daily(
        load_start,
        end_date,
    )
    calendar = load_trade_calendar(
        load_start,
        end_date,
    )

    return compute_spot_main(
        mapping,
        spots,
        calendar,
        lookback,
    )


def calculate_factor(
    start_date: str,
    end_date: str,
    parameters: Mapping[str, object] | None = None,
) -> pd.DataFrame:
    """Return SpotMain as the standard ``raw_factor`` panel."""
    settings = dict(DEFAULT_PARAMETERS)
    settings.update(
        dict(parameters or {})
    )

    lookback = settings["lookback"]
    signal_min_days_to_maturity = validate_min_days_to_maturity(
        settings["signal_min_days_to_maturity"],
        "signal_min_days_to_maturity",
    )
    if not isinstance(lookback, int) or lookback <= 0:
        raise ValueError(
            "lookback must be a positive integer"
        )

    panel = load_spot_main(
        start_date,
        end_date,
        lookback,
        signal_min_days_to_maturity,
    )

    result = panel.copy()
    result["raw_factor"] = result["spotmain"]

    requested_start = pd.to_datetime(
        start_date
    )
    requested_end = pd.to_datetime(
        end_date
    )

    return result.loc[
        result["trade_date"].between(
            requested_start,
            requested_end,
        )
    ].reset_index(drop=True)
