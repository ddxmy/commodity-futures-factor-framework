"""Liquidity-ranked main/sub-main carry factor.

The signal annualizes the close-price spread between the first- and
second-ranked contracts by their signed maturity gap, then averages it over
a rolling window. A negative maturity gap remains meaningful when liquidity
ranks do not follow maturity order.
"""

from collections.abc import Mapping

import numpy as np
import pandas as pd

from src.p01_market_data import (
    load_trade_calendar,
    validate_min_days_to_maturity,
)
from src.p02_contract_selection import build_contract_mapping


FACTOR_NAME = "carry"

DEFAULT_PARAMETERS: dict[str, object] = {
    "lookback": 90,
    "signal_min_days_to_maturity": 0,
}


def compute_main_sub_carry(
    contract_mapping: pd.DataFrame,
    trade_calendar: pd.DataFrame,
    lookback: int,
) -> pd.DataFrame:
    """Compute daily main/sub-main carry and its rolling mean."""
    if not isinstance(lookback, int) or lookback <= 0:
        raise ValueError("lookback must be a positive integer")

    required_columns = {
        "trade_date",
        "fut_code",
        "ts_code_A",
        "ts_code_B",
        "close_A",
        "close_B",
        "d_AB",
    }
    missing_columns = required_columns - set(contract_mapping.columns)
    if missing_columns:
        raise ValueError(
            "contract_mapping is missing columns: "
            + ", ".join(sorted(missing_columns))
        )
    if "trade_date" not in trade_calendar.columns:
        raise ValueError("trade_calendar is missing trade_date")

    df = contract_mapping.copy()
    df['trade_date'] = pd.to_datetime(df['trade_date'])

    if df.duplicated(["trade_date", "fut_code"]).any():
        raise ValueError(
            "contract mapping contains duplicate date-commodity keys"
        )

    df = df.sort_values(['fut_code', 'trade_date']).reset_index(drop=True)

    df['d_AB'] = df['d_AB'].replace(0, np.nan)

    df['daily_carry'] = (
        -(df['close_B'] - df['close_A'])
        / df['close_A']
        * 365.0
        / df['d_AB']
    )

    calendar_dates = pd.to_datetime(
        trade_calendar['trade_date']
    )
    products = sorted(
        df['fut_code'].dropna().unique()
    )
    panel = pd.MultiIndex.from_product(
        [calendar_dates, products],
        names=['trade_date', 'fut_code']
    ).to_frame(index=False)

    df = panel.merge(
        df,
        on=['trade_date', 'fut_code'],
        how='left',
        validate='one_to_one',
    )
    df['has_contract_mapping'] = df['ts_code_A'].notna()
    df = df.sort_values(
        ['fut_code', 'trade_date'],
    ).reset_index(drop=True)

    df['main_sub_carry'] = (
        df.groupby('fut_code')['daily_carry']
        .transform(
            lambda x: x.rolling(
                window=lookback,
                min_periods=lookback,
            ).mean()
        )
    )

    return df



def load_main_sub_carry(
    start_date: str,
    end_date: str,
    lookback: int,
    signal_min_days_to_maturity: int,
) -> pd.DataFrame:
    """Load enough history and return the rolling carry panel."""
    requested_start = pd.to_datetime(start_date)
    buffer_days = lookback * 2 + 30
    load_start = (
        requested_start - pd.Timedelta(days=buffer_days)
    ).strftime("%Y%m%d")

    mapping = build_contract_mapping(
        load_start,
        end_date,
        min_days_to_maturity=signal_min_days_to_maturity,
    )
    calendar = load_trade_calendar(load_start, end_date)
    return compute_main_sub_carry(mapping, calendar, lookback)


def calculate_factor(
    start_date: str,
    end_date: str,
    parameters: Mapping[str, object] | None = None,
) -> pd.DataFrame:
    """Return main/sub-main carry as the standard ``raw_factor`` panel."""
    settings = dict(DEFAULT_PARAMETERS)
    settings.update(dict(parameters or {}))
    lookback = settings["lookback"]
    signal_min_days_to_maturity = validate_min_days_to_maturity(
        settings["signal_min_days_to_maturity"],
        "signal_min_days_to_maturity",
    )
    if not isinstance(lookback, int) or lookback <= 0:
        raise ValueError("lookback must be a positive integer")

    df = load_main_sub_carry(
        start_date,
        end_date,
        lookback,
        signal_min_days_to_maturity,
    )
    result = df.copy()
    result["raw_factor"] = df["main_sub_carry"]
    requested_start = pd.to_datetime(start_date)
    requested_end = pd.to_datetime(end_date)
    return result.loc[
        (result["trade_date"] >= requested_start)
        & (result["trade_date"] <= requested_end)
    ].reset_index(drop=True)
