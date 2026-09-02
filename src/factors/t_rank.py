"""Robust momentum factor based on standardized daily return ranks.

For each trading day, the signal ranks the close-to-close returns of the
volume-ranked main contracts across commodities and standardizes those ranks
using the theoretical mean and population standard deviation of ``1..N``.
The final factor is the complete rolling mean of the standardized rank scores.

Contract returns are calculated on fixed contracts before the daily main
contract is selected, so a change in the volume-ranked main contract does not
create a synthetic cross-contract return. A signal dated T is executed by the
shared framework no earlier than the open on T+1.
"""

from collections.abc import Mapping

import numpy as np
import pandas as pd

from src.p01_market_data import (
    load_trade_calendar,
    validate_min_days_to_maturity,
)
from src.p02_contract_selection import build_contract_mapping


FACTOR_NAME = "t_rank"

DEFAULT_PARAMETERS: dict[str, object] = {
    "lookback": 10,
    "signal_min_days_to_maturity": 0,
}


def compute_t_rank(
    contract_mapping: pd.DataFrame,
    trade_calendar: pd.DataFrame,
    lookback: int,
) -> pd.DataFrame:
    """Compute standardized daily return ranks and their rolling mean."""
    if (
        isinstance(lookback, bool)
        or not isinstance(lookback, int)
        or lookback <= 0
    ):
        raise ValueError(
            "lookback must be a positive integer"
        )

    required_mapping_columns = {
        "trade_date",
        "fut_code",
        "ts_code_A",
        "daily_return_A",
    }
    missing_mapping_columns = (
        required_mapping_columns
        - set(contract_mapping.columns)
    )
    if missing_mapping_columns:
        raise ValueError(
            "contract_mapping is missing columns: "
            + ", ".join(
                sorted(missing_mapping_columns)
            )
        )

    if "trade_date" not in trade_calendar.columns:
        raise ValueError(
            "trade_calendar is missing column: trade_date"
        )

    mapping = contract_mapping.copy()
    mapping["trade_date"] = pd.to_datetime(
        mapping["trade_date"],
        errors="raise",
    )

    if mapping.duplicated(
        ["trade_date", "fut_code"]
    ).any():
        raise ValueError(
            "contract mapping contains duplicate "
            "date-commodity keys"
        )

    calendar_dates = pd.to_datetime(
        trade_calendar["trade_date"],
        errors="raise",
    )

    if calendar_dates.duplicated().any():
        raise ValueError(
            "trade calendar contains duplicate dates"
        )

    if not calendar_dates.is_monotonic_increasing:
        raise ValueError(
            "trade calendar is not sorted by trade_date"
        )

    products = sorted(
        mapping["fut_code"]
        .dropna()
        .unique()
    )

    panel = pd.MultiIndex.from_product(
        [
            calendar_dates,
            products,
        ],
        names=[
            "trade_date",
            "fut_code",
        ],
    ).to_frame(index=False)

    result = panel.merge(
        mapping,
        on=[
            "trade_date",
            "fut_code",
        ],
        how="left",
        validate="one_to_one",
    )

    result["has_contract_mapping"] = (
        result["ts_code_A"].notna()
    )

    result = result.sort_values(
        [
            "trade_date",
            "fut_code",
        ]
    ).reset_index(drop=True)

    result['participant_count'] = (
        result.groupby(
            'trade_date',
            sort=False,
        )['daily_return_A']
        .transform('count')
    )

    result['return_rank'] = (
        result.groupby(
            'trade_date',
            sort=False
        )['daily_return_A']
        .rank(
            method='average',
            ascending=True,
        )
    )

    result['rank_mean'] = (
        result['participant_count'] + 1
    ) / 2.0

    valid_cross_section = (
        result['participant_count'] >= 2
    )

    rank_variance = (
        (
            result['participant_count'] + 1
        )
        * (
            result['participant_count'] - 1
        )
        / 12.0
    ).where(
        valid_cross_section,
        np.nan,
    )

    result['rank_std'] = np.sqrt(
        rank_variance
    )

    result['rank_score'] = (
        result['return_rank']
        - result['rank_mean']
    ) / result['rank_std']

    result = result.sort_values(
        [
            'fut_code',
            'trade_date',
        ]
    ).reset_index(drop=True)

    result['t_rank'] = (
        result.groupby(
            'fut_code',
            sort=False,
        )['rank_score']
        .transform(
            lambda values: values.rolling(
                window=lookback,
                min_periods=lookback,
            ).mean()
        )
    )

    return result.sort_values(
        [
            'trade_date',
            'fut_code',
        ]
    ).reset_index(drop=True)




def load_t_rank(
    start_date: str,
    end_date: str,
    lookback: int,
    signal_min_days_to_maturity: int,
) -> pd.DataFrame:
    """Load buffered contract history and return the T_Rank panel."""
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
    return compute_t_rank(mapping, calendar, lookback)


def calculate_factor(
    start_date: str,
    end_date: str,
    parameters: Mapping[str, object] | None = None,
) -> pd.DataFrame:
    """Return T_Rank as the standard ``raw_factor`` panel."""
    settings = dict(DEFAULT_PARAMETERS)
    settings.update(dict(parameters or {}))

    lookback = settings["lookback"]
    signal_min_days_to_maturity = validate_min_days_to_maturity(
        settings["signal_min_days_to_maturity"],
        "signal_min_days_to_maturity",
    )
    if (
        isinstance(lookback, bool)
        or not isinstance(lookback, int)
        or lookback <= 0
    ):
        raise ValueError("lookback must be a positive integer")

    panel = load_t_rank(
        start_date,
        end_date,
        lookback,
        signal_min_days_to_maturity,
    )
    if "t_rank" not in panel.columns:
        raise ValueError("factor panel is missing column: t_rank")

    result = panel.copy()
    result["raw_factor"] = result["t_rank"]
    requested_start = pd.to_datetime(start_date)
    requested_end = pd.to_datetime(end_date)
    return result.loc[
        result["trade_date"].between(requested_start, requested_end)
    ].reset_index(drop=True)
