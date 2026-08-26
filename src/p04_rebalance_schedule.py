"""Rebalance-date selection for the shared research pipeline."""

from collections.abc import Iterable

import pandas as pd


WEEKDAY_NAMES = {"MON", "TUE", "WED", "THU", "FRI"}


def get_rebalance_dates(
    trade_dates: Iterable,
    freq: int | str,
) -> set[pd.Timestamp]:
    """Return first-day-anchored integer or week-ending rebalance dates."""
    dates = pd.DatetimeIndex(
        pd.to_datetime(sorted(trade_dates))
    ).drop_duplicates()
    if dates.empty:
        return set()

    if isinstance(freq, int) and not isinstance(freq, bool):
        if freq <= 0:
            raise ValueError("freq must be a positive integer")
        return set(dates[::freq])

    if isinstance(freq, str):
        normalized = freq.strip().upper()
        if normalized.startswith("W-"):
            weekday = normalized[2:]
            if weekday not in WEEKDAY_NAMES:
                raise ValueError(
                    "weekly freq must use MON, TUE, WED, THU or FRI"
                )
            periods = dates.to_period(normalized)
            selected = pd.Series(dates, index=periods).groupby(level=0).max()
            return set(pd.DatetimeIndex(selected))

    raise ValueError("freq must be a positive integer or a value such as W-FRI")
