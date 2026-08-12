"""Rebalance-date selection for the shared research pipeline."""

from collections.abc import Iterable

import pandas as pd


def get_rebalance_dates(
    trade_dates: Iterable,
    freq: int | str,
) -> set[pd.Timestamp]:
    """Return rebalance dates selected by trading-day interval or weekday."""
    dates = pd.Series(pd.to_datetime(sorted(trade_dates)))
    if dates.empty:
        return set()

    if isinstance(freq, int):
        if freq <= 0:
            raise ValueError("freq must be a positive integer")
        return set(dates.iloc[::freq])

    if isinstance(freq, str) and freq.startswith("W-"):
        weekday_map = {"MON": 0, "TUE": 1, "WED": 2, "THU": 3, "FRI": 4}
        weekday = freq[2:].upper()
        if weekday not in weekday_map:
            raise ValueError("weekly freq must use MON, TUE, WED, THU or FRI")
        target = weekday_map[weekday]
        date_index = pd.DatetimeIndex(dates)
        mask = date_index.dayofweek == target
        return set(date_index[mask])

    raise ValueError("freq must be a positive integer or a value such as W-FRI")
