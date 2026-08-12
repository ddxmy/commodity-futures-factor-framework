"""Align low-frequency factor observations to an exchange trade calendar."""

import numpy as np
import pandas as pd


REQUIRED_SOURCE_COLUMNS = {"fut_code", "available_date", "raw_factor"}


def align_factor_to_calendar(
    data: pd.DataFrame,
    trade_calendar: pd.DataFrame,
    publication_lag: int = 1,
    max_staleness: int | None = None,
) -> pd.DataFrame:
    """Map each commodity's latest available observation to trading dates.

    ``publication_lag=1`` means that a value becomes usable on the first
    trading day strictly after its publication date. Values are carried
    forward independently by commodity and never matched from the future.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    if not isinstance(trade_calendar, pd.DataFrame):
        raise TypeError("trade_calendar must be a pandas DataFrame")
    if not isinstance(publication_lag, int) or publication_lag < 0:
        raise ValueError("publication_lag must be a nonnegative integer")
    if max_staleness is not None and (
        not isinstance(max_staleness, int) or max_staleness < 0
    ):
        raise ValueError("max_staleness must be a nonnegative integer or None")

    missing_source = REQUIRED_SOURCE_COLUMNS - set(data.columns)
    if missing_source:
        raise ValueError(
            "data is missing columns: " + ", ".join(sorted(missing_source))
        )
    if "trade_date" not in trade_calendar.columns:
        raise ValueError("trade_calendar is missing trade_date")

    source = data.copy()
    source["available_date"] = pd.to_datetime(
        source["available_date"], errors="raise"
    )
    if source[["fut_code", "available_date"]].isna().any().any():
        raise ValueError("fut_code and available_date cannot be missing")
    if source.duplicated(["fut_code", "available_date"]).any():
        raise ValueError("duplicate publication dates exist for one commodity")

    calendar = pd.DatetimeIndex(
        pd.to_datetime(trade_calendar["trade_date"], errors="raise")
    ).drop_duplicates().sort_values()
    if calendar.empty:
        raise ValueError("trade_calendar cannot be empty")

    calendar_values = calendar.to_numpy(dtype="datetime64[ns]")
    available_values = source["available_date"].to_numpy(dtype="datetime64[ns]")
    if publication_lag == 0:
        effective_positions = np.searchsorted(
            calendar_values, available_values, side="left"
        )
    else:
        effective_positions = (
            np.searchsorted(calendar_values, available_values, side="right")
            + publication_lag
            - 1
        )

    source["_effective_position"] = effective_positions
    source = source[source["_effective_position"] < len(calendar)].copy()
    source["effective_date"] = calendar.take(
        source["_effective_position"].astype(int)
    ).to_numpy()

    aligned_parts = []
    for fut_code, commodity_data in source.groupby("fut_code", sort=True):
        base = pd.DataFrame({"trade_date": calendar})
        base["fut_code"] = fut_code
        base["_trade_position"] = np.arange(len(calendar))

        observations = commodity_data.sort_values("effective_date").copy()
        observations = observations.rename(
            columns={"_effective_position": "_source_position"}
        )
        aligned = pd.merge_asof(
            base.sort_values("trade_date"),
            observations.drop(columns="fut_code"),
            left_on="trade_date",
            right_on="effective_date",
            direction="backward",
            allow_exact_matches=True,
        )
        aligned["staleness_days"] = (
            aligned["_trade_position"] - aligned["_source_position"]
        )

        if max_staleness is not None:
            stale = aligned["staleness_days"].gt(max_staleness)
            aligned.loc[stale, "raw_factor"] = np.nan

        aligned_parts.append(aligned)

    if not aligned_parts:
        columns = ["trade_date", "fut_code", "raw_factor", "available_date"]
        return pd.DataFrame(columns=columns)

    result = pd.concat(aligned_parts, ignore_index=True)
    uses_future_data = (
        result["available_date"].notna()
        & result["available_date"].gt(result["trade_date"])
    )
    if uses_future_data.any():
        raise ValueError("aligned data contains future observations")

    return (
        result.drop(columns=["_trade_position", "_source_position"])
        .sort_values(["trade_date", "fut_code"])
        .reset_index(drop=True)
    )
