"""Validation helpers for the standard factor-data contract."""

from collections.abc import Iterable

import numpy as np
import pandas as pd


FACTOR_COLUMNS = ["trade_date", "fut_code", "raw_factor"]


def validate_factor_data(
    factor_data: pd.DataFrame,
    trade_calendar: pd.DataFrame | Iterable | None = None,
) -> pd.DataFrame:
    """Validate and normalize a factor plugin's output.

    Missing factor values are allowed because they represent unavailable
    observations. Infinite values and duplicate date-commodity keys are not.
    """
    if not isinstance(factor_data, pd.DataFrame):
        raise TypeError("factor_data must be a pandas DataFrame")

    missing_columns = set(FACTOR_COLUMNS) - set(factor_data.columns)
    if missing_columns:
        raise ValueError(
            "factor_data is missing columns: "
            + ", ".join(sorted(missing_columns))
        )

    out = factor_data.copy()
    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="raise")

    if out[["trade_date", "fut_code"]].isna().any().any():
        raise ValueError("trade_date and fut_code cannot contain missing values")
    if out.duplicated(["trade_date", "fut_code"]).any():
        raise ValueError("factor_data contains duplicate date-commodity keys")
    if not pd.api.types.is_numeric_dtype(out["raw_factor"]):
        raise TypeError("raw_factor must have a numeric dtype")

    finite_values = out["raw_factor"].dropna().to_numpy(dtype=float)
    if not np.isfinite(finite_values).all():
        raise ValueError("raw_factor cannot contain infinite values")

    if "available_date" in out.columns:
        out["available_date"] = pd.to_datetime(
            out["available_date"], errors="raise"
        )
        uses_future_data = (
            out["available_date"].notna()
            & out["available_date"].gt(out["trade_date"])
        )
        if uses_future_data.any():
            raise ValueError("available_date cannot be later than trade_date")

    if trade_calendar is not None:
        if isinstance(trade_calendar, pd.DataFrame):
            if "trade_date" not in trade_calendar.columns:
                raise ValueError("trade_calendar is missing trade_date")
            valid_dates = pd.to_datetime(trade_calendar["trade_date"])
        else:
            valid_dates = pd.to_datetime(list(trade_calendar))
        invalid_dates = ~out["trade_date"].isin(pd.DatetimeIndex(valid_dates))
        if invalid_dates.any():
            raise ValueError("factor_data contains dates outside the trade calendar")

    return out.sort_values(["trade_date", "fut_code"]).reset_index(drop=True)

