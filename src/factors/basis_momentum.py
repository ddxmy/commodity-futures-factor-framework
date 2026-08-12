"""Liquidity-ranked basis momentum factor.

The signal annualizes the return difference between adjacent contracts by
their signed maturity gap. A negative gap is meaningful: it automatically
keeps the near-minus-far direction when liquidity ranks change maturity order.
"""

from collections.abc import Mapping

import numpy as np
import pandas as pd

from config.settings import LOOKBACK
from src.p01_market_data import load_trade_calendar
from src.p02_contract_selection import build_contract_mapping


FACTOR_NAME = "basis_momentum"
DEFAULT_PARAMETERS: dict[str, object] = {
    "variant": "AB",
    "lookback": LOOKBACK,
}
VALID_VARIANTS = {"AB", "BC", "BLEND"}


def compute_basis_components(
    contract_mapping: pd.DataFrame,
    trade_calendar: pd.DataFrame,
    lookback: int,
) -> pd.DataFrame:
    """Compute rolling AB and BC signals on a complete trade-date panel."""
    if not isinstance(lookback, int) or lookback <= 0:
        raise ValueError("lookback must be a positive integer")

    required_columns = {
        "trade_date",
        "fut_code",
        "ts_code_A",
        "daily_return_A",
        "daily_return_B",
        "daily_return_C",
        "d_AB",
        "d_BC",
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
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.sort_values(["fut_code", "trade_date"])
    if df.duplicated(["trade_date", "fut_code"]).any():
        raise ValueError("contract mapping contains duplicate date-commodity keys")

    df["d_AB"] = df["d_AB"].replace(0, np.nan)
    df["d_BC"] = df["d_BC"].replace(0, np.nan)
    df["daily_factor_AB"] = (
        (df["daily_return_A"] - df["daily_return_B"]) / df["d_AB"] * 365
    )
    df["daily_factor_BC"] = (
        (df["daily_return_B"] - df["daily_return_C"]) / df["d_BC"] * 365
    )

    calendar_dates = pd.to_datetime(trade_calendar["trade_date"])
    products = sorted(df["fut_code"].dropna().unique())
    panel = pd.MultiIndex.from_product(
        [calendar_dates, products],
        names=["trade_date", "fut_code"],
    ).to_frame(index=False)
    df = panel.merge(
        df,
        on=["trade_date", "fut_code"],
        how="left",
        validate="one_to_one",
    )
    df["has_contract_mapping"] = df["ts_code_A"].notna()
    df = df.sort_values(["fut_code", "trade_date"]).reset_index(drop=True)

    for component in ["AB", "BC"]:
        daily_column = f"daily_factor_{component}"
        factor_column = f"factor_{component}"
        df[factor_column] = df.groupby("fut_code")[daily_column].transform(
            lambda values: values.rolling(
                window=lookback,
                min_periods=lookback,
            ).mean()
        )
    return df


def load_basis_components(
    start_date: str,
    end_date: str,
    lookback: int,
) -> pd.DataFrame:
    """Load enough history and return both rolling basis components."""
    requested_start = pd.to_datetime(start_date)
    buffer_days = lookback * 2 + 30
    load_start = (
        requested_start - pd.Timedelta(days=buffer_days)
    ).strftime("%Y%m%d")

    mapping = build_contract_mapping(load_start, end_date)
    calendar = load_trade_calendar(load_start, end_date)
    return compute_basis_components(mapping, calendar, lookback)


def calculate_factor(
    start_date: str,
    end_date: str,
    parameters: Mapping[str, object] | None = None,
) -> pd.DataFrame:
    """Return one selected basis-momentum variant as ``raw_factor``."""
    settings = dict(DEFAULT_PARAMETERS)
    settings.update(dict(parameters or {}))
    variant = str(settings["variant"]).upper()
    lookback = settings["lookback"]
    if variant not in VALID_VARIANTS:
        raise ValueError("variant must be AB, BC or BLEND")
    if not isinstance(lookback, int) or lookback <= 0:
        raise ValueError("lookback must be a positive integer")

    df = load_basis_components(start_date, end_date, lookback)
    if variant == "AB":
        raw_factor = df["factor_AB"]
    elif variant == "BC":
        raw_factor = df["factor_BC"]
    else:
        raw_factor = (df["factor_AB"] + df["factor_BC"]) / 2.0

    result = df.copy()
    result["raw_factor"] = raw_factor
    requested_start = pd.to_datetime(start_date)
    return result.loc[result["trade_date"] >= requested_start].reset_index(drop=True)
