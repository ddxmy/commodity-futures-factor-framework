"""Template for a daily cross-sectional commodity factor plugin.

Copy this file, rename it, and replace the factor-specific loading and
calculation functions. The public ``calculate_factor`` function must return
one row per trading date and commodity with ``trade_date``, ``fut_code``, and
numeric ``raw_factor`` columns. Additional diagnostic columns may be retained.
"""

from collections.abc import Mapping

import numpy as np
import pandas as pd

from src.p01_market_data import validate_min_days_to_maturity


FACTOR_NAME = "factor_template"
FACTOR_VALUE_COLUMN = "factor_value"

DEFAULT_PARAMETERS: dict[str, object] = {
    "lookback": 90,
    "signal_min_days_to_maturity": 0,
}


def compute_factor(
    input_data: pd.DataFrame,
    trade_calendar: pd.DataFrame,
    lookback: int,
) -> pd.DataFrame:
    """Return a full daily panel containing ``FACTOR_VALUE_COLUMN``."""
    raise NotImplementedError(
        "replace compute_factor after copying the factor template"
    )


def load_factor_data(
    start_date: str,
    end_date: str,
    lookback: int,
    signal_min_days_to_maturity: int,
) -> pd.DataFrame:
    """Load buffered inputs and return the computed daily factor panel."""
    raise NotImplementedError(
        "replace load_factor_data after copying the factor template"
    )


def calculate_factor(
    start_date: str,
    end_date: str,
    parameters: Mapping[str, object] | None = None,
) -> pd.DataFrame:
    """Return the standard factor panel for the requested period."""
    settings = dict(DEFAULT_PARAMETERS)
    settings.update(dict(parameters or {}))
    lookback = settings["lookback"]
    signal_min_days_to_maturity = validate_min_days_to_maturity(
        settings["signal_min_days_to_maturity"],
        "signal_min_days_to_maturity",
    )
    if not isinstance(lookback, int) or lookback <= 0:
        raise ValueError("lookback must be a positive integer")

    panel = load_factor_data(
        start_date,
        end_date,
        lookback,
        signal_min_days_to_maturity,
    )
    if FACTOR_VALUE_COLUMN not in panel.columns:
        raise ValueError(
            f"factor panel is missing column: {FACTOR_VALUE_COLUMN}"
        )

    result = panel.copy()
    result["raw_factor"] = result[FACTOR_VALUE_COLUMN]
    requested_start = pd.to_datetime(start_date)
    requested_end = pd.to_datetime(end_date)
    return result.loc[
        result["trade_date"].between(requested_start, requested_end)
    ].reset_index(drop=True)
