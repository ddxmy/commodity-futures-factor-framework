"""Minimal template for a new factor plugin.

Copy this file, rename it, and replace only the data loading and factor formula.
The public ``calculate_factor`` function must return one row per date and
commodity with the columns ``trade_date``, ``fut_code`` and ``raw_factor``.
"""

from collections.abc import Mapping

import pandas as pd


FACTOR_NAME = "factor_template"
DEFAULT_PARAMETERS: dict[str, object] = {}


def calculate_factor(
    start_date: str,
    end_date: str,
    parameters: Mapping[str, object] | None = None,
) -> pd.DataFrame:
    """Return the standard factor panel for the requested period."""
    raise NotImplementedError(
        "Copy _factor_template.py and implement calculate_factor in the new module"
    )

