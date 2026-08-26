"""Select liquidity-ranked contracts and build the A/B/C research panel."""

import numpy as np
import pandas as pd

from config.settings import MIN_DAYS_TO_MATURITY
from src.p01_market_data import load_contract_daily


MAPPING_SOURCE_COLUMNS = [
    "trade_date",
    "fut_code",
    "ts_code",
    "vol",
    "oi",
    "amount",
    "close",
    "daily_return",
    "days_to_maturity",
    "avg_vol",
    "avg_oi",
    "avg_amount",
]


def _select_ranked_contract(
    contracts: pd.DataFrame,
    rank: int,
    suffix: str,
) -> pd.DataFrame:
    """Return one liquidity rank with suffixes used by factor formulas."""
    selected = contracts.loc[
        contracts["rank_by_vol"].eq(rank),
        MAPPING_SOURCE_COLUMNS,
    ].copy()
    selected = selected.rename(
        columns={
            "ts_code": f"ts_code_{suffix}",
            "vol": f"vol_{suffix}",
            "oi": f"oi_{suffix}",
            "amount": f"amount_{suffix}",
            "close": f"close_{suffix}",
            "daily_return": f"daily_return_{suffix}",
            "days_to_maturity": f"d_{suffix}",
            "avg_vol": f"avg_vol_{suffix}",
            "avg_oi": f"avg_oi_{suffix}",
            "avg_amount": f"avg_amount_{suffix}",
        }
    )
    return selected


def build_contract_mapping(
    start_date: str,
    end_date: str,
    min_days_to_maturity: int = MIN_DAYS_TO_MATURITY,
) -> pd.DataFrame:
    """Build the daily A/B/C panel ranked by contract trading volume.

    A, B and C denote the first, second and third contracts by liquidity;
    they do not imply a fixed maturity order. Signed maturity differences
    preserve the near-minus-far direction in the factor formula.
    """
    df = load_contract_daily(
        start_date,
        end_date,
        min_days_to_maturity=min_days_to_maturity,
    ).reset_index()
    df = df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    previous_close = df.groupby("ts_code")["close"].shift(1)
    df["daily_return"] = np.log(df["close"] / previous_close)

    top_three = df.loc[df["rank_by_vol"].le(3)].copy()
    result = _select_ranked_contract(top_three, 1, "A")
    result = result.merge(
        _select_ranked_contract(top_three, 2, "B"),
        on=["trade_date", "fut_code"],
        how="left",
        validate="one_to_one",
    )
    result = result.merge(
        _select_ranked_contract(top_three, 3, "C"),
        on=["trade_date", "fut_code"],
        how="left",
        validate="one_to_one",
    )

    result["d_AB"] = result["d_B"] - result["d_A"]
    result["d_BC"] = result["d_C"] - result["d_B"]

    output_columns = [
        "trade_date",
        "fut_code",
        "ts_code_A",
        "ts_code_B",
        "ts_code_C",
        "vol_A",
        "vol_B",
        "vol_C",
        "oi_A",
        "oi_B",
        "oi_C",
        "amount_A",
        "close_A",
        "close_B",
        "close_C",
        "d_A",
        "d_B",
        "d_C",
        "daily_return_A",
        "daily_return_B",
        "daily_return_C",
        "d_AB",
        "d_BC",
        "avg_vol_A",
        "avg_oi_A",
        "avg_amount_A",
    ]
    return result[output_columns]
