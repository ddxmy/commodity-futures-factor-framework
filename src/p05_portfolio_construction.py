"""Convert a standard factor panel into market-neutral target weights."""

import numpy as np
import pandas as pd

from config.research_config import ResearchConfig
from config.settings import LOOKBACK
from src.factor_loader import calculate_factor
from src.p03_factor_processing import validate_factor_data
from src.p04_rebalance_schedule import get_rebalance_dates


CONTRACT_CONTEXT_COLUMNS = [
    "trade_date",
    "fut_code",
    "ts_code_A",
    "avg_vol_A",
    "avg_oi_A",
    "avg_amount_A",
]


def build_target_weights(
    factor_data: pd.DataFrame,
    contract_data: pd.DataFrame,
    config: ResearchConfig,
    normalize: str = "rank",
    zscore_clip: float | None = None,
) -> pd.DataFrame:
    """Apply liquidity filters, normalize one factor and form long-short weights."""
    if normalize not in {"rank", "zscore"}:
        raise ValueError("normalize must be rank or zscore")
    if zscore_clip is not None and zscore_clip <= 0:
        raise ValueError("zscore_clip must be positive or None")

    factors = validate_factor_data(factor_data)
    duplicated_context_columns = [
        column
        for column in CONTRACT_CONTEXT_COLUMNS
        if column not in {"trade_date", "fut_code"} and column in factors.columns
    ]
    factors = factors.drop(columns=duplicated_context_columns)
    missing_context = set(CONTRACT_CONTEXT_COLUMNS) - set(contract_data.columns)
    if missing_context:
        raise ValueError(
            "contract_data is missing columns: "
            + ", ".join(sorted(missing_context))
        )

    context = contract_data[CONTRACT_CONTEXT_COLUMNS].copy()
    context["trade_date"] = pd.to_datetime(context["trade_date"])
    if context.duplicated(["trade_date", "fut_code"]).any():
        raise ValueError("contract_data contains duplicate date-commodity keys")

    df = factors.merge(
        context,
        on=["trade_date", "fut_code"],
        how="left",
        validate="one_to_one",
    )
    df = df.sort_values(["fut_code", "trade_date"])

    rebalance_dates = get_rebalance_dates(
        sorted(df["trade_date"].unique()),
        config.rebalance_freq,
    )
    df["is_rebalance"] = df["trade_date"].isin(rebalance_dates)
    df["liquidity_ratio_A"] = df["avg_vol_A"] / df["avg_oi_A"]

    liquidity_history = df[
        ["avg_vol_A", "avg_oi_A", "avg_amount_A"]
    ].notna().all(axis=1)
    df["passes_liquidity"] = (
        liquidity_history
        & df["avg_vol_A"].ge(config.min_vol)
        & df["avg_oi_A"].ge(config.min_oi)
        & df["avg_amount_A"].ge(config.min_amount)
        & df["liquidity_ratio_A"].ge(config.liquidity_min)
    )

    df["weight_factor"] = df["raw_factor"]
    excluded = df["is_rebalance"] & ~df["passes_liquidity"]
    df.loc[excluded, "weight_factor"] = np.nan

    df["daily_count"] = df.groupby("trade_date")["weight_factor"].transform(
        "count"
    )
    df.loc[df["daily_count"].lt(config.min_assets), "weight_factor"] = np.nan

    if normalize == "rank":
        df["factor_rank"] = df.groupby("trade_date")["weight_factor"].rank(
            method="average",
            ascending=True,
        )
        df["factor"] = (
            2.0
            * (df["factor_rank"] - 1.0)
            / (df["daily_count"] - 1.0)
            - 1.0
        )
    else:
        df["factor"] = df.groupby("trade_date")["weight_factor"].transform(
            lambda values: (values - values.mean()) / values.std()
        )
        if zscore_clip is not None:
            df["factor"] = df["factor"].clip(-zscore_clip, zscore_clip)

    df["weight"] = 0.0
    for _, group in df.groupby("trade_date"):
        long_mask = group["factor"].gt(0)
        short_mask = group["factor"].lt(0)
        long_sum = group.loc[long_mask, "factor"].sum()
        short_sum = group.loc[short_mask, "factor"].abs().sum()

        if long_sum > 0:
            long_index = group.index[long_mask]
            df.loc[long_index, "weight"] = (
                0.5 * group.loc[long_index, "factor"] / long_sum
            )
        if short_sum > 0:
            short_index = group.index[short_mask]
            df.loc[short_index, "weight"] = (
                0.5 * group.loc[short_index, "factor"] / short_sum
            )

    output_columns = [
        "trade_date",
        "fut_code",
        "raw_factor",
        "weight_factor",
        "factor",
        "weight",
        "ts_code_A",
        "avg_vol_A",
        "avg_oi_A",
        "avg_amount_A",
        "liquidity_ratio_A",
        "passes_liquidity",
        "is_rebalance",
    ]
    return df[output_columns]


def generate_weights(
    start_date: str,
    end_date: str,
    factor_type: str = "AB",
    lookback: int = LOOKBACK,
    normalize: str = "rank",
    rebalance_freq: int | str = 5,
    zscore_clip: float | None = None,
) -> pd.DataFrame:
    """Compatibility wrapper for the original basis-momentum API."""
    factor_data = calculate_factor(
        factor_name="basis_momentum",
        start_date=start_date,
        end_date=end_date,
        parameters={"variant": factor_type, "lookback": lookback},
    )
    contract_data = factor_data[CONTRACT_CONTEXT_COLUMNS]
    config = ResearchConfig(rebalance_freq=rebalance_freq)
    return build_target_weights(
        factor_data=factor_data,
        contract_data=contract_data,
        config=config,
        normalize=normalize,
        zscore_clip=zscore_clip,
    )
