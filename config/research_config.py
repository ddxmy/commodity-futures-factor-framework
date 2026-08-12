"""Shared parameters for cross-sectional futures factor research."""

from dataclasses import dataclass

from config.settings import (
    COST_RATE,
    LIQUIDITY_MIN,
    MIN_AMOUNT,
    MIN_DAYS_TO_MATURITY,
    MIN_OI,
    MIN_VOL,
)


@dataclass(frozen=True)
class ResearchConfig:
    """Parameters shared by portfolio construction, backtesting and reports."""

    rebalance_freq: int | str = 1
    cost_rate: float = COST_RATE
    min_assets: int = 10
    group_count: int = 5
    rolling_ic_window: int = 20
    nw_lags: int = 5
    min_vol: float = MIN_VOL
    min_oi: float = MIN_OI
    min_amount: float = MIN_AMOUNT
    liquidity_min: float = LIQUIDITY_MIN
    min_days_to_maturity: int = MIN_DAYS_TO_MATURITY
    turnover_limit: float = 0.15
    max_abs_weight: float = 0.05

    def __post_init__(self) -> None:
        """Reject invalid settings before any database work begins."""
        valid_integer_frequency = (
            isinstance(self.rebalance_freq, int) and self.rebalance_freq > 0
        )
        valid_weekly_frequency = (
            isinstance(self.rebalance_freq, str)
            and self.rebalance_freq.upper()
            in {"W-MON", "W-TUE", "W-WED", "W-THU", "W-FRI"}
        )
        if not (valid_integer_frequency or valid_weekly_frequency):
            raise ValueError(
                "rebalance_freq must be a positive integer or weekday frequency"
            )
        if self.cost_rate < 0:
            raise ValueError("cost_rate cannot be negative")
        if self.min_assets < 2:
            raise ValueError("min_assets must be at least 2")
        if self.group_count < 2:
            raise ValueError("group_count must be at least 2")
        if self.rolling_ic_window <= 0:
            raise ValueError("rolling_ic_window must be positive")
        if self.nw_lags < 0:
            raise ValueError("nw_lags cannot be negative")
        if self.min_days_to_maturity < 0:
            raise ValueError("min_days_to_maturity cannot be negative")
        if not 0 < self.turnover_limit <= 1:
            raise ValueError("turnover_limit must be in (0, 1]")
        if not 0 < self.max_abs_weight <= 1:
            raise ValueError("max_abs_weight must be in (0, 1]")
