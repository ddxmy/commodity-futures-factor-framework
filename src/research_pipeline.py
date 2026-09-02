"""One-pass orchestration for cross-sectional futures factor research."""

from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from config.research_config import ResearchConfig
from config.settings import LIQ_LOOKBACK, RESULT_DIR
from src.factor_loader import calculate_factor
from src.p01_market_data import load_contract_prices, load_trade_calendar
from src.p02_contract_selection import build_contract_mapping
from src.p03_factor_processing import validate_factor_data
from src.p05_portfolio_construction import (
    CONTRACT_CONTEXT_COLUMNS,
    build_target_weights,
)
from src.p06_backtest_engine import run_backtest_from_weights
from src.p07_factor_evaluation import (
    assign_five_groups,
    build_factor_test_panel_from_data,
    calculate_group_nav,
    calculate_group_returns,
    calculate_ic_series,
    periods_per_year,
    summarize_ic_statistics,
)
from src.p08_reporting import (
    plot_group_nav,
    plot_ic_history,
    plot_strategy_nav,
    save_result_tables,
)


def build_factor_label(
    factor_name: str,
    factor_parameters: Mapping[str, object],
) -> str:
    """Return a readable label used in strategy names and output folders."""
    if factor_name == "basis_momentum":
        variant = str(factor_parameters.get("variant", "AB")).upper()
        lookback = factor_parameters.get("lookback", "default")
        return f"{variant}_L{lookback}"
    if factor_name == "t_rank":
        lookback = factor_parameters.get("lookback", 10)
        return f"t_rank_L{lookback}"
    return factor_name


def build_output_directory(
    result_dir: str | Path,
    factor_name: str,
    factor_label: str,
    start_date: str,
    end_date: str,
    rebalance_freq: int | str = 1,
) -> Path:
    """Return the factor-period folder and schedule-specific child folder."""
    base = (
        Path(result_dir)
        / factor_name
        / f"{factor_label}-{start_date}-{end_date}"
    )
    return base / build_rebalance_directory_name(rebalance_freq)


def build_rebalance_directory_name(rebalance_freq: int | str) -> str:
    """Return a stable result subdirectory for one rebalance schedule."""
    if rebalance_freq == 1:
        return "daily"
    if isinstance(rebalance_freq, int) and not isinstance(rebalance_freq, bool):
        return f"every_{rebalance_freq}_trading_days"
    normalized = str(rebalance_freq).upper()
    if normalized == "W-FRI":
        return "weekly_last_trading_day"
    return normalized.lower().replace("-", "_")


def prepare_contract_context(
    factor_data: pd.DataFrame,
    start_date: str,
    end_date: str,
    min_days_to_maturity: int | None = None,
) -> pd.DataFrame:
    """Return trading-contract and liquidity fields independent of the factor."""
    can_reuse_factor_context = (
        min_days_to_maturity is None
        and set(CONTRACT_CONTEXT_COLUMNS).issubset(factor_data.columns)
    )
    if can_reuse_factor_context:
        context = factor_data[CONTRACT_CONTEXT_COLUMNS].copy()
    else:
        requested_start = pd.to_datetime(start_date)
        buffer_days = LIQ_LOOKBACK * 3 + 30
        load_start = (
            requested_start - pd.Timedelta(days=buffer_days)
        ).strftime("%Y%m%d")
        mapping_kwargs = {}
        if min_days_to_maturity is not None:
            mapping_kwargs["min_days_to_maturity"] = (
                min_days_to_maturity
            )
        context = build_contract_mapping(
            load_start,
            end_date,
            **mapping_kwargs,
        )
        context = context.loc[
            context["trade_date"].ge(requested_start),
            CONTRACT_CONTEXT_COLUMNS,
        ].copy()

    if context.duplicated(["trade_date", "fut_code"]).any():
        raise ValueError("contract context contains duplicate date-commodity keys")
    return context


def run_strategy_comparison_from_data(
    factor_data: pd.DataFrame,
    contract_data: pd.DataFrame,
    prices: pd.DataFrame,
    config: ResearchConfig,
    factor_label: str,
    zscore_clip: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run Rank and clipped Z-score portfolios from one factor calculation."""
    settings = [("rank", None), ("zscore", zscore_clip)]
    metric_records = []
    nav_series = []
    rank_weights = None

    for normalize, clip_value in settings:
        weights = build_target_weights(
            factor_data=factor_data,
            contract_data=contract_data,
            config=config,
            normalize=normalize,
            zscore_clip=clip_value,
        )
        if normalize == "rank":
            rank_weights = weights.copy()

        nav, metrics, _ = run_backtest_from_weights(
            weights=weights,
            prices=prices,
            cost_rate=config.cost_rate,
        )
        if nav.empty:
            raise ValueError(f"{normalize} strategy produced no NAV")

        suffix = "rank" if normalize == "rank" else f"zscore_clip{zscore_clip:g}"
        strategy_name = f"{factor_label}_{suffix}"
        record = {
            "name": strategy_name,
            "normalize": normalize,
            "zscore_clip": clip_value,
            "trading_days": len(nav),
            "total_return": float(nav.iloc[-1] - 1.0),
            **metrics,
        }
        metric_records.append(record)
        named_nav = nav.copy()
        named_nav.name = strategy_name
        nav_series.append(named_nav)

    strategy_nav = pd.concat(nav_series, axis=1)
    strategy_nav.index.name = "trade_date"
    if rank_weights is None:
        raise RuntimeError("rank weights were not generated")
    return pd.DataFrame(metric_records), strategy_nav, rank_weights


def evaluate_factor_from_data(
    factor_data: pd.DataFrame,
    contract_data: pd.DataFrame,
    trade_calendar: pd.DataFrame,
    prices: pd.DataFrame,
    config: ResearchConfig,
) -> dict[str, pd.DataFrame]:
    """Calculate IC, Rank IC, five-group returns and cumulative NAV."""
    panel = build_factor_test_panel_from_data(
        factor_data=factor_data,
        contract_data=contract_data,
        trade_calendar=trade_calendar,
        prices=prices,
        config=config,
    )
    if panel.empty:
        raise ValueError("factor evaluation panel is empty")

    ic_series = calculate_ic_series(panel)
    ic_summary = summarize_ic_statistics(
        ic_series,
        nw_lags=config.nw_lags,
        annualization_periods=periods_per_year(config.rebalance_freq),
    )
    grouped_panel = assign_five_groups(panel, group_count=config.group_count)
    group_returns = calculate_group_returns(grouped_panel)
    group_nav = calculate_group_nav(group_returns)
    return {
        "ic_summary": ic_summary,
        "ic_series": ic_series,
        "group_returns": group_returns,
        "group_nav": group_nav,
    }


def run_factor_research(
    factor_name: str,
    start_date: str,
    end_date: str,
    factor_parameters: Mapping[str, object] | None = None,
    research_config: ResearchConfig | None = None,
    result_dir: str | Path = RESULT_DIR,
    zscore_clip: float = 3.0,
) -> Path:
    """Run the complete report while calculating the factor exactly once."""
    if zscore_clip <= 0:
        raise ValueError("zscore_clip must be positive")
    parameters = dict(factor_parameters or {})
    config = research_config or ResearchConfig()
    parameters.setdefault(
        "signal_min_days_to_maturity",
        config.signal_min_days_to_maturity,
    )

    trade_calendar = load_trade_calendar(start_date, end_date)
    factor_data = calculate_factor(
        factor_name=factor_name,
        start_date=start_date,
        end_date=end_date,
        parameters=parameters,
    )
    factor_data = validate_factor_data(factor_data, trade_calendar)
    contract_data = prepare_contract_context(
        factor_data,
        start_date,
        end_date,
        min_days_to_maturity=config.trade_min_days_to_maturity,
    )
    prices = load_contract_prices(start_date, end_date)
    factor_label = build_factor_label(factor_name, parameters)

    strategy_metrics, strategy_nav, _ = run_strategy_comparison_from_data(
        factor_data=factor_data,
        contract_data=contract_data,
        prices=prices,
        config=config,
        factor_label=factor_label,
        zscore_clip=zscore_clip,
    )
    factor_results = evaluate_factor_from_data(
        factor_data=factor_data,
        contract_data=contract_data,
        trade_calendar=trade_calendar,
        prices=prices,
        config=config,
    )

    output_dir = build_output_directory(
        result_dir,
        factor_name,
        factor_label,
        start_date,
        end_date,
        config.rebalance_freq,
    )
    run_metadata = {
        "factor_name": factor_name,
        "start_date": start_date,
        "end_date": end_date,
        "zscore_clip": zscore_clip,
        **{f"factor_{key}": value for key, value in parameters.items()},
        **asdict(config),
    }
    save_result_tables(
        output_dir,
        run_metadata,
        strategy_metrics,
        strategy_nav,
        factor_results,
    )
    plot_strategy_nav(strategy_nav, output_dir / "strategy_nav.png")
    plot_group_nav(factor_results["group_nav"], output_dir / "five_group_nav.png")
    plot_ic_history(
        factor_results["ic_series"],
        config.rolling_ic_window,
        output_dir / f"ic_rankic_rolling{config.rolling_ic_window}.png",
    )
    return output_dir
