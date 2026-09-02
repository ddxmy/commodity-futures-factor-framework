"""Compare 1/5/10-day holding periods for three inverse-volatility portfolios."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config.settings import COST_RATE, RESULT_DIR
from src.p01_market_data import (
    load_contract_prices,
    load_trade_calendar,
)
from src.p02_contract_selection import build_contract_mapping
from src.p06_backtest_engine import compute_metrics, run_backtest_from_weights
from src.run_five_factor_strategy_v1 import (
    END_DATE,
    FACTOR_LABELS,
    LOAD_START,
    MIN_ASSETS,
    START_DATE,
    TRADE_MIN_DAYS_TO_MATURITY,
    SIGNAL_MIN_DAYS_TO_MATURITY,
    VOLATILITY_FLOOR,
    build_factor_panels,
    build_semiannual_pool,
    build_strategy_panel,
    configure_plot_style,
)


OUTPUT_DIR = (
    Path(RESULT_DIR)
    / "five_factor_holding_periods_20200101_20260701"
)

HOLDING_PERIODS = (1, 5, 10)

METHODS = {
    "all_rank_invvol": {
        "label": "全截面排名",
        "tail_fraction": None,
    },
    "tail10_invvol": {
        "label": "前后10%",
        "tail_fraction": 0.10,
    },
    "tail20_invvol": {
        "label": "前后20%",
        "tail_fraction": 0.20,
    },
}

HOLDING_COLORS = {
    1: "#247BA0",
    5: "#E07A5F",
    10: "#43AA8B",
}

FACTOR_COLORS = {
    "basis_momentum": "#277DA1",
    "carry": "#43AA8B",
    "s_warehouse": "#F9C74F",
    "spotmain": "#F9844A",
    "t_rank": "#9B5DE5",
}


def build_inverse_volatility_weights(
    strategy_panel: pd.DataFrame,
    method: str,
    holding_period: int,
) -> pd.DataFrame:
    """Form one target portfolio only on every H-th trading date."""
    if method not in METHODS:
        raise ValueError(f"unknown method: {method}")
    if holding_period not in HOLDING_PERIODS:
        raise ValueError(f"unsupported holding period: {holding_period}")

    out = strategy_panel.copy()
    out["weight"] = 0.0
    dates = pd.DatetimeIndex(out["trade_date"].drop_duplicates().sort_values())
    rebalance_dates = dates[::holding_period]
    out["is_rebalance"] = out["trade_date"].isin(rebalance_dates)

    rebalance_panel = out.loc[out["is_rebalance"]]
    for _, index in rebalance_panel.groupby("trade_date", sort=True).groups.items():
        day = out.loc[index]
        eligible = day.loc[
            day["in_pool"]
            & day["raw_factor"].notna()
            & day["vol20"].notna()
            & day["vol20"].gt(0)
            & day["ts_code_A"].notna()
        ].copy()
        if len(eligible) < MIN_ASSETS:
            continue

        inverse_vol = 1.0 / eligible["vol20"].clip(
            lower=VOLATILITY_FLOOR
        )
        tail_fraction = METHODS[method]["tail_fraction"]

        if tail_fraction is None:
            ranks = eligible["raw_factor"].rank(method="average")
            centered_rank = ranks - (len(eligible) + 1) / 2.0
            raw_weight = centered_rank * inverse_vol
        else:
            tail_count = max(
                1,
                int(np.ceil(len(eligible) * float(tail_fraction))),
            )
            ordered = eligible.sort_values(
                ["raw_factor", "fut_code"],
                ascending=[True, True],
            )
            short_index = ordered.head(tail_count).index
            long_index = ordered.tail(tail_count).index
            raw_weight = pd.Series(0.0, index=eligible.index)
            raw_weight.loc[long_index] = inverse_vol.loc[long_index]
            raw_weight.loc[short_index] = -inverse_vol.loc[short_index]

        long_total = raw_weight.clip(lower=0).sum()
        short_total = -raw_weight.clip(upper=0).sum()
        if long_total <= 0 or short_total <= 0:
            continue

        normalized = pd.Series(0.0, index=eligible.index)
        normalized.loc[raw_weight > 0] = (
            raw_weight.loc[raw_weight > 0] / long_total * 0.5
        )
        normalized.loc[raw_weight < 0] = (
            raw_weight.loc[raw_weight < 0] / short_total * 0.5
        )
        out.loc[normalized.index, "weight"] = normalized

    return out[
        [
            "trade_date",
            "fut_code",
            "weight",
            "is_rebalance",
            "ts_code_A",
            "raw_factor",
            "in_pool",
            "vol20",
        ]
    ]


def calculate_horizon_ic(
    strategy_panel: pd.DataFrame,
    prices: pd.DataFrame,
    calendar: pd.DataFrame,
    holding_period: int,
) -> tuple[float, float, int]:
    """Match each rebalance signal to its next-open H-day locked return."""
    dates = pd.DatetimeIndex(
        calendar.loc[
            calendar["trade_date"].between(
                pd.Timestamp(START_DATE),
                pd.Timestamp(END_DATE),
            ),
            "trade_date",
        ]
        .drop_duplicates()
        .sort_values()
    )
    signal_positions = np.arange(0, len(dates), holding_period)
    schedule_records = []
    for current, following in zip(signal_positions[:-1], signal_positions[1:]):
        if following + 1 >= len(dates):
            break
        schedule_records.append(
            {
                "trade_date": dates[current],
                "entry_date": dates[current + 1],
                "exit_date": dates[following + 1],
            }
        )
    schedule = pd.DataFrame(schedule_records)
    if schedule.empty:
        return np.nan, np.nan, 0

    panel = strategy_panel.merge(schedule, on="trade_date", how="inner")
    panel = panel.loc[panel["in_pool"]].copy()
    entry_prices = prices[["trade_date", "ts_code", "open"]].rename(
        columns={
            "trade_date": "entry_date",
            "ts_code": "ts_code_A",
            "open": "entry_open",
        }
    )
    exit_prices = prices[["trade_date", "ts_code", "open"]].rename(
        columns={
            "trade_date": "exit_date",
            "ts_code": "ts_code_A",
            "open": "exit_open",
        }
    )
    panel = panel.merge(
        entry_prices,
        on=["entry_date", "ts_code_A"],
        how="left",
        validate="many_to_one",
    ).merge(
        exit_prices,
        on=["exit_date", "ts_code_A"],
        how="left",
        validate="many_to_one",
    )
    panel["forward_return"] = panel["exit_open"] / panel["entry_open"] - 1.0

    records = []
    for trade_date, day in panel.groupby("trade_date"):
        valid = day[["raw_factor", "forward_return"]].dropna()
        if (
            len(valid) >= MIN_ASSETS
            and valid["raw_factor"].nunique() > 1
            and valid["forward_return"].nunique() > 1
        ):
            records.append(
                {
                    "trade_date": trade_date,
                    "ic": valid["raw_factor"].corr(valid["forward_return"]),
                    "rank_ic": valid["raw_factor"].corr(
                        valid["forward_return"],
                        method="spearman",
                    ),
                }
            )
    ic = pd.DataFrame(records)
    if ic.empty:
        return np.nan, np.nan, 0
    return float(ic["ic"].mean()), float(ic["rank_ic"].mean()), len(ic)


def summarize_single_strategy(
    factor_name: str,
    method: str,
    holding_period: int,
    weights: pd.DataFrame,
    nav: pd.Series,
    metrics: dict[str, float],
    ic_summary: tuple[float, float, int],
) -> dict[str, float | int | str]:
    active_targets = weights.loc[
        weights["is_rebalance"] & weights["weight"].ne(0)
    ]
    counts = active_targets.groupby("trade_date")["weight"].agg(
        long_count=lambda values: int((values > 0).sum()),
        short_count=lambda values: int((values < 0).sum()),
    )
    ic, rank_ic, ic_periods = ic_summary
    record: dict[str, float | int | str] = {
        "factor": factor_name,
        "factor_label": FACTOR_LABELS[factor_name],
        "method": method,
        "method_label": METHODS[method]["label"],
        "holding_period": holding_period,
        "final_nav": float(nav.iloc[-1]),
        "ic": ic,
        "rank_ic": rank_ic,
        "ic_periods": ic_periods,
        "rebalance_count": int(
            weights.loc[weights["is_rebalance"], "trade_date"].nunique()
        ),
        "average_long_count": float(counts["long_count"].mean()),
        "average_short_count": float(counts["short_count"].mean()),
    }
    record.update(metrics)
    return record


def build_composite(
    method: str,
    holding_period: int,
    navs: dict[str, pd.Series],
    diagnostics: dict[str, pd.DataFrame],
) -> tuple[pd.Series, dict[str, float | int | str]]:
    nav_frame = pd.concat(navs, axis=1).ffill().fillna(1.0)
    composite_nav = nav_frame.mean(axis=1)
    composite_nav.name = "equal_weight_composite"
    metrics = compute_metrics(composite_nav.pct_change().fillna(0.0))

    prior_nav = nav_frame.shift(1).fillna(1.0)
    total_capital = prior_nav.sum(axis=1)
    weighted_turnover = pd.Series(0.0, index=nav_frame.index)
    weighted_cost = pd.Series(0.0, index=nav_frame.index)
    for factor_name in nav_frame.columns:
        daily = diagnostics[factor_name].reindex(nav_frame.index).fillna(0.0)
        sleeve_share = prior_nav[factor_name] / total_capital
        weighted_turnover += sleeve_share * daily["turnover"]
        weighted_cost += sleeve_share * daily["cost"]

    record: dict[str, float | int | str] = {
        "method": method,
        "method_label": METHODS[method]["label"],
        "holding_period": holding_period,
        "final_nav": float(composite_nav.iloc[-1]),
        "annual_turnover": float(weighted_turnover.mean() * 252),
        "annual_cost": float(weighted_cost.mean() * 252),
    }
    record.update(metrics)
    return composite_nav, record


def plot_composite_comparison(
    composites: dict[tuple[str, int], pd.Series],
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(17, 5.6), sharey=False)
    for axis, (method, settings) in zip(axes, METHODS.items()):
        for holding_period in HOLDING_PERIODS:
            nav = composites[(method, holding_period)]
            axis.plot(
                nav.index,
                nav.values,
                label=f"H={holding_period}",
                color=HOLDING_COLORS[holding_period],
                linewidth=1.8,
            )
        axis.axhline(1.0, color="#666666", linewidth=0.7, alpha=0.6)
        axis.set_title(settings["label"])
        axis.set_ylabel("NAV")
        axis.legend(frameon=False)
    figure.suptitle("五因子等权组合：持仓周期比较", fontsize=17)
    figure.tight_layout()
    figure.savefig(
        OUTPUT_DIR / "composite_holding_period_comparison.png",
        bbox_inches="tight",
    )
    plt.close(figure)


def plot_factor_comparisons(
    factor_navs: dict[tuple[str, int], dict[str, pd.Series]],
) -> None:
    for factor_name, factor_label in FACTOR_LABELS.items():
        figure, axes = plt.subplots(1, 3, figsize=(17, 5.4), sharey=False)
        for axis, (method, settings) in zip(axes, METHODS.items()):
            for holding_period in HOLDING_PERIODS:
                nav = factor_navs[(method, holding_period)][factor_name]
                axis.plot(
                    nav.index,
                    nav.values,
                    label=f"H={holding_period}",
                    color=HOLDING_COLORS[holding_period],
                    linewidth=1.6,
                )
            axis.axhline(1.0, color="#666666", linewidth=0.7, alpha=0.6)
            axis.set_title(settings["label"])
            axis.set_ylabel("NAV")
            axis.legend(frameon=False)
        figure.suptitle(f"{factor_label}：持仓周期比较", fontsize=17)
        figure.tight_layout()
        figure.savefig(
            OUTPUT_DIR / f"holding_comparison_{factor_name}.png",
            bbox_inches="tight",
        )
        plt.close(figure)


def plot_all_nav_grid(
    factor_navs: dict[tuple[str, int], dict[str, pd.Series]],
    composites: dict[tuple[str, int], pd.Series],
) -> None:
    figure, axes = plt.subplots(3, 3, figsize=(18, 15), sharex=True)
    for row, (method, settings) in enumerate(METHODS.items()):
        for column, holding_period in enumerate(HOLDING_PERIODS):
            axis = axes[row, column]
            key = (method, holding_period)
            for factor_name, label in FACTOR_LABELS.items():
                nav = factor_navs[key][factor_name]
                axis.plot(
                    nav.index,
                    nav.values,
                    label=label,
                    color=FACTOR_COLORS[factor_name],
                    linewidth=1.0,
                    alpha=0.82,
                )
            composite = composites[key]
            axis.plot(
                composite.index,
                composite.values,
                label="五因子等权总 NAV",
                color="#111111",
                linewidth=2.4,
            )
            axis.axhline(1.0, color="#666666", linewidth=0.7, alpha=0.6)
            axis.set_title(f"{settings['label']}，H={holding_period}")
            axis.set_ylabel("NAV")
            if row == 0 and column == 0:
                axis.legend(frameon=False, fontsize=8, ncol=2)
    figure.suptitle("五因子逆波动率策略：选品比例与持仓周期", fontsize=19)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "all_strategy_nav_grid.png", bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    configure_plot_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pool_statistics, pool_members = build_semiannual_pool()
    calendar = load_trade_calendar(LOAD_START, END_DATE)
    signal_mapping = build_contract_mapping(
        LOAD_START,
        END_DATE,
        min_days_to_maturity=SIGNAL_MIN_DAYS_TO_MATURITY,
    )
    trade_mapping = build_contract_mapping(
        LOAD_START,
        END_DATE,
        min_days_to_maturity=TRADE_MIN_DAYS_TO_MATURITY,
    )
    factor_panels = build_factor_panels(signal_mapping, calendar)
    prices = load_contract_prices(START_DATE, END_DATE)

    strategy_panels = {
        factor_name: build_strategy_panel(
            factor_panel,
            trade_mapping,
            calendar,
            pool_members,
        )
        for factor_name, factor_panel in factor_panels.items()
    }
    ic_cache = {
        (factor_name, holding_period): calculate_horizon_ic(
            strategy_panel,
            prices,
            calendar,
            holding_period,
        )
        for factor_name, strategy_panel in strategy_panels.items()
        for holding_period in HOLDING_PERIODS
    }

    factor_navs: dict[tuple[str, int], dict[str, pd.Series]] = {}
    diagnostics_by_key: dict[tuple[str, int], dict[str, pd.DataFrame]] = {}
    metric_records = []

    for method in METHODS:
        for holding_period in HOLDING_PERIODS:
            key = (method, holding_period)
            factor_navs[key] = {}
            diagnostics_by_key[key] = {}
            for factor_name, strategy_panel in strategy_panels.items():
                weights = build_inverse_volatility_weights(
                    strategy_panel,
                    method,
                    holding_period,
                )
                nav, metrics, _, diagnostics = run_backtest_from_weights(
                    weights,
                    prices,
                    cost_rate=COST_RATE,
                    return_diagnostics=True,
                )
                factor_navs[key][factor_name] = nav
                diagnostics_by_key[key][factor_name] = diagnostics
                metric_records.append(
                    summarize_single_strategy(
                        factor_name,
                        method,
                        holding_period,
                        weights,
                        nav,
                        metrics,
                        ic_cache[(factor_name, holding_period)],
                    )
                )

    composites = {}
    composite_records = []
    for key, navs in factor_navs.items():
        method, holding_period = key
        composite, metrics = build_composite(
            method,
            holding_period,
            navs,
            diagnostics_by_key[key],
        )
        composites[key] = composite
        composite_records.append(metrics)

    nav_output = pd.DataFrame(index=next(iter(composites.values())).index)
    for (method, holding_period), navs in factor_navs.items():
        for factor_name, nav in navs.items():
            nav_output[
                f"{method}__H{holding_period}__{factor_name}"
            ] = nav
        nav_output[
            f"{method}__H{holding_period}__equal_weight_composite"
        ] = composites[(method, holding_period)]

    pool_statistics.to_csv(
        OUTPUT_DIR / "semiannual_pool_statistics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pool_members.to_csv(
        OUTPUT_DIR / "semiannual_pool_members.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(metric_records).to_csv(
        OUTPUT_DIR / "factor_holding_period_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(composite_records).to_csv(
        OUTPUT_DIR / "composite_holding_period_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    nav_output.to_csv(
        OUTPUT_DIR / "daily_nav.csv",
        encoding="utf-8-sig",
    )

    plot_composite_comparison(composites)
    plot_factor_comparisons(factor_navs)
    plot_all_nav_grid(factor_navs, composites)

    print(f"Results written to: {OUTPUT_DIR}")
    print(pd.DataFrame(composite_records).to_string(index=False))


if __name__ == "__main__":
    main()
