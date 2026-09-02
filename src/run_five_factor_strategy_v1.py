"""Run the first report-ready five-factor commodity strategy comparison."""

from __future__ import annotations

from contextlib import closing
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config.settings import COST_RATE, DB_PATH, RESULT_DIR
from src.factors.basis_momentum import compute_basis_components
from src.factors.carry import compute_main_sub_carry
from src.factors.s_warehouse import compute_s_warehouse
from src.factors.spotmain import compute_spot_main
from src.factors.t_rank import compute_t_rank
from src.p01_market_data import (
    load_contract_prices,
    load_spot_daily,
    load_trade_calendar,
    load_warehouse_daily,
)
from src.p02_contract_selection import build_contract_mapping
from src.p06_backtest_engine import compute_metrics, run_backtest_from_weights


START_DATE = "20200101"
END_DATE = "20260701"
LOAD_START = "20190101"
POOL_SIZE = 40
MIN_VALID_RATE = 0.80
VOLATILITY_WINDOW = 20
MIN_ASSETS = 10
TAIL_FRACTION = 0.10
VOLATILITY_FLOOR = 0.001
SIGNAL_MIN_DAYS_TO_MATURITY = 0
TRADE_MIN_DAYS_TO_MATURITY = 45

OUTPUT_DIR = Path(RESULT_DIR) / "five_factor_strategy_v1_20200101_20260701"

FACTOR_LABELS = {
    "basis_momentum": "Basis Momentum",
    "carry": "Carry",
    "s_warehouse": "S_Warehouse",
    "spotmain": "SpotMain",
    "t_rank": "T_Rank",
}

METHOD_LABELS = {
    "all_rank_invvol": "全截面排名 + 20日逆波动率",
    "tail10_invvol": "前后10% + 20日逆波动率",
}


def configure_plot_style() -> None:
    """Apply the project-wide Chinese and Latin font convention."""
    plt.rcParams.update(
        {
            "font.family": ["Times New Roman", "Kaiti SC"],
            "axes.unicode_minus": False,
            "figure.dpi": 140,
            "savefig.dpi": 220,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.20,
            "grid.linewidth": 0.7,
        }
    )


def semester_starts() -> pd.DatetimeIndex:
    starts = []
    for year in range(2020, 2027):
        for month in (1, 7):
            candidate = pd.Timestamp(year=year, month=month, day=1)
            if candidate <= pd.Timestamp(END_DATE):
                starts.append(candidate)
    return pd.DatetimeIndex(starts)


def load_pool_source() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load historical main-contract liquidity without using future semesters."""
    query = """
    SELECT
        u.trade_date,
        u.fut_code,
        u.main_contract,
        d.amount,
        d.vol,
        d.oi
    FROM tradable_universe AS u
    LEFT JOIN fut_daily AS d
      ON d.trade_date = u.trade_date
     AND d.ts_code = u.main_contract
    WHERE u.is_tradable = 1
      AND u.trade_date BETWEEN ? AND ?
    ORDER BY u.trade_date, u.fut_code
    """
    calendar_query = """
    SELECT DISTINCT cal_date AS trade_date
    FROM trade_cal
    WHERE is_open = 1
      AND cal_date BETWEEN ? AND ?
    ORDER BY cal_date
    """
    with closing(sqlite3.connect(DB_PATH)) as connection:
        liquidity = pd.read_sql_query(
            query,
            connection,
            params=("20190701", "20260630"),
        )
        calendar = pd.read_sql_query(
            calendar_query,
            connection,
            params=("20190701", "20260630"),
        )
    liquidity["trade_date"] = pd.to_datetime(liquidity["trade_date"])
    calendar["trade_date"] = pd.to_datetime(calendar["trade_date"])
    return liquidity, calendar


def build_semiannual_pool() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select the top 40 products using only the preceding six months."""
    source, calendar = load_pool_source()
    statistics = []
    members = []

    for effective_start in semester_starts():
        history_start = effective_start - pd.DateOffset(months=6)
        history_end = effective_start - pd.Timedelta(days=1)
        period = source.loc[
            source["trade_date"].between(history_start, history_end)
        ].copy()
        market_days = int(
            calendar["trade_date"].between(history_start, history_end).sum()
        )

        summary = (
            period.groupby("fut_code", as_index=False)
            .agg(
                valid_days=("amount", lambda values: int((values > 0).sum())),
                mean_amount=("amount", "mean"),
                median_amount=("amount", "median"),
                mean_volume=("vol", "mean"),
                mean_open_interest=("oi", "mean"),
            )
        )
        summary["valid_rate"] = summary["valid_days"] / market_days
        eligible = summary.loc[
            summary["valid_rate"].ge(MIN_VALID_RATE)
            & summary["mean_amount"].gt(0)
        ].copy()
        eligible = eligible.sort_values(
            ["mean_amount", "fut_code"],
            ascending=[False, True],
        ).reset_index(drop=True)
        eligible["liquidity_rank"] = np.arange(1, len(eligible) + 1)
        selected = eligible.head(POOL_SIZE).copy()
        selected["effective_start"] = effective_start
        selected["effective_end"] = (
            effective_start + pd.DateOffset(months=6) - pd.Timedelta(days=1)
        )
        members.append(selected)

        rank35 = eligible.iloc[34]["mean_amount"] if len(eligible) >= 35 else np.nan
        rank40 = eligible.iloc[39]["mean_amount"] if len(eligible) >= 40 else np.nan
        statistics.append(
            {
                "semester": f"{effective_start.year}H{1 if effective_start.month == 1 else 2}",
                "history_start": history_start,
                "history_end": history_end,
                "market_days": market_days,
                "observed_products": summary["fut_code"].nunique(),
                "eligible_products": len(eligible),
                "selected_products": len(selected),
                "rank35_mean_amount": rank35,
                "rank40_mean_amount": rank40,
            }
        )

    return pd.DataFrame(statistics), pd.concat(members, ignore_index=True)


def build_factor_panels(
    signal_mapping: pd.DataFrame,
    calendar: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Calculate the five existing factors on one shared data snapshot."""
    basis = compute_basis_components(signal_mapping, calendar, lookback=120)
    carry = compute_main_sub_carry(signal_mapping, calendar, lookback=90)
    spot = compute_spot_main(
        signal_mapping,
        load_spot_daily(LOAD_START, END_DATE),
        calendar,
        lookback=90,
    )
    rank = compute_t_rank(signal_mapping, calendar, lookback=10)
    warehouse = compute_s_warehouse(
        load_warehouse_daily(LOAD_START, END_DATE),
        calendar,
        lookback=90,
        smooth_window=20,
        min_observations=18,
    )

    factor_columns = {
        "basis_momentum": (basis, "factor_AB"),
        "carry": (carry, "main_sub_carry"),
        "s_warehouse": (warehouse, "s_warehouse"),
        "spotmain": (spot, "spotmain"),
        "t_rank": (rank, "t_rank"),
    }
    output = {}
    requested_start = pd.Timestamp(START_DATE)
    requested_end = pd.Timestamp(END_DATE)
    for name, (panel, column) in factor_columns.items():
        compact = panel[["trade_date", "fut_code", column]].copy()
        compact = compact.rename(columns={column: "raw_factor"})
        compact = compact.loc[
            compact["trade_date"].between(requested_start, requested_end)
        ].reset_index(drop=True)
        output[name] = compact
    return output


def attach_pool_membership(
    panel: pd.DataFrame,
    pool_members: pd.DataFrame,
) -> pd.DataFrame:
    out = panel.copy()
    out["semester_start"] = pd.to_datetime(
        {
            "year": out["trade_date"].dt.year,
            "month": np.where(out["trade_date"].dt.month <= 6, 1, 7),
            "day": 1,
        }
    )
    membership = pool_members[["effective_start", "fut_code"]].drop_duplicates()
    membership["in_pool"] = True
    out = out.merge(
        membership,
        left_on=["semester_start", "fut_code"],
        right_on=["effective_start", "fut_code"],
        how="left",
        validate="many_to_one",
    )
    out["in_pool"] = out["in_pool"].astype("boolean").fillna(False).astype(bool)
    return out.drop(columns=["effective_start"])


def build_strategy_panel(
    factor_panel: pd.DataFrame,
    trade_mapping: pd.DataFrame,
    calendar: pd.DataFrame,
    pool_members: pd.DataFrame,
) -> pd.DataFrame:
    products = sorted(trade_mapping["fut_code"].dropna().unique())
    requested_dates = calendar.loc[
        calendar["trade_date"].between(
            pd.Timestamp(START_DATE), pd.Timestamp(END_DATE)
        ),
        "trade_date",
    ]
    grid = pd.MultiIndex.from_product(
        [requested_dates, products],
        names=["trade_date", "fut_code"],
    ).to_frame(index=False)

    mapping_columns = [
        "trade_date",
        "fut_code",
        "ts_code_A",
        "daily_return_A",
    ]
    mapping = trade_mapping[mapping_columns].copy()
    mapping = mapping.sort_values(["fut_code", "trade_date"])
    mapping["vol20"] = mapping.groupby("fut_code")["daily_return_A"].transform(
        lambda values: values.rolling(
            VOLATILITY_WINDOW,
            min_periods=VOLATILITY_WINDOW,
        ).std()
    )

    out = grid.merge(
        mapping[["trade_date", "fut_code", "ts_code_A", "vol20"]],
        on=["trade_date", "fut_code"],
        how="left",
        validate="one_to_one",
    )
    out = out.merge(
        factor_panel,
        on=["trade_date", "fut_code"],
        how="left",
        validate="one_to_one",
    )
    return attach_pool_membership(out, pool_members)


def inverse_volatility_weights(
    strategy_panel: pd.DataFrame,
    method: str,
) -> pd.DataFrame:
    out = strategy_panel.copy()
    out["weight"] = 0.0

    for _, index in out.groupby("trade_date", sort=True).groups.items():
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

        effective_vol = eligible["vol20"].clip(lower=VOLATILITY_FLOOR)
        inverse_vol = 1.0 / effective_vol

        if method == "all_rank_invvol":
            ranks = eligible["raw_factor"].rank(method="average")
            centered = ranks - (len(eligible) + 1) / 2.0
            raw_weight = centered * inverse_vol
        elif method == "tail10_invvol":
            tail_count = max(1, int(np.ceil(len(eligible) * TAIL_FRACTION)))
            ordered = eligible.sort_values(
                ["raw_factor", "fut_code"],
                ascending=[True, True],
            )
            short_index = ordered.head(tail_count).index
            long_index = ordered.tail(tail_count).index
            raw_weight = pd.Series(0.0, index=eligible.index)
            raw_weight.loc[long_index] = inverse_vol.loc[long_index]
            raw_weight.loc[short_index] = -inverse_vol.loc[short_index]
        else:
            raise ValueError(f"unknown portfolio method: {method}")

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

    out["is_rebalance"] = True
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


def calculate_daily_ic(
    strategy_panel: pd.DataFrame,
    prices: pd.DataFrame,
    calendar: pd.DataFrame,
) -> tuple[float, float, int]:
    dates = calendar.loc[
        calendar["trade_date"].between(
            pd.Timestamp(START_DATE), pd.Timestamp(END_DATE)
        ),
        "trade_date",
    ].reset_index(drop=True)
    schedule = pd.DataFrame(
        {
            "trade_date": dates,
            "entry_date": dates.shift(-1),
            "exit_date": dates.shift(-2),
        }
    )
    panel = strategy_panel.merge(schedule, on="trade_date", how="left")
    entry = prices[["trade_date", "ts_code", "open"]].rename(
        columns={"trade_date": "entry_date", "ts_code": "ts_code_A", "open": "entry_open"}
    )
    exit_prices = prices[["trade_date", "ts_code", "open"]].rename(
        columns={"trade_date": "exit_date", "ts_code": "ts_code_A", "open": "exit_open"}
    )
    panel = panel.merge(
        entry,
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
    valid_panel = panel.loc[panel["in_pool"]].copy()
    for trade_date, day in valid_panel.groupby("trade_date"):
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
                        valid["forward_return"], method="spearman"
                    ),
                }
            )
    ic = pd.DataFrame(records)
    if ic.empty:
        return np.nan, np.nan, 0
    return float(ic["ic"].mean()), float(ic["rank_ic"].mean()), len(ic)


def summarize_strategy(
    factor_name: str,
    method: str,
    weights: pd.DataFrame,
    nav: pd.Series,
    metrics: dict[str, float],
    strategy_panel: pd.DataFrame,
    prices: pd.DataFrame,
    calendar: pd.DataFrame,
) -> dict[str, float | str]:
    ic, rank_ic, ic_days = calculate_daily_ic(strategy_panel, prices, calendar)
    active = weights.loc[weights["weight"].ne(0)]
    daily_counts = active.groupby("trade_date")["weight"].agg(
        long_count=lambda values: int((values > 0).sum()),
        short_count=lambda values: int((values < 0).sum()),
    )
    record: dict[str, float | str] = {
        "factor": factor_name,
        "factor_label": FACTOR_LABELS[factor_name],
        "method": method,
        "method_label": METHOD_LABELS[method],
        "final_nav": float(nav.iloc[-1]),
        "ic": ic,
        "rank_ic": rank_ic,
        "ic_days": ic_days,
        "average_long_count": float(daily_counts["long_count"].mean()),
        "average_short_count": float(daily_counts["short_count"].mean()),
    }
    record.update(metrics)
    return record


def build_composite(
    method: str,
    navs: dict[str, pd.Series],
    diagnostics: dict[str, pd.DataFrame],
) -> tuple[pd.Series, dict[str, float]]:
    nav_frame = pd.concat(navs, axis=1).ffill().fillna(1.0)
    composite_nav = nav_frame.mean(axis=1)
    composite_nav.name = "equal_weight_composite"
    composite_return = composite_nav.pct_change().fillna(0.0)
    metrics = compute_metrics(composite_return)

    prior_nav = nav_frame.shift(1).fillna(1.0)
    capital_sum = prior_nav.sum(axis=1)
    weighted_turnover = pd.Series(0.0, index=nav_frame.index)
    weighted_cost = pd.Series(0.0, index=nav_frame.index)
    for factor_name in nav_frame.columns:
        daily = diagnostics[factor_name].reindex(nav_frame.index).fillna(0.0)
        sleeve_weight = prior_nav[factor_name] / capital_sum
        weighted_turnover += sleeve_weight * daily["turnover"]
        weighted_cost += sleeve_weight * daily["cost"]
    metrics.update(
        {
            "final_nav": float(composite_nav.iloc[-1]),
            "annual_turnover": float(weighted_turnover.mean() * 252),
            "annual_cost": float(weighted_cost.mean() * 252),
            "method": method,
            "method_label": METHOD_LABELS[method],
        }
    )
    return composite_nav, metrics


def plot_factor_comparisons(navs_by_method: dict[str, dict[str, pd.Series]]) -> None:
    palette = {"all_rank_invvol": "#247BA0", "tail10_invvol": "#E07A5F"}
    for factor_name, factor_label in FACTOR_LABELS.items():
        figure, axis = plt.subplots(figsize=(10.5, 5.2))
        for method, label in METHOD_LABELS.items():
            axis.plot(
                navs_by_method[method][factor_name].index,
                navs_by_method[method][factor_name].values,
                label=label,
                color=palette[method],
                linewidth=1.7,
            )
        axis.axhline(1.0, color="#555555", linewidth=0.8, alpha=0.7)
        axis.set_title(f"{factor_label} 策略净值")
        axis.set_ylabel("NAV")
        axis.legend(frameon=False)
        figure.tight_layout()
        figure.savefig(OUTPUT_DIR / f"nav_{factor_name}.png", bbox_inches="tight")
        plt.close(figure)


def plot_summary(
    navs_by_method: dict[str, dict[str, pd.Series]],
    composites: dict[str, pd.Series],
) -> None:
    colors = {
        "basis_momentum": "#277DA1",
        "carry": "#43AA8B",
        "s_warehouse": "#F9C74F",
        "spotmain": "#F9844A",
        "t_rank": "#9B5DE5",
    }
    figure, axes = plt.subplots(1, 2, figsize=(16, 6.2), sharey=False)
    for axis, (method, method_label) in zip(axes, METHOD_LABELS.items()):
        for factor_name, label in FACTOR_LABELS.items():
            nav = navs_by_method[method][factor_name]
            axis.plot(
                nav.index,
                nav.values,
                label=label,
                color=colors[factor_name],
                linewidth=1.1,
                alpha=0.82,
            )
        composite = composites[method]
        axis.plot(
            composite.index,
            composite.values,
            label="五因子等权总 NAV",
            color="#111111",
            linewidth=2.6,
        )
        axis.axhline(1.0, color="#666666", linewidth=0.7, alpha=0.6)
        axis.set_title(method_label)
        axis.set_ylabel("NAV")
        axis.legend(frameon=False, fontsize=9, ncol=2)
    figure.suptitle("五因子逆波动率策略汇总", fontsize=17)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "all_factor_nav_summary.png", bbox_inches="tight")
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

    methods = list(METHOD_LABELS)
    navs_by_method: dict[str, dict[str, pd.Series]] = {
        method: {} for method in methods
    }
    diagnostics_by_method: dict[str, dict[str, pd.DataFrame]] = {
        method: {} for method in methods
    }
    metric_records = []
    coverage_records = []

    strategy_panels = {}
    for factor_name, factor_panel in factor_panels.items():
        strategy_panel = build_strategy_panel(
            factor_panel,
            trade_mapping,
            calendar,
            pool_members,
        )
        strategy_panels[factor_name] = strategy_panel
        valid = strategy_panel.loc[
            strategy_panel["in_pool"] & strategy_panel["raw_factor"].notna()
        ]
        coverage_by_day = valid.groupby("trade_date")["fut_code"].nunique()
        coverage_records.append(
            {
                "factor": factor_name,
                "first_valid_date": valid["trade_date"].min(),
                "last_valid_date": valid["trade_date"].max(),
                "average_daily_coverage": coverage_by_day.mean(),
                "minimum_daily_coverage": coverage_by_day.min(),
                "maximum_daily_coverage": coverage_by_day.max(),
            }
        )

        for method in methods:
            weights = inverse_volatility_weights(strategy_panel, method)
            nav, metrics, _, diagnostics = run_backtest_from_weights(
                weights,
                prices,
                cost_rate=COST_RATE,
                return_diagnostics=True,
            )
            navs_by_method[method][factor_name] = nav
            diagnostics_by_method[method][factor_name] = diagnostics
            metric_records.append(
                summarize_strategy(
                    factor_name,
                    method,
                    weights,
                    nav,
                    metrics,
                    strategy_panel,
                    prices,
                    calendar,
                )
            )

    composites = {}
    composite_records = []
    for method in methods:
        composite, metrics = build_composite(
            method,
            navs_by_method[method],
            diagnostics_by_method[method],
        )
        composites[method] = composite
        composite_records.append(metrics)

    nav_output = pd.DataFrame(index=composites[methods[0]].index)
    for method in methods:
        for factor_name, nav in navs_by_method[method].items():
            nav_output[f"{method}__{factor_name}"] = nav
        nav_output[f"{method}__equal_weight_composite"] = composites[method]

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
    pd.DataFrame(coverage_records).to_csv(
        OUTPUT_DIR / "factor_coverage.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(metric_records).to_csv(
        OUTPUT_DIR / "factor_strategy_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(composite_records).to_csv(
        OUTPUT_DIR / "composite_strategy_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    nav_output.to_csv(
        OUTPUT_DIR / "daily_nav.csv",
        encoding="utf-8-sig",
    )

    plot_factor_comparisons(navs_by_method)
    plot_summary(navs_by_method, composites)

    print(f"Results written to: {OUTPUT_DIR}")
    print(pd.DataFrame(metric_records).to_string(index=False))
    print(pd.DataFrame(composite_records).to_string(index=False))


if __name__ == "__main__":
    main()
