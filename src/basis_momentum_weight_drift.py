"""Compare Basis Momentum before and after holdings-weight drift."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config.settings import COST_RATE, RESULT_DIR
from src.factors.basis_momentum import compute_basis_components
from src.p01_market_data import load_contract_prices, load_trade_calendar
from src.p02_contract_selection import build_contract_mapping
from src.p06_backtest_engine import compute_metrics, run_backtest_from_weights
import src.run_five_factor_strategy_v1 as five_factor


START_DATE = "20200101"
END_DATE = "20251231"
LOAD_START_DATE = "20190101"
OUTPUT_DIR = (
    Path(RESULT_DIR)
    / "basis_momentum_weight_drift_20200101_20251231"
)

METHODS = {
    "all_rank_invvol": "全截面",
    "tail10_invvol": "前后10%",
}


def _execution_targets(weights: pd.DataFrame) -> pd.DataFrame:
    required = {
        "trade_date",
        "fut_code",
        "weight",
        "is_rebalance",
        "ts_code_A",
    }
    missing = required - set(weights.columns)
    if missing:
        raise ValueError(
            "weights is missing columns: " + ", ".join(sorted(missing))
        )
    out = weights.copy()
    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="raise")
    if out.duplicated(["trade_date", "fut_code"]).any():
        raise ValueError("weights contains duplicate date-product keys")
    out = out.sort_values(["fut_code", "trade_date"])
    out["target_weight"] = out["weight"].where(out["is_rebalance"])
    out["desired_exec_weight"] = (
        out.groupby("fut_code")["target_weight"].shift(1).ffill().fillna(0.0)
    )
    out["target_ts_code"] = out["ts_code_A"].where(out["is_rebalance"])
    out["desired_trade_ts_code"] = (
        out.groupby("fut_code")["target_ts_code"].shift(1).ffill()
    )
    return out.sort_values(["trade_date", "fut_code"]).reset_index(drop=True)


def run_drift_aware_backtest(
    weights: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    cost_rate: float = COST_RATE,
    return_positions: bool = False,
):
    """Track fractional futures exposure units and rebalance at each open."""
    if cost_rate < 0:
        raise ValueError("cost_rate cannot be negative")
    targets = _execution_targets(weights)

    required_prices = {"trade_date", "ts_code", "open", "close"}
    missing_prices = required_prices - set(prices.columns)
    if missing_prices:
        raise ValueError(
            "prices is missing columns: " + ", ".join(sorted(missing_prices))
        )
    prepared_prices = prices[list(required_prices)].copy()
    prepared_prices["trade_date"] = pd.to_datetime(
        prepared_prices["trade_date"], errors="raise"
    )
    if prepared_prices.duplicated(["trade_date", "ts_code"]).any():
        raise ValueError("prices contains duplicate date-contract keys")
    price_lookup = prepared_prices.set_index(
        ["trade_date", "ts_code"]
    )[["open", "close"]].to_dict("index")

    equity_close = 1.0
    positions: dict[str, dict[str, float | str]] = {}
    daily_records: list[dict[str, float | pd.Timestamp]] = []
    position_records: list[dict[str, float | str | bool | pd.Timestamp | None]] = []

    def contract_prices(trade_date: pd.Timestamp, contract: str) -> tuple[float, float]:
        values = price_lookup.get((trade_date, contract))
        if values is None:
            raise ValueError(f"missing price for {trade_date.date()} {contract}")
        open_price = float(values["open"])
        close_price = float(values["close"])
        if not np.isfinite(open_price) or open_price <= 0:
            raise ValueError(f"invalid open for {trade_date.date()} {contract}")
        if not np.isfinite(close_price) or close_price <= 0:
            raise ValueError(f"invalid close for {trade_date.date()} {contract}")
        return open_price, close_price

    def held_contract_prices(
        trade_date: pd.Timestamp,
        contract: str,
        last_close: float,
    ) -> tuple[float, float, bool]:
        values = price_lookup.get((trade_date, contract))
        if values is None:
            raise ValueError(f"missing price for {trade_date.date()} {contract}")
        close_price = float(values["close"])
        if not np.isfinite(close_price) or close_price <= 0:
            raise ValueError(f"invalid close for {trade_date.date()} {contract}")
        open_price = float(values["open"])
        is_tradable = np.isfinite(open_price) and open_price > 0
        valuation_open = open_price if is_tradable else float(last_close)
        return valuation_open, close_price, bool(is_tradable)

    for trade_date, day in targets.groupby("trade_date", sort=True):
        equity_previous_close = equity_close
        opening_prices: dict[str, tuple[float, float, bool]] = {}
        opening_contracts = {
            fut_code: str(position["contract"])
            for fut_code, position in positions.items()
        }
        overnight_pnl = 0.0
        for fut_code, position in positions.items():
            contract = str(position["contract"])
            open_price, close_price, is_tradable = held_contract_prices(
                trade_date,
                contract,
                float(position["last_close"]),
            )
            opening_prices[fut_code] = (
                open_price,
                close_price,
                is_tradable,
            )
            overnight_pnl += float(position["units"]) * (
                open_price - float(position["last_close"])
            )

        equity_open = equity_previous_close + overnight_pnl
        if not np.isfinite(equity_open) or equity_open <= 0:
            raise ValueError(f"non-positive opening equity on {trade_date.date()}")

        turnover_cash = 0.0
        active_turnover_cash = 0.0
        roll_turnover_cash = 0.0
        day_position_records = []

        for row in day.itertuples(index=False):
            fut_code = row.fut_code
            target_weight = float(row.desired_exec_weight)
            desired_contract = (
                None
                if pd.isna(row.desired_trade_ts_code)
                else str(row.desired_trade_ts_code)
            )
            old = positions.get(fut_code)
            old_contract = None if old is None else str(old["contract"])
            old_units = 0.0 if old is None else float(old["units"])
            old_open = 0.0 if old is None else opening_prices[fut_code][0]
            old_tradable = old is None or opening_prices[fut_code][2]
            pretrade_notional = old_units * old_open
            pretrade_weight = pretrade_notional / equity_open
            wants_position = not np.isclose(target_weight, 0.0)
            is_roll = (
                old is not None
                and wants_position
                and desired_contract != old_contract
            )
            blocked_trade = False

            desired_open = np.nan
            new_tradable = not wants_position
            if wants_position:
                values = price_lookup.get((trade_date, desired_contract))
                if values is None or not np.isfinite(values["open"]) or values["open"] <= 0:
                    new_tradable = False
                else:
                    desired_open = float(values["open"])
                    new_tradable = True

            target_notional = target_weight * equity_open
            needs_trade = (
                (old is None and wants_position)
                or (old is not None and not wants_position)
                or is_roll
                or (
                    old is not None
                    and wants_position
                    and not is_roll
                    and not np.isclose(target_notional, pretrade_notional)
                )
            )
            if needs_trade:
                blocked_trade = (
                    (old is not None and not old_tradable)
                    or (wants_position and not new_tradable)
                )
            product_turnover_cash = 0.0
            if not blocked_trade:
                if old is None and wants_position:
                    product_turnover_cash = abs(target_notional)
                    positions[fut_code] = {
                        "contract": desired_contract,
                        "units": target_notional / desired_open,
                        "last_close": np.nan,
                    }
                elif old is not None and not wants_position:
                    product_turnover_cash = abs(pretrade_notional)
                    positions.pop(fut_code)
                elif old is not None and wants_position and is_roll:
                    product_turnover_cash = (
                        abs(pretrade_notional) + abs(target_notional)
                    )
                    positions[fut_code] = {
                        "contract": desired_contract,
                        "units": target_notional / desired_open,
                        "last_close": np.nan,
                    }
                elif old is not None and wants_position:
                    product_turnover_cash = abs(
                        target_notional - pretrade_notional
                    )
                    positions[fut_code]["units"] = target_notional / old_open

            turnover_cash += product_turnover_cash
            if is_roll and not blocked_trade:
                roll_turnover_cash += product_turnover_cash
            else:
                active_turnover_cash += product_turnover_cash
            day_position_records.append(
                {
                    "trade_date": trade_date,
                    "fut_code": fut_code,
                    "prev_trade_ts_code": old_contract,
                    "trade_ts_code": (
                        None
                        if fut_code not in positions
                        else str(positions[fut_code]["contract"])
                    ),
                    "pretrade_weight": pretrade_weight,
                    "target_weight": target_weight,
                    "turnover_cash": product_turnover_cash,
                    "turnover": product_turnover_cash / equity_open,
                    "actual_active_turnover": (
                        0.0 if is_roll else product_turnover_cash / equity_open
                    ),
                    "actual_roll_turnover": (
                        product_turnover_cash / equity_open if is_roll else 0.0
                    ),
                    "blocked_trade": blocked_trade,
                }
            )

        transaction_cost_cash = turnover_cash * cost_rate
        intraday_pnl = 0.0
        for fut_code, position in positions.items():
            contract = str(position["contract"])
            if (
                fut_code in opening_prices
                and contract == opening_contracts[fut_code]
            ):
                open_price, close_price = opening_prices[fut_code][:2]
            else:
                open_price, close_price = contract_prices(trade_date, contract)
            intraday_pnl += float(position["units"]) * (
                close_price - open_price
            )
            position["last_close"] = close_price

        equity_close = equity_open + intraday_pnl - transaction_cost_cash
        gross_return = (overnight_pnl + intraday_pnl) / equity_previous_close
        cost_return = transaction_cost_cash / equity_previous_close
        net_return = gross_return - cost_return
        daily_records.append(
            {
                "trade_date": trade_date,
                "gross_return": gross_return,
                "turnover": turnover_cash / equity_open,
                "cost": cost_return,
                "daily_return": net_return,
                "actual_active_turnover": active_turnover_cash / equity_open,
                "actual_roll_turnover": roll_turnover_cash / equity_open,
                "actual_mandatory_exit_turnover": 0.0,
                "actual_total_turnover": turnover_cash / equity_open,
                "equity_open": equity_open,
                "equity_close": equity_close,
            }
        )
        position_records.extend(day_position_records)

    daily = pd.DataFrame(daily_records).set_index("trade_date").sort_index()
    daily_return = daily["daily_return"].copy()
    nav = (1.0 + daily_return).cumprod()
    nav.name = "nav"
    metrics = compute_metrics(daily_return)
    gross = daily["gross_return"]
    annual_gross_return = float(gross.mean() * 252)
    annual_gross_volatility = float(gross.std() * np.sqrt(252))
    metrics.update(
        {
            "annual_gross_return": annual_gross_return,
            "gross_sharpe": (
                annual_gross_return / annual_gross_volatility
                if annual_gross_volatility > 0
                else np.nan
            ),
            "annual_turnover": float(daily["turnover"].mean() * 252),
            "annual_cost": float(daily["cost"].mean() * 252),
            "cost_share_of_gross_return": (
                float(daily["cost"].mean() * 252) / annual_gross_return
                if annual_gross_return > 0
                else np.nan
            ),
            "annual_active_turnover": float(
                daily["actual_active_turnover"].mean() * 252
            ),
            "annual_roll_turnover": float(
                daily["actual_roll_turnover"].mean() * 252
            ),
            "annual_mandatory_exit_turnover": 0.0,
        }
    )
    if return_positions:
        return nav, metrics, daily_return, daily, pd.DataFrame(position_records)
    return nav, metrics, daily_return, daily


def _prepare_basis_momentum_weights() -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    _, pool_members = five_factor.build_semiannual_pool()
    calendar = load_trade_calendar(LOAD_START_DATE, END_DATE)
    signal_mapping = build_contract_mapping(
        LOAD_START_DATE,
        END_DATE,
        min_days_to_maturity=five_factor.SIGNAL_MIN_DAYS_TO_MATURITY,
    )
    trade_mapping = build_contract_mapping(
        LOAD_START_DATE,
        END_DATE,
        min_days_to_maturity=five_factor.TRADE_MIN_DAYS_TO_MATURITY,
    )
    basis = compute_basis_components(signal_mapping, calendar, lookback=120)
    factor_panel = basis[["trade_date", "fut_code", "factor_AB"]].rename(
        columns={"factor_AB": "raw_factor"}
    )
    factor_panel = factor_panel.loc[
        factor_panel["trade_date"].between(
            pd.Timestamp(START_DATE), pd.Timestamp(END_DATE)
        )
    ]
    strategy_panel = five_factor.build_strategy_panel(
        factor_panel,
        trade_mapping,
        calendar,
        pool_members,
    )
    weights = {
        method: five_factor.inverse_volatility_weights(strategy_panel, method)
        for method in METHODS
    }
    return weights, load_contract_prices(START_DATE, END_DATE)


def _comparison_plot(
    navs: dict[str, dict[str, pd.Series]],
    diagnostics: dict[str, dict[str, pd.DataFrame]],
) -> None:
    five_factor.configure_plot_style()
    figure, axes = plt.subplots(2, 2, figsize=(14, 8), sharex="col")
    colors = {"static_target": "#7A7A7A", "holdings_drift": "#247BA0"}
    labels = {"static_target": "原权重差", "holdings_drift": "持仓漂移后"}
    for column, method in enumerate(METHODS):
        for version in labels:
            axes[0, column].plot(
                navs[version][method].index,
                navs[version][method],
                label=labels[version],
                color=colors[version],
                linewidth=1.8,
            )
            cumulative_cost = diagnostics[version][method]["cost"].cumsum()
            axes[1, column].plot(
                cumulative_cost.index,
                cumulative_cost,
                label=labels[version],
                color=colors[version],
                linewidth=1.8,
            )
        axes[0, column].set_title(METHODS[method])
        axes[0, column].set_ylabel("NAV")
        axes[1, column].set_ylabel("累计成本")
        axes[1, column].set_xlabel("交易日期")
        axes[0, column].legend(frameon=False)
    figure.suptitle("Basis Momentum：权重漂移敏感性", fontsize=17)
    figure.tight_layout()
    figure.savefig(
        OUTPUT_DIR / "basis_momentum_weight_drift_comparison.png",
        bbox_inches="tight",
    )
    plt.close(figure)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    weights_by_method, prices = _prepare_basis_momentum_weights()
    versions = ("static_target", "holdings_drift")
    navs = {version: {} for version in versions}
    diagnostics = {version: {} for version in versions}
    records = []
    daily_output = pd.DataFrame()

    for method, weights in weights_by_method.items():
        baseline_nav, baseline_metrics, _, baseline_daily = (
            run_backtest_from_weights(
                weights,
                prices,
                cost_rate=COST_RATE,
                return_diagnostics=True,
            )
        )
        drift_nav, drift_metrics, _, drift_daily, drift_positions = (
            run_drift_aware_backtest(
                weights,
                prices,
                cost_rate=COST_RATE,
                return_positions=True,
            )
        )
        results = {
            "static_target": (baseline_nav, baseline_metrics, baseline_daily),
            "holdings_drift": (drift_nav, drift_metrics, drift_daily),
        }
        for version, (nav, metrics, daily) in results.items():
            navs[version][method] = nav
            diagnostics[version][method] = daily
            daily_output[f"{version}__{method}_nav"] = nav
            daily_output[f"{version}__{method}_return"] = daily["daily_return"]
            record = {
                "version": version,
                "method": method,
                "method_label": METHODS[method],
                "final_nav": float(nav.iloc[-1]),
            }
            record.update(metrics)
            records.append(record)
            daily.to_csv(
                OUTPUT_DIR / f"{version}__{method}_daily_diagnostics.csv",
                encoding="utf-8-sig",
            )
        drift_positions.to_csv(
            OUTPUT_DIR / f"holdings_drift__{method}_position_diagnostics.csv",
            index=False,
            encoding="utf-8-sig",
        )

    metrics = pd.DataFrame(records)
    metrics.to_csv(
        OUTPUT_DIR / "basis_momentum_weight_drift_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    comparison_columns = [
        "final_nav",
        "annual_return",
        "annual_gross_return",
        "sharpe",
        "max_drawdown",
        "annual_turnover",
        "annual_active_turnover",
        "annual_roll_turnover",
        "annual_cost",
    ]
    comparison_records = []
    for method in METHODS:
        method_metrics = metrics.loc[metrics["method"].eq(method)].set_index(
            "version"
        )
        for metric in comparison_columns:
            baseline = float(method_metrics.loc["static_target", metric])
            drift = float(method_metrics.loc["holdings_drift", metric])
            comparison_records.append(
                {
                    "method": method,
                    "method_label": METHODS[method],
                    "metric": metric,
                    "static_target": baseline,
                    "holdings_drift": drift,
                    "difference": drift - baseline,
                    "relative_difference": (
                        drift / baseline - 1.0
                        if not np.isclose(baseline, 0.0)
                        else np.nan
                    ),
                }
            )
    pd.DataFrame(comparison_records).to_csv(
        OUTPUT_DIR / "basis_momentum_weight_drift_differences.csv",
        index=False,
        encoding="utf-8-sig",
    )
    daily_output.to_csv(
        OUTPUT_DIR / "basis_momentum_weight_drift_daily_nav.csv",
        encoding="utf-8-sig",
    )
    _comparison_plot(navs, diagnostics)
    print(f"Results written to: {OUTPUT_DIR}")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
