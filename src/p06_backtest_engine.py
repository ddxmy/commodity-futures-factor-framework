"""Execution, P&L attribution and metrics for futures factor portfolios."""

import numpy as np
import pandas as pd

from config.settings import COST_RATE, LOOKBACK
from src.p01_market_data import load_contract_prices
from src.p05_portfolio_construction import generate_weights
from src.portfolio.turnover_optimizer import optimize_portfolio


def resolve_executed_positions(targets, prices):
    """根据开盘价可用性，将目标仓位转换为实际成交仓位。"""
    out = targets.sort_values(["fut_code", "trade_date"]).copy()

    open_prices = prices[["trade_date", "ts_code", "open"]].copy()
    duplicate_prices = open_prices.duplicated(["trade_date", "ts_code"]).any()
    if duplicate_prices:
        raise ValueError("开盘价数据存在重复的日期合约")

    open_lookup = open_prices.set_index(["trade_date", "ts_code"])["open"].to_dict()

    def has_open(trade_date, ts_code):
        if ts_code is None or pd.isna(ts_code):
            return False
        open_price = open_lookup.get((trade_date, ts_code), np.nan)
        return pd.notna(open_price) and open_price > 0

    previous_weights = []
    executed_weights = []
    previous_contracts = []
    executed_contracts = []
    blocked_trades = []
    delayed_rolls = []

    for _, group in out.groupby("fut_code", sort=False):
        actual_weight = 0.0
        actual_contract = None

        for _, row in group.iterrows():
            desired_weight = row["desired_exec_weight"]
            desired_weight = 0.0 if pd.isna(desired_weight) else float(desired_weight)
            desired_contract = row["desired_trade_ts_code"]
            desired_contract = None if pd.isna(desired_contract) else desired_contract

            previous_weight = actual_weight
            previous_contract = actual_contract
            had_position = not np.isclose(previous_weight, 0.0)
            wants_position = not np.isclose(desired_weight, 0.0)
            changes_contract = (
                had_position
                and wants_position
                and desired_contract != previous_contract
            )
            changes_weight = not np.isclose(desired_weight, previous_weight)
            needs_trade = changes_weight or changes_contract

            if not needs_trade:
                can_execute = True
            elif had_position and wants_position and changes_contract:
                can_execute = (
                    has_open(row["trade_date"], previous_contract)
                    and has_open(row["trade_date"], desired_contract)
                )
            elif had_position and not wants_position:
                can_execute = has_open(row["trade_date"], previous_contract)
            else:
                can_execute = has_open(row["trade_date"], desired_contract)

            blocked_trade = needs_trade and not can_execute
            delayed_roll = changes_contract and blocked_trade

            if can_execute and needs_trade:
                actual_weight = desired_weight if wants_position else 0.0
                actual_contract = desired_contract if wants_position else None

            previous_weights.append(previous_weight)
            executed_weights.append(actual_weight)
            previous_contracts.append(previous_contract)
            executed_contracts.append(actual_contract)
            blocked_trades.append(blocked_trade)
            delayed_rolls.append(delayed_roll)

    out["prev_exec_weight"] = previous_weights
    out["exec_weight"] = executed_weights
    out["prev_trade_ts_code"] = previous_contracts
    out["trade_ts_code"] = executed_contracts
    out["blocked_trade"] = blocked_trades
    out["delayed_roll"] = delayed_rolls
    return out


def build_optimized_execution_path(
    signals,
    prices,
    turnover_limit=0.15,
    max_abs_weight=0.05,
):
    """Execute prior targets at the open, then optimize at the close."""
    required_signal_columns = {
        "trade_date",
        "fut_code",
        "factor",
        "passes_liquidity",
        "is_rebalance",
        "ts_code_A",
    }
    missing_signal_columns = (
        required_signal_columns - set(signals.columns)
    )
    if missing_signal_columns:
        raise ValueError(
            "signals缺少字段："
            + ", ".join(sorted(missing_signal_columns))
        )

    required_price_columns = {
        "trade_date",
        "ts_code",
        "open",
    }
    missing_price_columns = (
        required_price_columns - set(prices.columns)
    )
    if missing_price_columns:
        raise ValueError(
            "prices缺少字段："
            + ", ".join(sorted(missing_price_columns))
        )

    out = signals.copy()
    out["trade_date"] = pd.to_datetime(out["trade_date"])
    out = out.sort_values(
        ["trade_date", "fut_code"]
    ).reset_index(drop=True)

    if out.duplicated(["trade_date", "fut_code"]).any():
        raise ValueError("信号数据存在重复的日期品种")

    open_prices = prices[
        ["trade_date", "ts_code", "open"]
    ].copy()
    open_prices["trade_date"] = pd.to_datetime(
        open_prices["trade_date"]
    )
    if open_prices.duplicated(
        ["trade_date", "ts_code"]
    ).any():
        raise ValueError("开盘价数据存在重复的日期合约")

    open_lookup = open_prices.set_index(
        ["trade_date", "ts_code"]
    )["open"].to_dict()

    def has_open(trade_date, ts_code):
        if ts_code is None or pd.isna(ts_code):
            return False
        open_price = open_lookup.get(
            (trade_date, ts_code),
            np.nan,
        )
        return pd.notna(open_price) and open_price > 0

    fut_codes = out["fut_code"].drop_duplicates().tolist()

    actual_weights = {
        fut_code: 0.0 for fut_code in fut_codes
    }
    actual_contracts = {
        fut_code: None for fut_code in fut_codes
    }

    # 收盘形成的目标仓位，留到下一交易日开盘尝试执行。
    pending_weights = {
        fut_code: 0.0 for fut_code in fut_codes
    }
    pending_contracts = {
        fut_code: None for fut_code in fut_codes
    }
    pending_mandatory_exits = {
        fut_code: False for fut_code in fut_codes
    }

    execution_records = []
    optimizer_records = []
    portfolio_initialized = False

    for trade_date, day in out.groupby(
        "trade_date",
        sort=True,
    ):
        day = day.set_index("fut_code", drop=False)

        if day["is_rebalance"].isna().any():
            raise ValueError("is_rebalance不能包含缺失值")
        if day["is_rebalance"].nunique() != 1:
            raise ValueError("同一天的is_rebalance必须一致")

        # 第一步：开盘执行上一交易日收盘后形成的目标仓位。
        for fut_code, row in day.iterrows():
            desired_weight = float(
                pending_weights[fut_code]
            )
            desired_contract = pending_contracts[fut_code]

            previous_weight = actual_weights[fut_code]
            previous_contract = actual_contracts[fut_code]

            had_position = not np.isclose(
                previous_weight,
                0.0,
            )
            wants_position = not np.isclose(
                desired_weight,
                0.0,
            )
            changes_contract = (
                had_position
                and wants_position
                and desired_contract != previous_contract
            )
            changes_weight = not np.isclose(
                desired_weight,
                previous_weight,
            )
            needs_trade = changes_weight or changes_contract

            if not needs_trade:
                can_execute = True
            elif changes_contract:
                can_execute = (
                    has_open(trade_date, previous_contract)
                    and has_open(trade_date, desired_contract)
                )
            elif had_position and not wants_position:
                can_execute = has_open(
                    trade_date,
                    previous_contract,
                )
            else:
                can_execute = has_open(
                    trade_date,
                    desired_contract,
                )

            blocked_trade = needs_trade and not can_execute
            delayed_roll = changes_contract and blocked_trade

            if needs_trade and can_execute:
                actual_weights[fut_code] = (
                    desired_weight if wants_position else 0.0
                )
                actual_contracts[fut_code] = (
                    desired_contract if wants_position else None
                )

            record = row.to_dict()
            record.update(
                {
                    "desired_exec_weight": desired_weight,
                    "desired_trade_ts_code": desired_contract,
                    "prev_exec_weight": previous_weight,
                    "exec_weight": actual_weights[fut_code],
                    "prev_trade_ts_code": previous_contract,
                    "trade_ts_code": actual_contracts[fut_code],
                    "blocked_trade": blocked_trade,
                    "delayed_roll": delayed_roll,
                    "mandatory_exit_requested": (
                        pending_mandatory_exits[fut_code]
                    ),
                }
            )
            execution_records.append(record)

        if sum(abs(weight) for weight in actual_weights.values()) > 1e-10:
            portfolio_initialized = True

        # 第二步：收盘读取当日信号和真实仓位，形成次日目标。
        if bool(day["is_rebalance"].iloc[0]):
            eligible = (
                day["passes_liquidity"].fillna(False)
                & day["factor"].notna()
                & day["ts_code_A"].notna()
            ).astype(bool)

            scores = (
                day["factor"]
                .where(eligible, 0.0)
                .fillna(0.0)
                .astype(float)
            )
            previous_weights = pd.Series(
                {
                    fut_code: actual_weights[fut_code]
                    for fut_code in day.index
                },
                index=day.index,
                dtype=float,
            )
            contract_changed = pd.Series(
                {
                    fut_code: (
                        not np.isclose(
                            actual_weights[fut_code],
                            0.0,
                        )
                        and actual_contracts[fut_code] is not None
                        and pd.notna(day.loc[fut_code, "ts_code_A"])
                        and actual_contracts[fut_code]
                        != day.loc[fut_code, "ts_code_A"]
                    )
                    for fut_code in day.index
                },
                index=day.index,
                dtype=bool,
            )

            optimized_weights, diagnostics = optimize_portfolio(
                scores=scores,
                previous_weights=previous_weights,
                eligible=eligible,
                contract_changed=contract_changed,
                turnover_limit=turnover_limit,
                max_abs_weight=max_abs_weight,
                is_initial=not portfolio_initialized,
            )

            for fut_code in day.index:
                target_weight = float(
                    optimized_weights.loc[fut_code]
                )
                has_target = not np.isclose(
                    target_weight,
                    0.0,
                )

                pending_weights[fut_code] = (
                    target_weight if has_target else 0.0
                )
                pending_contracts[fut_code] = (
                    day.loc[fut_code, "ts_code_A"]
                    if has_target
                    else None
                )
                pending_mandatory_exits[fut_code] = bool(
                    not np.isclose(
                        previous_weights.loc[fut_code],
                        0.0,
                    )
                    and not eligible.loc[fut_code]
                )

            diagnostics = diagnostics.copy()
            diagnostics["signal_date"] = trade_date
            optimizer_records.append(diagnostics)

    execution_path = pd.DataFrame(execution_records)
    execution_path = execution_path.sort_values(
        ["fut_code", "trade_date"]
    ).reset_index(drop=True)

    optimizer_diagnostics = pd.DataFrame(optimizer_records)
    if not optimizer_diagnostics.empty:
        optimizer_diagnostics = (
            optimizer_diagnostics
            .set_index("signal_date")
            .sort_index()
        )

    return execution_path, optimizer_diagnostics


def compute_metrics(daily_pnl):
    """Calculate annualized return, risk, drawdown and tail-risk metrics."""
    annual_return = daily_pnl.mean() * 252
    annual_volatility = daily_pnl.std() * np.sqrt(252)
    sharpe = annual_return / annual_volatility if annual_volatility > 0 else np.nan
    cum = (1 + daily_pnl).cumprod()
    max_dd = (cum / cum.cummax() - 1).min()
    calmar = annual_return / abs(max_dd) if max_dd < 0 else np.nan
    downside = daily_pnl[daily_pnl < 0]
    downside_volatility = downside.std() * np.sqrt(252)
    sortino = annual_return / downside_volatility if downside_volatility > 0 else np.nan
    win_rate = (daily_pnl > 0).mean()
    avg_win = daily_pnl[daily_pnl > 0].mean()
    avg_loss = daily_pnl[daily_pnl < 0].mean()
    profit_loss_ratio = avg_win / abs(avg_loss) if avg_loss != 0 else np.nan
    VaR = daily_pnl.quantile(0.05)
    CVaR = daily_pnl[daily_pnl <= VaR].mean()

    return {
        'annual_return': annual_return,
        'annual_volatility': annual_volatility,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'calmar': calmar,
        'sortino': sortino,
        'win_rate': win_rate,
        'profit_loss_ratio': profit_loss_ratio,
        'VaR_95': VaR,
        'CVaR_95': CVaR,
    }


def classify_actual_turnover(position_results):
    """将实际成交换手拆成主动调仓、换月和流动性退出。"""
    out = position_results.copy()

    if "turnover" not in out.columns:
        raise ValueError("position_results缺少turnover字段")

    mandatory_exit = out.get(
        "mandatory_exit_requested",
        pd.Series(False, index=out.index),
    ).fillna(False).astype(bool)

    true_roll = (
        out["prev_trade_ts_code"].notna()
        & out["trade_ts_code"].notna()
        & out["trade_ts_code"].ne(
            out["prev_trade_ts_code"]
        )
    )

    out["actual_total_turnover"] = out["turnover"]
    out["actual_mandatory_exit_turnover"] = out[
        "turnover"
    ].where(mandatory_exit, 0.0)
    out["actual_roll_turnover"] = out[
        "turnover"
    ].where(~mandatory_exit & true_roll, 0.0)
    out["actual_active_turnover"] = (
        out["actual_total_turnover"]
        - out["actual_mandatory_exit_turnover"]
        - out["actual_roll_turnover"]
    )

    return out


def aggregate_daily_diagnostics(position_results):
    """汇总逐品种毛收益、换手和成本，得到每日组合结果。"""
    value_columns = ["gross_pnl", "turnover", "cost"]
    turnover_columns = [
        "actual_active_turnover",
        "actual_roll_turnover",
        "actual_mandatory_exit_turnover",
        "actual_total_turnover",
    ]
    value_columns.extend(
        column
        for column in turnover_columns
        if column in position_results.columns
    )
    for column in ["blocked_trade", "delayed_roll"]:
        if column in position_results.columns:
            value_columns.append(column)

    daily = (
        position_results.groupby("trade_date")
        [value_columns]
        .sum()
        .rename(columns={"gross_pnl": "gross_return"})
    )

    daily["daily_return"] = daily["gross_return"] - daily["cost"]
    return daily


def run_backtest(
    start_date,
    end_date,
    factor_type='AB',
    rebalance_freq=5,
    lookback=LOOKBACK,
    normalize='rank',
    zscore_clip=None,
    use_optimizer=False,
    turnover_limit=0.15,
    max_abs_weight=0.05,
    return_diagnostics=False,
    cost_rate=COST_RATE,
    prepared_weights=None,
    prepared_prices=None,
    _weight_loader=None,
    _price_loader=None,
):
    """Run the original next-open execution and three-part P&L model.

    By default the function retains the legacy convenience API and loads
    basis-momentum weights internally. ``prepared_weights`` and
    ``prepared_prices`` let the shared pipeline reuse data already calculated
    for another factor without changing any execution or cost convention.
    """
    if use_optimizer and normalize != "rank":
        raise ValueError("组合优化器当前只支持rank因子")

    if cost_rate < 0:
        raise ValueError("cost_rate cannot be negative")

    if prepared_weights is None:
        weight_loader = _weight_loader or generate_weights
        df = weight_loader(
            start_date=start_date,
            end_date=end_date,
            factor_type=factor_type,
            lookback=lookback,
            normalize=normalize,
            rebalance_freq=rebalance_freq,
            zscore_clip=zscore_clip,
        ).copy()
    else:
        df = prepared_weights.copy()
    
    df = df.sort_values(['fut_code', 'trade_date'])

    if prepared_prices is None:
        price_loader = _price_loader or load_contract_prices
        prices = price_loader(start_date, end_date)
    else:
        prices = prepared_prices.copy()

    optimizer_diagnostics = pd.DataFrame()
    if use_optimizer:
        df, optimizer_diagnostics = build_optimized_execution_path(
            signals=df,
            prices=prices,
            turnover_limit=turnover_limit,
            max_abs_weight=max_abs_weight,
        )
    else:
        # ----- 1. 信号 → 次日计划仓位 -----
        df['target_weight'] = df['weight'].where(
            df['is_rebalance'],
            np.nan,
        )
        df['desired_exec_weight'] = (
            df.groupby('fut_code')['target_weight'].shift(1)
        )
        df['desired_exec_weight'] = (
            df.groupby('fut_code')['desired_exec_weight']
            .ffill()
            .fillna(0.0)
        )

        # 锁定交易合约：调仓日确定，持仓期不变。
        df['target_ts_code'] = df['ts_code_A'].where(
            df['is_rebalance'],
            np.nan,
        )
        df['desired_trade_ts_code'] = (
            df.groupby('fut_code')['target_ts_code'].shift(1)
        )
        df['desired_trade_ts_code'] = (
            df.groupby('fut_code')['desired_trade_ts_code'].ffill()
        )

        # 开盘价缺失时不假设成交，实际仓位和合约继续沿用上一日状态。
        df = resolve_executed_positions(df, prices)

    trade_prices = prices.copy().rename(
        columns={
            'ts_code': 'trade_ts_code',
            'open': 'trade_open',
            'close': 'trade_close',
            'prev_close': 'trade_prev_close',
        }
    )

    trade_prices = trade_prices[
        [
            "trade_date",
            "trade_ts_code",
            "trade_open",
            "trade_close",
            "trade_prev_close",
        ]
    ]

    rows_before = len(df)

    df = df.merge(
        trade_prices,
        on=["trade_date", "trade_ts_code"],
        how="left",
        validate='many_to_one',
    )

    if len(df) != rows_before:
        raise ValueError("merging trade prices changed the number of rows")

    old_prices = prices.copy().rename(
        columns={
            'ts_code': 'prev_trade_ts_code',
            'open': 'old_open',
            'close': 'old_close',
            'prev_close': 'old_prev_close',
        }
    )

    old_prices = old_prices[
        [
            "trade_date",
            "prev_trade_ts_code",
            "old_open",
            "old_close",
            "old_prev_close",
        ]
    ]

    rows_before = len(df)

    df = df.merge(
        old_prices,
        on=['trade_date', 'prev_trade_ts_code'],
        how='left',
        validate='many_to_one',
    )

    if len(df) != rows_before:
        raise ValueError("merging previous-contract prices changed the number of rows")

    df = df.sort_values(
        ["fut_code", "trade_date"]
    ).reset_index(drop=True)    

    df['trade_daily_return'] = (
        df['trade_close'] / df['trade_prev_close'] - 1
    )

    df['trade_intraday_return'] = (
        df['trade_close'] / df['trade_open'] - 1
    )

    df['trade_overnight_return'] = (
        df['trade_open'] / df['trade_prev_close'] - 1
    )

    df['old_overnight_return'] = (
        df['old_open'] / df['old_prev_close'] - 1
    )

    # ----- 2. 三分法拆分（方向感知） -----
    exec_sign = np.sign(df['exec_weight'])
    prev_sign = np.sign(df['prev_exec_weight'])
    exec_abs  = df['exec_weight'].abs()
    prev_abs  = df['prev_exec_weight'].abs()

    same_sign = (exec_sign == prev_sign)
    cont_abs = np.where(same_sign, np.minimum(exec_abs, prev_abs), 0.0)

    df['continuation'] = cont_abs * exec_sign
    df['new_entry']    = (exec_abs - cont_abs) * exec_sign
    df['new_exit']     = (prev_abs - cont_abs) * prev_sign

    # ----- 3. 换月日覆盖（延续仓必须为0，全平旧 + 全开新） -----
    contract_changed = (
        df["prev_trade_ts_code"].notna()
        & df["trade_ts_code"].ne(df["prev_trade_ts_code"])
    )
    df.loc[contract_changed, 'continuation'] = 0.0
    df.loc[contract_changed, 'new_entry']    = df.loc[contract_changed, 'exec_weight']
    df.loc[contract_changed, 'new_exit']     = df.loc[contract_changed, 'prev_exec_weight']

    missing_continuation = (
        df["continuation"].ne(0)
        & df["trade_daily_return"].isna()
    )

    missing_entry = (
        df["new_entry"].ne(0)
        & df["trade_intraday_return"].isna()
    )

    missing_exit = (
        df["new_exit"].ne(0)
        & df["old_overnight_return"].isna()
    )

    missing_return_counts = {
        "continuation": int(missing_continuation.sum()),
        "entry": int(missing_entry.sum()),
        "exit": int(missing_exit.sum()),
    }
    if any(missing_return_counts.values()):
        raise ValueError(
            "position returns are missing: "
            + ", ".join(
                f"{name}={count}"
                for name, count in missing_return_counts.items()
            )
        )

    # ----- 4. 盈亏归因 -----
    df['continuation_pnl'] = np.where(
        df['continuation'].ne(0),
        df['continuation'] * df['trade_daily_return'],
        0.0,
    )

    df['entry_pnl'] = np.where(
        df['new_entry'].ne(0),
        df['new_entry'] * df['trade_intraday_return'],
        0.0,
    )

    df['exit_pnl'] = np.where(
        df['new_exit'].ne(0),
        df['new_exit'] * df['old_overnight_return'],
        0.0,
    )

    df['gross_pnl'] = (
        df['continuation_pnl']+
        df['entry_pnl']+
        df['exit_pnl']
    )

    # ----- 5. 换手与成本（直接用三分法结果）-----
    df['turnover'] = df['new_entry'].abs() + df['new_exit'].abs()
    df['cost'] = df['turnover'] * cost_rate
    df = classify_actual_turnover(df)

    daily_diagnostics = aggregate_daily_diagnostics(df)

    optimizer_columns = {
        "optimized_turnover": "optimizer_budget_turnover",
        "effective_turnover_limit": "optimizer_limit",
        "constraint_binding": "optimizer_constraint_binding",
        "turnover_limit_relaxed": "optimizer_limit_relaxed",
    }
    if not optimizer_diagnostics.empty:
        optimizer_daily = optimizer_diagnostics[
            list(optimizer_columns)
        ].rename(columns=optimizer_columns)
        daily_diagnostics = daily_diagnostics.join(
            optimizer_daily,
            how="left",
        )
    else:
        for column in optimizer_columns.values():
            daily_diagnostics[column] = np.nan

    daily_pnl = daily_diagnostics['daily_return'].copy()
    nav = (1 + daily_pnl).cumprod()
    nav.name = 'nav'

    metrics = compute_metrics(daily_pnl)

    gross_return = daily_diagnostics['gross_return']
    annual_gross_return = gross_return.mean() * 252
    annual_gross_volatility = gross_return.std() * np.sqrt(252)
    gross_sharpe = (
        annual_gross_return / annual_gross_volatility
        if annual_gross_volatility > 0
        else np.nan
    )
    annual_cost = daily_diagnostics['cost'].mean() * 252

    metrics.update(
        {
            'annual_gross_return': annual_gross_return,
            'gross_sharpe': gross_sharpe,
            'annual_turnover': daily_diagnostics['turnover'].mean() * 252,
            'annual_cost': annual_cost,
            'cost_share_of_gross_return': (
                annual_cost / annual_gross_return
                if annual_gross_return > 0
                else np.nan
            ),
            'annual_active_turnover': (
                daily_diagnostics['actual_active_turnover'].mean()
                * 252
            ),
            'annual_roll_turnover': (
                daily_diagnostics['actual_roll_turnover'].mean()
                * 252
            ),
            'annual_mandatory_exit_turnover': (
                daily_diagnostics[
                    'actual_mandatory_exit_turnover'
                ].mean()
                * 252
            ),
            'optimizer_binding_rate': (
                daily_diagnostics[
                    'optimizer_constraint_binding'
                ].dropna().mean()
                if daily_diagnostics[
                    'optimizer_constraint_binding'
                ].notna().any()
                else np.nan
            ),
            'optimizer_relaxation_count': int(
                daily_diagnostics[
                    'optimizer_limit_relaxed'
                ].eq(True).sum()
            ),
        }
    )

    if return_diagnostics:
        return nav, metrics, daily_pnl, daily_diagnostics

    return nav, metrics, daily_pnl


def run_backtest_from_weights(
    weights: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    cost_rate: float = COST_RATE,
    use_optimizer: bool = False,
    turnover_limit: float = 0.15,
    max_abs_weight: float = 0.05,
    return_diagnostics: bool = False,
):
    """Run the existing execution model from already prepared target weights."""
    if weights.empty:
        raise ValueError("weights cannot be empty")
    if prices.empty:
        raise ValueError("prices cannot be empty")
    dates = pd.to_datetime(weights["trade_date"])
    start_date = dates.min().strftime("%Y%m%d")
    end_date = dates.max().strftime("%Y%m%d")
    return run_backtest(
        start_date=start_date,
        end_date=end_date,
        normalize="rank",
        use_optimizer=use_optimizer,
        turnover_limit=turnover_limit,
        max_abs_weight=max_abs_weight,
        return_diagnostics=return_diagnostics,
        cost_rate=cost_rate,
        prepared_weights=weights,
        prepared_prices=prices,
    )
