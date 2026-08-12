"""横截面因子评价：日期对齐、IC、分组收益与汇总指标。"""

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

from config.research_config import ResearchConfig
from src.p01_market_data import (
    load_contract_prices,
    load_trade_calendar,
)
from src.p05_portfolio_construction import build_target_weights, generate_weights


def build_execution_schedule(trade_dates, signal_dates) -> pd.DataFrame:
    """把信号日映射为下一交易日开仓和下一期执行日平仓。"""
    dates = (
        pd.DatetimeIndex(pd.to_datetime(trade_dates))
        .drop_duplicates()
        .sort_values()
    )
    signals = (
        pd.DatetimeIndex(pd.to_datetime(signal_dates))
        .drop_duplicates()
        .sort_values()
    )

    schedule = pd.DataFrame({"signal_date": signals})

    # 每个信号映射到严格晚于信号日的第一个交易日开盘。
    entry_dates = [dates[dates > signal_date].min() for signal_date in signals]
    schedule["entry_date"] = entry_dates

    # 本期平仓日等于下一期的开仓日，确保各期首尾衔接。
    schedule["exit_date"] = schedule['entry_date'].shift(-1)

    return (
        schedule.dropna(subset=["entry_date", "exit_date"])
        .reset_index(drop=True)
    )


def attach_locked_forward_returns(
    signals,
    schedule,
    prices,
    drop_missing=False,
) -> pd.DataFrame:
    """使用信号日锁定的合约计算开仓至平仓的收益率。"""
    df = signals.merge(
        schedule,
        how='inner',
        on='signal_date',
        validate='many_to_one',
    )

    entry_prices = prices.rename(
        columns={
            'trade_date': 'entry_date',
            'ts_code': 'trade_ts_code',
            'open': 'entry_open',
        }
    )

    entry_prices = entry_prices[
        ['entry_date', 'trade_ts_code', 'entry_open']
    ]

    df = df.merge(
        entry_prices,
        on=['entry_date', 'trade_ts_code'],
        how='left',
        validate='many_to_one',
    )

    exit_prices = prices.rename(
        columns={
            'trade_date': 'exit_date',
            'ts_code': 'trade_ts_code',
            'open': 'exit_open',
        }
    )

    exit_prices = exit_prices[
        ['exit_date', 'trade_ts_code', 'exit_open']
    ]

    df = df.merge(
        exit_prices,
        on=['exit_date', 'trade_ts_code'],
        how='left',
        validate='many_to_one',
    )

    missing_prices = df[
        ['entry_open', 'exit_open']
    ].isna().any(axis=1)

    invalid_prices = (
        df[['entry_open', 'exit_open']] <= 0
    ).any(axis=1)

    unusable_prices = (
        missing_prices | invalid_prices
    )

    if drop_missing:
        df = df.loc[
            ~unusable_prices,
        ].copy()
    else:
        if missing_prices.any():
            raise ValueError("some positions are missing entry or exit prices")
        if invalid_prices.any():
            raise ValueError("entry and exit prices must be positive")

    df['forward_return'] = (
        df['exit_open'] / df['entry_open'] - 1
    )

    return df.reset_index(drop=True)


def build_factor_test_panel(
    start_date,
    end_date,
    factor_type,
    lookback,
    rebalance_freq=1,
    min_assets=10,
    prepared_weights=None,
    prepared_calendar=None,
    prepared_prices=None,
    _weight_loader=None,
    _calendar_loader=None,
    _price_loader=None,
) -> pd.DataFrame:
    """构建横截面因子值与下一持有期收益的对齐面板。"""
    if (
        not isinstance(rebalance_freq, int)
        or not 1 <= rebalance_freq <= 20
    ):
        raise ValueError(
            "rebalance_freq必须是1到20之间的整数"
        )

    if min_assets < 2:
        raise ValueError(
            "min_assets至少为2"
        )

    if prepared_weights is None:
        weight_loader = _weight_loader or generate_weights
        df = weight_loader(
            start_date=start_date,
            end_date=end_date,
            factor_type=factor_type,
            lookback=lookback,
            normalize="rank",
            rebalance_freq=rebalance_freq,
        ).copy()
    else:
        df = prepared_weights.copy()

    eligible = (
        df['is_rebalance']
        & df['weight_factor'].notna()
    )

    panel = df.loc[
        eligible,
        [
            'trade_date',
            'fut_code',
            'weight_factor',
            'ts_code_A',
        ]
    ].copy()

    panel = panel.rename(
        columns={
            'trade_date': 'signal_date',
            'weight_factor': 'raw_factor',
            'ts_code_A': 'trade_ts_code',
        }
    )

    has_duplicate_asset = panel.duplicated(
        subset=['signal_date', 'fut_code']
    ).any()

    if has_duplicate_asset:
        raise ValueError("factor data contains duplicate date-commodity keys")

    panel['asset_count'] = (
        panel.groupby('signal_date')['fut_code']
        .transform('count')
    )

    panel = panel[
        panel['asset_count'] >= min_assets
    ].copy()

    if prepared_calendar is None:
        calendar_loader = _calendar_loader or load_trade_calendar
        calendar = calendar_loader(start_date, end_date)
    else:
        calendar = prepared_calendar.copy()
    schedule = build_execution_schedule(
        trade_dates=calendar['trade_date'],
        signal_dates=panel['signal_date'],
    )

    if prepared_prices is None:
        price_loader = _price_loader or load_contract_prices
        prices = price_loader(start_date, end_date)
    else:
        prices = prepared_prices.copy()

    panel = attach_locked_forward_returns(
        signals=panel,
        schedule=schedule,
        prices=prices,
        drop_missing=True,
    )

    panel['asset_count'] = (
        panel.groupby('signal_date')['fut_code']
        .transform('count')
    )

    panel = panel[
        panel['asset_count'] >= min_assets
    ].copy()

    panel = panel.sort_values(
        ['signal_date', 'fut_code']
    ).reset_index(drop=True)

    return panel


def build_factor_test_panel_from_data(
    factor_data: pd.DataFrame,
    contract_data: pd.DataFrame,
    trade_calendar: pd.DataFrame,
    prices: pd.DataFrame,
    config: ResearchConfig,
) -> pd.DataFrame:
    """Build the evaluation panel from one already calculated factor panel."""
    weights = build_target_weights(
        factor_data=factor_data,
        contract_data=contract_data,
        config=config,
        normalize="rank",
    )
    dates = pd.to_datetime(weights["trade_date"])
    return build_factor_test_panel(
        start_date=dates.min().strftime("%Y%m%d"),
        end_date=dates.max().strftime("%Y%m%d"),
        factor_type="prepared",
        lookback=1,
        rebalance_freq=config.rebalance_freq,
        min_assets=config.min_assets,
        prepared_weights=weights,
        prepared_calendar=trade_calendar,
        prepared_prices=prices,
    )


def calculate_ic_series(panel) -> pd.DataFrame:
    """按信号日计算 Pearson IC 与 Spearman Rank IC。"""
    records = []

    for signal_date, group in panel.groupby('signal_date'):
        valid = group[
            ['raw_factor', 'forward_return']
        ].dropna()

        has_variation = (
            len(valid) >= 2
            and valid['raw_factor'].nunique() > 1
            and valid['forward_return'].nunique() > 1
        )

        if has_variation:
            ic = valid['raw_factor'].corr(
                valid['forward_return'],
                method='pearson',
            )
            rank_ic = valid['raw_factor'].corr(
                valid['forward_return'],
                method='spearman',
            )
        else:
            ic = np.nan
            rank_ic = np.nan

        records.append(
            {
                'signal_date': signal_date,
                'asset_count': len(valid),
                'ic': ic,
                'rank_ic': rank_ic,
            }
        )

    result = pd.DataFrame(records)

    return (
        result.sort_values('signal_date')
        .reset_index(drop=True)
    )


def summarize_ic_statistics(
    ic_series,
    nw_lags=5,
) -> pd.DataFrame:
    """汇总 IC 与 Rank IC 的统计量和显著性检验。"""
    if (
        not isinstance(nw_lags, int)
        or nw_lags < 0
    ):
        raise ValueError(
            "nw_lags必须是大于等于0的整数"
        )
    records = []

    for metric in ['ic', 'rank_ic']:
        values = (
            ic_series[metric]
            .dropna()
            .astype(float)
        )

        mean_value = values.mean()
        std_value = values.std(ddof=1)
        effective_lags = min(
            nw_lags,
            max(len(values) - 1, 0),
        )

        if (
            len(values) >= 2
            and std_value > 0
        ):
            icir_raw = mean_value / std_value
            icir_annualized = icir_raw * np.sqrt(252)
            t_test = stats.ttest_1samp(
                values,
                popmean=0.0,
            )

            t_stat = float(t_test.statistic)
            p_value = float(t_test.pvalue)

            constant = np.ones(
                (len(values), 1)
            )

            nw_model = sm.OLS(
                values.to_numpy(),
                constant,
            ).fit(
                cov_type="HAC",
                cov_kwds={
                    'maxlags': effective_lags,
                }
            )

            nw_t_stat = float(
                nw_model.tvalues[0]
            )
            nw_p_value = float(
                nw_model.pvalues[0]
            )

        else:
            icir_raw = np.nan
            icir_annualized = np.nan
            t_stat = np.nan
            p_value = np.nan
            nw_t_stat = np.nan
            nw_p_value = np.nan

        records.append(
            {
                'metric': metric,
                'observations': len(values),
                'mean': mean_value,
                'std': std_value,
                'positive_rate': (values > 0).mean(),
                'minimum': values.min(),
                'maximum': values.max(),
                'icir_raw': icir_raw,
                'icir_annualized': icir_annualized,
                't_stat': t_stat,
                'p_value': p_value,
                'nw_t_stat': nw_t_stat,
                'nw_p_value': nw_p_value,
                'nw_lags': effective_lags,
            }
        )

    return pd.DataFrame(records)

def assign_five_groups(
    panel,
    group_count=5,
) -> pd.DataFrame:
    """按信号日将商品从高因子到低因子分为等数量组。"""
    if group_count < 2:
        raise ValueError(
            "group_count 必须大于等于2"
        )

    grouped = panel.sort_values(
        ['signal_date', 'raw_factor', 'fut_code'],
        ascending=[True, False, True],
    ).copy()

    position = (
        grouped.groupby('signal_date')
        .cumcount()
    )

    asset_count = (
        grouped.groupby('signal_date')['fut_code']
        .transform('count')
    )

    has_too_few_assets = (
        asset_count < group_count
    ).any()

    if has_too_few_assets:
        raise ValueError(
            '部分信号日的商品数少于分组数'
        )

    grouped['group'] = (
        position * group_count
        // asset_count
        + 1
    )

    return grouped.reset_index(drop=True)


def calculate_group_returns(grouped_panel) -> pd.DataFrame:
    """计算每日五组等权收益及多空组合收益。"""
    group_returns = (
        grouped_panel.groupby(
            ['signal_date', 'group']
        )['forward_return']
        .mean()
        .unstack('group')
        .rename(
            columns=lambda group: f'G{int(group)}'
        )
        .reset_index()
    )

    group_returns['spread_raw'] = (
        group_returns['G1']
        - group_returns['G5']
    )

    group_returns['long_g1'] = (
        0.5 * group_returns['G1']
    )

    group_returns['short_g5'] = (
        -0.5 * group_returns['G5']
    )

    group_returns['long_short'] = (
        group_returns['long_g1']
        + group_returns['short_g5']
    )

    return group_returns


def calculate_group_nav(group_returns) -> pd.DataFrame:
    """将五组及多空组合收益复利为累计净值。"""
    return_columns = [
        'G1', 'G2', 'G3', 'G4', 'G5',
        'spread_raw', 'long_g1', 'short_g5', 'long_short',
    ]

    ordered_returns = (
        group_returns.sort_values('signal_date')
        .reset_index(drop=True)
    )

    nav = ordered_returns[
        ['signal_date']
    ].copy()

    nav[return_columns] = (
        1.0 + ordered_returns[return_columns]
        ).cumprod()

    return nav


def summarize_factor_periods(
    ic_series,
    group_returns,
    periods,
    nw_lags=5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按命名时期汇总 IC 统计和各组合绩效。"""
    portfolio_columns = [
        'G1', 'G2', 'G3', 'G4', 'G5',
        'spread_raw', 'long_g1', 'short_g5', 'long_short',
    ]

    ic_summaries = []
    performance_records = []

    for period, date_range in periods.items():
        start_date = pd.Timestamp(date_range[0])
        end_date = pd.Timestamp(date_range[1])

        ic_mask = ic_series['signal_date'].between(
            start_date,
            end_date,
            inclusive='both',
        )

        period_ic = ic_series.loc[
            ic_mask
        ].copy()

        period_ic_summary = summarize_ic_statistics(
            period_ic, nw_lags=nw_lags
        )

        period_ic_summary.insert(
            0,
            'period',
            period,
        )

        ic_summaries.append(period_ic_summary)

        return_mask = group_returns[
            'signal_date'
        ].between(
            start_date,
            end_date,
            inclusive='both',
        )

        period_returns = (
            group_returns.loc[return_mask]
            .sort_values('signal_date')
            .copy()
        )

        for portfolio in portfolio_columns:
            values = (
                period_returns[portfolio]
                .dropna()
                .astype(float)
            )

            observations = len(values)

            if observations > 0:
                nav = (
                    1.0 + values
                ).cumprod()

                total_return = (
                    nav.iloc[-1] - 1.0
                )

                if nav.iloc[-1] > 0:
                    annual_return = (
                        nav.iloc[-1]
                        ** (252 / observations)
                        - 1.0
                    )
                else:
                    annual_return = np.nan

                if observations >= 2:
                    return_std = values.std(ddof=1)
                    annual_volatility = return_std * np.sqrt(252)
                    if return_std > 0:
                        sharpe = values.mean() / return_std * np.sqrt(252)
                    else:
                        sharpe = np.nan
                else:
                    annual_volatility = np.nan
                    sharpe = np.nan

                wealth = pd.concat(
                    [
                        pd.Series([1.0]),
                        nav.reset_index(drop=True),
                    ],
                    ignore_index=True,
                )

                running_peak = wealth.cummax()

                drawdown = (
                    wealth / running_peak
                    - 1.0
                )

                max_drawdown = drawdown.min()

            else:
                total_return = np.nan
                annual_return = np.nan
                annual_volatility = np.nan
                sharpe = np.nan
                max_drawdown = np.nan

            performance_records.append(
                {
                    'period': period,
                    'portfolio': portfolio,
                    'observations': observations,
                    'total_return': total_return,
                    'annual_return': annual_return,
                    'annual_volatility': annual_volatility,
                    'sharpe': sharpe,
                    'max_drawdown': max_drawdown,
                }
            )

    ic_summary = pd.concat(
        ic_summaries,
        ignore_index=True,
    )

    performance_summary = pd.DataFrame(
        performance_records
    )

    return ic_summary, performance_summary
