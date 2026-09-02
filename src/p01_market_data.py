"""Read and prepare daily futures data from the shared SQLite database."""

from contextlib import closing
import sqlite3

import pandas as pd

from config.settings import DB_PATH, LIQ_LOOKBACK, MIN_DAYS_TO_MATURITY


def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    """Open a SQLite connection to the configured futures database."""
    return sqlite3.connect(db_path)


def validate_min_days_to_maturity(
    value: int,
    name: str = "min_days_to_maturity",
) -> int:
    """Return one validated non-negative calendar-day cutoff."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def load_contract_daily(
    start_date: str,
    end_date: str,
    db_path: str = DB_PATH,
    min_days_to_maturity: int = MIN_DAYS_TO_MATURITY,
) -> pd.DataFrame:
    """Return eligible contracts and rolling liquidity measures by date."""
    min_days_to_maturity = validate_min_days_to_maturity(
        min_days_to_maturity
    )

    query = """
    SELECT
        u.trade_date,
        u.fut_code,
        b.ts_code,
        d.close,
        d.open,
        d.vol,
        d.oi,
        d.amount,
        b.delist_date
    FROM tradable_universe u
    JOIN fut_basic b
      ON b.exchange = u.exchange
     AND b.fut_code = u.fut_code
    JOIN fut_daily d
      ON d.ts_code = b.ts_code
     AND d.trade_date = u.trade_date
    WHERE u.is_tradable = 1
      AND u.trade_date BETWEEN ? AND ?
      AND b.list_date <= u.trade_date
      AND b.delist_date > u.trade_date
      AND d.open IS NOT NULL
      AND d.open > 0
      AND d.close IS NOT NULL
      AND d.close > 0
    ORDER BY u.trade_date, u.fut_code, d.vol DESC
    """
    with closing(get_connection(db_path)) as conn:
        df = pd.read_sql_query(query, conn, params=(start_date, end_date))

    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df['delist_date'] = pd.to_datetime(
        df['delist_date'],
        errors='coerce',
        )
    df['days_to_maturity'] = (
        df['delist_date'] - df['trade_date']
        ).dt.days

    df = df[
        df['days_to_maturity'] >= min_days_to_maturity
    ].copy()

    calendar = load_trade_calendar(start_date, end_date, db_path=db_path).copy()

    calendar['trade_index'] = range(len(calendar))

    df = df.merge(
        calendar,
        on='trade_date',
        how='left',
        validate='many_to_one',
    )

    if not df["trade_index"].notna().all():
        raise ValueError("some futures prices cannot be matched to the trade calendar")

    df = df.sort_values(
        ['ts_code', 'trade_date']
    )

    duplicate_contract_date = df.duplicated(
        ['ts_code', 'trade_date']
    ).sum()

    if duplicate_contract_date != 0:
        raise ValueError("a fixed contract contains duplicate trading dates")

    df["avg_vol"] = (
        df.groupby("ts_code")["vol"]
        .transform(
            lambda values: values.rolling(
                window=LIQ_LOOKBACK,
                min_periods=LIQ_LOOKBACK,
            ).mean()
        )
    )

    df["avg_oi"] = (
        df.groupby("ts_code")["oi"]
        .transform(
            lambda values: values.rolling(
                window=LIQ_LOOKBACK,
                min_periods=LIQ_LOOKBACK,
            ).mean()
        )
    )

    df["avg_amount"] = (
        df.groupby("ts_code")["amount"]
        .transform(
            lambda values: values.rolling(
                window=LIQ_LOOKBACK,
                min_periods=LIQ_LOOKBACK,
            ).mean()
        )
    )

    df['window_start_index'] = (
        df.groupby('ts_code')['trade_index']
        .shift(LIQ_LOOKBACK - 1)
    )

    df['has_complete_liquidity_window'] = (
        df['trade_index']
        - df['window_start_index']
        == LIQ_LOOKBACK - 1
    )

    liquidity_columns = [
        'avg_vol',
        'avg_oi',
        'avg_amount',
    ]

    df.loc[
        ~df['has_complete_liquidity_window'],
        liquidity_columns,
    ] = pd.NA

    df = df[(df['vol'] > 0) & (df['oi'] > 0)].copy()

    df = df.sort_values(
        ['trade_date', 'fut_code', 'vol', 'oi', 'ts_code'],
        ascending=[True, True, False, False, True],
    )

    df['rank_by_vol'] = (
        df.groupby(['trade_date', 'fut_code'], sort=False)
        .cumcount()
        .add(1)
    )
    
    return df.set_index("trade_date").sort_index()


def load_contract_prices(
    start_date: str,
    end_date: str,
    db_path: str = DB_PATH,
) -> pd.DataFrame:
    """
    读取固定期货合约的开盘价和收盘价。

    返回字段：
    trade_date、ts_code、open、close、prev_close。
    """
    requested_start = pd.to_datetime(start_date)

    # 提前读取30个自然日，用于给回测第一天寻找上一交易日收盘价。
    load_start = (
        requested_start - pd.Timedelta(days=30)
    ).strftime('%Y%m%d')

    query = """
    SELECT
        trade_date,
        ts_code,
        open,
        close
    FROM fut_daily
    WHERE trade_date BETWEEN ? AND ?
    ORDER BY ts_code, trade_date
    """

    with closing(get_connection(db_path)) as conn:
        prices = pd.read_sql_query(
            query,
            conn,
            params=(load_start, end_date),
        )

    prices['trade_date'] = pd.to_datetime(prices['trade_date'])

    prices = prices.sort_values(
        ['ts_code', 'trade_date']
    )

    prices['prev_close'] = (
        prices.groupby('ts_code')['close']
        .shift(1)
    )

    prices = prices[
        prices['trade_date'] >= requested_start
    ].copy()

    return prices


def load_spot_daily(
    start_date: str,
    end_date: str,
    db_path: str = DB_PATH,
) -> pd.DataFrame:
    """Return raw daily spot observations for the requested period."""
    query = """
    SELECT
        trade_date,
        fut_code,
        spot_price,
        source
    FROM spot_price
    WHERE trade_date BETWEEN ? AND ?
    ORDER BY trade_date, fut_code
    """
    with closing(get_connection(db_path)) as connection:
        spot = pd.read_sql_query(
            query,
            connection,
            params=(start_date, end_date),
        )

    spot["trade_date"] = pd.to_datetime(
        spot["trade_date"],
        errors="raise",
    )
    return spot


def load_warehouse_daily(
    start_date: str,
    end_date: str,
    db_path: str = DB_PATH,
) -> pd.DataFrame:
    """Return audited product-day warehouse receipts for the period."""
    query = """
    SELECT
        trade_date,
        fut_code,
        exchange,
        warehouse_total,
        source_row_count,
        used_row_count,
        quality_status,
        quality_note,
        source
    FROM warehouse_daily
    WHERE trade_date BETWEEN ? AND ?
    ORDER BY trade_date, fut_code
    """
    with closing(get_connection(db_path)) as connection:
        warehouse = pd.read_sql_query(
            query,
            connection,
            params=(start_date, end_date),
        )

    warehouse["trade_date"] = pd.to_datetime(
        warehouse["trade_date"],
        errors="raise",
    )
    return warehouse


def load_trade_calendar(
    start_date: str,
    end_date: str,
    db_path: str = DB_PATH,
) -> pd.DataFrame:
    """
    读取回测区间内的期货市场开市日期。

    四个交易所的日历可能重复，因此只返回不重复的交易日。
    """
    query = """
    SELECT DISTINCT
        cal_date AS trade_date
    FROM trade_cal
    WHERE is_open = 1
      AND cal_date BETWEEN ? AND ?
    ORDER BY cal_date
    """

    with closing(get_connection(db_path)) as conn:
        calendar = pd.read_sql_query(
            query,
            conn,
            params=(start_date, end_date),
        )

    calendar["trade_date"] = pd.to_datetime(
        calendar["trade_date"]
    )

    return calendar
