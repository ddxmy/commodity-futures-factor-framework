"""Rolling out-of-sample Ridge combination for five commodity factors."""

from __future__ import annotations

from contextlib import closing
from pathlib import Path
import sqlite3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

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
from src.p06_backtest_engine import run_backtest_from_weights
from src.run_five_factor_strategy_v1 import (
    calculate_daily_ic,
    configure_plot_style,
    inverse_volatility_weights,
)


MODEL_START_DATE = "20150101"
PREDICTION_START_DATE = "20200101"
END_DATE = "20251231"
LOAD_START_DATE = "20140101"
POOL_SIZE = 40
MIN_VALID_RATE = 0.80
LOOKBACK_YEARS = 5
RIDGE_PENALTIES = (
    0.0001,
    0.001,
    0.01,
    0.1,
    1.0,
    10.0,
    100.0,
)

FACTOR_COLUMNS = (
    "basis_momentum",
    "carry",
    "s_warehouse",
    "spotmain",
    "t_rank",
)

FEATURE_COLUMNS = tuple(f"{name}_score" for name in FACTOR_COLUMNS)

METHODS = {
    "all_rank_invvol": "全截面",
    "tail10_invvol": "前后10%",
}

OUTPUT_DIR = (
    Path(RESULT_DIR)
    / "ridge_five_factor_rolling5y_20200101_20251231"
)


def map_factor_ranks(
    panel: pd.DataFrame,
    factor_columns: list[str] | tuple[str, ...],
) -> pd.DataFrame:
    """Map each factor's valid daily cross-section linearly to [-1, 1]."""
    required = {"trade_date", "fut_code", *factor_columns}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(
            "panel is missing columns: " + ", ".join(sorted(missing))
        )

    result = panel.copy()
    result["trade_date"] = pd.to_datetime(result["trade_date"], errors="raise")
    if result.duplicated(["trade_date", "fut_code"]).any():
        raise ValueError("panel contains duplicate date-product keys")

    for factor_name in factor_columns:
        values = pd.to_numeric(result[factor_name], errors="coerce")
        values = values.where(np.isfinite(values))
        valid_count = values.notna().groupby(result["trade_date"]).transform("sum")
        ranks = values.groupby(result["trade_date"]).rank(
            method="average",
            ascending=True,
        )
        score = 2.0 * (ranks - 1.0) / (valid_count - 1.0) - 1.0
        result[f"{factor_name}_score"] = score.where(valid_count >= 2, 0.0).fillna(0.0)

    return result


def build_equal_weight_signal(
    panel: pd.DataFrame,
    feature_columns: list[str] | tuple[str, ...],
) -> pd.DataFrame:
    """Sum equally weighted mapped factor scores into one composite signal."""
    if not feature_columns:
        raise ValueError("feature_columns cannot be empty")
    missing = set(feature_columns) - set(panel.columns)
    if missing:
        raise ValueError(
            "panel is missing feature columns: " + ", ".join(sorted(missing))
        )
    result = panel.copy()
    scores = result[list(feature_columns)].apply(
        pd.to_numeric,
        errors="coerce",
    )
    result["equal_weight_signal"] = scores.fillna(0.0).sum(axis=1)
    return result


def build_next_open_labels(
    signals: pd.DataFrame,
    trade_calendar: pd.DataFrame,
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """Attach next-open to following-open returns on the signal-date contract."""
    required_signals = {"trade_date", "fut_code", "ts_code_A"}
    missing_signals = required_signals - set(signals.columns)
    if missing_signals:
        raise ValueError(
            "signals is missing columns: "
            + ", ".join(sorted(missing_signals))
        )
    if "trade_date" not in trade_calendar.columns:
        raise ValueError("trade_calendar is missing trade_date")
    required_prices = {"trade_date", "ts_code", "open"}
    missing_prices = required_prices - set(prices.columns)
    if missing_prices:
        raise ValueError(
            "prices is missing columns: " + ", ".join(sorted(missing_prices))
        )

    result = signals.copy()
    result["trade_date"] = pd.to_datetime(result["trade_date"], errors="raise")
    if result.duplicated(["trade_date", "fut_code"]).any():
        raise ValueError("signals contains duplicate date-product keys")

    dates = pd.DatetimeIndex(
        pd.to_datetime(trade_calendar["trade_date"], errors="raise")
    ).drop_duplicates().sort_values()
    schedule = pd.DataFrame(
        {
            "trade_date": dates,
            "entry_date": pd.Series(dates, dtype="datetime64[ns]").shift(-1),
            "exit_date": pd.Series(dates, dtype="datetime64[ns]").shift(-2),
        }
    )
    result = result.merge(
        schedule,
        on="trade_date",
        how="left",
        validate="many_to_one",
    )

    prepared_prices = prices[["trade_date", "ts_code", "open"]].copy()
    prepared_prices["trade_date"] = pd.to_datetime(
        prepared_prices["trade_date"], errors="raise"
    )
    if prepared_prices.duplicated(["trade_date", "ts_code"]).any():
        raise ValueError("prices contains duplicate date-contract keys")

    entry = prepared_prices.rename(
        columns={
            "trade_date": "entry_date",
            "ts_code": "ts_code_A",
            "open": "entry_open",
        }
    )
    exit_prices = prepared_prices.rename(
        columns={
            "trade_date": "exit_date",
            "ts_code": "ts_code_A",
            "open": "exit_open",
        }
    )
    result = result.merge(
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
    valid_prices = result["entry_open"].gt(0) & result["exit_open"].gt(0)
    result["forward_return"] = (
        result["exit_open"] / result["entry_open"] - 1.0
    ).where(valid_prices)
    return result


def training_sample_for_year(
    labeled_panel: pd.DataFrame,
    prediction_year: int,
    lookback_years: int = 5,
) -> pd.DataFrame:
    """Return only labels fully realized before one prediction year."""
    if isinstance(prediction_year, bool) or not isinstance(prediction_year, int):
        raise ValueError("prediction_year must be an integer")
    if (
        isinstance(lookback_years, bool)
        or not isinstance(lookback_years, int)
        or lookback_years <= 0
    ):
        raise ValueError("lookback_years must be a positive integer")
    required = {"trade_date", "exit_date", "forward_return"}
    missing = required - set(labeled_panel.columns)
    if missing:
        raise ValueError(
            "labeled_panel is missing columns: "
            + ", ".join(sorted(missing))
        )

    panel = labeled_panel.copy()
    panel["trade_date"] = pd.to_datetime(panel["trade_date"], errors="raise")
    panel["exit_date"] = pd.to_datetime(panel["exit_date"], errors="coerce")
    prediction_start = pd.Timestamp(year=prediction_year, month=1, day=1)
    training_start = pd.Timestamp(
        year=prediction_year - lookback_years,
        month=1,
        day=1,
    )
    mask = (
        panel["trade_date"].ge(training_start)
        & panel["trade_date"].lt(prediction_start)
        & panel["exit_date"].lt(prediction_start)
        & panel["forward_return"].notna()
    )
    return panel.loc[mask].copy().reset_index(drop=True)


def build_forward_year_folds(
    training_panel: pd.DataFrame,
    min_train_years: int = 2,
) -> list[dict[str, object]]:
    """Build expanding annual folds whose training years precede validation."""
    if "trade_date" not in training_panel.columns:
        raise ValueError("training_panel is missing trade_date")
    if (
        isinstance(min_train_years, bool)
        or not isinstance(min_train_years, int)
        or min_train_years <= 0
    ):
        raise ValueError("min_train_years must be a positive integer")

    dates = pd.to_datetime(training_panel["trade_date"], errors="raise")
    years = sorted(dates.dt.year.unique().tolist())
    folds = []
    for position in range(min_train_years, len(years)):
        train_years = tuple(years[:position])
        validation_year = years[position]
        folds.append(
            {
                "train_years": train_years,
                "validation_year": validation_year,
                "train_index": training_panel.index[
                    dates.dt.year.isin(train_years)
                ],
                "validation_index": training_panel.index[
                    dates.dt.year.eq(validation_year)
                ],
            }
        )
    return folds


def _mean_daily_rank_ic(
    dates: pd.Series,
    predictions: np.ndarray,
    realized: pd.Series,
) -> float:
    evaluation = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(dates).to_numpy(),
            "prediction": np.asarray(predictions, dtype=float),
            "realized": pd.to_numeric(realized, errors="coerce").to_numpy(),
        }
    )
    daily_values = []
    for _, day in evaluation.groupby("trade_date"):
        valid = day[["prediction", "realized"]].dropna()
        if (
            len(valid) >= 2
            and valid["prediction"].nunique() > 1
            and valid["realized"].nunique() > 1
        ):
            daily_values.append(
                valid["prediction"].corr(valid["realized"], method="spearman")
            )
    return float(np.mean(daily_values)) if daily_values else np.nan


def ridge_alpha_from_penalty(penalty: float, training_rows: int) -> float:
    """Convert a per-sample penalty to sklearn's sum-of-squares alpha."""
    if not np.isfinite(penalty) or penalty <= 0:
        raise ValueError("penalty must be a positive finite number")
    if isinstance(training_rows, bool) or training_rows <= 0:
        raise ValueError("training_rows must be positive")
    return float(penalty) * int(training_rows)


def select_ridge_alpha(
    training_panel: pd.DataFrame,
    feature_columns: list[str] | tuple[str, ...],
    alphas: list[float] | tuple[float, ...],
) -> tuple[float, pd.DataFrame]:
    """Choose a per-sample Ridge penalty by forward annual RankIC."""
    if not alphas:
        raise ValueError("alphas cannot be empty")
    numeric_alphas = sorted({float(alpha) for alpha in alphas})
    if any(not np.isfinite(alpha) or alpha <= 0 for alpha in numeric_alphas):
        raise ValueError("alphas must be positive finite numbers")

    folds = build_forward_year_folds(training_panel, min_train_years=2)
    records = []
    for penalty in numeric_alphas:
        fold_scores = []
        for fold in folds:
            train = training_panel.loc[fold["train_index"]]
            validation = training_panel.loc[fold["validation_index"]]
            if train.empty or validation.empty:
                continue
            effective_alpha = ridge_alpha_from_penalty(penalty, len(train))
            model = Ridge(alpha=effective_alpha, fit_intercept=True)
            model.fit(
                train[list(feature_columns)].to_numpy(dtype=float),
                train["forward_return"].to_numpy(dtype=float),
            )
            prediction = model.predict(
                validation[list(feature_columns)].to_numpy(dtype=float)
            )
            score = _mean_daily_rank_ic(
                validation["trade_date"],
                prediction,
                validation["forward_return"],
            )
            records.append(
                {
                    "penalty_lambda": penalty,
                    "effective_alpha": effective_alpha,
                    "validation_year": fold["validation_year"],
                    "rank_ic": score,
                }
            )
            if np.isfinite(score):
                fold_scores.append(score)

        mean_score = float(np.mean(fold_scores)) if fold_scores else np.nan
        records.append(
            {
                "penalty_lambda": penalty,
                "effective_alpha": np.nan,
                "validation_year": "mean",
                "rank_ic": mean_score,
            }
        )

    diagnostics = pd.DataFrame(records)
    means = diagnostics.loc[diagnostics["validation_year"].eq("mean")].copy()
    finite = means.loc[np.isfinite(means["rank_ic"])]
    if finite.empty:
        selected = numeric_alphas[0]
    else:
        selected = float(
            finite.sort_values(
                ["rank_ic", "penalty_lambda"],
                ascending=[False, True],
            ).iloc[0]["penalty_lambda"]
        )
    return selected, diagnostics


def fit_rolling_ridge_predictions(
    labeled_panel: pd.DataFrame,
    feature_columns: list[str] | tuple[str, ...],
    prediction_years: list[int] | tuple[int, ...],
    alphas: list[float] | tuple[float, ...],
    lookback_years: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit one historical Ridge model per year and return OOS predictions."""
    required = {
        "trade_date",
        "exit_date",
        "fut_code",
        "forward_return",
        *feature_columns,
    }
    missing = required - set(labeled_panel.columns)
    if missing:
        raise ValueError(
            "labeled_panel is missing columns: "
            + ", ".join(sorted(missing))
        )

    panel = labeled_panel.copy()
    panel["trade_date"] = pd.to_datetime(panel["trade_date"], errors="raise")
    predictions = []
    summaries = []
    cv_records = []

    for prediction_year in prediction_years:
        training = training_sample_for_year(
            panel,
            prediction_year=prediction_year,
            lookback_years=lookback_years,
        )
        prediction = panel.loc[
            panel["trade_date"].dt.year.eq(prediction_year)
        ].copy()
        if training.empty:
            raise ValueError(f"no training rows for prediction year {prediction_year}")
        if prediction.empty:
            raise ValueError(f"no prediction rows for year {prediction_year}")

        training = training.dropna(subset=[*feature_columns, "forward_return"])
        prediction = prediction.dropna(subset=list(feature_columns))
        penalty, cv = select_ridge_alpha(training, feature_columns, alphas)
        cv.insert(0, "prediction_year", prediction_year)
        cv_records.append(cv)

        effective_alpha = ridge_alpha_from_penalty(penalty, len(training))
        model = Ridge(alpha=effective_alpha, fit_intercept=True)
        model.fit(
            training[list(feature_columns)].to_numpy(dtype=float),
            training["forward_return"].to_numpy(dtype=float),
        )
        prediction["ridge_prediction"] = model.predict(
            prediction[list(feature_columns)].to_numpy(dtype=float)
        )
        prediction["prediction_year"] = prediction_year
        predictions.append(prediction)

        summary = {
            "prediction_year": prediction_year,
            "training_start": training["trade_date"].min(),
            "training_end": training["trade_date"].max(),
            "training_rows": len(training),
            "prediction_rows": len(prediction),
            "penalty_lambda": penalty,
            "alpha": effective_alpha,
            "intercept": float(model.intercept_),
        }
        summary.update(
            {
                f"coef_{feature_name}": float(coefficient)
                for feature_name, coefficient in zip(feature_columns, model.coef_)
            }
        )
        summaries.append(summary)

    return (
        pd.concat(predictions, ignore_index=True),
        pd.DataFrame(summaries),
        pd.concat(cv_records, ignore_index=True),
    )


def _semester_starts(
    start_date: str,
    end_date: str,
) -> pd.DatetimeIndex:
    start = pd.to_datetime(start_date, format="%Y%m%d", errors="raise")
    end = pd.to_datetime(end_date, format="%Y%m%d", errors="raise")
    starts = []
    for year in range(start.year, end.year + 1):
        for month in (1, 7):
            candidate = pd.Timestamp(year=year, month=month, day=1)
            if start <= candidate <= end:
                starts.append(candidate)
    return pd.DatetimeIndex(starts)


def build_historical_semiannual_pool(
    start_date: str = MODEL_START_DATE,
    end_date: str = END_DATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rebuild the lagged six-month Top-40 pool back to model training."""
    requested_start = pd.to_datetime(start_date, format="%Y%m%d")
    history_start = requested_start - pd.DateOffset(months=6)
    requested_end = pd.to_datetime(end_date, format="%Y%m%d")
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
    parameters = (
        history_start.strftime("%Y%m%d"),
        requested_end.strftime("%Y%m%d"),
    )
    with closing(sqlite3.connect(DB_PATH)) as connection:
        source = pd.read_sql_query(query, connection, params=parameters)
        calendar = pd.read_sql_query(
            calendar_query,
            connection,
            params=parameters,
        )
    source["trade_date"] = pd.to_datetime(source["trade_date"], errors="raise")
    calendar["trade_date"] = pd.to_datetime(
        calendar["trade_date"], errors="raise"
    )

    statistics = []
    members = []
    for effective_start in _semester_starts(start_date, end_date):
        period_start = effective_start - pd.DateOffset(months=6)
        period_end = effective_start - pd.Timedelta(days=1)
        period = source.loc[
            source["trade_date"].between(period_start, period_end)
        ]
        market_days = int(
            calendar["trade_date"].between(period_start, period_end).sum()
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
        selected["effective_end"] = min(
            effective_start + pd.DateOffset(months=6) - pd.Timedelta(days=1),
            requested_end,
        )
        members.append(selected)
        statistics.append(
            {
                "effective_start": effective_start,
                "history_start": period_start,
                "history_end": period_end,
                "market_days": market_days,
                "eligible_products": len(eligible),
                "selected_products": len(selected),
                "rank40_mean_amount": (
                    eligible.iloc[39]["mean_amount"]
                    if len(eligible) >= 40
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(statistics), pd.concat(members, ignore_index=True)


def calculate_raw_factor_panels(
    signal_mapping: pd.DataFrame,
    calendar: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Calculate the five approved factor definitions for 2015-2025."""
    basis = compute_basis_components(signal_mapping, calendar, lookback=120)
    carry = compute_main_sub_carry(signal_mapping, calendar, lookback=90)
    warehouse = compute_s_warehouse(
        load_warehouse_daily(LOAD_START_DATE, END_DATE),
        calendar,
        lookback=90,
        smooth_window=20,
        min_observations=18,
    )
    spot = compute_spot_main(
        signal_mapping,
        load_spot_daily(LOAD_START_DATE, END_DATE),
        calendar,
        lookback=90,
    )
    rank = compute_t_rank(signal_mapping, calendar, lookback=10)
    sources = {
        "basis_momentum": (basis, "factor_AB"),
        "carry": (carry, "main_sub_carry"),
        "s_warehouse": (warehouse, "s_warehouse"),
        "spotmain": (spot, "spotmain"),
        "t_rank": (rank, "t_rank"),
    }
    output = {}
    start = pd.Timestamp(MODEL_START_DATE)
    end = pd.Timestamp(END_DATE)
    for factor_name, (panel, value_column) in sources.items():
        compact = panel[["trade_date", "fut_code", value_column]].copy()
        compact = compact.rename(columns={value_column: factor_name})
        output[factor_name] = compact.loc[
            compact["trade_date"].between(start, end)
        ].reset_index(drop=True)
    return output


def build_model_feature_panel(
    factor_panels: dict[str, pd.DataFrame],
    trade_mapping: pd.DataFrame,
    calendar: pd.DataFrame,
    pool_members: pd.DataFrame,
) -> pd.DataFrame:
    """Merge pool, contracts and factors, then apply daily [-1, 1] ranks."""
    dates = calendar.loc[
        calendar["trade_date"].between(
            pd.Timestamp(MODEL_START_DATE),
            pd.Timestamp(END_DATE),
        ),
        "trade_date",
    ]
    products = sorted(trade_mapping["fut_code"].dropna().unique())
    panel = pd.MultiIndex.from_product(
        [dates, products],
        names=["trade_date", "fut_code"],
    ).to_frame(index=False)
    panel = panel.merge(
        trade_mapping[["trade_date", "fut_code", "ts_code_A"]],
        on=["trade_date", "fut_code"],
        how="left",
        validate="one_to_one",
    )
    for factor_name in FACTOR_COLUMNS:
        panel = panel.merge(
            factor_panels[factor_name],
            on=["trade_date", "fut_code"],
            how="left",
            validate="one_to_one",
        )

    panel["semester_start"] = pd.to_datetime(
        {
            "year": panel["trade_date"].dt.year,
            "month": np.where(panel["trade_date"].dt.month <= 6, 1, 7),
            "day": 1,
        }
    )
    membership = pool_members[["effective_start", "fut_code"]].drop_duplicates()
    membership["in_pool"] = True
    panel = panel.merge(
        membership,
        left_on=["semester_start", "fut_code"],
        right_on=["effective_start", "fut_code"],
        how="left",
        validate="many_to_one",
    )
    panel["in_pool"] = (
        panel["in_pool"].astype("boolean").fillna(False).astype(bool)
    )
    panel = panel.loc[panel["in_pool"]].copy()
    panel = map_factor_ranks(panel, FACTOR_COLUMNS)
    return panel.drop(columns=["effective_start"])


def build_ridge_strategy_panel(
    predictions: pd.DataFrame,
    trade_mapping: pd.DataFrame,
    calendar: pd.DataFrame,
    pool_members: pd.DataFrame,
    signal_column: str = "ridge_prediction",
) -> pd.DataFrame:
    """Create the complete execution grid expected by the shared engine."""
    if signal_column not in predictions.columns:
        raise ValueError(f"predictions is missing {signal_column}")
    requested_dates = calendar.loc[
        calendar["trade_date"].between(
            pd.Timestamp(PREDICTION_START_DATE),
            pd.Timestamp(END_DATE),
        ),
        "trade_date",
    ]
    products = sorted(trade_mapping["fut_code"].dropna().unique())
    panel = pd.MultiIndex.from_product(
        [requested_dates, products],
        names=["trade_date", "fut_code"],
    ).to_frame(index=False)

    mapping = trade_mapping[
        ["trade_date", "fut_code", "ts_code_A", "daily_return_A"]
    ].copy()
    mapping = mapping.sort_values(["fut_code", "trade_date"])
    mapping["vol20"] = mapping.groupby("fut_code")["daily_return_A"].transform(
        lambda values: values.rolling(20, min_periods=20).std()
    )
    panel = panel.merge(
        mapping[["trade_date", "fut_code", "ts_code_A", "vol20"]],
        on=["trade_date", "fut_code"],
        how="left",
        validate="one_to_one",
    )
    panel = panel.merge(
        predictions[["trade_date", "fut_code", signal_column]],
        on=["trade_date", "fut_code"],
        how="left",
        validate="one_to_one",
    ).rename(columns={signal_column: "raw_factor"})

    panel["semester_start"] = pd.to_datetime(
        {
            "year": panel["trade_date"].dt.year,
            "month": np.where(panel["trade_date"].dt.month <= 6, 1, 7),
            "day": 1,
        }
    )
    membership = pool_members[["effective_start", "fut_code"]].drop_duplicates()
    membership["in_pool"] = True
    panel = panel.merge(
        membership,
        left_on=["semester_start", "fut_code"],
        right_on=["effective_start", "fut_code"],
        how="left",
        validate="many_to_one",
    )
    panel["in_pool"] = (
        panel["in_pool"].astype("boolean").fillna(False).astype(bool)
    )
    return panel.drop(columns=["effective_start"])


def summarize_backtest(
    method: str,
    strategy_panel: pd.DataFrame,
    weights: pd.DataFrame,
    nav: pd.Series,
    metrics: dict[str, float],
    prices: pd.DataFrame,
    calendar: pd.DataFrame,
) -> dict[str, float | str]:
    ic, rank_ic, ic_days = calculate_daily_ic(
        strategy_panel,
        prices,
        calendar,
    )
    active = weights.loc[weights["weight"].ne(0)]
    counts = active.groupby("trade_date")["weight"].agg(
        long_count=lambda values: int((values > 0).sum()),
        short_count=lambda values: int((values < 0).sum()),
    )
    result: dict[str, float | str] = {
        "method": method,
        "method_label": METHODS[method],
        "final_nav": float(nav.iloc[-1]),
        "ic": ic,
        "rank_ic": rank_ic,
        "ic_days": ic_days,
        "average_long_count": float(counts["long_count"].mean()),
        "average_short_count": float(counts["short_count"].mean()),
    }
    result.update(metrics)
    return result


def add_relative_coefficients(
    coefficient_summary: pd.DataFrame,
    coefficient_columns: list[str] | tuple[str, ...],
) -> pd.DataFrame:
    """Add annual coefficients normalized by their absolute sum."""
    result = coefficient_summary.copy()
    scale = result[list(coefficient_columns)].abs().sum(axis=1).replace(0, np.nan)
    for column in coefficient_columns:
        result[f"relative_{column}"] = result[column] / scale
    return result


def plot_results(
    signal_navs: dict[str, dict[str, pd.Series]],
    coefficient_summary: pd.DataFrame,
) -> None:
    configure_plot_style()
    benchmark_path = (
        Path(RESULT_DIR)
        / "five_factor_strategy_20200101_20251231_h1"
        / "daily_nav.csv"
    )
    benchmark = pd.read_csv(
        benchmark_path,
        index_col=0,
        parse_dates=True,
    )

    figure, axes = plt.subplots(1, 2, figsize=(15, 5.7))
    for axis, method in zip(axes, METHODS):
        benchmark_column = f"{method}__equal_weight_composite"
        axis.plot(
            benchmark.index,
            benchmark[benchmark_column],
            label="五因子等资金",
            color="#7A7A7A",
            linewidth=1.6,
        )
        axis.plot(
            signal_navs["equal_weight_signal"][method].index,
            signal_navs["equal_weight_signal"][method].values,
            label="五因子等权信号",
            color="#E07A5F",
            linewidth=1.8,
        )
        axis.plot(
            signal_navs["ridge"][method].index,
            signal_navs["ridge"][method].values,
            label="滚动 Ridge",
            color="#247BA0",
            linewidth=2.1,
        )
        axis.axhline(1.0, color="#555555", linewidth=0.7, alpha=0.6)
        axis.set_title(METHODS[method])
        axis.set_ylabel("NAV")
        axis.legend(frameon=False)
    figure.suptitle("五因子信号合成方法对比", fontsize=17)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "ridge_nav_comparison.png", bbox_inches="tight")
    figure.savefig(
        OUTPUT_DIR / "signal_combination_nav_comparison.png",
        bbox_inches="tight",
    )
    plt.close(figure)

    coefficient_columns = [f"coef_{name}" for name in FEATURE_COLUMNS]
    coefficient_labels = [
        "Basis Momentum",
        "Carry",
        "S_Warehouse",
        "SpotMain",
        "T_Rank",
    ]
    relative_columns = [f"relative_{column}" for column in coefficient_columns]
    figure, axis = plt.subplots(figsize=(11, 5.8))
    for column, label in zip(relative_columns, coefficient_labels):
        axis.plot(
            coefficient_summary["prediction_year"],
            coefficient_summary[column],
            marker="o",
            linewidth=1.6,
            label=label,
        )
    axis.axhline(0.0, color="#555555", linewidth=0.8)
    axis.set_title("滚动 Ridge 年度相对系数")
    axis.set_xlabel("预测年份")
    axis.set_ylabel("Relative Coefficient")
    axis.legend(frameon=False, ncol=2)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "ridge_coefficients.png", bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pool_statistics, pool_members = build_historical_semiannual_pool()
    calendar = load_trade_calendar(LOAD_START_DATE, END_DATE)
    signal_mapping = build_contract_mapping(
        LOAD_START_DATE,
        END_DATE,
        min_days_to_maturity=0,
    )
    trade_mapping = build_contract_mapping(
        LOAD_START_DATE,
        END_DATE,
        min_days_to_maturity=45,
    )
    factor_panels = calculate_raw_factor_panels(signal_mapping, calendar)
    model_panel = build_model_feature_panel(
        factor_panels,
        trade_mapping,
        calendar,
        pool_members,
    )
    prices = load_contract_prices(MODEL_START_DATE, END_DATE)
    labeled = build_next_open_labels(model_panel, calendar, prices)
    predictions, coefficients, cv_diagnostics = fit_rolling_ridge_predictions(
        labeled,
        feature_columns=FEATURE_COLUMNS,
        prediction_years=list(range(2020, 2026)),
        alphas=RIDGE_PENALTIES,
        lookback_years=LOOKBACK_YEARS,
    )
    predictions = build_equal_weight_signal(predictions, FEATURE_COLUMNS)
    coefficient_columns = [f"coef_{name}" for name in FEATURE_COLUMNS]
    coefficients = add_relative_coefficients(coefficients, coefficient_columns)
    strategy_panels = {
        "ridge": build_ridge_strategy_panel(
            predictions,
            trade_mapping,
            calendar,
            pool_members,
        ),
        "equal_weight_signal": build_ridge_strategy_panel(
            predictions,
            trade_mapping,
            calendar,
            pool_members,
            signal_column="equal_weight_signal",
        ),
    }
    backtest_prices = load_contract_prices(PREDICTION_START_DATE, END_DATE)

    metrics_records = []
    signal_navs: dict[str, dict[str, pd.Series]] = {
        signal_name: {} for signal_name in strategy_panels
    }
    comparison_daily = pd.DataFrame()
    for signal_name, strategy_panel in strategy_panels.items():
        for method in METHODS:
            weights = inverse_volatility_weights(strategy_panel, method)
            nav, metrics, daily_return, diagnostics = run_backtest_from_weights(
                weights,
                backtest_prices,
                cost_rate=COST_RATE,
                return_diagnostics=True,
            )
            signal_navs[signal_name][method] = nav
            comparison_daily[f"{signal_name}__{method}_return"] = daily_return
            comparison_daily[f"{signal_name}__{method}_nav"] = nav
            summary = summarize_backtest(
                method,
                strategy_panel,
                weights,
                nav,
                metrics,
                backtest_prices,
                calendar,
            )
            summary["signal_method"] = signal_name
            metrics_records.append(summary)
            diagnostics.to_csv(
                OUTPUT_DIR / f"{signal_name}__{method}_daily_diagnostics.csv",
                encoding="utf-8-sig",
            )

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
    coefficients.to_csv(
        OUTPUT_DIR / "ridge_yearly_coefficients.csv",
        index=False,
        encoding="utf-8-sig",
    )
    cv_diagnostics.to_csv(
        OUTPUT_DIR / "ridge_alpha_cv.csv",
        index=False,
        encoding="utf-8-sig",
    )
    predictions[
        [
            "trade_date",
            "fut_code",
            "prediction_year",
            "ridge_prediction",
            "equal_weight_signal",
        ]
    ].to_csv(
        OUTPUT_DIR / "signal_oos_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    predictions[
        ["trade_date", "fut_code", "prediction_year", "ridge_prediction"]
    ].to_csv(
        OUTPUT_DIR / "ridge_oos_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    comparison_metrics = pd.DataFrame(metrics_records)
    comparison_metrics.to_csv(
        OUTPUT_DIR / "signal_combination_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    comparison_metrics.loc[
        comparison_metrics["signal_method"].eq("ridge")
    ].drop(columns="signal_method").to_csv(
        OUTPUT_DIR / "ridge_strategy_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    comparison_metrics.loc[
        comparison_metrics["signal_method"].eq("equal_weight_signal")
    ].to_csv(
        OUTPUT_DIR / "equal_weight_signal_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    comparison_daily.to_csv(
        OUTPUT_DIR / "signal_combination_daily_nav.csv",
        encoding="utf-8-sig",
    )
    comparison_daily[
        [column for column in comparison_daily if column.startswith("ridge__")]
    ].rename(columns=lambda column: column.removeprefix("ridge__")).to_csv(
        OUTPUT_DIR / "ridge_daily_nav.csv",
        encoding="utf-8-sig",
    )
    plot_results(signal_navs, coefficients)

    print(f"Results written to: {OUTPUT_DIR}")
    print(comparison_metrics.to_string(index=False))
    print(coefficients.to_string(index=False))


if __name__ == "__main__":
    main()
