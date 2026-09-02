"""Rolling out-of-sample tree-model combinations for five commodity factors."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from config.settings import COST_RATE, RESULT_DIR
from src.p01_market_data import load_contract_prices, load_trade_calendar
from src.p02_contract_selection import build_contract_mapping
from src.p06_backtest_engine import run_backtest_from_weights
from src.ridge_five_factor_strategy import (
    END_DATE,
    FEATURE_COLUMNS,
    LOAD_START_DATE,
    LOOKBACK_YEARS,
    METHODS,
    PREDICTION_START_DATE,
    RIDGE_PENALTIES,
    _mean_daily_rank_ic,
    build_equal_weight_signal,
    build_forward_year_folds,
    build_historical_semiannual_pool,
    build_model_feature_panel,
    build_next_open_labels,
    build_ridge_strategy_panel,
    calculate_raw_factor_panels,
    fit_rolling_ridge_predictions,
    inverse_volatility_weights,
    summarize_backtest,
    training_sample_for_year,
)
from src.run_five_factor_strategy_v1 import configure_plot_style


RANDOM_SEED = 20260826
SUPPORTED_TREE_MODELS = ("lightgbm", "xgboost")

LIGHTGBM_PARAMETER_GRID = (
    {
        "n_estimators": 150,
        "learning_rate": 0.03,
        "max_depth": 3,
        "num_leaves": 7,
        "min_child_samples": 200,
    },
    {
        "n_estimators": 150,
        "learning_rate": 0.03,
        "max_depth": 4,
        "num_leaves": 15,
        "min_child_samples": 100,
    },
    {
        "n_estimators": 100,
        "learning_rate": 0.03,
        "max_depth": 3,
        "num_leaves": 7,
        "min_child_samples": 400,
    },
)

XGBOOST_PARAMETER_GRID = (
    {
        "n_estimators": 150,
        "learning_rate": 0.03,
        "max_depth": 2,
        "min_child_weight": 100,
    },
    {
        "n_estimators": 150,
        "learning_rate": 0.03,
        "max_depth": 3,
        "min_child_weight": 100,
    },
    {
        "n_estimators": 100,
        "learning_rate": 0.03,
        "max_depth": 2,
        "min_child_weight": 300,
    },
)

OUTPUT_DIR = (
    Path(RESULT_DIR)
    / "tree_five_factor_comparison_20200101_20251231"
)

MODEL_COLUMNS = {
    "equal_weight": "equal_weight_signal",
    "ridge": "ridge_prediction",
    "lightgbm": "lightgbm_prediction",
    "xgboost": "xgboost_prediction",
}

MODEL_LABELS = {
    "equal_weight": "Equal Weight",
    "ridge": "Ridge",
    "lightgbm": "LightGBM",
    "xgboost": "XGBoost",
}


def build_tree_model(model_name: str, parameters: dict[str, Any]):
    """Build a deterministic CPU regressor with conservative defaults."""
    if model_name == "lightgbm":
        from lightgbm import LGBMRegressor

        defaults = {
            "objective": "regression",
            "learning_rate": 0.05,
            "n_estimators": 150,
            "max_depth": 3,
            "num_leaves": 7,
            "min_child_samples": 200,
            "subsample": 0.8,
            "subsample_freq": 1,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
            "random_state": RANDOM_SEED,
            "n_jobs": 1,
            "deterministic": True,
            "force_col_wise": True,
            "verbosity": -1,
        }
        defaults.update(parameters)
        return LGBMRegressor(**defaults)

    if model_name == "xgboost":
        from xgboost import XGBRegressor

        defaults = {
            "objective": "reg:squarederror",
            "learning_rate": 0.05,
            "n_estimators": 150,
            "max_depth": 3,
            "min_child_weight": 100,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
            "random_state": RANDOM_SEED,
            "n_jobs": 1,
            "tree_method": "hist",
        }
        defaults.update(parameters)
        return XGBRegressor(**defaults)

    raise ValueError(f"unsupported tree model: {model_name}")


def select_tree_parameters(
    training_panel: pd.DataFrame,
    feature_columns: list[str] | tuple[str, ...],
    model_name: str,
    parameter_grid: list[dict[str, Any]] | tuple[dict[str, Any], ...],
) -> tuple[int, dict[str, Any], pd.DataFrame]:
    """Select one candidate by mean forward annual Rank IC."""
    if model_name not in SUPPORTED_TREE_MODELS:
        raise ValueError(f"unsupported tree model: {model_name}")
    if not parameter_grid:
        raise ValueError("parameter_grid cannot be empty")

    folds = build_forward_year_folds(training_panel, min_train_years=2)
    records: list[dict[str, Any]] = []
    candidate_means: list[tuple[int, float]] = []
    for candidate, parameters in enumerate(parameter_grid):
        fold_scores = []
        for fold in folds:
            train = training_panel.loc[fold["train_index"]]
            validation = training_panel.loc[fold["validation_index"]]
            if train.empty or validation.empty:
                continue
            model = build_tree_model(model_name, dict(parameters))
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
                    "model": model_name,
                    "candidate": candidate,
                    "parameters": json.dumps(parameters, sort_keys=True),
                    "validation_year": fold["validation_year"],
                    "rank_ic": score,
                }
            )
            if np.isfinite(score):
                fold_scores.append(float(score))

        mean_score = float(np.mean(fold_scores)) if fold_scores else np.nan
        candidate_means.append((candidate, mean_score))
        records.append(
            {
                "model": model_name,
                "candidate": candidate,
                "parameters": json.dumps(parameters, sort_keys=True),
                "validation_year": "mean",
                "rank_ic": mean_score,
            }
        )

    finite = [item for item in candidate_means if np.isfinite(item[1])]
    selected_candidate = (
        sorted(finite, key=lambda item: (-item[1], item[0]))[0][0]
        if finite
        else 0
    )
    return (
        selected_candidate,
        dict(parameter_grid[selected_candidate]),
        pd.DataFrame(records),
    )


def fit_rolling_tree_predictions(
    labeled_panel: pd.DataFrame,
    feature_columns: list[str] | tuple[str, ...],
    prediction_years: list[int] | tuple[int, ...],
    model_name: str,
    parameter_grid: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    lookback_years: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit one tree model per year using only the prior rolling window."""
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
            "labeled_panel is missing columns: " + ", ".join(sorted(missing))
        )
    if model_name not in SUPPORTED_TREE_MODELS:
        raise ValueError(f"unsupported tree model: {model_name}")

    panel = labeled_panel.copy()
    panel["trade_date"] = pd.to_datetime(panel["trade_date"], errors="raise")
    prediction_frames = []
    summaries = []
    cv_frames = []
    prediction_column = f"{model_name}_prediction"

    for prediction_year in prediction_years:
        training = training_sample_for_year(
            panel,
            prediction_year=prediction_year,
            lookback_years=lookback_years,
        ).dropna(subset=[*feature_columns, "forward_return"])
        prediction = panel.loc[
            panel["trade_date"].dt.year.eq(prediction_year)
        ].dropna(subset=list(feature_columns)).copy()
        if training.empty:
            raise ValueError(f"no training rows for prediction year {prediction_year}")
        if prediction.empty:
            raise ValueError(f"no prediction rows for year {prediction_year}")

        selected, parameters, diagnostics = select_tree_parameters(
            training,
            feature_columns,
            model_name,
            parameter_grid,
        )
        diagnostics.insert(0, "prediction_year", prediction_year)
        cv_frames.append(diagnostics)

        model = build_tree_model(model_name, parameters)
        model.fit(
            training[list(feature_columns)].to_numpy(dtype=float),
            training["forward_return"].to_numpy(dtype=float),
        )
        prediction[prediction_column] = model.predict(
            prediction[list(feature_columns)].to_numpy(dtype=float)
        )
        prediction["prediction_year"] = prediction_year
        prediction_frames.append(prediction)
        summaries.append(
            {
                "model": model_name,
                "prediction_year": prediction_year,
                "training_start": training["trade_date"].min(),
                "training_end": training["trade_date"].max(),
                "training_rows": len(training),
                "prediction_rows": len(prediction),
                "selected_candidate": selected,
                "parameters": json.dumps(parameters, sort_keys=True),
            }
        )

    return (
        pd.concat(prediction_frames, ignore_index=True),
        pd.DataFrame(summaries),
        pd.concat(cv_frames, ignore_index=True),
    )


def _daily_rank_ic(
    panel: pd.DataFrame,
    prediction_column: str,
) -> pd.Series:
    values = {}
    for trade_date, day in panel.groupby("trade_date"):
        valid = day[[prediction_column, "forward_return"]].dropna()
        if (
            len(valid) >= 2
            and valid[prediction_column].nunique() > 1
            and valid["forward_return"].nunique() > 1
        ):
            values[trade_date] = valid[prediction_column].corr(
                valid["forward_return"], method="spearman"
            )
    return pd.Series(values, dtype=float).sort_index()


def paired_rank_ic_test(
    panel: pd.DataFrame,
    model_column: str,
    baseline_column: str,
    max_lags: int = 5,
) -> dict[str, float | int | str]:
    """Test the mean paired daily Rank-IC difference with Newey-West errors."""
    model_ic = _daily_rank_ic(panel, model_column).rename("model")
    baseline_ic = _daily_rank_ic(panel, baseline_column).rename("baseline")
    paired = pd.concat([model_ic, baseline_ic], axis=1).dropna()
    difference = (paired["model"] - paired["baseline"]).to_numpy(dtype=float)
    observations = len(difference)
    if observations == 0:
        mean_difference = np.nan
        standard_error = np.nan
        t_stat = np.nan
        p_value = np.nan
    else:
        mean_difference = float(difference.mean())
        centered = difference - mean_difference
        lag_count = min(max(int(max_lags), 0), observations - 1)
        long_run_variance = float(np.mean(centered * centered))
        for lag in range(1, lag_count + 1):
            weight = 1.0 - lag / (lag_count + 1.0)
            covariance = float(np.mean(centered[lag:] * centered[:-lag]))
            long_run_variance += 2.0 * weight * covariance
        long_run_variance = max(long_run_variance, 0.0)
        standard_error = float(np.sqrt(long_run_variance / observations))
        if standard_error == 0.0:
            t_stat = float(np.sign(mean_difference) * np.inf)
            p_value = 0.0 if mean_difference != 0.0 else 1.0
        else:
            t_stat = mean_difference / standard_error
            p_value = float(
                2.0 * stats.t.sf(abs(t_stat), df=max(observations - 1, 1))
            )

    return {
        "model": model_column,
        "baseline": baseline_column,
        "observations": observations,
        "mean_rank_ic_difference": mean_difference,
        "newey_west_lags": min(max(int(max_lags), 0), max(observations - 1, 0)),
        "standard_error": standard_error,
        "t_stat": t_stat,
        "p_value": p_value,
    }


def merge_model_predictions(
    baseline: pd.DataFrame,
    lightgbm: pd.DataFrame,
    xgboost: pd.DataFrame,
) -> pd.DataFrame:
    """Combine predictions after enforcing an identical unique key set."""
    keys = ["trade_date", "fut_code"]
    baseline_columns = [
        *keys,
        "forward_return",
        "ridge_prediction",
        "equal_weight_signal",
    ]
    model_specs = (
        (lightgbm, "lightgbm_prediction"),
        (xgboost, "xgboost_prediction"),
    )
    missing = set(baseline_columns) - set(baseline.columns)
    if missing:
        raise ValueError(
            "baseline is missing columns: " + ", ".join(sorted(missing))
        )
    if baseline.duplicated(keys).any():
        raise ValueError("baseline contains duplicate prediction keys")

    result = baseline[baseline_columns].copy()
    expected_keys = pd.MultiIndex.from_frame(result[keys])
    for frame, prediction_column in model_specs:
        required = {*keys, prediction_column}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(
                f"{prediction_column} frame is missing columns: "
                + ", ".join(sorted(missing))
            )
        if frame.duplicated(keys).any():
            raise ValueError(f"{prediction_column} contains duplicate prediction keys")
        actual_keys = pd.MultiIndex.from_frame(frame[keys])
        if set(actual_keys) != set(expected_keys):
            raise ValueError("prediction keys do not match across models")
        result = result.merge(
            frame[[*keys, prediction_column]],
            on=keys,
            how="left",
            validate="one_to_one",
        )
    return result


def plot_model_comparison(
    model_navs: dict[str, dict[str, pd.Series]],
) -> None:
    """Save one same-scale NAV panel for both portfolio methods."""
    configure_plot_style()
    colors = {
        "equal_weight": "#7A7A7A",
        "ridge": "#247BA0",
        "lightgbm": "#E07A5F",
        "xgboost": "#6A4C93",
    }
    figure, axes = plt.subplots(1, 2, figsize=(15, 5.8))
    for axis, method in zip(axes, METHODS):
        for model_name in MODEL_COLUMNS:
            nav = model_navs[model_name][method]
            axis.plot(
                nav.index,
                nav.values,
                label=MODEL_LABELS[model_name],
                color=colors[model_name],
                linewidth=2.0 if model_name != "equal_weight" else 1.5,
            )
        axis.axhline(1.0, color="#555555", linewidth=0.7, alpha=0.6)
        axis.set_title(METHODS[method])
        axis.set_ylabel("NAV")
        axis.legend(frameon=False)
    figure.suptitle("五因子滚动样本外模型对比", fontsize=17)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "model_nav_comparison.png", bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    """Run the four-model experiment and write a new isolated result set."""
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise FileExistsError(
            f"output directory is not empty; refusing to overwrite: {OUTPUT_DIR}"
        )

    prediction_years = list(range(2020, 2026))
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
    prices = load_contract_prices(LOAD_START_DATE, END_DATE)
    labeled = build_next_open_labels(model_panel, calendar, prices)

    baseline, ridge_summary, ridge_cv = fit_rolling_ridge_predictions(
        labeled,
        feature_columns=FEATURE_COLUMNS,
        prediction_years=prediction_years,
        alphas=RIDGE_PENALTIES,
        lookback_years=LOOKBACK_YEARS,
    )
    baseline = build_equal_weight_signal(baseline, FEATURE_COLUMNS)
    lightgbm, lightgbm_summary, lightgbm_cv = fit_rolling_tree_predictions(
        labeled,
        feature_columns=FEATURE_COLUMNS,
        prediction_years=prediction_years,
        model_name="lightgbm",
        parameter_grid=LIGHTGBM_PARAMETER_GRID,
        lookback_years=LOOKBACK_YEARS,
    )
    xgboost, xgboost_summary, xgboost_cv = fit_rolling_tree_predictions(
        labeled,
        feature_columns=FEATURE_COLUMNS,
        prediction_years=prediction_years,
        model_name="xgboost",
        parameter_grid=XGBOOST_PARAMETER_GRID,
        lookback_years=LOOKBACK_YEARS,
    )
    predictions = merge_model_predictions(baseline, lightgbm, xgboost)

    strategy_panels = {
        model_name: build_ridge_strategy_panel(
            predictions,
            trade_mapping,
            calendar,
            pool_members,
            signal_column=prediction_column,
        )
        for model_name, prediction_column in MODEL_COLUMNS.items()
    }
    backtest_prices = load_contract_prices(PREDICTION_START_DATE, END_DATE)
    metric_records = []
    comparison_daily = pd.DataFrame()
    model_navs: dict[str, dict[str, pd.Series]] = {
        model_name: {} for model_name in MODEL_COLUMNS
    }
    diagnostics_to_write: dict[str, pd.DataFrame] = {}
    for model_name, strategy_panel in strategy_panels.items():
        for method in METHODS:
            weights = inverse_volatility_weights(strategy_panel, method)
            nav, metrics, daily_return, diagnostics = run_backtest_from_weights(
                weights,
                backtest_prices,
                cost_rate=COST_RATE,
                return_diagnostics=True,
            )
            model_navs[model_name][method] = nav
            comparison_daily[f"{model_name}__{method}_return"] = daily_return
            comparison_daily[f"{model_name}__{method}_nav"] = nav
            summary = summarize_backtest(
                method,
                strategy_panel,
                weights,
                nav,
                metrics,
                backtest_prices,
                calendar,
            )
            summary["model"] = model_name
            summary["model_label"] = MODEL_LABELS[model_name]
            metric_records.append(summary)
            diagnostics_to_write[f"{model_name}__{method}_diagnostics.csv"] = (
                diagnostics
            )

    paired_tests = []
    for model_name, prediction_column in MODEL_COLUMNS.items():
        if model_name == "ridge":
            continue
        test_result = paired_rank_ic_test(
            predictions,
            model_column=prediction_column,
            baseline_column="ridge_prediction",
            max_lags=5,
        )
        test_result["model_name"] = model_name
        paired_tests.append(test_result)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    predictions.to_csv(
        OUTPUT_DIR / "oos_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.concat(
        [
            ridge_summary.assign(model="ridge"),
            lightgbm_summary,
            xgboost_summary,
        ],
        ignore_index=True,
        sort=False,
    ).to_csv(
        OUTPUT_DIR / "yearly_model_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.concat(
        [
            ridge_cv.assign(model="ridge"),
            lightgbm_cv,
            xgboost_cv,
        ],
        ignore_index=True,
        sort=False,
    ).to_csv(
        OUTPUT_DIR / "forward_validation_diagnostics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(metric_records).to_csv(
        OUTPUT_DIR / "model_comparison_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(paired_tests).to_csv(
        OUTPUT_DIR / "paired_rank_ic_newey_west.csv",
        index=False,
        encoding="utf-8-sig",
    )
    comparison_daily.to_csv(
        OUTPUT_DIR / "model_comparison_daily_nav.csv",
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
    for filename, diagnostics in diagnostics_to_write.items():
        diagnostics.to_csv(OUTPUT_DIR / filename, encoding="utf-8-sig")
    plot_model_comparison(model_navs)

    metrics_frame = pd.DataFrame(metric_records)
    print(f"Results written to: {OUTPUT_DIR}")
    print(
        metrics_frame[
            [
                "model",
                "method",
                "rank_ic",
                "annual_return",
                "sharpe",
                "max_drawdown",
                "annual_turnover",
            ]
        ].to_string(index=False)
    )
    print(pd.DataFrame(paired_tests).to_string(index=False))


if __name__ == "__main__":
    main()
