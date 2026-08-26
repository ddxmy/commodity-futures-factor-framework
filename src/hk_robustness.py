"""Lightweight H x K robustness analysis for cross-sectional factors."""

import argparse
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from config.research_config import ResearchConfig
from config.settings import COST_RATE, RESULT_DIR
from main import validate_dates
from src.factor_loader import calculate_factor, load_factor_module
from src.p01_market_data import load_contract_prices, load_trade_calendar
from src.p03_factor_processing import validate_factor_data
from src.p05_portfolio_construction import build_target_weights
from src.p06_backtest_engine import run_backtest_from_weights
from src.p07_factor_evaluation import (
    build_factor_test_panel,
    calculate_ic_series,
    periods_per_year,
    summarize_ic_statistics,
)
from src.research_pipeline import prepare_contract_context


DEFAULT_START_DATE = "20220101"
DEFAULT_END_DATE = "20251231"
DEFAULT_K_VALUES = (30, 60, 90, 120)
DEFAULT_H_VALUES = (1, 5, 10)
ROBUSTNESS_COLUMNS = [
    "K",
    "H",
    "trading_days",
    "ic_observations",
    "total_return",
    "annual_return",
    "net_sharpe",
    "max_drawdown",
    "annual_turnover",
    "mean_ic",
    "mean_rank_ic",
]
ROBUSTNESS_FILENAMES = {
    "robustness_details.csv",
    "robustness_summary.csv",
    "net_sharpe_heatmap.png",
    "rank_ic_heatmap.png",
    "annual_return_heatmap.png",
}

plt.rcParams["font.family"] = ["Times New Roman", "Kaiti SC"]
plt.rcParams["axes.unicode_minus"] = False


def validate_grid_values(
    values: list[int] | tuple[int, ...],
    name: str,
) -> tuple[int, ...]:
    """Return a validated, order-preserving positive integer grid."""
    checked = tuple(values)
    if not checked:
        raise ValueError(f"{name} values cannot be empty")
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in checked
    ):
        raise ValueError(f"{name} values must be integers")
    if any(value <= 0 for value in checked):
        raise ValueError(f"{name} values must be positive")
    if len(set(checked)) != len(checked):
        raise ValueError(f"{name} values must be unique")
    return checked


def run_robustness_cell(
    factor_data: pd.DataFrame,
    contract_data: pd.DataFrame,
    trade_calendar: pd.DataFrame,
    prices: pd.DataFrame,
    config: ResearchConfig,
    k: int,
    h: int,
) -> dict[str, object]:
    """Run one first-day-anchored Rank strategy and matching IC evaluation."""
    weights = build_target_weights(
        factor_data=factor_data,
        contract_data=contract_data,
        config=config,
        normalize="rank",
    )
    nav, metrics, _ = run_backtest_from_weights(
        weights=weights,
        prices=prices,
        cost_rate=config.cost_rate,
    )
    if nav.empty:
        raise ValueError(f"K={k}, H={h} produced no strategy NAV")

    dates = pd.to_datetime(weights["trade_date"])
    panel = build_factor_test_panel(
        start_date=dates.min().strftime("%Y%m%d"),
        end_date=dates.max().strftime("%Y%m%d"),
        factor_type="prepared",
        lookback=1,
        rebalance_freq=h,
        min_assets=config.min_assets,
        prepared_weights=weights,
        prepared_calendar=trade_calendar,
        prepared_prices=prices,
    )
    if panel.empty:
        raise ValueError(f"K={k}, H={h} produced no IC panel")
    ic_series = calculate_ic_series(panel)
    ic_summary = summarize_ic_statistics(
        ic_series,
        nw_lags=config.nw_lags,
        annualization_periods=periods_per_year(h),
    ).set_index("metric")

    return {
        "K": k,
        "H": h,
        "trading_days": len(nav),
        "ic_observations": int(
            ic_summary.loc["rank_ic", "observations"]
        ),
        "total_return": float(nav.iloc[-1] - 1.0),
        "annual_return": float(metrics["annual_return"]),
        "net_sharpe": float(metrics["sharpe"]),
        "max_drawdown": float(metrics["max_drawdown"]),
        "annual_turnover": float(metrics["annual_turnover"]),
        "mean_ic": float(ic_summary.loc["ic", "mean"]),
        "mean_rank_ic": float(ic_summary.loc["rank_ic", "mean"]),
    }


def calculate_hk_robustness(
    factor_name: str,
    start_date: str,
    end_date: str,
    k_values: list[int] | tuple[int, ...] = DEFAULT_K_VALUES,
    h_values: list[int] | tuple[int, ...] = DEFAULT_H_VALUES,
    base_config: ResearchConfig | None = None,
) -> pd.DataFrame:
    """Calculate one Rank robustness record for every requested K x H cell."""
    ks = validate_grid_values(k_values, "K")
    hs = validate_grid_values(h_values, "H")
    validate_dates(start_date, end_date)
    module = load_factor_module(factor_name)
    if "lookback" not in dict(getattr(module, "DEFAULT_PARAMETERS", {})):
        raise ValueError(
            f"factor {factor_name} does not expose a lookback parameter"
        )

    calendar = load_trade_calendar(start_date, end_date)
    prices = load_contract_prices(start_date, end_date)
    template = base_config or ResearchConfig()
    records = []

    for k in ks:
        factor_data = calculate_factor(
            factor_name=factor_name,
            start_date=start_date,
            end_date=end_date,
            parameters={
                "lookback": k,
                "signal_min_days_to_maturity": (
                    template.signal_min_days_to_maturity
                ),
            },
        )
        factor_data = validate_factor_data(factor_data, calendar)
        contract_data = prepare_contract_context(
            factor_data,
            start_date,
            end_date,
            min_days_to_maturity=(
                template.trade_min_days_to_maturity
            ),
        )
        for h in hs:
            config = replace(template, rebalance_freq=h)
            try:
                record = run_robustness_cell(
                    factor_data=factor_data,
                    contract_data=contract_data,
                    trade_calendar=calendar,
                    prices=prices,
                    config=config,
                    k=k,
                    h=h,
                )
            except Exception as error:
                raise RuntimeError(
                    f"robustness failed for K={k}, H={h}"
                ) from error
            records.append(record)

    result = pd.DataFrame(records, columns=ROBUSTNESS_COLUMNS)
    expected = len(ks) * len(hs)
    if len(result) != expected or result.duplicated(["K", "H"]).any():
        raise RuntimeError("robustness result grid is incomplete or duplicated")
    return result


def _metric_matrix(
    details: pd.DataFrame,
    metric: str,
    k_values: tuple[int, ...],
    h_values: tuple[int, ...],
) -> pd.DataFrame:
    """Return one complete K-by-H matrix for a robustness metric."""
    if details.duplicated(["K", "H"]).any():
        raise ValueError("robustness details contain duplicate K x H cells")
    matrix = details.pivot(index="K", columns="H", values=metric)
    matrix = matrix.reindex(index=k_values, columns=h_values)
    if (
        matrix.shape != (len(k_values), len(h_values))
        or matrix.isna().any().any()
    ):
        raise ValueError(f"{metric} heatmap grid is incomplete")
    return matrix


def build_robustness_summary(
    details: pd.DataFrame,
    h_values: list[int] | tuple[int, ...],
) -> pd.DataFrame:
    """Return one K row with flattened H columns for headline metrics."""
    hs = validate_grid_values(h_values, "H")
    if details.duplicated(["K", "H"]).any():
        raise ValueError("robustness details contain duplicate K x H cells")
    ks = tuple(sorted(int(value) for value in details["K"].unique()))
    summary = pd.DataFrame({"K": ks})
    for metric in ["net_sharpe", "mean_rank_ic", "annual_return"]:
        matrix = _metric_matrix(details, metric, ks, hs)
        for h in hs:
            summary[f"{metric}_H{h}"] = matrix[h].to_numpy()
    return summary


def plot_robustness_heatmap(
    details: pd.DataFrame,
    metric: str,
    title: str,
    output_path: str | Path,
    k_values: list[int] | tuple[int, ...],
    h_values: list[int] | tuple[int, ...],
) -> None:
    """Save one annotated K-by-H robustness heatmap."""
    ks = validate_grid_values(k_values, "K")
    hs = validate_grid_values(h_values, "H")
    matrix = _metric_matrix(details, metric, ks, hs)
    figure, axis = plt.subplots(figsize=(7.2, 5.2))
    image = axis.imshow(matrix.to_numpy(), aspect="auto", cmap="RdYlGn")
    axis.set_xticks(range(len(hs)), labels=hs)
    axis.set_yticks(range(len(ks)), labels=ks)
    axis.set_xlabel("Holding period H (trading days)")
    axis.set_ylabel("Lookback K (trading days)")
    axis.set_title(title)
    for row in range(len(ks)):
        for column in range(len(hs)):
            axis.text(
                column,
                row,
                f"{matrix.iloc[row, column]:.3f}",
                ha="center",
                va="center",
                color="#1F1F1F",
            )
    figure.colorbar(image, ax=axis, shrink=0.85)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_robustness_results(
    details: pd.DataFrame,
    output_dir: str | Path,
    k_values: list[int] | tuple[int, ...],
    h_values: list[int] | tuple[int, ...],
) -> Path:
    """Validate a complete grid, refuse overwrites, then save all results."""
    ks = validate_grid_values(k_values, "K")
    hs = validate_grid_values(h_values, "H")
    for metric in ["net_sharpe", "mean_rank_ic", "annual_return"]:
        _metric_matrix(details, metric, ks, hs)
    summary = build_robustness_summary(details, hs)

    output = Path(output_dir)
    existing = [
        name for name in ROBUSTNESS_FILENAMES if (output / name).exists()
    ]
    if existing:
        raise FileExistsError(
            "refusing to overwrite robustness outputs: "
            + ", ".join(sorted(existing))
        )

    output.mkdir(parents=True, exist_ok=True)
    details.sort_values(["K", "H"]).round(6).to_csv(
        output / "robustness_details.csv",
        index=False,
    )
    summary.round(6).to_csv(
        output / "robustness_summary.csv",
        index=False,
    )
    plot_robustness_heatmap(
        details,
        "net_sharpe",
        "Fee-adjusted Sharpe",
        output / "net_sharpe_heatmap.png",
        ks,
        hs,
    )
    plot_robustness_heatmap(
        details,
        "mean_rank_ic",
        "Mean RankIC",
        output / "rank_ic_heatmap.png",
        ks,
        hs,
    )
    plot_robustness_heatmap(
        details,
        "annual_return",
        "Fee-adjusted Annual Return",
        output / "annual_return_heatmap.png",
        ks,
        hs,
    )
    return output


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Read robustness command-line options without loading market data."""
    parser = argparse.ArgumentParser(
        description="Run H x K factor robustness"
    )
    parser.add_argument("--factor", default="carry")
    parser.add_argument("--start", default=DEFAULT_START_DATE)
    parser.add_argument("--end", default=DEFAULT_END_DATE)
    parser.add_argument(
        "--k-values",
        nargs="+",
        type=int,
        default=list(DEFAULT_K_VALUES),
    )
    parser.add_argument(
        "--h-values",
        nargs="+",
        type=int,
        default=list(DEFAULT_H_VALUES),
    )
    parser.add_argument("--cost-rate", type=float, default=COST_RATE)
    parser.add_argument("--min-assets", type=int, default=10)
    parser.add_argument("--nw-lags", type=int, default=5)
    parser.add_argument(
        "--signal-min-days-to-maturity",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--trade-min-days-to-maturity",
        type=int,
        default=45,
    )
    parser.add_argument("--result-dir", type=Path, default=Path(RESULT_DIR))
    return parser.parse_args(argv)


def build_robustness_output_dir(
    result_dir: str | Path,
    factor_name: str,
    start_date: str,
    end_date: str,
    k_values: list[int] | tuple[int, ...],
    h_values: list[int] | tuple[int, ...],
) -> Path:
    """Return a grid-specific robustness output directory."""
    ks = validate_grid_values(k_values, "K")
    hs = validate_grid_values(h_values, "H")
    k_label = "_".join(str(value) for value in ks)
    h_label = "_".join(str(value) for value in hs)
    run_name = (
        f"{factor_name}-robustness-{start_date}-{end_date}"
        f"-K{k_label}-H{h_label}"
    )
    return Path(result_dir) / factor_name / run_name


def run_hk_robustness(
    factor_name: str = "carry",
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    k_values: list[int] | tuple[int, ...] = DEFAULT_K_VALUES,
    h_values: list[int] | tuple[int, ...] = DEFAULT_H_VALUES,
    base_config: ResearchConfig | None = None,
    result_dir: str | Path = RESULT_DIR,
) -> Path:
    """Calculate and save one complete H x K robustness report."""
    details = calculate_hk_robustness(
        factor_name,
        start_date,
        end_date,
        k_values,
        h_values,
        base_config,
    )
    output = build_robustness_output_dir(
        result_dir=result_dir,
        factor_name=factor_name,
        start_date=start_date,
        end_date=end_date,
        k_values=k_values,
        h_values=h_values,
    )
    return save_robustness_results(
        details,
        output,
        k_values=k_values,
        h_values=h_values,
    )


def main(argv: list[str] | None = None) -> Path:
    """Run the approved robustness grid and print its result directory."""
    args = parse_arguments(argv)
    config = ResearchConfig(
        cost_rate=args.cost_rate,
        min_assets=args.min_assets,
        nw_lags=args.nw_lags,
        signal_min_days_to_maturity=(
            args.signal_min_days_to_maturity
        ),
        trade_min_days_to_maturity=(
            args.trade_min_days_to_maturity
        ),
    )
    output = run_hk_robustness(
        factor_name=args.factor,
        start_date=args.start,
        end_date=args.end,
        k_values=args.k_values,
        h_values=args.h_values,
        base_config=config,
        result_dir=args.result_dir,
    )
    print(f"Robustness complete: {output}")
    return output


if __name__ == "__main__":
    main()
