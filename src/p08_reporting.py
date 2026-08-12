"""CSV and chart output for a completed single-factor research run."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


plt.rcParams["font.family"] = ["Times New Roman", "Kaiti SC"]
plt.rcParams["axes.unicode_minus"] = False


def calculate_rolling_ic(
    ic_series: pd.DataFrame,
    rolling_window: int,
) -> pd.DataFrame:
    """Return IC and Rank IC with complete-window rolling means."""
    if not isinstance(rolling_window, int) or rolling_window <= 0:
        raise ValueError("rolling_window must be a positive integer")
    required = {"signal_date", "ic", "rank_ic"}
    missing = required - set(ic_series.columns)
    if missing:
        raise ValueError("ic_series is missing columns: " + ", ".join(sorted(missing)))

    rolling = ic_series.sort_values("signal_date").reset_index(drop=True).copy()
    rolling["ic_rolling"] = rolling["ic"].rolling(
        rolling_window, min_periods=rolling_window
    ).mean()
    rolling["rank_ic_rolling"] = rolling["rank_ic"].rolling(
        rolling_window, min_periods=rolling_window
    ).mean()
    return rolling


def save_result_tables(
    output_dir: str | Path,
    config: dict,
    strategy_metrics: pd.DataFrame,
    strategy_nav: pd.DataFrame,
    factor_results: dict[str, pd.DataFrame],
) -> None:
    """Save run configuration and the seven standard result tables."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    required_results = {"ic_summary", "ic_series", "group_returns", "group_nav"}
    missing = required_results - set(factor_results)
    if missing:
        raise ValueError("factor_results is missing keys: " + ", ".join(sorted(missing)))

    pd.DataFrame([config]).to_csv(output_path / "run_config.csv", index=False)
    strategy_metrics.round(6).to_csv(
        output_path / "strategy_metrics.csv", index=False
    )
    strategy_nav.to_csv(output_path / "strategy_nav.csv")
    factor_results["ic_summary"].round(6).to_csv(
        output_path / "ic_summary.csv", index=False
    )
    factor_results["ic_series"].to_csv(
        output_path / "ic_series.csv", index=False
    )
    factor_results["group_returns"].to_csv(
        output_path / "group_returns.csv", index=False
    )
    factor_results["group_nav"].to_csv(
        output_path / "group_nav.csv", index=False
    )


def _normalize_plot_series(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize each nonempty series to one at its first valid observation."""
    normalized = data.copy()
    for column in normalized.columns:
        valid = normalized[column].dropna()
        if valid.empty:
            raise ValueError(f"plot series {column} has no valid observations")
        first_value = valid.iloc[0]
        if first_value == 0:
            raise ValueError(f"plot series {column} starts at zero")
        normalized[column] = normalized[column] / first_value
    return normalized


def plot_strategy_nav(strategy_nav: pd.DataFrame, output_path: str | Path) -> None:
    """Plot Rank and Z-score strategy NAV on a common starting level."""
    normalized_nav = _normalize_plot_series(strategy_nav)
    figure, axis = plt.subplots(figsize=(12, 5.5))
    normalized_nav.plot(
        ax=axis,
        color=["#3B78A8", "#E07A34"],
        linewidth=1.6,
    )
    axis.axhline(1.0, color="#777777", linestyle="--", linewidth=1.0)
    axis.set_title("Rank 与 Z-score 策略净值")
    axis.set_xlabel("日期")
    axis.set_ylabel("净值（共同起点 = 1）")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_group_nav(group_nav: pd.DataFrame, output_path: str | Path) -> None:
    """Plot cumulative NAV from the highest-factor G1 to lowest-factor G5."""
    group_columns = ["G1", "G2", "G3", "G4", "G5"]
    missing = set(group_columns) - set(group_nav.columns)
    if missing:
        raise ValueError("group_nav is missing columns: " + ", ".join(sorted(missing)))
    plot_data = group_nav.set_index("signal_date")[group_columns]
    plot_data = _normalize_plot_series(plot_data)

    figure, axis = plt.subplots(figsize=(12, 5.5))
    plot_data.plot(
        ax=axis,
        color=["#B51F35", "#E76F51", "#999999", "#62A8D1", "#2166AC"],
        linewidth=1.5,
    )
    axis.axhline(1.0, color="#444444", linewidth=0.9)
    axis.set_title("五组累计净值")
    axis.set_xlabel("信号日期")
    axis.set_ylabel("累计净值")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, ncol=5)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_ic_history(
    ic_series: pd.DataFrame,
    rolling_window: int,
    output_path: str | Path,
) -> None:
    """Plot daily IC, Rank IC and their rolling means."""
    plot_data = calculate_rolling_ic(ic_series, rolling_window).set_index(
        "signal_date"
    )
    figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(
        plot_data.index,
        plot_data["ic"],
        color="#9EBBDD",
        alpha=0.45,
        linewidth=1.0,
        label="每日 IC",
    )
    axes[0].plot(
        plot_data.index,
        plot_data["ic_rolling"],
        color="#2E6DB4",
        linewidth=1.8,
        label=f"{rolling_window} 日均值",
    )
    axes[0].axhline(0.0, color="#333333", linewidth=0.9)
    axes[0].set_title("IC")
    axes[0].set_ylabel("相关系数")
    axes[0].grid(alpha=0.2)
    axes[0].legend(frameon=False)

    axes[1].plot(
        plot_data.index,
        plot_data["rank_ic"],
        color="#E9A6A6",
        alpha=0.45,
        linewidth=1.0,
        label="每日 Rank IC",
    )
    axes[1].plot(
        plot_data.index,
        plot_data["rank_ic_rolling"],
        color="#C43D3D",
        linewidth=1.8,
        label=f"{rolling_window} 日均值",
    )
    axes[1].axhline(0.0, color="#333333", linewidth=0.9)
    axes[1].set_title("Rank IC")
    axes[1].set_xlabel("信号日期")
    axes[1].set_ylabel("相关系数")
    axes[1].grid(alpha=0.2)
    axes[1].legend(frameon=False)

    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
