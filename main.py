"""Command-line entry point for the standard single-factor research pipeline."""

import argparse
from pathlib import Path

import pandas as pd

from config.research_config import ResearchConfig
from config.settings import RESULT_DIR
from src.research_pipeline import run_factor_research


DEFAULT_START_DATE = "20190101"
DEFAULT_END_DATE = "20260710"
WEEKLY_FREQUENCIES = {"W-MON", "W-TUE", "W-WED", "W-THU", "W-FRI"}


def _parse_parameter_value(value: str) -> object:
    """Convert a command-line value to bool, None, int, float or string."""
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "none":
        return None
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def parse_factor_parameters(items: list[str] | None) -> dict[str, object]:
    """Parse repeated ``key=value`` factor parameters into a dictionary."""
    parameters: dict[str, object] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"factor parameter must use key=value: {item}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key or key in parameters:
            raise ValueError(f"invalid or duplicate factor parameter: {key}")
        parameters[key] = _parse_parameter_value(value.strip())
    return parameters


def parse_rebalance_frequency(value: str) -> int | str:
    """Parse a positive trading-day interval or supported weekly schedule."""
    normalized = value.strip().upper()
    try:
        interval = int(normalized)
    except ValueError:
        if normalized in WEEKLY_FREQUENCIES:
            return normalized
        raise argparse.ArgumentTypeError(
            "rebalance frequency must be a positive integer or W-MON to W-FRI"
        )
    if interval <= 0:
        raise argparse.ArgumentTypeError(
            "rebalance frequency must be a positive integer"
        )
    return interval


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Read command-line options without running database work."""
    parser = argparse.ArgumentParser(
        description="Run a cross-sectional futures single-factor report",
    )
    parser.add_argument("--factor", default="basis_momentum")
    parser.add_argument(
        "--factor-param",
        action="append",
        default=None,
        help="Factor-specific key=value; may be repeated",
    )
    parser.add_argument("--start", default=DEFAULT_START_DATE)
    parser.add_argument("--end", default=DEFAULT_END_DATE)
    parser.add_argument(
        "--rebalance-freq",
        type=parse_rebalance_frequency,
        default=1,
    )
    parser.add_argument("--cost-rate", type=float, default=0.0005)
    parser.add_argument("--min-assets", type=int, default=10)
    parser.add_argument("--group-count", type=int, default=5)
    parser.add_argument("--rolling-window", type=int, default=20)
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
    parser.add_argument("--zscore-clip", type=float, default=3.0)
    parser.add_argument("--result-dir", type=Path, default=Path(RESULT_DIR))
    return parser.parse_args(argv)


def validate_dates(start_date: str, end_date: str) -> None:
    """Validate compact YYYYMMDD dates before accessing the database."""
    parsed = []
    for name, value in [("start", start_date), ("end", end_date)]:
        if not isinstance(value, str) or len(value) != 8 or not value.isdigit():
            raise ValueError(f"{name} must use YYYYMMDD format")
        parsed.append(pd.to_datetime(value, format="%Y%m%d", errors="raise"))
    if parsed[0] >= parsed[1]:
        raise ValueError("start must be earlier than end")


def main(argv: list[str] | None = None) -> Path:
    """Build configuration, run one factor and print its result directory."""
    args = parse_arguments(argv)
    validate_dates(args.start, args.end)
    factor_parameters = parse_factor_parameters(args.factor_param)
    if args.factor == "basis_momentum":
        factor_parameters.setdefault("variant", "AB")
        factor_parameters.setdefault("lookback", 252)

    config = ResearchConfig(
        rebalance_freq=args.rebalance_freq,
        cost_rate=args.cost_rate,
        min_assets=args.min_assets,
        group_count=args.group_count,
        rolling_ic_window=args.rolling_window,
        nw_lags=args.nw_lags,
        signal_min_days_to_maturity=(
            args.signal_min_days_to_maturity
        ),
        trade_min_days_to_maturity=(
            args.trade_min_days_to_maturity
        ),
    )
    print(f"Running {args.factor}: {args.start} to {args.end}")
    output_dir = run_factor_research(
        factor_name=args.factor,
        start_date=args.start,
        end_date=args.end,
        factor_parameters=factor_parameters,
        research_config=config,
        result_dir=args.result_dir,
        zscore_clip=args.zscore_clip,
    )
    print(f"Research complete: {output_dir}")
    return output_dir


if __name__ == "__main__":
    main()
