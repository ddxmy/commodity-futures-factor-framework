"""Close-to-close execution sensitivity for prepared factor weights."""

import numpy as np
import pandas as pd


def run_close_to_close_backtest(
    weights: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    execution_lag: int,
    cost_rate: float,
) -> pd.DataFrame:
    """Return daily close-to-close P&L with close execution after the return."""
    if execution_lag not in {0, 1}:
        raise ValueError("execution_lag must be 0 or 1")
    if cost_rate < 0:
        raise ValueError("cost_rate cannot be negative")

    required_weight_columns = {
        "trade_date",
        "fut_code",
        "weight",
        "ts_code_A",
        "is_rebalance",
    }
    missing_weights = required_weight_columns - set(weights.columns)
    if missing_weights:
        raise ValueError(
            "weights are missing columns: " + ", ".join(sorted(missing_weights))
        )
    required_price_columns = {
        "trade_date",
        "ts_code",
        "close",
        "prev_close",
    }
    missing_prices = required_price_columns - set(prices.columns)
    if missing_prices:
        raise ValueError(
            "prices are missing columns: " + ", ".join(sorted(missing_prices))
        )

    targets = weights.copy()
    targets["trade_date"] = pd.to_datetime(targets["trade_date"])
    targets = targets.sort_values(["fut_code", "trade_date"])
    if targets.duplicated(["trade_date", "fut_code"]).any():
        raise ValueError("weights contain duplicate date-commodity keys")

    targets["target_weight"] = targets["weight"].where(
        targets["is_rebalance"]
    )
    targets["target_contract"] = targets["ts_code_A"].where(
        targets["is_rebalance"]
    ).astype("string")
    targets["desired_weight"] = targets.groupby("fut_code")[
        "target_weight"
    ].transform(lambda values: values.shift(execution_lag).ffill()).fillna(0.0)
    targets["desired_contract"] = targets.groupby("fut_code")[
        "target_contract"
    ].transform(lambda values: values.shift(execution_lag).ffill())
    targets.loc[
        targets["desired_weight"].abs().le(1e-12), "desired_contract"
    ] = None

    quote_data = prices.copy()
    quote_data["trade_date"] = pd.to_datetime(quote_data["trade_date"])
    if quote_data.duplicated(["trade_date", "ts_code"]).any():
        raise ValueError("prices contain duplicate date-contract keys")
    quote_data["close"] = pd.to_numeric(quote_data["close"], errors="coerce")
    quote_data["prev_close"] = pd.to_numeric(
        quote_data["prev_close"], errors="coerce"
    )
    quote_data["close_return"] = (
        quote_data["close"] / quote_data["prev_close"] - 1.0
    )
    close_lookup = quote_data.set_index(["trade_date", "ts_code"])[
        "close"
    ].to_dict()
    return_lookup = quote_data.set_index(["trade_date", "ts_code"])[
        "close_return"
    ].to_dict()

    fut_codes = targets["fut_code"].drop_duplicates().tolist()
    actual_weights = {fut_code: 0.0 for fut_code in fut_codes}
    actual_contracts = {fut_code: None for fut_code in fut_codes}
    records = []

    def has_close(trade_date, contract):
        if contract is None or pd.isna(contract):
            return False
        value = close_lookup.get((trade_date, contract), np.nan)
        return pd.notna(value) and value > 0

    for trade_date, day in targets.groupby("trade_date", sort=True):
        gross_return = 0.0
        turnover = 0.0
        blocked_trades = 0

        for row in day.itertuples(index=False):
            fut_code = row.fut_code
            previous_weight = actual_weights[fut_code]
            previous_contract = actual_contracts[fut_code]

            if not np.isclose(previous_weight, 0.0):
                close_return = return_lookup.get(
                    (trade_date, previous_contract), np.nan
                )
                if pd.isna(close_return):
                    raise ValueError(
                        "active close-to-close return is missing: "
                        f"{trade_date:%Y-%m-%d} {previous_contract}"
                    )
                gross_return += previous_weight * float(close_return)

            desired_weight = float(row.desired_weight)
            desired_contract = row.desired_contract
            if pd.isna(desired_contract):
                desired_contract = None
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
            elif changes_contract:
                can_execute = has_close(
                    trade_date, previous_contract
                ) and has_close(trade_date, desired_contract)
            elif had_position and not wants_position:
                can_execute = has_close(trade_date, previous_contract)
            else:
                can_execute = has_close(trade_date, desired_contract)

            if needs_trade and can_execute:
                if changes_contract:
                    turnover += abs(previous_weight) + abs(desired_weight)
                else:
                    turnover += abs(desired_weight - previous_weight)
                actual_weights[fut_code] = desired_weight
                actual_contracts[fut_code] = (
                    desired_contract if wants_position else None
                )
            elif needs_trade:
                blocked_trades += 1

        cost = turnover * cost_rate
        daily_return = gross_return - cost
        records.append(
            {
                "trade_date": trade_date,
                "gross_return": gross_return,
                "turnover": turnover,
                "cost": cost,
                "daily_return": daily_return,
                "blocked_trades": blocked_trades,
            }
        )

    result = pd.DataFrame(records)
    result["nav"] = (1.0 + result["daily_return"]).cumprod()
    return result
