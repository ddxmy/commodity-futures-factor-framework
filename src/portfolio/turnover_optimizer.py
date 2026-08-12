"""Turnover-constrained optimizer for optional portfolio experiments."""

import cvxpy as cp
import numpy as np
import pandas as pd


def optimize_portfolio(
    scores: pd.Series,
    previous_weights: pd.Series,
    eligible: pd.Series,
    contract_changed: pd.Series,
    turnover_limit: float = 0.15,
    max_abs_weight: float = 0.05,
    is_initial: bool = False,
) -> tuple[pd.Series, dict]:
    """Maximize factor exposure subject to neutrality and turnover limits."""

    series_inputs = {
        "scores": scores,
        "previous_weights": previous_weights,
        "eligible": eligible,
        "contract_changed": contract_changed,
    }

    if not all(
        isinstance(value, pd.Series)
        for value in series_inputs.values()
    ):
        raise TypeError(
            "优化器的四个输入必须都是pd.Series"
        )

    if scores.empty:
        raise ValueError(
            "优化器输入不能为空"
        )

    if not scores.index.is_unique:
        raise ValueError(
            "商品索引不能重复"
        )

    has_misaligned_index = any(
        not value.index.equals(scores.index)
        for value in series_inputs.values()
    )

    if has_misaligned_index:
        raise ValueError(
            "所有输入必须使用完全相同的商品索引和顺序"
        )

    numeric_inputs = {
        "scores": scores,
        "previous_weights": previous_weights,
    }

    for name, values in numeric_inputs.items():
        is_finite = np.isfinite(
            values.to_numpy(dtype=float)
        ).all()

        if not is_finite:
            raise ValueError(
                f"{name}包含缺失值或无穷值"
            )

    if eligible.isna().any():
        raise ValueError(
            "eligible不能包含缺失值"
        )

    if contract_changed.isna().any():
        raise ValueError(
            "contract_changed不能包含缺失值"
        )

    if not 0 < turnover_limit <= 1:
        raise ValueError(
            "turnover_limit必须位于(0, 1]之间"
        )

    if not 0 < max_abs_weight <= 1:
        raise ValueError(
            "max_abs_weight必须位于(0, 1]之间"
        )

    asset_index = scores.index

    score_values = scores.to_numpy(
        dtype=float
    )

    previous_values = previous_weights.to_numpy(
        dtype=float
    )

    eligible_values = eligible.to_numpy(
        dtype=bool
    )

    # 流动性退出是强制交易，不占用主动调仓的换手预算。
    base_weights = previous_weights.where(
        eligible,
        0.0,
    )
    base_values = base_weights.to_numpy(
        dtype=float
    )
    mandatory_exit_turnover = float(
        np.abs(
            previous_values - base_values
        ).sum()
    )

    # 只有已有仓位且仍可交易的品种更换合约时，才计为换月。
    roll_mask = (
        contract_changed.astype(bool)
        & eligible.astype(bool)
        & base_weights.ne(0.0)
    )
    roll_values = roll_mask.to_numpy(
        dtype=bool
    )

    asset_count = len(scores)

    effective_turnover_limit = (
        1.0 if is_initial
        else turnover_limit
    )

    weight = cp.Variable(asset_count)

    objective = cp.Maximize(
        score_values @ weight
    )

    normal_mask = (
        ~roll_values
    ).astype(float)
    roll_mask_numeric = (
        roll_values
    ).astype(float)

    normal_turnover = cp.sum(
        cp.multiply(
            normal_mask,
            cp.abs(
                weight - base_values
            ),
        )
    )
    roll_turnover = cp.sum(
        cp.multiply(
            roll_mask_numeric,
            np.abs(base_values)
            + cp.abs(weight),
        )
    )
    optimized_turnover_expression = (
        normal_turnover
        + roll_turnover
    )

    portfolio_constraints = [
        cp.sum(weight) == 0,
        cp.norm1(weight) <= 1.0,
        cp.abs(weight) <= max_abs_weight,
    ]

    if (~eligible_values).any():
        portfolio_constraints.append(
            weight[~eligible_values] == 0
        )

    turnover_limit_parameter = cp.Parameter(
        nonneg=True,
        value=effective_turnover_limit,
    )
    constraints = [
        *portfolio_constraints,
        optimized_turnover_expression
        <= turnover_limit_parameter,
    ]

    problem = cp.Problem(
        objective,
        constraints,
    )

    # 该模型是线性规划，SciPy 会调用 HiGHS 获得稳定的数值解。
    problem.solve(
        solver=cp.SCIPY
    )

    valid_statuses = {
        cp.OPTIMAL,
        cp.OPTIMAL_INACCURATE,
    }
    infeasible_statuses = {
        cp.INFEASIBLE,
        cp.INFEASIBLE_INACCURATE,
    }
    turnover_limit_relaxed = False

    if problem.status not in valid_statuses:
        if problem.status not in infeasible_statuses:
            raise RuntimeError(
                f"组合优化失败：{problem.status}"
            )

        minimum_turnover_problem = cp.Problem(
            cp.Minimize(
                optimized_turnover_expression
            ),
            portfolio_constraints,
        )
        minimum_turnover_problem.solve(
            solver=cp.SCIPY
        )

        if minimum_turnover_problem.status not in valid_statuses:
            raise RuntimeError(
                "无法求得最低可行换手："
                f"{minimum_turnover_problem.status}"
            )

        minimum_required_turnover = float(
            minimum_turnover_problem.value
        )
        effective_turnover_limit = max(
            effective_turnover_limit,
            minimum_required_turnover + 1e-8,
        )
        turnover_limit_parameter.value = (
            effective_turnover_limit
        )
        turnover_limit_relaxed = True

        problem.solve(
            solver=cp.SCIPY
        )

        if problem.status not in valid_statuses:
            raise RuntimeError(
                "放宽换手后组合优化仍失败："
                f"{problem.status}"
            )

    weight_values = np.asarray(
        weight.value,
        dtype=float,
    )

    weight_values[
        np.abs(weight_values) < 1e-10
    ] = 0.0

    optimized_weights = pd.Series(
        weight_values,
        index=asset_index,
        name="optimized_weight",
    )

    normal_turnover_value = float(
        (
            normal_mask
            * np.abs(
                weight_values - base_values
            )
        ).sum()
    )
    roll_turnover_value = float(
        (
            roll_mask_numeric
            * (
                np.abs(base_values)
                + np.abs(weight_values)
            )
        ).sum()
    )
    optimized_turnover = (
        normal_turnover_value
        + roll_turnover_value
    )

    tolerance = 1e-6
    constraint_violations = []

    if abs(optimized_weights.sum()) > tolerance:
        constraint_violations.append("净仓位不为0")
    if optimized_weights.abs().sum() > 1.0 + tolerance:
        constraint_violations.append("总名义仓位超过100%")
    if optimized_weights.abs().max() > max_abs_weight + tolerance:
        constraint_violations.append("单品种仓位超过上限")
    if (
        (~eligible_values).any()
        and optimized_weights.loc[~eligible_values].abs().max()
        > tolerance
    ):
        constraint_violations.append("流动性不合格品种仍有仓位")
    if optimized_turnover > effective_turnover_limit + tolerance:
        constraint_violations.append("优化换手超过有效上限")

    if constraint_violations:
        raise RuntimeError(
            "优化结果未通过约束检查："
            + "；".join(constraint_violations)
        )

    diagnostics = {
        "solver_status": problem.status,
        "objective_value": float(problem.value),
        "configured_turnover_limit": turnover_limit,
        "effective_turnover_limit": effective_turnover_limit,
        "optimized_turnover": optimized_turnover,
        "mandatory_exit_turnover": mandatory_exit_turnover,
        "gross_weight": float(
            optimized_weights.abs().sum()
        ),
        "net_weight": float(
            optimized_weights.sum()
        ),
        "constraint_binding": (
            abs(
                optimized_turnover
                - effective_turnover_limit
            ) <= 1e-6
        ),
        "turnover_limit_relaxed": turnover_limit_relaxed,
    }

    return optimized_weights, diagnostics
