# 因子表现记录

本目录保存需要长期维护的研究结论。原始 CSV、净值和图表仍保存在
`results/`；这里记录因子定义、统一口径、样本内外表现、参数稳定性和当前判断。

## 当前档案

| 因子 | 当前状态 | 研究样本 | 近期样本 | 数据截至 |
|---|---|---|---|---|
| [基差动量](basis_momentum.md) | Degrading | 2010–2022表现强且显著 | 2023年至今仍为正，但明显衰减 | 2026-07-10 |
| [Carry](carry.md) | Degrading | K120在多个H下表现较强 | 2023年至今接近失效 | 2026-07-10 |

状态含义：

- `Promising`：初步有效，尚缺独立样本验证。
- `Validated`：近期样本仍有效，且参数邻域稳定。
- `Degrading`：早期有效，但近期预测能力或策略收益明显衰减。
- `Rejected`：近期失效，或结果只依赖单一参数。
- `Needs Review`：数据时点、路径或执行口径仍有待核对。

## 证据层级

```text
results/<factor>/...              原始回测证据
results/factor_registry.csv       所有标准报告的逐实验快照
docs/factor_performance/<factor>  长期研究判断
```

`results/factor_registry.csv` 当前由各报告目录中的 `run_config.csv`、
`strategy_metrics.csv` 和 `ic_summary.csv` 汇总。`results/` 不受 Git 管理，
因此 CSV 是便于筛选的本地快照，因子档案才是应随代码提交的长期记录。

## 固定记录口径

每条标准实验由以下字段共同确定：

```text
factor + variant + K + H + start_date + end_date + portfolio + cost
```

样本角色按日期机械划分：

- `research_sample`：结束日期不晚于 `20221231`。
- `holdout_sample`：开始日期不早于 `20230101`。
- `mixed_sample`：跨越上述分界，不能用于严格的前后样本比较。

这里的 `holdout_sample` 只表示固定日期切分，不保证研究者从未观察过这段数据。

## 新结果如何登记

1. 保留完整报告目录，不手工修改报告 CSV。
2. 将标准报告的 Rank 和 Z-score 两行都加入 `results/factor_registry.csv`。
3. 在对应因子档案中更新研究样本、近期样本和H×K结论。
4. 记录不利结果、异常路径和口径变化，不只记录最优参数。
5. 因子公式、数据字段或执行规则变化时，新建可区分的版本，避免与旧结果混用。

## Registry字段

| 字段 | 含义 |
|---|---|
| `factor`, `variant` | 因子模块与变体 |
| `K` | 因子回看窗口，单位为交易日 |
| `H` | 调仓间隔；整数为交易日，`W-FRI`为当周最后一个交易日 |
| `sample_role` | `research_sample`、`holdout_sample`或`mixed_sample` |
| `portfolio` | `rank`或`zscore` |
| `annual_return`, `sharpe`, `max_drawdown` | 费后策略表现 |
| `mean_rank_ic`, `rank_ic_nw_p_value` | RankIC均值及Newey–West显著性 |
| `annual_turnover`, `annual_cost` | 年化换手与年化成本 |
| `cost_share_of_gross_return` | 成本占毛收益比例 |
| `result_directory` | 可追溯的原始报告目录 |

## 相关文档

- [运行命令与结果文件示例](../运行与结果示例.md)
- [项目入口说明](../../README.md)
