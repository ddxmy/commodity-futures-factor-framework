# 商品期货横截面单因子研究模板

本项目提供一套可复用的国内商品期货日频横截面单因子流程。因子模块负责生成原始因子值，公共框架统一处理流动性、权重、交易执行、成本、IC、分组回测和报告。

## 流程结构

```text
P01 读取市场数据
 -> P02 选择主力交易合约和 A/B/C 期限合约
 -> 因子插件计算 raw_factor
 -> P03 验证标准因子面板
 -> P04 生成调仓日期
 -> P05 构建 Rank 和 Z-score 多空权重
 -> P06 次日开盘执行并计算换手、成本和收益
 -> P07 计算 IC、Rank IC 和五组收益
 -> P08 输出 CSV 和图表
```

正式入口是 [`main.py`](main.py)，公共研究流程使用 `p01` 至 `p08` 模块。

## 运行基差动量

```bash
python main.py \
  --factor basis_momentum \
  --factor-param variant=AB \
  --factor-param lookback=252 \
  --start 20190101 \
  --end 20260710
```

`variant` 可以使用 `AB`、`BC` 或 `BLEND`。默认进行日频调仓，同时输出 Rank 和截面 Z-score 策略。

结果保存在：

```text
results/AB_L252-20190101-20260710/
```

其中包括运行参数、策略指标、策略净值、IC、Rank IC、五组收益、五组净值和三张图。

## 新增一个因子

1. 参考 [`src/factors/_factor_template.py`](src/factors/_factor_template.py)。
2. 在 `src/factors/` 新建文件，例如 `inventory.py`。
3. 实现 `calculate_factor()`。
4. 在 `tests/` 增加因子公式和数据时点测试。
5. 使用 `python main.py --factor inventory ...` 运行。

每个因子必须返回：

```text
trade_date | fut_code | raw_factor
```

- `trade_date`：因子进入当日横截面的交易日。
- `fut_code`：商品品种代码，例如 `RB`、`CU`。
- `raw_factor`：尚未进行 Rank 或 Z-score 的原始因子值。

允许保留额外的审计字段，例如：

```text
observation_date | available_date | source_value
```

公共框架不会读取具体公式，只识别标准字段。因此新增因子不需要修改 P05、P06、P07 或报告代码。

## 基本面数据的日期

库存、仓单、产量等数据必须区分：

```text
observation_date -> available_date -> trade_date -> execution_date
```

- `observation_date`：数据描述的经济时期。
- `available_date`：当时最早能够获得数据的日期。
- `trade_date`：因子进入横截面的日期。
- `execution_date`：实际执行日期，当前默认为下一交易日开盘。

可以使用 [`src/data_alignment.py`](src/data_alignment.py) 中的 `align_factor_to_calendar()`。默认 `publication_lag=1`，即不知道准确公布时间时，从公布后的第一个交易日开始使用。`max_staleness` 用于禁止长期沿用过期数据。

## 因子参数与公共参数

因子参数通过重复的 `--factor-param key=value` 传入：

```bash
python main.py \
  --factor inventory \
  --factor-param lookback=60 \
  --factor-param publication_lag=1 \
  --start 20190101 \
  --end 20260710
```

公共参数由 [`config/research_config.py`](config/research_config.py) 管理，包括：

- 调仓频率和交易成本。
- 最低横截面商品数。
- 流动性门槛。
- 分组数量和滚动 IC 窗口。
- 可选组合优化器参数。

因子参数只描述公式；公共参数描述所有因子共同遵守的研究口径。

## 时间和交易口径

- `t` 日收盘后形成因子和目标权重。
- `t+1` 日开盘尝试执行。
- 开盘价缺失时不假设成交，实际仓位继续沿用。
- 持仓期锁定交易合约。
- 换月时平旧合约并开新合约，换手和成本均计入。
- 基础成本为单边万五。

## 测试

运行全部测试：

```bash
python -m unittest discover -s tests -v
```

检查语法：

```bash
python -m compileall config src main.py
```

新因子至少应测试：

- 公式方向和手算结果。
- 日期品种唯一性。
- 公布滞后和未来函数。
- 缺失值和极端值。
- 参数边界。

## 组合优化器

换手约束优化器位于 [`src/portfolio/turnover_optimizer.py`](src/portfolio/turnover_optimizer.py)。它是 P06 的可选工具，基础单因子报告默认不启用。多因子组合研究可以通过 `use_optimizer=True` 单独比较。
