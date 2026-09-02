# 商品期货五因子简单策略

本项目实现一套国内商品期货日频截面策略。策略使用五个期货因子，每半年根据历史成交额选取 Top 40 商品，在截面排序后加入单品种逆波动率调整，对比全截面与前后 10% 两种组合构建方法。

仓库同时保留了通用单因子研究流程，可用于检查因子的 IC、Rank IC、五组收益、参数鲁棒性和交易成本。

## 五个因子

| 因子 | 默认参数 | 主要含义 |
|---|---:|---|
| `basis_momentum` | AB，K=252 | A、B 流动性合约的收益差，按到期日差年化后取滚动均值 |
| `carry` | K=90 | A、B 合约收盘价差，按到期日差年化后取滚动均值 |
| `spotmain` | K=90 | 现货与 A 合约之间的年化基差及其滚动均值 |
| `s_warehouse` | K=90，平滑窗口20 | 平滑仓单量相对 K 个交易日前的变化，库存下降对应更高因子值 |
| `t_rank` | K=20 | 每日主力合约收益的截面标准化排名，再取滚动均值 |

A、B、C 表示当日按成交量排名的第一、第二和第三个合约，不是固定的近月或远月合约。合约收益在真实合约内计算，不使用换月价差构造收益。

## 策略设定

```text
前120个交易日平均成交额
        ↓
每年1月和7月选取 Top 40 商品
        ↓
五个因子分别在每日截面映射到 [-1, 1]
        ↓
因子分数 / 单品种20日波动率
        ↓
全截面组合  对比  前后10%组合
        ↓
多空两侧分别归一化为 +0.5 / -0.5
        ↓
五个因子策略各占20%，组成等权组合
```

默认交易口径：

- `T` 日收盘后形成因子和目标权重，`T+1` 日开盘尝试执行。
- 开盘价缺失时不假设成交，实际持仓继续保留。
- 交易合约要求至少剩余 45 个自然日，换月的平仓和开仓均计入换手。
- 单边交易成本默认为 `0.0005`。
- 每日调仓，收益率口径不假设具体初始资金。

## 快速运行

### 1. 安装依赖

```bash
python -m pip install -r requirements.txt
```

### 2. 准备数据库

项目默认从父目录读取：

```text
../data/db/huatai_quant.db
```

数据库路径由 [`config/settings.py`](config/settings.py) 中的 `DB_PATH` 管理。原始数据库体积较大，不随 Git 仓库提交；运行策略前需要在本地准备对应的 SQLite 数据库。

### 3. 运行五因子策略

```bash
python -m src.multi_factor_strategy \
  --start 20200101 \
  --end 20260701
```

重跑相同参数时，程序默认保护已有结果。确认允许覆盖本次策略所属文件时使用：

```bash
python -m src.multi_factor_strategy \
  --start 20200101 \
  --end 20260701 \
  --overwrite
```

程序不会删除结果目录中的其他文件。

## 核心输出

默认结果位于：

```text
results/multi_factor_strategy/
  top40_amount120_vol20-20200101-20260701/
    daily/
```

`daily/` 根目录保存：

- `strategy_metrics.csv`：五个因子与等权组合的全周期指标。
- `factor_ic_summary.csv`、`factor_ic_series.csv`、`factor_ic_annual.csv`：因子 IC 和 Rank IC。
- `universe_ranking.csv`、`universe_members.csv`、`universe_changes.csv`：商品池排名、成员和进出记录。
- `combined_nav_comparison.png` 及五张单因子方法对比图。

两个组合子目录为 `full_pool_invvol/` 和 `tail10_invvol/`，分别保存：

```text
daily_returns.csv
nav.csv
daily_diagnostics.csv
weights.csv
annual_metrics.csv
annual_returns.csv
nav_summary.png
```

详细命令、参数和结果文件见 [运行与结果说明](docs/运行与结果示例.md)。

## 代码结构

```text
main.py                         单因子通用入口
config/                         路径和研究参数
src/multi_factor_strategy.py    五因子主策略
src/factors/                    因子计算模块
src/p01_*.py 至 p08_*.py      数据、组合、回测、评价和报告
src/hk_robustness.py            H×K 参数鲁棒性
src/ridge_five_factor_strategy.py
src/tree_five_factor_strategy.py 线性与树模型因子合成对比
tests/                          单元测试
```

## 扩展研究

### 单因子报告

```bash
python main.py \
  --factor carry \
  --factor-param lookback=90 \
  --start 20190101 \
  --end 20260710 \
  --rebalance-freq 5
```

`main.py` 统一输出 Rank/Z-score 组合、IC/Rank IC、五组收益、净值和策略指标。新因子遵守 `trade_date | fut_code | raw_factor` 接口，可参考 [`src/factors/_factor_template.py`](src/factors/_factor_template.py)。

### H×K 鲁棒性

```bash
python -m src.hk_robustness \
  --factor carry \
  --start 20220101 \
  --end 20251231 \
  --k-values 30 60 90 120 \
  --h-values 1 5 10
```

### 因子表现档案

- [因子档案索引](docs/factor_performance/README.md)
- [基差动量](docs/factor_performance/basis_momentum.md)
- [Carry](docs/factor_performance/carry.md)

`results/` 保存可重新生成的本地结果，默认不受 Git 管理；`docs/factor_performance/` 只保存已经人工复核的长期研究记录。

## 测试

```bash
python -m unittest discover -s tests -v
python -m compileall -q config src main.py
```

测试覆盖因子公式、数据时点、合约选择、次日开盘执行、缺失成交、换手、成本、IC、组合权重和结果输出。

## 研究边界

- 该策略是日频研究框架，不包含滑点、涨跌停、保证金、合约乘数、品种费率差异、成交量参与率和持仓限额。
- 数据库、大量中间数据与生成结果未随仓库提交，仅克隆代码不能直接重建完整实证结果。
- 历史回测不代表未来表现，因子有效性需继续通过样本外和分阶段结果检查。
