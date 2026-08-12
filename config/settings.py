"""
项目级配置：路径、品种列表、参数等。
"""

import os

# ---- 路径 ----
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
RESULT_DIR = os.path.join(PROJECT_DIR, "results")

# 公用数据库保存在华泰量化根目录的 data/db/ 中。
PARENT_DIR = os.path.dirname(PROJECT_DIR)
DB_PATH = os.path.join(PARENT_DIR, "data", "db", "huatai_quant.db")

# ---- 品种列表（按需修改） ----
SYMBOLS = [
    # 例如: "RB", "HC", "I", "J", "JM", "MA", "TA", ...
]

# ---- 因子参数 ----
# 基差动量回溯窗口（交易日）
LOOKBACK = 120
# 周频和双周频回测预留足够时间，在合约失去流动性前完成换月。
MIN_DAYS_TO_MATURITY = 45

# ---- 策略参数 ----
COST_RATE = 0.0005      # 单边万5

# ---- 商品流动性参数 ----
LIQ_LOOKBACK = 20
MIN_VOL = 3000           # 弱化
MIN_OI = 20000           # 主力过滤器
MIN_AMOUNT = 5000        # 万元
LIQUIDITY_MIN = 0.1

# ---- 回测参数 ----
START_DATE = "2015-01-01"
END_DATE = "2020-12-31"
INITIAL_CAPITAL = 10_000_000  # 初始资金
