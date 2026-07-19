"""精度相关常量 — 被 base/precision.py 和 algorithms/bkz_params.py 共用。

放在此处是为了打破 precision.py → bkz/ 的循环导入依赖。
"""

# mpmath 精度配置
AUTO_PRECISION_DIM_LOW = 150
AUTO_PRECISION_DIM_HIGH = 250
MP_DPS_LOW = 50
MP_DPS_HIGH = 100
MP_STUCK_THRESHOLD = 100
MP_MAX_RETRY = 1

PRECISION_MODE_LOW = 0
PRECISION_MODE_AUTO = 1
PRECISION_MODE_HIGH = 2
