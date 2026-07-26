"""精度管理工具。"""

import numpy as np
import mpmath

# 向后兼容：统一从 domain.exceptions 导入
from ...domain.exceptions import PrecisionFailureError


def set_mp_precision(dps):
    """设置 mpmath 全局精度。"""
    mpmath.mp.dps = dps
