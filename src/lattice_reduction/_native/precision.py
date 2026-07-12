"""精度管理工具。"""

import numpy as np
import mpmath


class PrecisionFailureError(Exception):
    """精度不足，可通过升级 dps 重试。"""

    def __init__(self, reason, current_dps):
        self.reason = reason
        self.current_dps = current_dps
        super().__init__(f"Precision failure at dps={current_dps}: {reason}")


def set_mp_precision(dps):
    """设置 mpmath 全局精度。"""
    mpmath.mp.dps = dps
