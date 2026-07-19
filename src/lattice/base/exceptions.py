"""格约减模块统一异常定义。

将异常类放在 base 层，避免 adapter ↔ base 的循环导入。
"""

import numpy as np


class LatticeReductionError(Exception):
    """格约减模块统一异常基类。"""
    pass


class InvalidBasisError(LatticeReductionError):
    """格基格式或内容非法。"""
    pass


class ReductionFailedError(LatticeReductionError):
    """约减过程失败。"""
    pass


class PrecisionFailureError(LatticeReductionError):
    """精度不足，可通过升级 dps 重试。"""

    def __init__(self, reason, current_dps):
        self.reason = reason
        self.current_dps = current_dps
        super().__init__(f"Precision failure at dps={current_dps}: {reason}")


__all__ = [
    "LatticeReductionError",
    "InvalidBasisError",
    "ReductionFailedError",
    "PrecisionFailureError",
]