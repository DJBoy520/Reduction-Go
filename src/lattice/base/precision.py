"""精度管理器 — 维度分层、失败检测、重试控制。"""

import hashlib
import logging
import numpy as np

from .precision_constants import (
    AUTO_PRECISION_DIM_LOW, AUTO_PRECISION_DIM_HIGH,
    MP_DPS_LOW, MP_DPS_HIGH, MP_STUCK_THRESHOLD, MP_MAX_RETRY,
    PRECISION_MODE_LOW, PRECISION_MODE_AUTO, PRECISION_MODE_HIGH,
)
from .precision_errors import PrecisionFailureError
from .exceptions import ReductionFailedError

logger = logging.getLogger(__name__)


def get_initial_dps(dim, force_dps=None, auto_enable=True):
    """根据维度返回 (初始dps, 模式枚举值)。

    规则：
      - force_dps 不为 None → 强制使用，模式为 PRECISION_MODE_LOW
      - auto_enable=False → 用低精度，模式为 PRECISION_MODE_LOW
      - auto_enable=True → 按维度分层
    """
    if force_dps is not None:
        return force_dps, PRECISION_MODE_LOW

    if not auto_enable:
        return MP_DPS_LOW, PRECISION_MODE_LOW

    if dim > AUTO_PRECISION_DIM_HIGH:
        return MP_DPS_HIGH, PRECISION_MODE_HIGH
    elif dim >= AUTO_PRECISION_DIM_LOW:
        return MP_DPS_LOW, PRECISION_MODE_AUTO
    else:
        return MP_DPS_LOW, PRECISION_MODE_AUTO


def check_precision_failure(gsn, stuck_counter=0):
    """检测当前精度是否失效。

    Args:
        gsn: float64 numpy 数组（GSO 平方范数）
        stuck_counter: 连续无更新迭代次数

    Returns:
        (is_failure, reason_str)
    """
    if gsn is not None:
        if np.any(np.isnan(gsn)):
            return True, "GSO 范数包含 NaN"
        if np.any(np.isinf(gsn)):
            return True, "GSO 范数包含 Inf"
        if np.any(gsn < -1e-30):
            return True, f"GSO 范数存在显著负值 (min={np.min(gsn):.2e})"

    if stuck_counter >= MP_STUCK_THRESHOLD:
        return True, f"死循环: 连续 {stuck_counter} 次无实质变化"

    return False, ""


class PrecisionContext:
    """精度管理上下文 — 维度分层 + 失败检测 + 重试控制。"""

    def __init__(self, original_basis, dim, initial_dps, mode, max_retry=MP_MAX_RETRY):
        self.original_basis = original_basis.copy()
        self.dim = dim
        self.current_dps = initial_dps
        self.mode = mode
        self.retry_count = 0
        self.max_retry = max_retry
        self.stuck_counter = 0
        self._last_hash = None

    def update_streak(self, current_basis):
        """更新死循环计数：基变化则清零，不变则 +1。"""
        h = hashlib.md5(current_basis.tobytes()).hexdigest()
        if h == self._last_hash:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
        self._last_hash = h

    def can_upgrade(self):
        """是否允许升级精度。"""
        return self.mode == PRECISION_MODE_AUTO and self.retry_count < self.max_retry

    def upgrade_dps(self):
        """升级到高精度，返回新 dps。"""
        self.retry_count += 1
        self.current_dps = MP_DPS_HIGH
        self.stuck_counter = 0
        self._last_hash = None
        logger.warning(f"精度升级: dps={MP_DPS_HIGH} (第 {self.retry_count} 次重试)")
        return self.current_dps

    def handle_precision_failure(self, exc: PrecisionFailureError) -> int:
        """处理精度失败：若可升级则升级精度，否则抛出异常。

        Args:
            exc: 捕获到的精度失败异常，包含 reason 和 current_dps。

        Returns:
            升级后的新 dps 值。

        Raises:
            ReductionFailedError: 当精度已达上限无法继续升级时。
        """
        if not self.can_upgrade():
            raise ReductionFailedError(
                f"精度升级已达上限 (dim={self.dim}, dps={self.current_dps})，原因: {exc.reason}"
            ) from exc
        return self.upgrade_dps()

    def reset_basis(self):
        """重置回原始整数基。"""
        return self.original_basis.copy()

    def build_error_msg(self, reason):
        """生成最终失败提示。"""
        suggestion = f"建议: 通过 --mp-dps {max(self.current_dps * 2, 150)} 重试"
        if self.mode != PRECISION_MODE_AUTO:
            suggestion += "（当前已使用固定精度模式，需手动指定更大值）"
        return (
            f"格约减失败 (dim={self.dim}, dps={self.current_dps}): {reason}\n"
            f"{suggestion}"
        )


def wrap_precision_error(reason, current_dps):
    """封装为 ReductionFailedError。"""
    return ReductionFailedError(
        f"精度不足 (dps={current_dps}): {reason}。"
        f"建议: --mp-dps {current_dps * 2} 重试"
    )
