"""
LLL 接口适配层。

策略：
  1. fpylll 可用 → 使用 fpylll wrapper 模式（自动升级精度，支持大维度）
  2. fpylll 不可用 → 回退到 native JIT（int64 精确点积 + Kahan + GSO 刷新）
     - 适用于 dim ≤ ~180 或小 entry 的格基
     - 大维度 + 大 entry 时 float64 精度不足，会触发安全退出

fpylll wrapper 模式参考 fplll 的多精度策略：
  先用 float64 尝试 → 检测精度不足时自动升级到 long double → MPFR
"""

import logging
import numpy as np

from .._native.lll.L3fp import l3fp
from .._native.lll.L3fp_params import LOVASZ_CONDITION_PARAM
from .common_adapter import (
    to_column_basis, to_row_basis,
    validate_basis, validate_delta,
    ReductionFailedError,
)

logger = logging.getLogger(__name__)

# 检测 fpylll 是否可用
try:
    from fpylll import IntegerMatrix as _FpylllIntMatrix
    from fpylll import LLL as _FpylllLLL
    _FPYLLL_AVAILABLE = True
except ImportError:
    _FPYLLL_AVAILABLE = False

# native 路径的维度阈值：超过此值且 entry 大时 float64 精度可能不足
_NATIVE_DIM_THRESHOLD = 180


def _fpylll_reduce(B: np.ndarray, delta: float) -> None:
    """使用 fpylll wrapper 模式执行 LLL（自动升级精度）。"""
    dim = B.shape[0]
    M = _FpylllIntMatrix(dim, dim)
    for i in range(dim):
        for j in range(dim):
            M[i, j] = int(B[i, j])

    # wrapper 模式（method=None）自动从 float64 升级到 long double / MPFR
    _FpylllLLL.reduction(M, delta=delta)

    for i in range(dim):
        for j in range(dim):
            B[i, j] = int(M[i, j])


def _native_reduce(B: np.ndarray, delta: float) -> None:
    """纯 Python/Numba JIT LLL 约减（回退路径）。"""
    B_col = to_column_basis(B)
    try:
        l3fp(B_col, Lovasz_cond_param=delta)
    except Exception as e:
        raise ReductionFailedError(f"LLL 约减失败: {e}") from e
    B[:] = to_row_basis(B_col)


def lll_reduce(B: np.ndarray, delta: float = 0.999,
               float_type: str = "mpfr", precision: int = 200,
               method: str = "proved") -> None:
    """LLL 约减（适配层），兼容 fpylll LLL.reduction() 接口。

    优先使用 fpylll（自动精度升级，支持大维度）。
    fpylll 不可用时回退到 native JIT（float64，适用于小/中维度）。

    Args:
        B: numpy int64 数组 (行向量基), 原地修改
        delta: LLL delta 参数, 默认 0.999
        float_type: 浮点类型 (保留兼容性，fpylll wrapper 自动选择)
        precision: 浮点精度 (保留兼容性)
        method: 方法 (保留兼容性)

    Raises:
        InvalidBasisError: 格基格式非法
        ReductionFailedError: 约减过程失败
    """
    B = validate_basis(B, "B")
    delta = validate_delta(delta)
    dim = B.shape[0]

    # 检测 entry 大小（判断 native 路径是否有精度风险）
    max_entry = int(np.max(np.abs(B)))
    entry_bits = max_entry.bit_length() if max_entry > 0 else 0
    native_precision_risk = dim > _NATIVE_DIM_THRESHOLD and entry_bits > 20

    if _FPYLLL_AVAILABLE:
        engine = "fpylll"
    elif native_precision_risk:
        logger.warning(
            f"dim={dim}, entry_bits={entry_bits}: float64 精度可能不足，"
            f"建议安装 fpylll (pip install fpylll) 以获得多精度支持"
        )
        engine = "native"
    else:
        engine = "native"

    logger.debug(f"LLL: engine={engine}, dim={dim}, delta={delta}")

    if engine == "fpylll":
        try:
            _fpylll_reduce(B, delta)
        except Exception as e:
            logger.warning(f"fpylll LLL 失败 ({e})，回退到 native")
            _native_reduce(B, delta)
    else:
        _native_reduce(B, delta)

    logger.debug(f"LLL 完成: dim={dim}")


def lll_reduce_full(B: np.ndarray, delta: float = 0.999) -> tuple:
    """LLL 约减，返回完整 GSO 信息。

    Args:
        B: numpy int64 数组 (行向量基)
        delta: LLL delta 参数

    Returns:
        tuple: (reduced_basis, gs_coeff_matrix, gs_squared_norms)
    """
    B = validate_basis(B, "B")
    delta = validate_delta(delta)

    if _FPYLLL_AVAILABLE:
        _fpylll_reduce(B, delta)
        # 用 native 计算完整 GSO 信息（仅用于质量评估）
        B_col = to_column_basis(B)
        try:
            _, gs_coeff, gs_norms = l3fp(B_col, Lovasz_cond_param=delta)
        except Exception:
            return B, None, None
        return B, gs_coeff, gs_norms
    else:
        B_col = to_column_basis(B)
        try:
            reduced_col, gs_coeff, gs_norms = l3fp(B_col, Lovasz_cond_param=delta)
        except Exception as e:
            raise ReductionFailedError(f"LLL 约减失败: {e}") from e
        reduced_row = to_row_basis(reduced_col)
        return reduced_row, gs_coeff, gs_norms
