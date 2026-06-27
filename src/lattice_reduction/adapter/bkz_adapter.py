"""
BKZ 接口适配层。

策略：
  1. fpylll 可用 → 使用 fpylll BKZ 引擎（跳过内部 LLL，因为已由 lll_adapter 完成）
  2. fpylll 不可用 → 回退到纯 Python BKZ (Schnorr-Euchner)
     - 适用于小维度或小 block_size
     - 大维度时建议安装 fpylll
"""

import logging
import numpy as np

from .._native.bkz.bkz_schnorr_euchner_progress_check import bkz_se_pc
from .common_adapter import (
    to_column_basis, to_row_basis,
    validate_basis, validate_block_size,
    ReductionFailedError,
)

logger = logging.getLogger(__name__)

try:
    from fpylll import IntegerMatrix as _FpylllIntMatrix
    from fpylll import BKZ as _FpylllBKZ
    from fpylll import LLL as _FpylllLLL
    _FPYLLL_AVAILABLE = True
except (ImportError, RuntimeError):
    _FPYLLL_AVAILABLE = False


def _fpylll_bkz(B: np.ndarray, block_size: int, max_loops: int,
                auto_abort: bool) -> dict:
    """使用 fpylll 执行 BKZ 约减。

    注意：调用方应已完成 LLL 约减，这里用 NO_LLL flag 跳过内部 LLL。
    """
    dim = B.shape[0]
    M = _FpylllIntMatrix(dim, dim)
    for i in range(dim):
        for j in range(dim):
            M[i, j] = int(B[i, j])

    flags = _FpylllBKZ.DEFAULT | _FpylllBKZ.NO_LLL
    if auto_abort:
        flags |= _FpylllBKZ.AUTO_ABORT

    params = _FpylllBKZ.Param(
        block_size=block_size,
        max_loops=max_loops,
        flags=flags,
    )
    _FpylllBKZ.reduction(M, params)

    for i in range(dim):
        for j in range(dim):
            B[i, j] = int(M[i, j])

    norms = np.linalg.norm(B.astype(np.float64), axis=1)
    shortest = float(np.min(norms[norms > 0])) if np.any(norms > 0) else 0.0
    return {"completed_loops": max_loops, "shortest_norms": [shortest]}


def bkz_reduce(B: np.ndarray, block_size: int = 20, max_loops: int = 8,
               enum_algo: str = "1", auto_abort: bool = False,
               progressive: bool = False,
               float_type: str = "mpfr", precision: int = 200) -> dict:
    """BKZ 约减（适配层）。

    优先使用 fpylll BKZ（跳过内部 LLL，因为调用方应已完成 LLL）。
    fpylll 不可用时回退到纯 Python Schnorr-Euchner。

    Args:
        B: numpy int64 数组 (行向量基), 原地修改。应已 LLL 约减。
        block_size: BKZ 块大小
        max_loops: 最大循环次数
        enum_algo: 枚举算法 ("1"=SE-OG, "2"=SE, "3"=SH)
        auto_abort: 连续无改善时提前终止
        progressive: 渐进式 BKZ
        float_type: 浮点类型 (保留兼容性)
        precision: 浮点精度 (保留兼容性)

    Returns:
        dict: {"completed_loops": int, "shortest_norms": list[float]}
    """
    B = validate_basis(B, "B")
    dim = B.shape[0]
    block_size = validate_block_size(block_size, dim)

    if max_loops < 1:
        raise ReductionFailedError(f"max_loops 必须 >= 1，当前: {max_loops}")

    if _FPYLLL_AVAILABLE:
        logger.debug(f"BKZ: engine=fpylll, dim={dim}, block_size={block_size}")
        try:
            return _fpylll_bkz(B, block_size, max_loops, auto_abort)
        except Exception as e:
            logger.warning(f"fpylll BKZ 失败，回退到 native: {e}")

    # 纯 Python 回退
    logger.debug(f"BKZ: engine=native, dim={dim}, block_size={block_size}")
    B_col = to_column_basis(B)
    all_norms = []
    total_loops = 0

    if progressive:
        start_bs = min(3, dim)
        bs_list = list(range(start_bs, block_size + 1, max(1, (block_size - start_bs) // 5)))
        if bs_list[-1] != block_size:
            bs_list.append(block_size)
        logger.info(f"BKZ progressive: {bs_list}")
    else:
        bs_list = [block_size]

    for bs in bs_list:
        loops_for_bs = max_loops if bs == block_size else max(1, max_loops // 2)
        prev_norm = float("inf")
        no_improve = 0

        for loop_i in range(1, loops_for_bs + 1):
            try:
                reduced_col, _, _ = bkz_se_pc(B_col, bs, enum_algo)
            except Exception as e:
                raise ReductionFailedError(f"BKZ failed (bs={bs}, loop={loop_i}): {e}") from e

            norms = np.linalg.norm(reduced_col, axis=0)
            shortest = float(np.min(norms[norms > 0])) if np.any(norms > 0) else 0.0
            all_norms.append(shortest)
            total_loops += 1
            B_col = reduced_col

            if auto_abort and loop_i > 2:
                if shortest >= prev_norm * 0.999:
                    no_improve += 1
                    if no_improve >= 2:
                        logger.debug(f"BKZ auto-abort at bs={bs}, loop={loop_i}")
                        break
                else:
                    no_improve = 0
            prev_norm = shortest

    B[:] = to_row_basis(B_col)
    return {"completed_loops": total_loops, "shortest_norms": all_norms}
