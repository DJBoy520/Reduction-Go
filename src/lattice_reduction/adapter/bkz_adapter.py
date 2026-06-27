"""
BKZ 接口适配层。

将上游 bkz_se_pc() / bkz_se() 接口封装为与 fpylll BKZ.reduction() 兼容的接口。

fpylll 接口:
    param = BKZ.Param(block_size=20, max_loops=1, auto_abort=False)
    BKZ.reduction(B, param, float_type="mpfr", precision=200)
    - B: IntegerMatrix (行向量), 原地修改
    - 返回: None

适配接口:
    bkz_reduce(B, block_size=20, max_loops=8, enum_algo="1", auto_abort=False)
    - B: numpy array (行向量), 原地修改
    - 返回: dict (completed_loops, shortest_norms)
"""

import logging
import numpy as np

from .._native.bkz.bkz_schnorr_euchner_progress_check import bkz_se_pc
from .._native.bkz.bkz_schnorr_euchner import bkz_se
from .common_adapter import (
    to_column_basis, to_row_basis,
    validate_basis, validate_block_size,
    ReductionFailedError,
)

logger = logging.getLogger(__name__)


def bkz_reduce(B: np.ndarray, block_size: int = 20, max_loops: int = 8,
               enum_algo: str = "1", auto_abort: bool = False,
               float_type: str = "mpfr", precision: int = 200) -> dict:
    """BKZ 约减（适配层），兼容 fpylll BKZ.reduction() 接口。

    上游 BKZ 是单轮完整约减（内部已包含 LLL + 多轮 SVP 枚举）。
    为兼容 max_loops 控制，循环调用上游 BKZ，每轮检查改善情况。

    Args:
        B: numpy int64 数组 (行向量基), 原地修改
        block_size: BKZ 块大小
        max_loops: 最大循环次数
        enum_algo: 枚举算法选择 ("1"=SE-OG, "2"=SE, "3"=SH)
        auto_abort: 连续无改善时提前终止
        float_type: 浮点类型（保留兼容性）
        precision: 浮点精度（保留兼容性）

    Returns:
        dict: {
            "completed_loops": int,
            "shortest_norms": list[float],  # 每轮最短范数
        }

    Raises:
        InvalidBasisError: 格基格式或参数非法
        ReductionFailedError: 约减过程失败
    """
    # 参数校验
    B = validate_basis(B, "B")
    dim = B.shape[0]
    block_size = validate_block_size(block_size, dim)

    if max_loops < 1:
        raise ReductionFailedError(f"max_loops 必须 >= 1，当前: {max_loops}")

    logger.debug(f"BKZ 约减开始: dim={dim}, block_size={block_size}, max_loops={max_loops}")

    # 行向量 → 列向量
    B_col = to_column_basis(B)

    shortest_norms = []
    prev_norm = float("inf")
    no_improve_count = 0

    for loop_i in range(1, max_loops + 1):
        try:
            # 调用上游 BKZ（单轮完整 BKZ 约减）
            reduced_col, gs_coeff, gs_norms = bkz_se_pc(
                B_col, block_size, enum_algo
            )
        except Exception as e:
            raise ReductionFailedError(f"BKZ 约减失败 (loop {loop_i}): {e}") from e

        # 计算最短范数
        norms = np.linalg.norm(reduced_col, axis=0)
        shortest = float(np.min(norms[norms > 0])) if np.any(norms > 0) else 0.0
        shortest_norms.append(shortest)

        # 更新基
        B_col = reduced_col

        logger.debug(f"BKZ loop {loop_i}/{max_loops}: shortest_norm={shortest:.2f}")

        # auto_abort 检查
        if auto_abort and loop_i > 2:
            if shortest >= prev_norm * 0.999:  # 容差 0.1%
                no_improve_count += 1
                if no_improve_count >= 2:
                    logger.info(f"BKZ auto-abort: 连续无改善，提前终止于 loop {loop_i}")
                    break
            else:
                no_improve_count = 0
        prev_norm = shortest

    # 列向量 → 行向量，原地写回
    reduced_row = to_row_basis(B_col)
    B[:] = reduced_row

    logger.debug(f"BKZ 约减完成: {len(shortest_norms)} loops")

    return {
        "completed_loops": len(shortest_norms),
        "shortest_norms": shortest_norms,
    }
