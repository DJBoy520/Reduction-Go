"""BKZ 适配层 — 统一 mpmath 引擎 + 精度管理 + 逐迭代进度。"""

import logging
import numpy as np

from .._native.bkz.bkz_schnorr_euchner_progress_check import bkz_se_pc
from .._native.precision import PrecisionFailureError
from .._native.precision_manager import get_initial_dps, PrecisionContext
from .common_adapter import (
    to_column_basis, to_row_basis,
    validate_basis, validate_block_size, ReductionFailedError,
)

logger = logging.getLogger(__name__)


def bkz_reduce(B, block_size=20, max_loops=8, enum_algo="1",
               auto_abort=False, progressive=False,
               float_type="mpfr", precision=200,
               auto_precision=True, mp_dps=None,
               progress_cb=None):
    """BKZ 约减（原地修改 B）。

    Args:
        B: numpy int64 行向量基
        progress_cb: 逐迭代进度回调 fn(z, m, shortest_norm)

    Returns:
        dict: {"completed_loops": int, "shortest_norms": list[float]}
    """
    B = validate_basis(B, "B")
    dim = B.shape[0]
    block_size = validate_block_size(block_size, dim)
    if max_loops < 1:
        raise ReductionFailedError(f"max_loops 必须 >= 1，当前: {max_loops}")

    dps, mode = get_initial_dps(dim, mp_dps, auto_precision)
    logger.debug(f"BKZ: dim={dim}, bs={block_size}, loops={max_loops}, dps={dps}")

    B_col = to_column_basis(B)
    ctx = PrecisionContext(B_col, dim, dps, mode)

    if progressive:
        start_bs = min(3, dim)
        bs_list = list(range(start_bs, block_size + 1,
                             max(1, (block_size - start_bs) // 5)))
        if bs_list[-1] != block_size:
            bs_list.append(block_size)
    else:
        bs_list = [block_size]

    for attempt in range(mode + 1):
        try:
            B_col = ctx.reset_basis()
            result = _run_bkz_loops(B_col, bs_list, max_loops, enum_algo,
                                     auto_abort, ctx.current_dps, progress_cb)
            B[:] = to_row_basis(B_col)
            return result
        except PrecisionFailureError as e:
            if ctx.can_upgrade():
                logger.warning(f"BKZ 精度不足 ({e.reason}), 升级重试")
                ctx.upgrade_dps()
            else:
                raise ReductionFailedError(ctx.build_error_msg(e.reason)) from e

    raise ReductionFailedError(ctx.build_error_msg("超过最大重试次数"))


def _run_bkz_loops(B_col, bs_list, max_loops, enum_algo, auto_abort, dps,
                    progress_cb=None):
    """执行 BKZ 多轮循环。"""
    all_norms = []
    total_loops = 0

    for bs in bs_list:
        loops_for_bs = max_loops if bs == bs_list[-1] else max(1, max_loops // 2)
        prev_norm = float("inf")
        no_improve = 0

        for loop_i in range(1, loops_for_bs + 1):
            # 包装回调：加上 loop 信息
            def _cb(z, m, shortest, _loop=loop_i, _total=loops_for_bs):
                if progress_cb is not None:
                    progress_cb(z, m, shortest, _loop, _total)

            reduced_col, _, _ = bkz_se_pc(
                B_col, bs, enum_algo, dps=dps, progress_cb=_cb)

            norms = np.linalg.norm(reduced_col, axis=0)
            shortest = float(np.min(norms[norms > 0])) if np.any(norms > 0) else 0.0
            all_norms.append(shortest)
            total_loops += 1
            B_col[:] = reduced_col

            if auto_abort and loop_i > 2:
                if shortest >= prev_norm * 0.999:
                    no_improve += 1
                    if no_improve >= 2:
                        break
                else:
                    no_improve = 0
            prev_norm = shortest

    return {"completed_loops": total_loops, "shortest_norms": all_norms}
