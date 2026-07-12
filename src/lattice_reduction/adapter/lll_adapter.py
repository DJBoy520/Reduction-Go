"""LLL 适配层 — 统一 mpmath 引擎 + 精度管理。"""

import logging
import numpy as np

from .._native.lll_mp import lll_mp
from .._native.precision import PrecisionFailureError
from .._native.precision_manager import (
    get_initial_dps, PrecisionContext, wrap_precision_error,
)
from .common_adapter import (
    to_column_basis, to_row_basis,
    validate_basis, validate_delta, ReductionFailedError,
)

logger = logging.getLogger(__name__)


def lll_reduce(B, delta=0.999, float_type="mpfr", precision=200,
               method="proved", auto_precision=True, mp_dps=None,
               progress=None):
    """LLL 约减（原地修改 B）。

    Args:
        B: numpy int64 行向量基
        delta: LLL δ 参数
        auto_precision: 是否自动选择精度
        mp_dps: 强制指定 mpmath 精度位数
    """
    B = validate_basis(B, "B")
    delta = validate_delta(delta)
    dim = B.shape[0]

    dps, mode = get_initial_dps(dim, mp_dps, auto_precision)
    logger.debug(f"LLL: dim={dim}, delta={delta}, dps={dps}")

    B_col = to_column_basis(B)
    ctx = PrecisionContext(B_col, dim, dps, mode)

    for attempt in range(mode + 1):
        try:
            basis_copy = ctx.reset_basis()
            lll_mp(basis_copy, Lovasz_cond_param=delta, dps=ctx.current_dps,
                   progress=progress, return_gso=False)
            B[:] = to_row_basis(basis_copy)
            logger.debug(f"LLL 完成: dim={dim}, dps={ctx.current_dps}")
            return
        except PrecisionFailureError as e:
            if ctx.can_upgrade():
                logger.warning(f"LLL 精度不足 ({e.reason}), 升级重试")
                ctx.upgrade_dps()
            else:
                raise ReductionFailedError(ctx.build_error_msg(e.reason)) from e

    raise ReductionFailedError(ctx.build_error_msg("超过最大重试次数"))


def lll_reduce_full(B, delta=0.999, auto_precision=True, mp_dps=None,
                    progress=None):
    """LLL 约减，返回完整 GSO 信息。"""
    B = validate_basis(B, "B")
    delta = validate_delta(delta)
    dim = B.shape[0]

    dps, mode = get_initial_dps(dim, mp_dps, auto_precision)
    B_col = to_column_basis(B)
    ctx = PrecisionContext(B_col, dim, dps, mode)

    for attempt in range(mode + 1):
        try:
            basis_copy = ctx.reset_basis()
            reduced_col, gs_coeff, gs_norms = lll_mp(
                basis_copy, Lovasz_cond_param=delta, dps=ctx.current_dps,
                progress=progress, return_gso=True)
            return to_row_basis(reduced_col), gs_coeff, gs_norms
        except PrecisionFailureError as e:
            if ctx.can_upgrade():
                logger.warning(f"LLL 精度不足 ({e.reason}), 升级重试")
                ctx.upgrade_dps()
            else:
                raise ReductionFailedError(ctx.build_error_msg(e.reason)) from e

    raise ReductionFailedError(ctx.build_error_msg("超过最大重试次数"))
