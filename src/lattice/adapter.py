"""格约减适配层 — 统一接口 + 精度管理。

合并自：
  - adapter/common_adapter.py  → 通用工具函数 + 异常类
  - adapter/lll_adapter.py     → LLL 约减接口
  - adapter/bkz_adapter.py     → BKZ 约减接口
"""

import logging
import numpy as np

from .algorithms.lll import lll_mp
from .algorithms.bkz import bkz
from .base.exceptions import (
    LatticeReductionError, InvalidBasisError, ReductionFailedError,
)
from .base.precision_errors import PrecisionFailureError
from .base.precision import get_initial_dps, PrecisionContext, wrap_precision_error

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# 通用工具函数（原 common_adapter.py）
# ──────────────────────────────────────────────────────────────

def validate_basis(B: np.ndarray, name: str = "B") -> np.ndarray:
    """校验格基输入合法性。"""
    if B is None:
        raise InvalidBasisError(f"{name} 不能为 None")

    if not isinstance(B, np.ndarray):
        try:
            B = np.array(B, dtype=np.int64)
        except (ValueError, TypeError) as e:
            raise InvalidBasisError(f"{name} 无法转为 numpy 数组: {e}")

    if B.ndim != 2:
        raise InvalidBasisError(f"{name} 必须是二维矩阵，当前维度: {B.ndim}")

    if B.shape[0] != B.shape[1]:
        raise InvalidBasisError(f"{name} 必须是方阵，当前形状: {B.shape}")

    if B.shape[0] == 0:
        raise InvalidBasisError(f"{name} 不能为空矩阵")

    if np.any(np.isnan(B)):
        raise InvalidBasisError(f"{name} 包含 NaN 值")

    if np.any(np.isinf(B)):
        raise InvalidBasisError(f"{name} 包含 Inf 值")

    if B.dtype != np.int64:
        B = B.astype(np.int64)

    return B


def validate_delta(delta: float) -> float:
    """校验 LLL delta 参数，范围 (0.25, 1.0)。"""
    if not isinstance(delta, (int, float)):
        raise InvalidBasisError(f"delta 必须是数值类型，当前: {type(delta)}")
    if delta <= 0.25 or delta >= 1.0:
        raise InvalidBasisError(f"delta 必须在 (0.25, 1.0) 范围内，当前: {delta}")
    return float(delta)


def validate_block_size(block_size: int, dim: int) -> int:
    """校验 BKZ block_size 参数。"""
    if not isinstance(block_size, int):
        raise InvalidBasisError(f"block_size 必须是整数，当前: {type(block_size)}")
    if block_size < 2:
        raise InvalidBasisError(f"block_size 必须 >= 2，当前: {block_size}")
    if block_size > dim:
        raise InvalidBasisError(f"block_size ({block_size}) 不能超过维度 ({dim})")
    return block_size


def to_column_basis(B: np.ndarray) -> np.ndarray:
    """行向量基 → 列向量基（转置）。"""
    return B.T.copy()


def to_row_basis(B_col: np.ndarray) -> np.ndarray:
    """列向量基 → 行向量基（转置）。"""
    return B_col.T.copy()


# ──────────────────────────────────────────────────────────────
# LLL 约减接口（原 lll_adapter.py）
# ──────────────────────────────────────────────────────────────

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
            logger.debug(f"LLL 完成: dim={dim}, dps={ctx.current_dps}, attempts={attempt+1}")
            return B
        except PrecisionFailureError as e:
            logger.warning(f"LLL 精度失败 (attempt {attempt+1}): {e}")
            ctx.handle_precision_failure(e)
            if attempt == mode:
                raise ReductionFailedError(
                    f"LLL 约减失败: 精度迭代耗尽 ({mode+1} 次尝试)"
                ) from e


def lll_reduce_full(B, delta=0.999, float_type="mpfr", precision=200,
                    method="proved", auto_precision=True, mp_dps=None,
                    progress=None):
    """LLL 约减，返回完整信息（原地修改 B）。"""
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
            gso = lll_mp(basis_copy, Lovasz_cond_param=delta, dps=ctx.current_dps,
                         progress=progress, return_gso=True)
            B[:] = to_row_basis(basis_copy)
            logger.debug(f"LLL 完成: dim={dim}, dps={ctx.current_dps}, attempts={attempt+1}")
            return {"basis": B, "gso": gso, "dps": ctx.current_dps}
        except PrecisionFailureError as e:
            logger.warning(f"LLL 精度失败 (attempt {attempt+1}): {e}")
            ctx.handle_precision_failure(e)
            if attempt == mode:
                raise ReductionFailedError(
                    f"LLL 约减失败: 精度迭代耗尽 ({mode+1} 次尝试)"
                ) from e


# ──────────────────────────────────────────────────────────────
# BKZ 约减接口（原 bkz_adapter.py）
# ──────────────────────────────────────────────────────────────

def bkz_reduce(B, block_size=20, max_loops=8, enum_algo="1",
               auto_abort=False, progressive=False,
               float_type="mpfr", precision=200,
               auto_precision=True, mp_dps=None,
               progress_cb=None):
    """BKZ 约减（原地修改 B）。

    Args:
        B: numpy int64 行向量基
        progress_cb: 逐迭代进度回调 fn(current_iter, total_iters, shortest_norm)

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
        bs_list = list(range(start_bs, block_size + 1))
    else:
        bs_list = [block_size]

    shortest_norms = []
    completed_loops = 0

    for bs in bs_list:
        for attempt in range(mode + 1):
            try:
                basis_copy = ctx.reset_basis()

                # 包装 progress_cb 以收集 shortest_norms
                _collected = []
                def _progress_wrap(current, total, sn, _c=_collected, _ext_cb=progress_cb):
                    _c.append(sn)
                    if _ext_cb is not None:
                        _ext_cb(current, total, sn)

                _ret_basis, _ret_gsc, _ret_gsn = bkz(
                    basis_copy, bs, enum_algo,
                    dps=ctx.current_dps, progress_cb=_progress_wrap,
                )
                B[:] = to_row_basis(basis_copy)
                shortest_norms.extend(_collected)
                completed_loops += 1
                logger.debug(f"BKZ bs={bs} 完成: dps={ctx.current_dps}, attempts={attempt+1}")
                break
            except PrecisionFailureError as e:
                logger.warning(f"BKZ 精度失败 (attempt {attempt+1}): {e}")
                ctx.handle_precision_failure(e)
                if attempt == mode:
                    raise ReductionFailedError(
                        f"BKZ 约减失败: 精度迭代耗尽 ({mode+1} 次尝试)"
                    ) from e

    return {
        "completed_loops": completed_loops,
        "shortest_norms": shortest_norms,
    }


def evaluate_basis_quality(B, reduced=False):
    """评估格基质量，返回质量指标字典。"""
    B = np.asarray(B, dtype=np.float64)
    n, m = B.shape
    if n == 0:
        return {"det_ratio": 0.0, "orthogonal_defect": 0.0,
                "shortest_norm": 0.0, "reduced": reduced}

    gram = B @ B.T
    det_gram = np.linalg.det(gram)
    if det_gram <= 0:
        det_gram = 1e-300

    row_norms = np.linalg.norm(B, axis=1)
    hadamard = np.prod(row_norms)
    det_ratio = (det_gram ** (1.0 / n)) / hadamard if hadamard > 0 else 0.0

    frob = np.sqrt(np.sum(row_norms ** 2))
    orth_defect = frob / (det_gram ** (1.0 / (2 * n))) if det_gram > 0 else 0.0

    return {
        "det_ratio": float(det_ratio),
        "orthogonal_defect": float(orth_defect),
        "shortest_norm": float(np.min(row_norms)),
        "reduced": reduced,
    }


class ReductionResult:
    """约减结果容器，兼容 smoke_adapter 测试期望。"""
    def __init__(self, basis=None, success=True, message="", algorithm="lll",
                 quality=None, lattice_quality=None, norm_quality=None,
                 completed_loops=0, shortest_norms=None):
        self.basis = basis
        self.success = success
        self.message = message
        self.algorithm = algorithm
        self.quality = quality or {}
        self.lattice_quality = lattice_quality
        self.norm_quality = norm_quality
        self.completed_loops = completed_loops
        self.shortest_norms = shortest_norms or []

    def __repr__(self):
        return f"ReductionResult(success={self.success}, algorithm={self.algorithm})"


class LatticeReducer:
    """高级约减接口，兼容 smoke_adapter 测试期望。"""
    def __init__(self):
        pass

    def reduce(self, basis, delta=0.99, algorithm="lll", **kwargs):
        """执行约减并返回 ReductionResult。"""
        try:
            if algorithm == "bkz":
                result = bkz_reduce(basis, **kwargs)
            else:
                result = lll_reduce(basis, delta=delta, **kwargs)
            return ReductionResult(
                basis=result.get("basis"),
                success=result.get("success", True),
                message=result.get("message", ""),
                algorithm=algorithm,
                quality=result.get("quality", {}),
                lattice_quality=result.get("lattice_quality"),
                norm_quality=result.get("norm_quality"),
                completed_loops=result.get("completed_loops", 0),
                shortest_norms=result.get("shortest_norms", []),
            )
        except Exception as e:
            return ReductionResult(success=False, message=str(e), algorithm=algorithm)

    def lll(self, basis, delta=0.99, **kwargs):
        return self.reduce(basis, delta=delta, algorithm="lll", **kwargs)

    def bkz(self, basis, **kwargs):
        return self.reduce(basis, algorithm="bkz", **kwargs)
