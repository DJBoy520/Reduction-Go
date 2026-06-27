"""
LLL 接口适配层。

将上游 l3fp() 接口封装为与 fpylll LLL.reduction() 兼容的接口。

fpylll 接口:
    LLL.reduction(B, delta=0.999, float_type="mpfr", precision=200, method="proved")
    - B: IntegerMatrix (行向量), 原地修改
    - 返回: None (原地修改 B)

适配接口:
    lll_reduce(B, delta=0.999, float_type="mpfr", precision=200)
    - B: numpy array (行向量), 原地修改
    - 返回: None (原地修改 B)
"""

import numpy as np

from .._native.lll.L3fp import l3fp
from .._native.lll.L3fp_params import LOVASZ_CONDITION_PARAM
from .common_adapter import to_column_basis, to_row_basis


def lll_reduce(B: np.ndarray, delta: float = 0.999,
               float_type: str = "mpfr", precision: int = 200,
               method: str = "proved") -> None:
    """LLL 约减（适配层），兼容 fpylll LLL.reduction() 接口。

    将行向量基转为列向量基，调用上游 l3fp()，再转回行向量基原地写回。

    Args:
        B: numpy int64 数组 (行向量基), 原地修改
        delta: LLL delta 参数 (映射为 Lovasz_cond_param), 默认 0.999
        float_type: 浮点类型（上游仅支持 float64，此参数保留兼容性）
        precision: 浮点精度（上游使用固定 float64，此参数保留兼容性）
        method: 方法（上游不区分，此参数保留兼容性）
    """
    # 行向量 → 列向量
    B_col = to_column_basis(B)

    # 调用上游 LLL
    reduced_col, gs_coeff, gs_norms = l3fp(
        B_col,
        Lovasz_cond_param=delta,
    )

    # 列向量 → 行向量，原地写回
    reduced_row = to_row_basis(reduced_col)
    B[:] = reduced_row


def lll_reduce_full(B: np.ndarray, delta: float = 0.999) -> tuple:
    """LLL 约减，返回完整 GSO 信息。

    Args:
        B: numpy int64 数组 (行向量基)
        delta: LLL delta 参数

    Returns:
        tuple: (reduced_basis, gs_coeff_matrix, gs_squared_norms)
            - reduced_basis: numpy array (行向量基)
            - gs_coeff_matrix: numpy array (GSO 系数矩阵)
            - gs_squared_norms: numpy array (GSO 平方范数)
    """
    B_col = to_column_basis(B)
    reduced_col, gs_coeff, gs_norms = l3fp(B_col, Lovasz_cond_param=delta)
    reduced_row = to_row_basis(reduced_col)
    return reduced_row, gs_coeff, gs_norms
