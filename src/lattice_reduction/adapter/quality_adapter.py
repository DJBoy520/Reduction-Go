"""
格基质量评估接口适配层。

封装上游质量评估功能，提供简洁的评估接口。
"""

import numpy as np

from .._native.quality.basis_quality_characteristics import (
    compute_column_norms,
    compute_root_hermite_factor,
    compute_hermite_factor,
    compute_lattice_volume_log,
    compute_orthogonality_defect,
)
from .._native.quality.basis_quality_evaluation import (
    compute_basis_quality_characteristics,
)
from .common_adapter import to_column_basis


def evaluate_basis_quality(B: np.ndarray, reduced: bool = True) -> dict:
    """评估格基质量（适配层）。

    Args:
        B: numpy array (行向量基)
        reduced: 是否已约减（影响列范数排序）

    Returns:
        dict: {
            "shortest_norm": float,      # 最短列范数
            "root_hermite_factor": float, # 根 Hermite 因子
            "hermite_factor": float,      # Hermite 因子
            "orthogonality_defect": float, # 正交缺陷
            "log_volume": float,          # 格体积（对数）
            "column_norms": np.ndarray,   # 列范数
        }
    """
    # 行向量 → 列向量
    B_col = to_column_basis(B)

    # 基础特征
    col_norms = compute_column_norms(B_col)
    log_vol = compute_lattice_volume_log(B_col)
    dim = B_col.shape[1]

    # 质量评估
    shortest, rhf, od = compute_basis_quality_characteristics(B_col, reduced=reduced)

    # Hermite 因子
    hf = compute_hermite_factor(shortest, log_vol, dim)

    return {
        "shortest_norm": shortest,
        "root_hermite_factor": rhf,
        "hermite_factor": hf,
        "orthogonality_defect": od,
        "log_volume": log_vol,
        "column_norms": col_norms,
    }
