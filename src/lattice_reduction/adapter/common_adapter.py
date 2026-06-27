"""
通用格式转换与异常映射。

处理 fpylll IntegerMatrix ↔ numpy array 的转换，
以及行向量/列向量的转置差异。
"""

import numpy as np


def to_column_basis(B: np.ndarray) -> np.ndarray:
    """将行向量基转为列向量基（上游格式）。

    fpylll 使用行向量，上游 LatticeReductionAlgorithms 使用列向量。
    """
    return B.T.copy()


def to_row_basis(B_col: np.ndarray) -> np.ndarray:
    """将列向量基转为行向量基（fpylll 格式）。"""
    return B_col.T.copy()


def basis_to_numpy(B) -> np.ndarray:
    """将任意格式的格基转为 numpy int64 数组。

    支持：numpy array、list of lists、fpylll IntegerMatrix。
    """
    if isinstance(B, np.ndarray):
        return B.astype(np.int64)
    # fpylll IntegerMatrix 支持 list() 转换
    try:
        rows = []
        for i in range(B.nrows):
            rows.append([int(B[i, j]) for j in range(B.ncols)])
        return np.array(rows, dtype=np.int64)
    except AttributeError:
        # list of lists
        return np.array(B, dtype=np.int64)
