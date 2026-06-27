"""
通用格式转换与异常映射。

处理 fpylll IntegerMatrix ↔ numpy array 的转换，
以及行向量/列向量的转置差异。
"""

import numpy as np


class LatticeReductionError(Exception):
    """格约减模块统一异常基类。"""
    pass


class InvalidBasisError(LatticeReductionError):
    """格基格式或内容非法。"""
    pass


class ReductionFailedError(LatticeReductionError):
    """约减过程失败。"""
    pass


def validate_basis(B: np.ndarray, name: str = "B") -> np.ndarray:
    """校验格基输入合法性。

    Args:
        B: 输入格基
        name: 参数名（用于错误信息）

    Returns:
        np.ndarray: 校验后的 int64 数组

    Raises:
        InvalidBasisError: 格基格式或内容非法
    """
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
    """校验 LLL delta 参数。

    Args:
        delta: LLL delta 参数，范围 (0.25, 1.0)

    Returns:
        float: 校验后的 delta

    Raises:
        InvalidBasisError: delta 超出范围
    """
    if not isinstance(delta, (int, float)):
        raise InvalidBasisError(f"delta 必须是数值类型，当前: {type(delta)}")
    if delta <= 0.25 or delta >= 1.0:
        raise InvalidBasisError(f"delta 必须在 (0.25, 1.0) 范围内，当前: {delta}")
    return float(delta)


def validate_block_size(block_size: int, dim: int) -> int:
    """校验 BKZ block_size 参数。

    Args:
        block_size: BKZ 块大小
        dim: 格维度

    Returns:
        int: 校验后的 block_size

    Raises:
        InvalidBasisError: block_size 超出范围
    """
    if not isinstance(block_size, int) or block_size < 2:
        raise InvalidBasisError(f"block_size 必须 >= 2 的整数，当前: {block_size}")
    if block_size > dim:
        raise InvalidBasisError(f"block_size ({block_size}) 不能超过格维度 ({dim})")
    return block_size


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
