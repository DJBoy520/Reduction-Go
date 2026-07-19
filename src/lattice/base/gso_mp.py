"""
GSO 计算辅助模块。

包含随机格生成和 Gram-Schmidt 正交化。
"""

import numpy as np
import mpmath


def generate_random_lattice(n: int, d: int, bound: int = 10, dtype=np.int64) -> np.ndarray:
    """
    生成随机格基矩阵。

    参数:
        n: 基向量个数（行数）
        d: 向量维度（列数）
        bound: 随机整数范围的绝对值上限
        dtype: 数据类型

    返回:
        shape 为 (n, d) 的 numpy 数组，元素为随机整数
    """
    rng = np.random.default_rng()
    return rng.integers(-bound, bound + 1, size=(n, d), dtype=dtype)
