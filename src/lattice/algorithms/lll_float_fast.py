"""Float64 快速 LLL — 预处理阶段使用。

完全基于 numpy.float64，不使用 mpmath。用于在调用高精度 mpmath LLL 之前，
先把 Kannan 嵌入矩阵中巨大的 qI 块大致"拉正"。
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)


def _gso_float64(B):
    """计算 float64 GSO（Gram-Schmidt 正交化）。

    Returns:
        mu: (m, m) 正交化系数矩阵
        B_star_norms: (m,) 正交列向量的平方范数
    """
    m = B.shape[1]
    B_f = B.astype(np.float64)
    B_star = B_f.copy()
    mu = np.zeros((m, m), dtype=np.float64)
    B_star_norms = np.zeros(m, dtype=np.float64)

    for i in range(m):
        for j in range(i):
            dot_ij = np.dot(B_f[:, i], B_star[:, j])
            mu[j, i] = dot_ij / B_star_norms[j] if B_star_norms[j] > 0 else 0.0
            B_star[:, i] -= mu[j, i] * B_star[:, j]
        B_star_norms[i] = np.dot(B_star[:, i], B_star[:, i])

    return mu, B_star_norms, B_star


def lll_float_fast(basis_matrix, delta=0.9, max_iters=500):
    """Float64 快速 LLL 约减（预处理用）。

    使用全量 GSO 刷新策略（O(n^3)/swap），简单稳定。

    Args:
        basis_matrix: (n, m) int64 列向量基 — 原地修改
        delta: LLL delta 参数（默认 0.9，比标准 0.999 弱以提速）
        max_iters: 最大迭代次数（防止浮点误差死循环）

    Returns:
        basis_matrix: 约减后的 int64 基（转回 int64）
    """
    if basis_matrix.dtype != np.int64:
        basis_matrix = basis_matrix.astype(np.int64)
    m = basis_matrix.shape[1]
    if m < 2:
        return basis_matrix

    B = basis_matrix.astype(np.float64)
    mu, B_star_norms, _ = _gso_float64(B)

    k = 1
    iters = 0
    while k < m and iters < max_iters:
        iters += 1

        # Size reduction: 对 k 列向前归约
        for j in range(k - 1, -1, -1):
            if abs(mu[j, k]) > 0.5:
                r = int(round(mu[j, k]))
                B[:, k] -= r * B[:, j]
                # 更新 mu 列（仅影响当前列的系数）
                for i in range(j + 1):
                    mu[i, k] -= r * mu[i, j]

        # Lovasz 条件检查
        lovasz_left = delta * B_star_norms[k - 1]
        lovasz_right = B_star_norms[k] + mu[k - 1, k]**2 * B_star_norms[k - 1]

        if lovasz_left <= lovasz_right:
            k += 1
        else:
            # 交换 k-1 和 k 列
            B[:, [k - 1, k]] = B[:, [k, k - 1]]
            # 全量刷新 GSO（简单但稳定）
            mu, B_star_norms, _ = _gso_float64(B)
            k = max(k - 1, 1)

    # 转回 int64
    result = np.round(B).astype(np.int64)
    basis_matrix[:, :] = result
    logger.debug(f"lll_float_fast 完成: {iters} iters, k={k}/{m}")
    return basis_matrix
