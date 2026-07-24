"""纯 mpmath LLL 约减的多进程版本。

基向量 int64，GSO 全程 mpmath.mpf，精度由 dps 控制。
使用多进程加速 GSO 计算。
"""

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import numpy as np
import mpmath

from ..base.gso import (
    _int_dot,
    gso_step_mp, gso_full_refresh_mp, gso_swap_update_mp,
    init_gso_mp,
    gso_norms_to_float, gso_coeffs_to_floa
    _check_gso_failure,
)

logger = logging.getLogger(__name__)


def _compute_gso_row_mp(args):
    """计算单行 GSO 系数的辅助函数（用于多进程）。"""
    basis_matrix, row_idx, dim, dps = args
    mpmath.mp.dps = dps
    row = basis_matrix[row_idx]
    gsc_row = []
    for j in range(row_idx):
        col_j = basis_matrix[j]
        dot_ij = _int_dot(row, col_j)
        gsc_row.append(mpmath.mpf(dot_ij) / mpmath.mpf(basis_matrix[:, j].dot(basis_matrix[:, j])))
    gsc_row.append(mpmath.mpf(1))
    return row_idx, gsc_row


def _compute_gso_batch(args):
    """批量计算 GSO 系数的辅助函数。"""
    basis_matrix, start, end, dim, dps = args
    mpmath.mp.dps = dps
    results = []
    for row_idx in range(start, end):
        row = basis_matrix[row_idx]
        gsc_row = []
        for j in range(row_idx):
            col_j = basis_matrix[j]
            dot_ij = _int_dot(row, col_j)
            gsc_row.append(mpmath.mpf(dot_ij) / mpmath.mpf(basis_matrix[:, j].dot(basis_matrix[:, j])))
        gsc_row.append(mpmath.mpf(1))
        results.append((row_idx, gsc_row))
    return results


def _parallel_gso_refresh(basis_matrix, dim, dps):
    """使用多进程并行刷新 GSO 系数。"""
    mpmath.mp.dps = dps
    
    # 对于小矩阵，直接串行计算
    if dim <= 4:
        return gso_full_refresh_mp(basis_matrix, dim, dim, dps)
    
    # 对于大矩阵，使用多进程
    # 将行分组，每组由一个进程计算
    num_workers = min(mp.cpu_count(), dim // 2 + 1)
    rows_per_worker = (dim + num_workers - 1) // num_workers
    
    batches = []
    for i in range(num_workers):
        start = i * rows_per_worker
        end = min(start + rows_per_worker, dim)
        if start < end:
            batches.append((basis_matrix, start, end, dim, dps))
    
    results = {}
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_compute_gso_batch, batch) for batch in batches]
        for future in as_completed(futures):
            batch_results = future.result()
            for row_idx, gsc_row in batch_results:
                results[row_idx] = gsc_row
    
    # 组装结果
    gsc = np.empty((dim, dim), dtype=object)
    for row_idx in range(dim):
        gsc[row_idx, :len(results[row_idx])] = results[row_idx]
    
    return gsc


def lll_reduce_mp(
    basis_matrix: np.ndarray,
    delta: float = 0.75,
    dps: int = 30,
    return_gsc: bool = False,
    return_gsn: bool = False,
) -> tuple:
    """多进程版 LLL 约减（Schnorr-Euchner 1994）。

    基向量 int64，GSO 全程 mpmath.mpf，精度由 dps 控制。

    Args:
        basis_matrix: 基矩阵，形状 (n, d)，int64
        delta: LLL 参数，0.25 < delta < 1.0，典型值 0.75 或 0.99
        dps: mpmath 精度（小数位数）
        return_gsc: 是否返回 GSO 系数矩阵
        return_gsn: 是否返回 GSO 范数平方

    Returns:
        (reduced_basis,) 或 (reduced_basis, gsc) 或 (reduced_basis, gsc, gsn)
    """
    if not (0.25 < delta < 1.0):
        raise ValueError("delta must be in (0.25, 1.0)")

    n, d = basis_matrix.shape
    basis_matrix = basis_matrix.astype(np.int64)
    
    mpmath.mp.dps = dps

    # 初始化 GSO
    gsc, gsn = init_gso_mp(basis_matrix, n, d, dps)

    # LLL 约减主循环
    k = 1
    swaps = 0
    
    while k < n:
        # 正交化步骤
        gso_step_mp(basis_matrix, gsc, gsn, k, n, d, dps)
        
        # 检查 GSO 失败
        _check_gso_failure(gsn, k, d, n)
        
        # Size reduction
        for j in range(k - 1, -1, -1):
            mu_kj = gsc[k, j]
            if abs(mu_kj) > 0.5:
                # 执行行操作：b_k = b_k - round(mu_kj) * b_j
                q = int(round(float(mu_kj)))
                basis_matrix[k] -= q * basis_matrix[j]
                # 更新 GSO 系数
                for l in range(j + 1):
                    gsc[k, l] -= mpmath.mpf(q) * gsc[j, l]
                swaps += 1
        
        # Lovász 条件
        lhs = gsn[k, d]
        rhs = (delta - gsc[k, k-1]**2) * gsn[k-1, d]
        
        if lhs >= rhs:
            k += 1
        else:
            # 交换 b_k 和 b_{k-1}
            basis_matrix[[k, k-1]] = basis_matrix[[k-1, k]]
            gso_swap_update_mp(basis_matrix, gsc, gsn, k, n, d, dps)
            k = max(k - 1, 1)
            swaps += 1

    logger.debug("LLL-MP 约减完成：n=%d, swaps=%d, dps=%d", n, swaps, dps)

    # 返回结果
    if return_gsc and return_gsn:
        return basis_matrix, gso_coeffs_to_float(gsc, n, d), gso_norms_to_float(gsn, n, d)
    if return_gsc:
        return basis_matrix, gso_coeffs_to_float(gsc, n, d)
    if return_gsn:
        return basis_matrix, gso_norms_to_float(gsn, n, d)
    return basis_matrix
