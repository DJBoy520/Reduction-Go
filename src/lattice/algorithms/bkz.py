"""BKZ 约减（Schnorr-Euchner 1994）— 纯 mpmath 实现。

合并自原 _native/bkz/bkz_schnorr_euchner.py 和
_native/bkz/bkz_schnorr_euchner_progress_check.py。
通过 progress_cb 可选参数支持进度跟踪。
"""

import numpy as np
import mpmath

from .lll import lll_mp
from .deep_insert import l3fp_deep_insert
from ..base.enumeration import ENUM_ALGORITHMS
from ...domain.exceptions import PrecisionFailureError
from ..base.gso import (
    init_gso_mp,
    gso_full_refresh_mp,
    gso_incremental_refresh_mp,
    gso_norms_to_float,
    gso_coeffs_to_float,
)

DELTA = 0.999


def structural_changes(gs_norms_before, gs_norms_after, block_size):
    """检测 block 的 GSO 范数是否有实质性变化（mpmath 版本）。"""
    scale = max(max(gs_norms_before), max(gs_norms_after), mpmath.mpf(1))
    tol = mpmath.mpf(block_size) * mpmath.mpf('1e-12') * scale
    for a, b in zip(gs_norms_before, gs_norms_after):
        if abs(a - b) > tol:
            return False
    return True


def bkz(basis_matrix, block_size, enum_algo, dps=100, max_loops=8, auto_abort=False, progress_cb=None):
    """BKZ 约减（Schnorr-Euchner + 纯 mpmath）。

    Args:
        basis_matrix: (n, n) int64 列向量基。
        block_size: BKZ 块大小。
        enum_algo: 枚举算法 key。
        dps: mpmath 精度位数。
        max_loops: 最大 BKZ tour 轮数。
        auto_abort: 若某轮无任何向量更新则提前终止。
        progress_cb: 可选进度回调 fn(current_iter, total_iters, shortest_norm)，每轮迭代后调用。

    Returns:
        (basis_matrix, gsc_float64, gsn_float64)
    """
    svp_solver = ENUM_ALGORITHMS[enum_algo]
    total_dim = basis_matrix.shape[1]

    basis_matrix, _, _ = lll_mp(basis_matrix, dps=dps)

    with mpmath.workdps(dps):
        gsc_mp, gsn_mp = init_gso_mp(total_dim)
        has_zero = gso_full_refresh_mp(basis_matrix, gsc_mp, gsn_mp, total_dim)

        if has_zero:
            for col in range(total_dim - 1, -1, -1):
                if float(gsn_mp[col]) <= 0 or np.all(basis_matrix[:, col] == 0):
                    basis_matrix = np.delete(basis_matrix, col, axis=1)
            total_dim = basis_matrix.shape[1]
            gsc_mp, gsn_mp = init_gso_mp(total_dim)
            gso_full_refresh_mp(basis_matrix, gsc_mp, gsn_mp, total_dim)

    m = total_dim - 1

    z = 0
    j = -1
    completed_loops = 0
    tour_updated = False

    while z < m and completed_loops < max_loops:
        j += 1
        k = min(j + block_size - 1, m)
        if j == m:
            j = 0
            k = block_size
            completed_loops += 1
            if auto_abort and not tour_updated:
                break
            tour_updated = False

        block_size_actual = k - j + 1

        # 从 mpmath GSO 数据提取 block
        # block_gs_norms: list of mpf（用于 Lovasz 条件 / structural_changes）
        block_gs_norms = [mpmath.mpf(gsn_mp[idx]) for idx in range(j, k + 1)]
        # block_gs_coeffs: numpy float64（solver 需要 2D numpy 切片 gs_coeffs[t, t+1:s+1]）
        block_gs_coeffs = np.zeros((block_size_actual, block_size_actual), dtype=object)
        for ri, idx in enumerate(range(j, k + 1)):
            for ci, cidx in enumerate(range(j, k + 1)):
                block_gs_coeffs[ri, ci] = gsc_mp[idx][cidx]

        candidate_proj_len, candidate_coeff_vec = svp_solver(
            basis_matrix[:, j:k + 1],
            np.array(block_gs_norms, dtype=object),
            block_gs_coeffs,
        )

        if len(candidate_coeff_vec) != block_size_actual:
            raise RuntimeError(
                f"Enumerator returned coeff vec of length {len(candidate_coeff_vec)}, "
                f"expected {block_size_actual} (block [{j},{k}])"
            )
        block_end = min(k + 1, m)

        # Lovasz 条件检查（mpmath 精度）
        if mpmath.mpf(DELTA) * gsn_mp[j] > candidate_proj_len:
            block_gs_norms_before = [mpmath.mpf(gsn_mp[idx]) for idx in range(j, k + 1)]
            tour_updated = True
            b_new = np.dot(basis_matrix[:, j:k + 1], candidate_coeff_vec)
            injected_basis = np.insert(
                basis_matrix[:, :block_end + 1], j, np.transpose(b_new), axis=1
            )

            di_basis, di_gsc, di_gsn = l3fp_deep_insert(
                injected_basis_matrix=injected_basis,
                start_stage=j,
                Lovasz_cond_param=DELTA,
                f_c=True,
                block_start=j,
                dps=dps,
            )

            di_cols = di_basis.shape[1]
            basis_matrix[:, :di_cols] = di_basis

            with mpmath.workdps(dps):
                gso_incremental_refresh_mp(basis_matrix, gsc_mp, gsn_mp, j, total_dim - 1)

            block_gs_norms_after = [mpmath.mpf(gsn_mp[idx]) for idx in range(j, k + 1)]
            z = 0
            if not structural_changes(block_gs_norms_before, block_gs_norms_after, block_size):
                continue

        z += 1

        # 进度回调（统一签名：current_iter, total_iters, shortest_norm）
        if progress_cb is not None:
            shortest = float(min(gsn_mp[idx] for idx in range(total_dim) if gsn_mp[idx] > 0))
            progress_cb(completed_loops, max_loops, shortest)

    # 返回前转 float64
    gs_coeff_matrix = gso_coeffs_to_float(gsc_mp, total_dim, total_dim)
    gs_squared_norms = gso_norms_to_float(gsn_mp, total_dim)
    return basis_matrix, gs_coeff_matrix, gs_squared_norms


# 向后兼容别名
bkz_se = bkz
bkz_se_pc = bkz
