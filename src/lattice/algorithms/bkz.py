"""BKZ 约减（Schnorr-Euchner 1994）— 纯 mpmath 实现。

合并自原 _native/bkz/bkz_schnorr_euchner.py 和
_native/bkz/bkz_schnorr_euchner_progress_check.py。
通过 progress_cb 可选参数支持进度跟踪。
"""

import numpy as np

from .lll import lll_mp
from .deep_insert import l3fp_deep_insert
from ..base.enumeration import ENUM_ALGORITHMS
from ..base.precision_errors import PrecisionFailureError
from ..base.gso import (
    init_gso_mp,
    gso_full_refresh_mp,
    gso_norms_to_float,
    gso_coeffs_to_float,
)

DELTA = 3 / 4


def structural_changes(gs_norms_before, gs_norms_after, block_size):
    """检测 block 的 GSO 范数是否有实质性变化。"""
    scale = max(np.max(gs_norms_before), np.max(gs_norms_after), 1.0)
    tol = block_size * 1e-12 * scale
    return np.allclose(gs_norms_before, gs_norms_after, rtol=0, atol=tol)


def bkz(basis_matrix, block_size, enum_algo, dps=100, progress_cb=None):
    """BKZ 约减（Schnorr-Euchner + 纯 mpmath）。

    Args:
        basis_matrix: (n, n) int64 列向量基。
        block_size: BKZ 块大小。
        enum_algo: 枚举算法 key。
        dps: mpmath 精度位数。
        progress_cb: 可选进度回调 fn(z, m, shortest_norm)，每轮迭代后调用。

    Returns:
        (basis_matrix, gsc_float64, gsn_float64)
    """
    svp_solver = ENUM_ALGORITHMS[enum_algo]
    total_dim = basis_matrix.shape[1]

    basis_matrix, _, _ = lll_mp(basis_matrix, dps=dps)

    gsc_mp, gsn_mp = init_gso_mp(total_dim)
    has_zero = gso_full_refresh_mp(basis_matrix, gsc_mp, gsn_mp, total_dim, dps)

    if has_zero:
        for col in range(total_dim - 1, -1, -1):
            if float(gsn_mp[col]) <= 0 or np.all(basis_matrix[:, col] == 0):
                basis_matrix = np.delete(basis_matrix, col, axis=1)
        total_dim = basis_matrix.shape[1]
        gsc_mp, gsn_mp = init_gso_mp(total_dim)
        gso_full_refresh_mp(basis_matrix, gsc_mp, gsn_mp, total_dim, dps)

    m = total_dim - 1
    gs_coeff_matrix = gso_coeffs_to_float(gsc_mp, total_dim, total_dim)
    gs_squared_norms = gso_norms_to_float(gsn_mp, total_dim)

    z = 0
    j = -1
    max_total_iters = m * 2 + 10
    total_iters = 0

    while z < m and total_iters < max_total_iters:
        total_iters += 1
        j += 1
        k = min(j + block_size - 1, m)
        if j == m:
            j = 0
            k = block_size

        candidate_proj_len, candidate_coeff_vec = svp_solver(
            basis_matrix[:, j:k + 1],
            gs_squared_norms[j:k + 1],
            gs_coeff_matrix[:, j:k + 1],
        )
        block_end = min(k + 1, m)

        if DELTA * gs_squared_norms[j] > candidate_proj_len:
            block_gs_norms_before = gs_squared_norms[j:k + 1].copy()
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

            gsc_mp, gsn_mp = init_gso_mp(total_dim)
            gso_full_refresh_mp(basis_matrix, gsc_mp, gsn_mp, total_dim, dps)
            gs_coeff_matrix = gso_coeffs_to_float(gsc_mp, total_dim, total_dim)
            gs_squared_norms = gso_norms_to_float(gsn_mp, total_dim)

            block_gs_norms_after = gs_squared_norms[j:k + 1].copy()
            z = 0
            if not structural_changes(block_gs_norms_before, block_gs_norms_after, block_size):
                continue

        z += 1

        # 进度回调
        if progress_cb is not None:
            norms = np.linalg.norm(basis_matrix.astype(np.float64), axis=0)
            shortest = float(np.min(norms[norms > 0])) if np.any(norms > 0) else 0.0
            progress_cb(z, m, shortest)

    return basis_matrix, gs_coeff_matrix, gs_squared_norms


# 向后兼容别名
bkz_se = bkz
bkz_se_pc = bkz
