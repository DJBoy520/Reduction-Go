"""BKZ 约减（Schnorr-Euchner 基础版本，无进度跟踪）。

与 bkz_schnorr_euchner_progress_check.py 功能相同，但不带进度跟踪。
共享同一套纯 mpmath 深插入和 GSO 刷新机制。
"""

import numpy as np

from .bkz_params import DELTA
from ..lll_mp import lll_mp
from ..lll.L3fp_deep_insertion import l3fp_deep_insert
from ..enumeration import ENUM_ALGORITHMS
from ..gso_mp import (
    init_gso_mp,
    gso_full_refresh_mp,
    gso_norms_to_float,
    gso_coeffs_to_float,
)


def bkz_se(basis_matrix, block_size, enum_algo, dps=100):
    """BKZ 约减（基础版本，纯 mpmath）。

    Args:
        basis_matrix: (n, n) numpy array, 列向量基。
        block_size: BKZ 块大小。
        enum_algo: 枚举算法 key。
        dps: mpmath 十进制精度位数。

    Returns:
        (basis_matrix, gs_coeff_matrix, gs_squared_norms)
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
    while z < m:
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

            z = 0
        else:
            z += 1

    return basis_matrix, gs_coeff_matrix, gs_squared_norms
