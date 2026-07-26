"""LLL 深插入（纯 mpmath）。注入短向量后重排列，删除零向量。"""

import numpy as np
import mpmath

from ..base.gso import delete_zero_vector
from ...domain.params import LOVASZ_CONDITION_PARAM
from ..base.gso import init_gso_mp, gso_full_refresh_mp, gso_step_mp, gso_coeffs_to_float, gso_norms_to_float
from .lll import _size_reduction_lll, PrecisionFailureError

from ...domain.params import DEEP_INSERT_STUCK_THRESHOLD as MP_STUCK_THRESHOLD


def l3fp_deep_insert(injected_basis_matrix, gs_coeff_matrix=None,
                     gs_squared_norms=None, start_stage=0,
                     Lovasz_cond_param=LOVASZ_CONDITION_PARAM,
                     f_c=False, block_start=None, dps=100):
    """深插入（mpmath 高精度）。

    Args:
        injected_basis_matrix: (n, m) int64, 列向量基 + 注入向量。
        start_stage: 起始 stage（注入位置）。
        block_start: BKZ block 起始，限制搜索范围。
        dps: mpmath 精度位数。

    Returns:
        (basis_int, gsc_float64, gsn_float64) — 零向量已删除，维度 -1。
    """
    if injected_basis_matrix.dtype != np.int64:
        injected_basis_matrix = injected_basis_matrix.astype(np.int64)

    n_rows, m_cols = injected_basis_matrix.shape
    initial_cols = m_cols
    di_min = start_stage if block_start is None else block_start

    gsc, gsn = init_gso_mp(m_cols)

    stage = max(start_stage, 1)
    end_stage = m_cols
    max_iterations = end_stage * 30
    iterations = 0
    f_c_count = 0
    max_f_c = end_stage * 3
    stuck_counter = 0

    with mpmath.workdps(dps):
        gso_full_refresh_mp(injected_basis_matrix, gsc, gsn, m_cols)

        while stage < end_stage and iterations < max_iterations:
            iterations += 1
            gso_step_mp(injected_basis_matrix, gsc, gsn, stage)
            f_c, need_full_refresh, _ = _size_reduction_lll(stage, gsc, gsn, injected_basis_matrix)

            if need_full_refresh:
                gso_full_refresh_mp(injected_basis_matrix, gsc, gsn, end_stage)
                stuck_counter += 1
                if stuck_counter >= MP_STUCK_THRESHOLD:
                    raise PrecisionFailureError(
                        f"deep_insert: stuck ({stuck_counter}) at stage {stage}", dps)
                continue

            if f_c:
                f_c_count += 1
                if f_c_count > max_f_c:
                    raise PrecisionFailureError(
                        f"deep_insert: too many f_c ({f_c_count}) at stage {stage}", dps)
                gso_full_refresh_mp(injected_basis_matrix, gsc, gsn, end_stage)
                stuck_counter += 1
                if stuck_counter >= MP_STUCK_THRESHOLD:
                    raise PrecisionFailureError(
                        f"deep_insert: stuck ({stuck_counter}) at stage {stage}", dps)
                continue

            if np.all(injected_basis_matrix[:, stage] == 0):
                injected_basis_matrix = np.delete(injected_basis_matrix, stage, axis=1)
                end_stage = injected_basis_matrix.shape[1]
                gsc, gsn = init_gso_mp(end_stage)
                gso_full_refresh_mp(injected_basis_matrix, gsc, gsn, end_stage)
                stage = 1
                continue

            # 使用 object dtype 避免 np.dot 的 int64 溢出
            col = injected_basis_matrix[:, stage].astype(np.int64).astype(object)
            temp_norm = mpmath.mpf(int(col @ col))
            delta = mpmath.mpf(Lovasz_cond_param)
            i = di_min
            re_ordered = False
            while i < stage:
                if delta * gsn[i] <= temp_norm:
                    temp_norm -= gsc[i][stage] ** 2 * gsn[i]
                    i += 1
                else:
                    injected_basis_matrix[:, i:stage + 1] = np.roll(
                        injected_basis_matrix[:, i:stage + 1], shift=1, axis=1)
                    re_ordered = True
                    stage = max(i - 1, di_min)
                    break

            if not re_ordered:
                stage += 1
                stuck_counter = 0

    # 兜底：用 GSO 范数检测线性相关列
    if end_stage == initial_cols:
        gsc_chk, gsn_chk = init_gso_mp(end_stage)
        with mpmath.workdps(dps):
            gso_full_refresh_mp(injected_basis_matrix, gsc_chk, gsn_chk, end_stage)
        deleted = False
        for col in range(end_stage - 1, -1, -1):
            if float(gsn_chk[col]) <= 0 or np.all(injected_basis_matrix[:, col] == 0):
                injected_basis_matrix = np.delete(injected_basis_matrix, col, axis=1)
                end_stage -= 1
                deleted = True
        if not deleted:
            raise PrecisionFailureError(
                f"deep_insert: no zero vector after {iterations} iterations", dps)

    # ========================================================================
    # 阶段 5：统一后置清理 —— 删除所有零范数列
    # ========================================================================
    if end_stage > 0:
        with mpmath.workdps(dps):
            gsc_chk, gsn_chk = init_gso_mp(end_stage)
            gso_full_refresh_mp(injected_basis_matrix, gsc_chk, gsn_chk, end_stage)

        cols_to_delete = []
        for col in range(end_stage - 1, -1, -1):
            norm_val = float(gsn_chk[col])
            is_zero_norm = norm_val <= 1e-30
            is_zero_col = np.all(injected_basis_matrix[:, col] == 0)
            if is_zero_norm or is_zero_col:
                cols_to_delete.append(col)

        if cols_to_delete:
            injected_basis_matrix = np.delete(injected_basis_matrix, cols_to_delete, axis=1)
            end_stage = injected_basis_matrix.shape[1]

    # ========================================================================
    # 最终 GSO 计算
    # ========================================================================
    if end_stage > 0:
        with mpmath.workdps(dps):
            gsc_out, gsn_out = init_gso_mp(end_stage)
            gso_full_refresh_mp(injected_basis_matrix, gsc_out, gsn_out, end_stage)
    else:
        gsc_out, gsn_out = [], []

    return (injected_basis_matrix,
            gso_coeffs_to_float(gsc_out, end_stage, end_stage),
            gso_norms_to_float(gsn_out, end_stage))
