"""纯 mpmath LLL 约减（Schnorr-Euchner 1994）。

基向量 int64，GSO 全程 mpmath.mpf，精度由 dps 控制。
"""

import logging

import numpy as np
import mpmath

from ..base.gso import (
    _int_dot,
    gso_step_mp, gso_full_refresh_mp, gso_swap_update_mp,
    init_gso_mp,
    gso_norms_to_float, gso_coeffs_to_float,
)
from ..base.precision_errors import PrecisionFailureError

logger = logging.getLogger(__name__)

MP_STUCK_THRESHOLD = 10000
FAST_LIMIT = 2**50
HARD_LIMIT = 2**60


def _size_reduction_lll(stage, gsc, gsn, basis_int):
    """Size reduction — 增量更新 gsc，基向量原地修改。

    从后往前循环，将 |μ_{i,stage}| > 0.5 的投影系数归约到 [-0.5, 0.5]。
    每步修改基向量 b_stage 并增量更新 gsc[k][stage]，保持 GSO 一致性。
    精度由外层 workdps 上下文控制。

    Returns:
        (f_c, need_full_refresh):
            f_c 表示数值过大无法安全归约；
            need_full_refresh 表示 fallback 之后 GSO 依赖链已被破坏，
            需要在主循环中进行一次全量刷新。
    """
    f_c = False
    need_full_refresh = False

    size_reduced = False

    for i in range(stage - 1, -1, -1):
        mu_val = gsc[i][stage]
        mu_float = abs(float(mu_val))
        if mu_float <= 0.5:
            continue

        mu = int(mpmath.nint(mu_val))
        mu_abs = abs(mu)

        if mu_abs > HARD_LIMIT:
            f_c = True
            break

        if mu_abs > FAST_LIMIT:
            logger.warning(
                f"Size reduction fallback to large int: stage={stage}, i={i}, mu={mu}"
            )
            col_stage_obj = basis_int[:, stage].astype(object) - mu * basis_int[:, i].astype(object)
            if not np.all((col_stage_obj >= -(2**63)) & (col_stage_obj < 2**63)):
                f_c = True
                break
            basis_int[:, stage] = col_stage_obj.astype(np.int64)
            gso_step_mp(basis_int, gsc, gsn, stage)
            # gso_step_mp 已正确更新当前列的 GSO 依赖链，无需全量刷新。
            # 仅在 mu_abs > HARD_LIMIT 或 f_c 溢出时才需要全量刷新。
            continue

        old_mu = gsc[i][stage]
        basis_int[:, stage] -= mu * basis_int[:, i]
        size_reduced = True

        for k in range(i):
            gsc[k][stage] -= mu * gsc[k][i]
        gsc[i][stage] -= mu

        gsn[stage] += (
            old_mu * old_mu - gsc[i][stage] * gsc[i][stage]
        ) * gsn[i]
        if gsn[stage] <= 0:
            f_c = True
            break

    return f_c, need_full_refresh, size_reduced


def lll_mp(basis_matrix, gs_coeff_matrix=None, gs_squared_norms=None,
           start_stage=0, Lovasz_cond_param=0.75, dps=100, progress=None,
           return_gso=True):
    """纯 mpmath LLL 约减。

    Args:
        basis_matrix: (n, m) int64 列向量基。
        start_stage: 起始 stage。
        Lovasz_cond_param: δ 参数。
        dps: mpmath 精度位数。

    Returns:
        (basis_matrix, gsc_float64, gsn_float64)
    """
    if basis_matrix.dtype != np.int64:
        basis_matrix = basis_matrix.astype(np.int64)

    end_stage = basis_matrix.shape[1]
    dim = end_stage
    gsc, gsn = init_gso_mp(dim)

    if start_stage == 0:
        stage = 1
        max_valid_stage = 0  # 无列已计算
    else:
        stage = start_stage
        gso_full_refresh_mp(basis_matrix, gsc, gsn, start_stage + 1)
        max_valid_stage = start_stage  # 列 0..start_stage 已有效

    size_count = 0
    swap_count = 0
    gso_count = 0
    with mpmath.workdps(dps):
        delta = mpmath.mpf(Lovasz_cond_param)
        stuck_counter = 0
        max_stage_reached = start_stage
        max_iterations = end_stage ** 3
        iterations = 0

        while stage < end_stage and iterations < max_iterations:
            iterations += 1
            if progress is not None and (iterations % 200 == 0):
                progress.update_stats(stage, iterations, swap_count, size_count, gso_count)

            # 延迟失效：只有当前列尚未计算时才调用完整 GSO
            if stage > max_valid_stage:
                gso_step_mp(basis_matrix, gsc, gsn, stage)
                max_valid_stage = stage
                gso_count += 1

            f_c, need_full_refresh, size_reduced = _size_reduction_lll(stage, gsc, gsn, basis_matrix)
            size_count += 1

            # size_reduced 时 GSO 已由 _size_reduction_lll 增量路径保持一致，无需全量刷新。
            # 只在 need_full_refresh（int64 fallback 破坏依赖链）时才全量刷新。

            if need_full_refresh:
                gso_full_refresh_mp(basis_matrix, gsc, gsn, end_stage)
                max_valid_stage = end_stage - 1
                stage = max(stage - 1, 1)
                stuck_counter += 1
                continue

            if f_c:
                stage = max(stage - 1, 1)
                max_valid_stage = min(max_valid_stage, stage - 1)
                stuck_counter += 1
            else:
                mu = gsc[stage - 1][stage]
                if delta * gsn[stage - 1] > gsn[stage] + mu * mu * gsn[stage - 1]:
                    # 交换列 — 局部 O(dim) 更新 GSO，保持已有列有效
                    basis_matrix[:, [stage - 1, stage]] = basis_matrix[:, [stage, stage - 1]]
                    gso_swap_update_mp(basis_matrix, gsc, gsn, stage - 1, dim)
                    swap_count += 1
                    stage = max(stage - 1, 1)
                    max_valid_stage = min(max_valid_stage, stage - 1)
                    stuck_counter = 0
                else:
                    stage += 1
                    if stage > max_stage_reached:
                        max_stage_reached = stage
                    stuck_counter = 0

            if stuck_counter >= MP_STUCK_THRESHOLD:
                raise PrecisionFailureError(
                    f"LLL stuck: {stuck_counter} consecutive stalls at stage {stage}, "
                    f"max_stage_reached={max_stage_reached}, dps={dps}. "
                    f"建议通过 --mp-dps {max(dps * 2, 150)} 重试。",
                    dps)

        if iterations >= max_iterations:
            raise PrecisionFailureError(
                f"LLL exceeded max iterations ({max_iterations})", dps)

    if return_gso:
        with mpmath.workdps(dps):
            gso_full_refresh_mp(basis_matrix, gsc, gsn, end_stage)
        return basis_matrix, gso_coeffs_to_float(gsc, dim, dim), gso_norms_to_float(gsn, dim)
    return basis_matrix
