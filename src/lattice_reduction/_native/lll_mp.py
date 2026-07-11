"""纯 mpmath LLL 约减（Schnorr-Euchner 1994）。

基向量 int64，GSO 全程 mpmath.mpf，精度由 dps 控制。
"""

import logging

import numpy as np
import mpmath

from .gso_mp import (
    _int_dot,
    gso_step_mp, gso_full_refresh_mp, gso_swap_update_mp,
    init_gso_mp,
    gso_norms_to_float, gso_coeffs_to_float,
)
from .precision import PrecisionFailureError

logger = logging.getLogger(__name__)

MP_STUCK_THRESHOLD = 2000
LOCAL_FC_THRESHOLD = 5
FAST_LIMIT = 2**38
HARD_LIMIT = 2**60


def _size_reduction_lll(stage, gsc, gsn, basis_int, dps):
    """Size reduction — 增量更新 gsc，基向量原地修改。

    从后往前循环，将 |μ_{i,stage}| > 0.5 的投影系数归约到 [-0.5, 0.5]。
    每步修改基向量 b_stage 并增量更新 gsc[k][stage]，保持 GSO 一致性。
    gsn[stage] 不在此函数内更新——由外部 gso_step_mp 从整数点积重算，
    以保证绝对精确（避免大数相减的精度损失）。

    Returns:
        f_c: True 当 |μ| > tau_limit（数值过大无法安全归约）
    """
    f_c = False

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
        else:
            basis_int[:, stage] -= mu * basis_int[:, i]

        for k in range(i):
            gsc[k][stage] -= mu * gsc[k][i]
        gsc[i][stage] -= mu

        with mpmath.workdps(dps):
            norm_sq = mpmath.mpf(_int_dot(basis_int[:, stage], basis_int[:, stage]))
            projected = mpmath.fsum(
                [gsc[j][stage] ** 2 * gsn[j] for j in range(stage) if gsn[j] > 0]
            ) if stage > 0 else mpmath.mpf(0)
            gsn[stage] = norm_sq - projected
            if gsn[stage] <= 0:
                f_c = True
                break

    return f_c


def lll_mp(basis_matrix, gs_coeff_matrix=None, gs_squared_norms=None,
           start_stage=0, Lovasz_cond_param=0.79, dps=100):
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
        gso_full_refresh_mp(basis_matrix, gsc, gsn, start_stage + 1, dps)
        max_valid_stage = start_stage  # 列 0..start_stage 已有效

    with mpmath.workdps(dps):
        delta = mpmath.mpf(Lovasz_cond_param)
        stuck_counter = 0
        backtrack_counts = {}
        max_iterations = end_stage * end_stage * 50
        iterations = 0

        while stage < end_stage and iterations < max_iterations:
            iterations += 1

            # 延迟失效：仅在列过期时重算（跳过已有效的列）
            if stage > max_valid_stage:
                gso_step_mp(basis_matrix, gsc, gsn, stage, dps)
                max_valid_stage = stage

            f_c = _size_reduction_lll(stage, gsc, gsn, basis_matrix, dps)

            if f_c:
                max_valid_stage = stage  # stage 之后的列过期
                stage_before = stage
                stage = max(stage - 1, 1)
                stuck_counter += 1
                backtrack_counts[stage_before] = backtrack_counts.get(stage_before, 0) + 1

                if backtrack_counts[stage_before] >= LOCAL_FC_THRESHOLD:
                    logger.warning(
                        f"LLL local drift detected at stage {stage_before}: "
                        f"{backtrack_counts[stage_before]} repeated backtracks, dps={dps}. Initiating exact GSO refresh."
                    )
                    # Self-heal: discard incremental GSO for these columns and recompute exactly
                    gso_full_refresh_mp(basis_matrix, gsc, gsn, stage_before + 1, dps)
                    max_valid_stage = stage_before
                    backtrack_counts[stage_before] = 0

                if stuck_counter >= MP_STUCK_THRESHOLD:
                    raise PrecisionFailureError(
                        f"LLL stuck: {stuck_counter} f_c at stage {stage}, dps={dps}. "
                        f"建议通过 --mp-dps {max(dps * 2, 150)} 重试。",
                        dps)
                continue

            mu = gsc[stage - 1][stage]
            if delta * gsn[stage - 1] > gsn[stage] + mu * mu * gsn[stage - 1]:
                # 交换列 — 增量 O(dim) 更新替代全量 O(dim²) 重算
                basis_matrix[:, [stage - 1, stage]] = \
                    basis_matrix[:, [stage, stage - 1]]
                gso_swap_update_mp(
                    basis_matrix, gsc, gsn, stage - 1, dim, dps)
                max_valid_stage = stage  # stage 之后的列过期
                stage_before = stage
                stage = max(stage - 1, 1)
                stuck_counter += 1
                backtrack_counts[stage_before] = backtrack_counts.get(stage_before, 0) + 1

                if backtrack_counts[stage_before] >= LOCAL_FC_THRESHOLD:
                    logger.warning(
                        f"LLL local drift detected at stage {stage_before}: "
                        f"{backtrack_counts[stage_before]} repeated backtracks (swap), dps={dps}. Initiating exact GSO refresh."
                    )
                    # Self-heal: full refresh of GSO up to the problem column
                    gso_full_refresh_mp(basis_matrix, gsc, gsn, stage_before + 1, dps)
                    max_valid_stage = stage_before
                    backtrack_counts[stage_before] = 0

                if stuck_counter >= MP_STUCK_THRESHOLD:
                    raise PrecisionFailureError(
                        f"LLL stuck: {stuck_counter} swaps at stage {stage}, dps={dps}. "
                        f"建议通过 --mp-dps {max(dps * 2, 150)} 重试。",
                        dps)
            else:
                stuck_counter = 0
                # Successful progress: 不再全局清空 backtrack_counts，
                # 保留历史回退计数以便自愈机制在长期运行中生效。
                stage += 1

        if iterations >= max_iterations:
            raise PrecisionFailureError(
                f"LLL exceeded max iterations ({max_iterations})", dps)

    gso_full_refresh_mp(basis_matrix, gsc, gsn, end_stage, dps)
    return basis_matrix, gso_coeffs_to_float(gsc, dim, dim), gso_norms_to_float(gsn, dim)
