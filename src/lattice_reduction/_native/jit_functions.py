"""
Numba JIT 加速版核心数值计算函数。

所有 JIT 函数均为纯数值计算，无 Python 对象操作。
无 Numba 环境时自动降级为纯 Python 版本。
"""

import numpy as np
from .jit_compat import njit


@njit(cache=True)
def gso_step_jit(basis_slice, gsc, gs, stage):
    """GSO 单步计算（JIT 加速版）。

    全部手动循环，不调用 np.dot（避免非连续数组开销）。
    CGS: μ_{j,k} = (⟨b_k, b_j⟩ - Σ_{i<j} μ_{i,j}·μ_{i,k}·||b*_i||²) / ||b*_j||²
    """
    n = basis_slice.shape[0]

    if stage == 1:
        s = 0.0
        for r in range(n):
            s += basis_slice[r, 0] * basis_slice[r, 0]
        gs[0] = s

    # Compute ⟨b_k, b_j⟩ for all j < stage, and ||b_k||²
    dots = np.zeros(stage)
    norm_sq = 0.0
    for r in range(n):
        bk_r = basis_slice[r, stage]
        norm_sq += bk_r * bk_r
        for j in range(stage):
            dots[j] += bk_r * basis_slice[r, j]
    gs[stage] = norm_sq

    for j in range(stage):
        # correction_term = Σ_{i<j} μ_{i,j} · μ_{i,stage} · ||b*_i||²
        ct = 0.0
        for i in range(j):
            ct += gsc[i, j] * gsc[i, stage] * gs[i]

        gsc[j, stage] = (dots[j] - ct) / gs[j]
        gs[stage] -= gsc[j, stage] * gsc[j, stage] * gs[j]

    gsc[stage, stage] = 1.0


@njit(cache=True)
def size_reduction_jit(gsc_col, gsc_mat, basis_col, basis_mat, stage, tau_limit):
    """尺寸缩减核心（JIT 加速版）。"""
    threshold = 0.5
    f_c = False

    for i in range(stage - 1, -1, -1):
        if abs(gsc_col[i]) > threshold:
            mu = round(gsc_col[i])
            if abs(mu) > tau_limit:
                f_c = True

            for k in range(i):
                gsc_col[k] -= mu * gsc_mat[k, i]
            gsc_col[i] -= mu

            n = basis_col.shape[0]
            for r in range(n):
                basis_col[r] -= mu * basis_mat[r, i]

    return f_c


@njit(cache=True)
def enum_se_jit(block, gs_sq_norms, gsc_coeffs, search_radius):
    """SVP 枚举器（JIT 加速版）— Schnorr-Euchner 策略。"""
    block_size = gs_sq_norms.shape[0]
    dim = block.shape[0]

    coeff = np.zeros(block_size, dtype=np.int64)
    partial = np.zeros(block_size)
    best_radius = search_radius
    found = False
    best_coeff = np.zeros(block_size, dtype=np.int64)

    k = 0
    coeff[0] = 0
    partial[0] = 0.0

    while k >= 0:
        target = 0.0
        if k > 0:
            for i in range(k):
                target -= coeff[i] * gsc_coeffs[i, k]

        step = 1
        upward = True

        while True:
            if upward:
                coeff[k] = int(np.ceil(target)) + step // 2
            else:
                coeff[k] = int(np.ceil(target)) - (step + 1) // 2
            upward = not upward
            step += 1

            diff = coeff[k] - target
            new_partial = partial[k - 1] if k > 0 else 0.0
            new_partial += diff * diff * gs_sq_norms[k]

            if new_partial >= best_radius:
                if step > 2 * int(np.sqrt(best_radius / gs_sq_norms[k])) + 10:
                    break
                continue

            if k == block_size - 1:
                best_radius = new_partial
                found = True
                for i in range(block_size):
                    best_coeff[i] = coeff[i]
                break

            partial[k] = new_partial
            k += 1
            coeff[k] = 0
            break
        else:
            k -= 1

    return found, best_radius, best_coeff
