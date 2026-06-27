"""
Numba JIT 加速版核心数值计算函数。

所有 JIT 函数均为纯数值计算，无 Python 对象操作。
无 Numba 环境时自动降级为纯 Python 版本。
"""

import numpy as np
from .jit_compat import njit


@njit
def gso_step_jit(basis_slice, gsc, gs, stage):
    """GSO 单步计算（JIT 加速版）。

    CGS: μ_{j,k} = (⟨b_k, b_j⟩ - Σ_{i<j} μ_{i,j}·μ_{i,k}·||b*_i||²) / ||b*_j||²
    """
    n = basis_slice.shape[0]

    if stage == 1:
        s = 0.0
        for r in range(n):
            s += basis_slice[r, 0] * basis_slice[r, 0]
        gs[0] = s

    # Compute dot products ⟨b_k, b_j⟩ for j < stage
    dots = np.zeros(stage)
    for j in range(stage):
        s = 0.0
        for r in range(n):
            s += basis_slice[r, stage] * basis_slice[r, j]
        dots[j] = s

    # Initial squared norm
    gs[stage] = dots[0] if stage == 0 else np.dot(
        basis_slice[:, stage], basis_slice[:, stage]
    )
    # Recompute from dots for consistency
    s = 0.0
    for r in range(n):
        s += basis_slice[r, stage] * basis_slice[r, stage]
    gs[stage] = s

    for j in range(stage):
        # correction_term = Σ_{i<j} μ_{i,j} · μ_{i,stage} · ||b*_i||²
        ct = 0.0
        for i in range(j):
            ct += gsc[i, j] * gsc[i, stage] * gs[i]

        gsc[j, stage] = (dots[j] - ct) / gs[j]
        gs[stage] -= gsc[j, stage] * gsc[j, stage] * gs[j]

    gsc[stage, stage] = 1.0


@njit
def size_reduction_jit(gsc_col, gsc_mat, basis_col, basis_mat, stage, tau_limit):
    """尺寸缩减核心（JIT 加速版）。

    对 column `stage` 的所有前驱执行尺寸缩减。
    """
    threshold = 0.5
    f_c = False

    for i in range(stage - 1, -1, -1):
        if abs(gsc_col[i]) > threshold:
            mu = round(gsc_col[i])
            if abs(mu) > tau_limit:
                f_c = True

            # Update GSO coefficients
            for k in range(i):
                gsc_col[k] -= mu * gsc_mat[k, i]
            gsc_col[i] -= mu

            # Update basis vector
            n = basis_col.shape[0]
            for r in range(n):
                basis_col[r] -= mu * basis_mat[r, i]

    return f_c


@njit
def enum_se_jit(block, gs_sq_norms, gsc_coeffs, search_radius):
    """SVP 枚举器（JIT 加速版）— Schnorr-Euchner 策略。

    Args:
        block: (dim, block_size) 块基向量
        gs_sq_norms: (block_size,) GSO 平方范数
        gsc_coeffs: (block_size, block_size) GSO 系数
        search_radius: 初始搜索半径

    Returns:
        (found, radius, coeff) — found=True 表示找到短向量
    """
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
        # Compute target coefficient using Schnorr-Euchner stepping
        if k == 0:
            target = 0.0
        else:
            target = 0.0
            for i in range(k):
                target -= coeff[i] * gsc_coeffs[i, k]

        # Enumerate around target
        step = 1
        upward = True

        while True:
            if upward:
                coeff[k] = int(np.ceil(target)) + step // 2
            else:
                coeff[k] = int(np.ceil(target)) - (step + 1) // 2
            upward = not upward
            step += 1

            # Partial norm
            diff = coeff[k] - target
            new_partial = partial[k - 1] if k > 0 else 0.0
            new_partial += diff * diff * gs_sq_norms[k]

            if new_partial >= best_radius:
                # Prune: try next coefficient or backtrack
                if step > 2 * int(np.sqrt(best_radius / gs_sq_norms[k])) + 10:
                    break
                continue

            # Found shorter vector
            if k == block_size - 1:
                best_radius = new_partial
                found = True
                for i in range(block_size):
                    best_coeff[i] = coeff[i]
                break

            # Go deeper
            partial[k] = new_partial
            k += 1
            coeff[k] = 0
            break
        else:
            k -= 1

    return found, best_radius, best_coeff
