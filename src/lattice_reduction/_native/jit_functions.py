"""
Numba JIT 加速版核心数值计算函数。

所有 JIT 函数均为纯数值计算，无 Python 对象操作。
无 Numba 环境时自动降级为纯 Python 版本。

核心策略（参考 fplll）：
  1. 基向量始终用 int64（精确，无精度漂移）
  2. GSO 系数和范数用 float64（快速）
  3. 点积用 int64 精确计算（避免 float64 溢出 53bit）
  4. Kahan 补偿求和减小 GSO 减法链的累积误差
  5. 定期从精确整数重新计算完整 GSO（重置累积误差）
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
def gso_full_refresh_int(basis_int, gsc, gs, end_stage):
    """从精确整数点积完整重算 GSO（重置累积浮点误差）。

    参考 fplll 的精度修复策略：当浮点 GSO 的累积误差过大时，
    从精确整数基重新计算所有 GSO 系数和范数。

    复杂度: O(end_stage² × n)。调用频率：每 dim 次 swap 一次。
    """
    n = basis_int.shape[0]
    dots = np.zeros(end_stage, dtype=np.int64)

    # Column 0: gs[0] = ||b_0||²
    s = np.int64(0)
    for r in range(n):
        s += basis_int[r, 0] * basis_int[r, 0]
    gs[0] = float(s)
    gsc[0, 0] = 1.0

    for k in range(1, end_stage):
        # 精确整数点积 <b_k, b_j> for j < k, 和 ||b_k||²
        norm_sq = np.int64(0)
        for j in range(k):
            dots[j] = 0
        for r in range(n):
            bk_r = basis_int[r, k]
            norm_sq += bk_r * bk_r
            for j in range(k):
                dots[j] += bk_r * basis_int[r, j]

        gs[k] = float(norm_sq)

        # Kahan 补偿 GSO 减法链
        gs_c = 0.0
        for j in range(k):
            # 校正项: Σ_{i<j} gsc[i,j] · gsc[i,k] · gs[i]
            ct = 0.0
            ct_c = 0.0
            for i in range(j):
                term = gsc[i, j] * gsc[i, k] * gs[i]
                y = term - ct_c
                t = ct + y
                ct_c = (t - ct) - y
                ct = t

            if gs[j] > 1e-30:
                gsc[j, k] = (float(dots[j]) - ct) / gs[j]
            else:
                gsc[j, k] = 0.0

            # gs[k] -= gsc[j,k]² × gs[j]  (Kahan 补偿)
            term = gsc[j, k] * gsc[j, k] * gs[j]
            y = -term - gs_c
            t = gs[k] + y
            gs_c = (t - gs[k]) - y
            gs[k] = t

        gsc[k, k] = 1.0


@njit(cache=True)
def l3fp_jit_int(basis_int, gsc, gs, stage, lovasz_param, tau_limit):
    """JIT LLL — int64 精确点积 + Kahan GSO + 定期完整刷新。

    基向量始终保持 int64（精确），GSO 用 float64（快速）。
    定期调用 gso_full_refresh_int 重置累积浮点误差。
    参考 fplll 的精度管理策略。

    Args:
        basis_int: (n, m) int64, 列向量基, 原地修改
        gsc: (m, m) float64, GSO 系数, 原地修改
        gs: (m,) float64, GSO 平方范数, 原地修改
        stage: 起始 stage
        lovasz_param: δ 参数
        tau_limit: 浮点精度阈值

    Returns:
        (basis_int, gsc, gs) — 全部原地修改
    """
    n = basis_int.shape[0]
    end_stage = basis_int.shape[1]
    threshold = 0.5
    f_c = False
    dots_int = np.zeros(end_stage, dtype=np.int64)

    swap_count = 0
    stuck_counter = 0
    refresh_interval = end_stage  # 每 dim 次 swap 刷新一次
    max_iterations = end_stage * end_stage * 50  # 安全网
    iterations = 0

    while stage < end_stage and iterations < max_iterations:
        iterations += 1

        # ── GSO 步: 精确 int64 点积 + Kahan 补偿减法链 ──
        if stage == 1:
            s = np.int64(0)
            for r in range(n):
                s += basis_int[r, 0] * basis_int[r, 0]
            gs[0] = float(s)

        norm_sq = np.int64(0)
        for j in range(stage):
            dots_int[j] = 0
        for r in range(n):
            bk_r = basis_int[r, stage]
            norm_sq += bk_r * bk_r
            for j in range(stage):
                dots_int[j] += bk_r * basis_int[r, j]
        gs[stage] = float(norm_sq)

        # Kahan 补偿 GSO 减法链
        gs_c = 0.0
        for j in range(stage):
            ct = 0.0
            ct_c = 0.0
            for i in range(j):
                term = gsc[i, j] * gsc[i, stage] * gs[i]
                y = term - ct_c
                t = ct + y
                ct_c = (t - ct) - y
                ct = t

            if gs[j] > 1e-30:
                gsc[j, stage] = (float(dots_int[j]) - ct) / gs[j]
            else:
                gsc[j, stage] = 0.0

            term = gsc[j, stage] * gsc[j, stage] * gs[j]
            y = -term - gs_c
            t = gs[stage] + y
            gs_c = (t - gs[stage]) - y
            gs[stage] = t

        gsc[stage, stage] = 1.0

        # ── 尺寸缩减（整数基，精确） ──
        f_c = False
        for i in range(stage - 1, -1, -1):
            if abs(gsc[i, stage]) > threshold:
                mu = round(gsc[i, stage])
                if abs(mu) > tau_limit:
                    f_c = True
                for k in range(i):
                    gsc[k, stage] -= mu * gsc[k, i]
                gsc[i, stage] -= mu
                for r in range(n):
                    basis_int[r, stage] -= mu * basis_int[r, i]

        if f_c:
            stage = max(stage - 1, 1)
            continue

        # ── Lovász 条件 ──
        mu = gsc[stage - 1, stage]
        if lovasz_param * gs[stage - 1] > gs[stage] + mu * mu * gs[stage - 1]:
            # 交换列 stage-1 和 stage
            for r in range(n):
                tmp = basis_int[r, stage - 1]
                basis_int[r, stage - 1] = basis_int[r, stage]
                basis_int[r, stage] = tmp
            stage = max(stage - 1, 1)
            swap_count += 1
            stuck_counter += 1

            # 定期完整 GSO 刷新（重置累积误差）
            if swap_count % refresh_interval == 0:
                gso_full_refresh_int(basis_int, gsc, gs, end_stage)

            # 卡住检测：连续 swap 无进展时强制刷新
            if stuck_counter > refresh_interval * 2:
                gso_full_refresh_int(basis_int, gsc, gs, end_stage)
                stuck_counter = 0
        else:
            stuck_counter = 0
            stage += 1

    return basis_int, gsc, gs


@njit(cache=True)
def l3fp_jit(basis_matrix, gsc, gs, stage, lovasz_param, tau_limit):
    """JIT LLL — float64 基（用于 deep insertion 路径）。

    Deep insertion 返回 float64 基，此函数处理该路径。
    块大小通常较小（BKZ block_size），float64 精度足够。

    Args:
        basis_matrix: (n, m) float64, 列向量基, 原地修改
        gsc: (m, m) float64, GSO 系数, 原地修改
        gs: (m,) float64, GSO 平方范数, 原地修改
        stage: 起始 stage
        lovasz_param: δ 参数
        tau_limit: 浮点精度阈值

    Returns:
        (basis_matrix, gsc, gs) — 全部原地修改
    """
    n = basis_matrix.shape[0]
    end_stage = basis_matrix.shape[1]
    threshold = 0.5
    f_c = False
    dots = np.zeros(end_stage)

    while stage < end_stage:
        # ── GSO 步 ──
        if stage == 1:
            s = 0.0
            for r in range(n):
                s += basis_matrix[r, 0] * basis_matrix[r, 0]
            gs[0] = s

        norm_sq = 0.0
        for j in range(stage):
            dots[j] = 0.0
        for r in range(n):
            bk_r = basis_matrix[r, stage]
            norm_sq += bk_r * bk_r
            for j in range(stage):
                dots[j] += bk_r * basis_matrix[r, j]
        gs[stage] = norm_sq

        for j in range(stage):
            ct = 0.0
            for i in range(j):
                ct += gsc[i, j] * gsc[i, stage] * gs[i]
            gsc[j, stage] = (dots[j] - ct) / gs[j]
            gs[stage] -= gsc[j, stage] * gsc[j, stage] * gs[j]
        gsc[stage, stage] = 1.0

        # ── 尺寸缩减 ──
        f_c = False
        for i in range(stage - 1, -1, -1):
            if abs(gsc[i, stage]) > threshold:
                mu = round(gsc[i, stage])
                if abs(mu) > tau_limit:
                    f_c = True
                for k in range(i):
                    gsc[k, stage] -= mu * gsc[k, i]
                gsc[i, stage] -= mu
                for r in range(n):
                    basis_matrix[r, stage] -= mu * basis_matrix[r, i]

        if f_c:
            stage = max(stage - 1, 1)
            continue

        # ── Lovász 条件 ──
        mu = gsc[stage - 1, stage]
        if lovasz_param * gs[stage - 1] > gs[stage] + mu * mu * gs[stage - 1]:
            for r in range(n):
                tmp = basis_matrix[r, stage - 1]
                basis_matrix[r, stage - 1] = basis_matrix[r, stage]
                basis_matrix[r, stage] = tmp
            stage = max(stage - 1, 1)
        else:
            stage += 1

    return basis_matrix, gsc, gs


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
