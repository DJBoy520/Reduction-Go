"""
纯 mpmath 实现的 GSO 计算模块。

所有 GSO 系数和范数全程使用 mpmath.mpf，不经过 float64 中间值。
基向量保持 int64（精确整数），点积用 Python 大整数（精确）。

精度由外层 mpmath.workdps 上下文控制（在 lll_mp / bkz 入口设置），
本模块函数不切换精度。
"""

import numpy as np
import mpmath

from .precision_errors import PrecisionFailureError


_INT64_DOT_SAFE = 2 ** 62


def _int_dot(col_a: np.ndarray, col_b: np.ndarray) -> int:
    """精确整数点积（Python int，无 int64 溢出）。

    快速路径: 先用 np.int64 原生 C 级 dot 计算，
    仅当 max|a_i| * max|b_i| * n >= 2^62 时才降级到 object 大数点积。
    绝大多数 LLL/BKZ 场景走快速路径，性能提升 3~5 倍。
    """
    a = np.asarray(col_a, dtype=np.int64)
    b = np.asarray(col_b, dtype=np.int64)
    n = a.shape[0]

    # Python int 无溢出，安全做乘法判断
    max_a = int(np.max(np.abs(a)))
    max_b = int(np.max(np.abs(b)))
    if max_a * max_b * n < _INT64_DOT_SAFE:
        return int(np.dot(a, b))

    return int(a.astype(object) @ b.astype(object))


def gso_full_refresh_mp(
    basis_int: np.ndarray,
    gsc: list,   # 2D list-of-lists of mpmath.mpf
    gsn: list,   # 1D list of mpmath.mpf
    end_stage: int,
) -> bool:
    """从精确整数基完整重算 GSO（纯 mpmath 高精度）。

    基向量 int64（精确），点积 Python int（精确），
    GSO 系数和范数 mpmath.mpf（任意精度）。

    复杂度: O(end_stage² × n)。201 维 dps=100 约 6 秒。

    精度由外层 workdps 上下文控制，本函数不切换精度。

    Args:
        basis_int: (n, m) int64 numpy 数组，列向量基
        gsc: (m, m) list-of-lists of mpf（原地修改）
        gsn: (m,) list of mpf（原地修改）
        end_stage: 计算到第几列

    Returns:
        has_zero: True 如果发现线性相关列（gsn <= 0）
    """
    zero = mpmath.mpf(0)
    one = mpmath.mpf(1)
    has_zero = False

    # Column 0
    gsn[0] = mpmath.mpf(_int_dot(basis_int[:, 0], basis_int[:, 0]))
    gsc[0][0] = one

    for k in range(1, end_stage):
        # 批量计算点积（object dtype 防 int64 溢出，一次转换 + 一次矩阵-向量乘法）
        col_k = basis_int[:, k].astype(np.int64).astype(object)
        gsn[k] = mpmath.mpf(int(col_k @ col_k))

        mat = basis_int[:, :k].astype(np.int64).astype(object)
        dots = mat.T @ col_k  # (k,) object array — 一次算完所有 <b_j, b_k>

        for j in range(k):
            dot_jk = mpmath.mpf(int(dots[j]))

            # correction: Σ_{i<j} μ_{i,j} · μ_{i,k} · ||b*_i||²
            if j > 0:
                terms = []
                for i in range(j):
                    if gsn[i] > zero:
                        terms.append(gsc[i][j] * gsc[i][k] * gsn[i])
                correction = mpmath.fsum(terms) if terms else zero
            else:
                correction = zero

            if gsn[j] > zero:
                gsc[j][k] = (dot_jk - correction) / gsn[j]
                gsn[k] -= gsc[j][k] ** 2 * gsn[j]
            else:
                # 零范数（线性相关）
                gsc[j][k] = zero

        gsc[k][k] = one

        if gsn[k] <= zero:
            has_zero = True

    return has_zero


def gso_step_mp(
    basis_int: np.ndarray,
    gsc: list,   # 2D list-of-lists of mpf
    gsn: list,   # 1D list of mpf
    stage: int,
):
    """GSO 单步计算（纯 mpmath 高精度）。

    原地修改 gsc 和 gsn 的第 stage 列。
    精度由外层 workdps 上下文控制。

    Args:
        basis_int: (n, m) int64 numpy 数组
        gsc: (m, m) list-of-lists of mpf
        gsn: (m,) list of mpf
        stage: 当前 stage 索引
    """
    zero = mpmath.mpf(0)
    one = mpmath.mpf(1)

    # 始终刷新 gsn[0]（处理列交换后第一列变化的情况）
    if stage >= 1:
        gsn[0] = mpmath.mpf(_int_dot(basis_int[:, 0], basis_int[:, 0]))
        gsc[0][0] = one

    if stage == 0:
        return

    # 批量计算点积（object dtype 防 int64 溢出，一次转换 + 一次矩阵-向量乘法）
    col_s = basis_int[:, stage].astype(np.int64).astype(object)
    gsn[stage] = mpmath.mpf(int(col_s @ col_s))

    mat = basis_int[:, :stage].astype(np.int64).astype(object)
    dots = mat.T @ col_s  # (stage,) object array — 一次算完所有 <b_j, b_stage>

    for j in range(stage):
        dot_jk = mpmath.mpf(int(dots[j]))

        if j > 0:
            terms = []
            for i in range(j):
                if gsn[i] > zero:
                    terms.append(gsc[i][j] * gsc[i][stage] * gsn[i])
            correction = mpmath.fsum(terms) if terms else zero
        else:
            correction = zero

        if gsn[j] > zero:
            gsc[j][stage] = (dot_jk - correction) / gsn[j]
            gsn[stage] -= gsc[j][stage] ** 2 * gsn[j]
        else:
            gsc[j][stage] = zero

    gsc[stage][stage] = one


def gso_swap_update_mp(
    basis_int: np.ndarray,
    gsc: list,   # 2D list-of-lists of mpf
    gsn: list,   # 1D list of mpf
    s: int,
    dim: int,
):
    """交换列 s 与 s+1，增量更新 GSO，复杂度 O(dim)。

    标准 LLL 两列交换的 GSO 增量更新。
    精度由外层 workdps 上下文控制。

    交换公式（格体积不变性）：
        B'_s   = B_{s+1} + μ² · B_s
        B'_{s+1} = B_s · B_{s+1} / B'_s        ← 体积守恒
        μ'(s,s+1) = μ · B_s / B'_s
        对 j > s+1: μ'(s,j)   = μ(s+1,j) - μ · μ(s,j)
                     μ'(s+1,j) = μ(s,j) + μ'(s,s+1) · μ'(s,j)
    """
    zero = mpmath.mpf(0)
    mu = gsc[s][s + 1]
    B_s_old = gsn[s]
    B_s1_old = gsn[s + 1]
    B_s_new = B_s1_old + mu * mu * B_s_old

    # 除零防御：极端精度耗尽时 B_s_new 可能趋近于 0
    if B_s_new <= zero:
        raise PrecisionFailureError(
            f"GSO swap: B'_s <= 0 at s={s}, precision exhausted",
            mpmath.mp.dps,
        )

    # 格体积不变性：B'_s · B'_{s+1} = B_s · B_{s+1}
    B_s1_new = B_s_old * B_s1_old / B_s_new

    gsn[s] = B_s_new
    gsn[s + 1] = B_s1_new
    gsc[s][s + 1] = mu * B_s_old / B_s_new
    gsc[s][s] = mpmath.mpf(1)
    gsc[s + 1][s] = mpmath.mpf(0)
    gsc[s + 1][s + 1] = mpmath.mpf(1)

    # 更新后续列 (j > s+1) 的投影系数
    for j in range(s + 2, dim):
        old_mu_sj = gsc[s][j]
        old_mu_s1j = gsc[s + 1][j]
        gsc[s][j] = old_mu_s1j - mu * old_mu_sj
        gsc[s + 1][j] = old_mu_sj + gsc[s][s + 1] * gsc[s][j]


def init_gso_mp(dim: int):
    """初始化 mpmath GSO 数组。

    零值初始化不依赖精度，由外层 workdps 上下文决定精度。

    Returns:
        gsc: (dim, dim) list-of-lists of mpf(0)
        gsn: (dim,) list of mpf(0)
    """
    zero = mpmath.mpf(0)
    one = mpmath.mpf(1)
    gsc = [[zero for _ in range(dim)] for _ in range(dim)]
    gsn = [zero for _ in range(dim)]
    gsc[0][0] = one
    return gsc, gsn


def gso_norms_to_float(gsn_mp: list, n: int) -> np.ndarray:
    """将 mpf 范数列表转为 float64 numpy 数组。"""
    return np.array([float(gsn_mp[i]) for i in range(n)], dtype=np.float64)


def gso_coeffs_to_float(gsc_mp: list, m: int, n: int) -> np.ndarray:
    """将 mpf 系数矩阵转为 float64 numpy 数组。"""
    result = np.zeros((m, n), dtype=np.float64)
    for i in range(m):
        for j in range(n):
            result[i, j] = float(gsc_mp[i][j])
    return result


def delete_zero_vector(
    basis_matrix: np.ndarray,
    gs_coeff_matrix: np.ndarray,
    gs_squared_norms: np.ndarray,
    pos: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """删除第 pos 列（零向量），返回更新后的三元组。

    gs_squared_norms 是一维范数数组，不能传 axis=1。
    """
    basis_matrix = np.delete(basis_matrix, pos, axis=1)
    gs_coeff_matrix = np.delete(gs_coeff_matrix, pos, axis=1)
    gs_squared_norms = np.delete(gs_squared_norms, pos)
    return basis_matrix, gs_coeff_matrix, gs_squared_norms
