"""Kannan 嵌入格基构造器

将 ML-DSA 公钥 (A, t) 构造为 Kannan 嵌入格基矩阵。

本模块是纯算法层，仅依赖 numpy，不依赖项目其他模块。
"""

from __future__ import annotations

import logging
import time
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger("lattice_attack")


def build_A_flat(A: np.ndarray) -> np.ndarray:
    """Flatten the polynomial matrix A (k, l, n) into A_flat (k*n, l*n).

    每个多项式 A[i,j] 展开为 n×n 负循环卷积矩阵 (mod x^n + 1)。
    """
    k, l, n = A.shape
    kn = k * n
    ln = l * n
    A_flat = np.zeros((kn, ln), dtype=np.int64)
    for i in range(k):
        for j in range(l):
            poly = A[i, j]
            for rr in range(n):
                for cc in range(n):
                    row = i * n + rr
                    col = j * n + cc
                    diff = rr - cc
                    idx = diff % n
                    if diff >= 0:
                        A_flat[row, col] = poly[idx]
                    else:
                        A_flat[row, col] = -poly[idx]
    return A_flat


def build_lattice_basis(
    A: np.ndarray,
    t: np.ndarray,
    q: int,
    n: int,
    k: int,
    l: int,
    sigma: float,
    mp_dps: int = 0,
    auto_precision: bool = False,
    t_recon: Optional[np.ndarray] = None,
    dual_mode: bool = False,
) -> dict:
    """构造 Kannan 嵌入格基矩阵 B。

    Parameters
    ----------
    A, t : 公钥矩阵和向量
    q, n, k, l : ML-DSA 参数
    sigma : 噪声标准差
    mp_dps : 任意精度位数 (0=自动)
    auto_precision : 自动精度调整
    t_recon : Power2Round 模式下已恢复的 t 值
    dual_mode : 双模格基（含方程约束）

    格基始终使用标准 Kannan 嵌入 (dim = ln + kn + 1)。

    Returns a dict with timing, results, and verification details.
    """
    result = {}
    kn = k * n
    ln = l * n
    dim = kn + ln + 1
    from .precision import get_initial_dps
    dps, _ = get_initial_dps(dim, mp_dps, auto_precision)

    # ── Build basis ──
    logger.info(f"[3/5] 构造格基矩阵 ({dim}×{dim})...")
    t0_build = time.time()
    B = build_lattice_basis_raw(A, t, q, n, k, l, sigma, t_recon=t_recon)
    result["build_time"] = time.time() - t0_build
    result["dps"] = dps
    result["dim"] = dim

    logger.info(f"  构造耗时: {result['build_time']:.2f}s")
    logger.info(f"  DPS: {dps}, dim: {dim}")

    # ── Verify basis ──
    from .classifier import verify_basis
    vres = verify_basis(B, A, t, q, n, k, l, sigma, t_recon=t_recon)
    result["verification"] = vres
    if not vres.passed:
        logger.warning(f"  ⚠ 格基验证失败: {vres.error}")
    else:
        logger.info("  ✓ 格基验证通过")

    return result


def build_lattice_basis_raw(
    A: np.ndarray,
    t: np.ndarray,
    q: int,
    n: int,
    k: int,
    l: int,
    sigma: float,
    t_recon: Optional[np.ndarray] = None,
) -> np.ndarray:
    """构造标准 Kannan 嵌入格基矩阵 (ln + kn + 1) × (ln + kn + 1)。

    格基结构:
        B = [  I_l  |  A_flat  |  0  ]
            [  0    |  I_k     |  0  ]
            [  t^T  |  s2^T    |  e ]

    Parameters
    ----------
    A : (k, l, n) 公钥多项式矩阵
    t : (l, n) 公钥向量
    q : 模数
    n, k, l : ML-DSA 参数
    sigma : 噪声标准差
    t_recon : Power2Round 模式下恢复的 t 值 (默认 None=标准模式)

    Returns
    -------
    B : (dim, dim) 整数格基矩阵
    """
    kn = k * n
    ln = l * n
    dim = kn + ln + 1

    # 展平 A
    A_flat = build_A_flat(A)

    # 构造格基
    B = np.zeros((dim, dim), dtype=np.int64)

    # 左上: I_l (ln × ln)
    for i in range(ln):
        B[i, i] = 1

    # 中上: A_flat (ln × kn)
    B[:ln, ln:ln + kn] = A_flat

    # 中中: I_k (kn × kn)
    for i in range(kn):
        B[ln + i, ln + i] = 1

    # 嵌入行: [t^T | s2^T | e]
    # 使用标准 Kannan 嵌入 (sigma_e = 1)
    t_flat = t.reshape(-1)
    if t_recon is not None:
        t_embed = t_recon.reshape(-1)
    else:
        t_embed = t_flat

    # 嵌入权重: 1 (sigma_e = 1)
    B[-1, :ln] = t_embed
    B[-1, -1] = 1

    return B
