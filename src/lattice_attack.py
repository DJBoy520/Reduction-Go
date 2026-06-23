"""
Lattice attack module.

Uses Kannan's embedding to recover short secret vectors from the
public key (A, t) where A_flat · s1 + s2 ≡ t (mod q).

Standard Kannan Embedding
=========================
Dim = ln + kn + 1. Target: (s1, s2, W).

Power2Round 合并误差模式
=========================
当公钥仅含 t1 时 (t = t1·2^d + t0),
令 s2' = s2 - t0, 则 A·s1 + s2' ≡ t_recon (mod q),
其中 t_recon = t1·2^d。
s2' 仍相对较小 (s2 ∈ [-η, η], t0 ∈ [-2^{d-1}, 2^{d-1}]),
使用标准 Kannan 嵌入, 目标向量设为 t_recon。
"""

import logging
import sys
import time

import numpy as np

from fpylll import IntegerMatrix, LLL, BKZ, FPLLL

from .poly_math import mat_vec_mul, vec_add_mod
from .progress import LLLProgress, BKZProgress

logger = logging.getLogger(__name__)


def _build_A_flat(A: np.ndarray) -> np.ndarray:
    """Flatten the polynomial matrix A (k, l, n) into A_flat (k*n, l*n).

    Each polynomial A[i,j] is expanded into an n×n negacyclic convolution
    matrix (mod x^n + 1).
    """
    k, l, n = A.shape
    kn, ln = k * n, l * n
    A_flat = np.zeros((kn, ln), dtype=np.int64)
    for i in range(k):
        for j in range(l):
            poly = A[i, j]
            for rr in range(n):
                for cc in range(n):
                    row = i * n + rr
                    col = j * n + cc
                    diff = rr - cc
                    if diff >= 0:
                        A_flat[row, col] = int(poly[diff])
                    else:
                        A_flat[row, col] = -int(poly[n + diff])
    return A_flat


def build_lattice_basis(A: np.ndarray, t: np.ndarray, q: int) -> IntegerMatrix:
    """构造 Kannan 嵌入格基矩阵。

    Args:
        A: 多项式矩阵 (k, l, n)
        t: 目标向量 (标准模式用完整 t; Power2Round 模式用 t_recon = t1·2^d)
        q: 模数
    """
    k, l, n = A.shape
    kn = k * n
    ln = l * n

    A_flat = _build_A_flat(A)
    t_flat = t.flatten().astype(np.int64)
    W = 1

    # ── Standard Kannan Embedding ──
    dim = ln + kn + 1

    B = IntegerMatrix(dim, dim)

    # 1. 左上角: I_{ln}
    for i in range(ln):
        B[i, i] = 1

    # 2. 中上角: -A_flat^T (转置)
    #    _build_A_flat 输出 (kn, ln)，格基中按列放置即为转置
    #    mat_vec_mul 实现的是 A_flat @ s1_flat，此处用 -A_flat^T 使
    #    格向量中间块 = -A_flat·c1 + q·c2 + t·c3，与方程一致
    for i in range(ln):
        for j in range(kn):
            B[i, ln + j] = -int(A_flat[j, i])

    # 3. 中间: q·I_{kn}
    for j in range(kn):
        B[ln + j, ln + j] = q

    # 4. 最后一行: [0, t^T, W]
    for j in range(kn):
        B[dim - 1, ln + j] = int(t_flat[j])
    B[dim - 1, dim - 1] = W

    return B


def verify_basis(A: np.ndarray, t_target: np.ndarray, q: int,
                 s1: np.ndarray, s2_target: np.ndarray):
    """验证 A_flat · s1 + s2_target ≡ t_target (mod q)。

    注意: s2_target 和 t_target 不一定是原始私钥。
    Power2Round 模式下它们分别是 s2' = s2 - t0 和 t_recon = t1·2^d。
    """
    k, l, n = A.shape
    A_flat = _build_A_flat(A)
    s1_flat = s1.flatten().astype(np.int64)
    s2_flat = s2_target.flatten().astype(np.int64)
    t_flat = t_target.flatten().astype(np.int64)

    residual = (A_flat @ s1_flat + s2_flat - t_flat) % q
    ok = bool(np.all(residual == 0))
    logger.info(f"verify_basis: {'A·s1 + s2_target ≡ t_target (mod q) ✓' if ok else '方程不成立 ✗'}")
    return ok


def run_attack(A: np.ndarray, t: np.ndarray, q: int,
               s1_real: np.ndarray, s2_real: np.ndarray,
               bkz_block_size: int = 20, bkz_max_loops: int = 8,
               bkz_threads: int = 6, no_bkz: bool = False,
               lll_delta: float = 0.999,
               bkz_auto_abort: bool = False,
               float_type: str = "mpfr",
               precision: int = 200) -> dict:
    """Run the full lattice attack: build basis → LLL → BKZ → extract & verify.

    t 参数: 标准模式传完整 t; Power2Round 模式传 t_recon = t1·2^d。
    格基始终使用标准 Kannan 嵌入 (dim = ln + kn + 1)。

    Returns a dict with timing, results, and verification details.
    """
    import os
    # fpylll 0.6.4 不支持 BKZ.Param(threads=...)，用 OMP_NUM_THREADS 控制并行
    if bkz_threads:
        os.environ.setdefault("OMP_NUM_THREADS", str(bkz_threads))

    result = {}
    k, l, n = A.shape
    kn = k * n
    ln = l * n
    dim = kn + ln + 1

    # ── Build basis ──
    logger.info(f"[3/5] 构造格基矩阵 ({dim}×{dim})...")
    t0_build = time.time()
    B = build_lattice_basis(A, t, q)
    result["build_time"] = time.time() - t0_build

    # Memory estimate
    mem_mb = dim * dim * 8 / (1024 ** 2)
    logger.debug(f"格基预估内存占用: {mem_mb:.2f} MB ({dim}×{dim}, int64)")

    # ── LLL ──
    logger.info(f"[4/5] LLL 约减 (维度 {dim}, {float_type}/{precision}bit)")

    lll_kwargs = {"delta": lll_delta}
    if float_type != "double":
        FPLLL.set_precision(precision)
        lll_kwargs["float_type"] = float_type
        lll_kwargs["precision"] = precision
        lll_kwargs["method"] = "proved"

    lll_progress = LLLProgress(dim, float_type, precision)
    lll_progress.start()
    LLL.reduction(B, **lll_kwargs)
    result["lll_time"] = lll_progress.finish()
    logger.info(f"    LLL 完成: {result['lll_time']:.3f}s")

    # ── BKZ ──
    if no_bkz:
        logger.info("[5/5] BKZ 已跳过 (--no-bkz)")
        result["bkz_time"] = 0.0
    else:
        logger.info(f"[5/5] BKZ 约减 (b={bkz_block_size}, max={bkz_max_loops})")

        # 计算 real_norm 用于 ratio 显示
        s1_c = s1_real.copy()
        s2_c = s2_real.copy()
        s1_c[s1_c >= q // 2] -= q
        s2_c[s2_c >= q // 2] -= q
        real_norm_for_ratio = float(np.sqrt(
            np.sum(s1_c.astype(np.int64) ** 2) +
            np.sum(s2_c.astype(np.int64) ** 2)
        ))

        bkz_progress = BKZProgress(dim, bkz_block_size, bkz_max_loops,
                                    float_type, precision)
        bkz_progress.start()

        bkz_kwargs = {}
        if float_type != "double":
            bkz_kwargs["float_type"] = float_type
            bkz_kwargs["precision"] = precision

        t_bkz = time.time()
        completed_loops = 0

        # fpylll 0.6.4 不支持 BKZ.Param(threads=...)，多线程不生效
        # 如需并行可用 OMP_NUM_THREADS 环境变量
        for loop_i in range(1, bkz_max_loops + 1):
            param = BKZ.Param(
                block_size=bkz_block_size,
                max_loops=1,
                auto_abort=False,
            )
            BKZ.reduction(B, param, **bkz_kwargs)
            completed_loops = loop_i

            # 从格基提取当前最短范数
            try:
                norms = []
                for row_i in range(dim):
                    row = [int(B[row_i, col]) for col in range(dim)]
                    n_sq = sum(x * x for x in row)
                    if n_sq > 0:
                        norms.append(n_sq ** 0.5)
                shortest = min(norms) if norms else 0.0
            except Exception:
                shortest = 0.0

            bkz_progress.update(loop_i, shortest_norm=shortest,
                                real_norm=real_norm_for_ratio)

            # auto_abort 检查: 连续无改善时提前退出
            if bkz_auto_abort and loop_i > 2:
                # 简单实现: 如果最短范数连续 2 轮没变小就停
                if not hasattr(bkz_progress, "_prev_norms"):
                    bkz_progress._prev_norms = []
                bkz_progress._prev_norms.append(shortest)
                if len(bkz_progress._prev_norms) >= 3:
                    recent = bkz_progress._prev_norms[-3:]
                    if recent[-1] >= recent[-2] >= recent[-3]:
                        logger.info(f"    BKZ auto-abort: 连续无改善，提前终止于 loop {loop_i}")
                        break

        result["bkz_time"] = bkz_progress.finish()
        result["bkz_loops"] = completed_loops
    logger.info(f"    BKZ 完成: {result['bkz_time']:.3f}s")

    # ── Extract candidates ──
    #
    # Kannan 嵌入格基结构:
    #   B = [ I_{ln}   -A_flat^T   0 ]  ← 左上角是 I
    #       [ 0        q·I_{kn}    0 ]
    #       [ 0        t^T         W ]
    #
    # 格向量 v = c · B = (c1, -A_flat·c1 + q·c2 + t·c3, W·c3)
    #
    # 当 c3=1, c1=s1 时:
    #   v[:ln] = s1 (直接读取，I 块保证无 q 倍数)
    #   v[ln:-1] = s2 + q·c2 (需要模 q 还原 s2)
    #
    # 当 c3=1, c1=-s1 时:
    #   v[:ln] = -s1, v[ln:-1] = -s2 + q·c2 (sign=-1 翻转)
    #
    logger.info("    搜索候选短向量 (Kannan 嵌入)...")
    # Center real secrets for comparison
    s1_real_c = s1_real.copy()
    s2_real_c = s2_real.copy()
    s1_real_c[s1_real_c >= q // 2] -= q
    s2_real_c[s2_real_c >= q // 2] -= q
    real_norm = float(np.sqrt(
        np.sum(s1_real_c.astype(np.int64) ** 2) +
        np.sum(s2_real_c.astype(np.int64) ** 2)
    ))

    candidates = []
    W = 1

    for row_idx in range(dim):
        row = [int(B[row_idx, col]) for col in range(dim)]

        # Kannan 嵌入: 最后一维是 W 或 -W
        weight = row[-1]
        if abs(weight) != W:
            continue

        sign = 1 if weight == W else -1

        # 读取: s1 来自左上角 I 块，直接读取
        #       s2 来自 q·I 块，值为 s2 + q·c2，需要模 q 还原
        s1_cand = np.array([sign * row[i] for i in range(ln)], dtype=np.int64)
        s2_cand = np.array([sign * row[ln + i] for i in range(kn)], dtype=np.int64)

        s1_cand_c = s1_cand.copy()
        s1_cand_c[s1_cand_c > q // 2] -= q
        # s2 必须先模 q（消除 c2 倍数），再中心化
        s2_cand_c = s2_cand % q
        s2_cand_c[s2_cand_c > q // 2] -= q

        # 验证方程: A·s1' + s2' ≡ t (mod q)
        lhs = vec_add_mod(mat_vec_mul(A, s1_cand_c.reshape(l, n), q),
                          s2_cand_c.reshape(k, n), q)
        eq_holds = np.array_equal(lhs % q, t % q)

        cand_norm = float(np.sqrt(
            np.sum(s1_cand_c.astype(np.int64) ** 2) +
            np.sum(s2_cand_c.astype(np.int64) ** 2)
        ))

        # perfect 判断: s1 必须匹配 (含 ±q 容错)
        s1_diff = np.abs(s1_cand_c.reshape(l, n).astype(np.int64) -
                         s1_real_c.astype(np.int64))
        s1_perfect = np.all((s1_diff == 0) | (s1_diff == q))
        s2_diff = np.abs(s2_cand_c.reshape(k, n).astype(np.int64) -
                         s2_real_c.astype(np.int64))
        s2_perfect = np.all((s2_diff == 0) | (s2_diff == q))
        perfect = s1_perfect and s2_perfect

        if eq_holds or cand_norm < real_norm * 2:
            if perfect:
                logger.critical(f"!!! 找到完美恢复私钥 !!! (行 {row_idx}, 范数={cand_norm:.4f})")
            elif s1_perfect:
                logger.critical(f"!!! s1 完美恢复 (s2 不匹配，可能是 s2'=s2-t0) !!! (行 {row_idx}, 范数={cand_norm:.4f})")
            elif eq_holds and real_norm > 0 and cand_norm / real_norm <= 1.2:
                logger.warning(f"找到替代短向量 (行 {row_idx}, 比值={cand_norm / real_norm:.4f})")

            candidates.append({
                "eq_holds": eq_holds,
                "cand_norm": cand_norm,
                "ratio": cand_norm / real_norm if real_norm > 0 else float("inf"),
                "perfect": perfect,
                "s1_perfect": s1_perfect,
                "s1_prime": s1_cand_c.reshape(l, n),
                "s2_prime": s2_cand_c.reshape(k, n),
            })

    result["candidates"] = candidates
    result["real_norm"] = real_norm
    return result


def classify_results(result: dict) -> list[dict]:
    """Apply the three-layer classification to each candidate."""
    classified = []
    for cand in result["candidates"]:
        if not cand["eq_holds"]:
            verdict = "无效解"
        elif cand.get("perfect", False):
            verdict = "完美恢复私钥"
        elif cand.get("s1_perfect", False):
            verdict = "s1 完美恢复 (s2 不匹配)"
        elif cand["ratio"] <= 1.2:
            verdict = "攻击成功，找到替代短向量"
        else:
            verdict = "满足方程但向量过长"
        classified.append({**cand, "verdict": verdict})
    return classified
