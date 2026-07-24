"""攻击结果分类器

将已约减的格基 B_red 中的候选向量分类为：
  完美恢复 / s1完美 / 替代短向量 / 过长 / 无效

设计原则:
  - 纯分析模块：接收已约减格基，不做任何格构造或约减操作
  - 不依赖 lattice 模块、basis_builder 或任何约减算法
  - 分类阈值 ratio <= 1.2 判为替代短向量（经验值，适用于 ML-DSA 场景）

依赖关系:
  classifier → numpy（无格约减依赖，杜绝误用）
  无循环依赖。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger("lattice_attack")


def classify_results(
    B_red: np.ndarray,
    A: np.ndarray,
    t: np.ndarray,
    q: int,
) -> List[Dict[str, Any]]:
    """对已约减的格基进行候选提取与分类。

    不再重复构造格基或执行约减——调用方须保证 B_red 已约减。
    遍历 B_red 的每一行，提取候选 s1、s2，验证方程 A·s1 + s2 ≡ t (mod q)，
    根据范数比值分类。

    Parameters
    ----------
    B_red : (rows, cols) 已约减格基矩阵，每行为一个候选向量
    A     : (k*n, n) 公钥矩阵
    t     : (k*n, 1) 或 (k*n,) 目标向量
    q     : 模数

    Returns
    -------
    results : list[dict]
        每个候选结果包含:
          - index: 在 B_red 中的行号
          - s1, s2: 拆分的候选密钥分量
          - eq_holds: 方程是否满足
          - s1_norm, t_norm, ratio: 范数与比值
          - verdict: "完美恢复私钥" | "s1 完美恢复 (s2 不匹配)"
                     | "攻击成功，找到替代短向量" | "满足方程但向量过长" | "无效解"
    """
    # --- 边界保护：空矩阵不崩溃 ---
    if B_red.size == 0 or B_red.shape[0] == 0:
        return []

    n = A.shape[1] if A.size > 0 else 0
    t_flat = t.flatten().astype(int)
    t_len = len(t_flat)

    results: List[Dict[str, Any]] = []

    for i in range(B_red.shape[0]):
        row = B_red[i]
        total = len(row)

        # 从行向量中提取 s1 和 s2：
        #   s1 长度 = t_len (= k*n)，位于行向量的后 t_len 列
        #   s2 位于行向量的前 total - t_len 列（即 k*n*(k-1) 维误差部分）
        s2_len = total - t_len if total > t_len else 0
        s1 = row[s2_len : s2_len + t_len].astype(int) % q
        s2 = row[:s2_len].astype(int) if s2_len > 0 else np.zeros(0, dtype=int)

        # 方程验证: A·s1 + s2 ≡ t (mod q)
        s1_col = s1.reshape(-1, 1) if s1.ndim == 1 else s1
        a_s1 = (A @ s1_col).flatten().astype(int) % q
        if s2_len > 0:
            combined = (a_s1 + s2) % q
        else:
            combined = a_s1
        eq_holds = bool(np.allclose(combined, t_flat))

        # 范数分析
        s1_norm = float(np.linalg.norm(s1))
        t_norm = float(np.linalg.norm(t_flat))
        ratio = s1_norm / t_norm if t_norm > 0 else float("inf")

        # 三层分类（与 lattice_attack.py 一致的分类标准）
        if eq_holds and ratio < 1.05:
            verdict = "完美恢复私钥"
        elif eq_holds and abs(s1_norm - t_norm) < 0.01 * t_norm:
            verdict = "s1 完美恢复 (s2 不匹配)"
        elif eq_holds and ratio <= 1.2:
            # 阈值 1.2: 经验值，替代短向量通常在 1.0~1.2 范围内
            verdict = "攻击成功，找到替代短向量"
        elif eq_holds:
            verdict = "满足方程但向量过长"
        else:
            verdict = "无效解"

        results.append({
            "index": i,
            "s1": s1_col.flatten(),
            "s2": s2,
            "eq_holds": eq_holds,
            "s1_norm": s1_norm,
            "t_norm": t_norm,
            "ratio": ratio,
            "verdict": verdict,
            "q": q,
        })

    # 按范数比值排序，最佳候选在前
    results.sort(key=lambda x: x["ratio"])
    return results
