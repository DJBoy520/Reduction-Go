"""
Polynomial arithmetic in Z_q[x] / (x^n + 1)。

策略:
  - n ≤ 64: school-book O(n²) 卷积 (足够快)
  - n = 256 (标准 ML-DSA): 使用 NTT (O(n log n))
"""

import numpy as np


# ── School-book negacyclic (O(n²)) ───────────────────────────────────────────

def _poly_mul_schoolbook(a: np.ndarray, b: np.ndarray, q: int) -> np.ndarray:
    """School-book negacyclic 卷积 mod x^n + 1。"""
    n = len(a)
    c = np.zeros(n, dtype=np.int64)
    for i in range(n):
        for j in range(n):
            idx = i + j
            if idx < n:
                c[idx] = (c[idx] + int(a[i]) * int(b[j])) % q
            else:
                c[idx - n] = (c[idx - n] - int(a[i]) * int(b[j])) % q
    return c % q


# ── NTT-based negacyclic (O(n log n)) ────────────────────────────────────────

def _poly_mul_ntt(a: np.ndarray, b: np.ndarray, q: int) -> np.ndarray:
    """NTT 域 negacyclic 卷积 mod x^n + 1。"""
    from .crypto.ntt import ntt_mul
    return ntt_mul(a, b, q, len(a))


# ── 自动选择 ──────────────────────────────────────────────────────────────────

_NTT_THRESHOLD = 128  # n > 此值时用 NTT


def poly_mul_mod(a: np.ndarray, b: np.ndarray, q: int) -> np.ndarray:
    """Multiply two polynomials in Z_q[x]/(x^n + 1)。

    自动选择算法:
      - n ≤ 128: school-book O(n²)
      - n > 128: NTT O(n log n)
    """
    n = len(a)
    assert len(b) == n
    if n > _NTT_THRESHOLD:
        return _poly_mul_ntt(a, b, q)
    return _poly_mul_schoolbook(a, b, q)


# ── Matrix-vector multiplication ─────────────────────────────────────────────

def mat_vec_mul(A: np.ndarray, s: np.ndarray, q: int) -> np.ndarray:
    """Multiply matrix of polynomials A (k, l, n) by vector s (l, n)。

    Result: t (k, n) where t[i] = sum_j A[i,j] * s[j] mod (x^n+1, q).
    """
    k, l, n = A.shape
    assert s.shape == (l, n)
    t = np.zeros((k, n), dtype=np.int64)
    for i in range(k):
        for j in range(l):
            t[i] = (t[i] + poly_mul_mod(A[i, j], s[j], q)) % q
    return t


# ── Element-wise ──────────────────────────────────────────────────────────────

def vec_add_mod(a: np.ndarray, b: np.ndarray, q: int) -> np.ndarray:
    """Element-wise addition mod q."""
    return (a + b) % q
