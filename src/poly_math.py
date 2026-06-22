"""
Polynomial arithmetic in Z_q[x] / (x^n + 1).

Implements negacyclic convolution (school-book O(n^2)) for polynomial
multiplication mod x^n + 1.  This is the simplified "toy" version;
can be replaced with NTT-based multiplication later.
"""

import numpy as np


# TODO: 当 n 较大时，用 NTT 替换 O(n^2) 乘法。
def poly_mul_mod(a: np.ndarray, b: np.ndarray, q: int) -> np.ndarray:
    """Multiply two polynomials in Z_q[x]/(x^n + 1).

    a, b: integer arrays of length n, coefficients mod q.
    Returns: integer array of length n, coefficients in [0, q).
    """
    n = len(a)
    assert len(b) == n
    c = np.zeros(n, dtype=np.int64)
    for i in range(n):
        for j in range(n):
            idx = i + j
            if idx < n:
                c[idx] = (c[idx] + int(a[i]) * int(b[j])) % q
            else:
                # x^n ≡ -1, so coefficient wraps and negates
                c[idx - n] = (c[idx - n] - int(a[i]) * int(b[j])) % q
    return c % q


# TODO: 当 n 较大时，用 NTT 替换 O(n^2) 乘法。
def mat_vec_mul(A: np.ndarray, s: np.ndarray, q: int) -> np.ndarray:
    """Multiply matrix of polynomials A (k, l, n) by vector of polynomials s (l, n).

    Result: t (k, n) where t[i] = sum_j A[i,j] * s[j] mod (x^n+1, q).
    """
    k, l, n = A.shape
    assert s.shape == (l, n)
    t = np.zeros((k, n), dtype=np.int64)
    for i in range(k):
        for j in range(l):
            t[i] = (t[i] + poly_mul_mod(A[i, j], s[j], q)) % q
    return t


def vec_add_mod(a: np.ndarray, b: np.ndarray, q: int) -> np.ndarray:
    """Element-wise addition mod q."""
    return (a + b) % q
