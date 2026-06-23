"""
NTT/INTT — Number Theoretic Transform for ML-DSA (FIPS 204 §4.3)。

NTT 算法移植自 Giacomo Pope 的 dilithium-py 项目：
  https://github.com/GiacomoPope/dilithium-py
  License: MIT OR Apache-2.0

negacyclic NTT (mod x^n + 1)，用于多项式乘法。

数学基础：
  - q = 8380417 = 2^23 - 2^13 + 1
  - n = 256 (标准多项式维度)
  - ζ = 1753 — primitive 2·n=512-th root of unity mod q (dilithium-py 使用)
  - ζ^256 = -1 mod q → negacyclic 性质
  - ntt_zetas[k] = ζ^{br(k)}，其中 br 是 8-bit bit-reversal
"""

import numpy as np

# ── 域参数 (来自 dilithium-py) ────────────────────────────────────────────────

Q = 8380417  # 2^23 - 2^13 + 1
N_REF = 256  # 参考维度 (FIPS 204 标准)

# dilithium-py 使用的 primitive 512-th root of unity
_ROOT_OF_UNITY = 1753

# n^{-1} mod q (用于 INTT 缩放)
_N_INV = pow(N_REF, -1, Q)

# 预计算 bit-reversed zeta 表 (来自 dilithium-py PolynomialRing.__init__)
def _br(i: int, k: int) -> int:
    """k-bit bit reversal."""
    bin_i = bin(i & (2**k - 1))[2:].zfill(k)
    return int(bin_i[::-1], 2)


_NTT_ZETAS = [pow(_ROOT_OF_UNITY, _br(i, 8), Q) for i in range(256)]


# ── Forward NTT (来自 dilithium-py Polynomial.to_ntt) ────────────────────────

def ntt(f: np.ndarray, n: int = N_REF) -> np.ndarray:
    """Forward negacyclic NTT。

    移植自 dilithium-py Polynomial.to_ntt()。
    输入系数表示，输出 NTT 域表示（bit-reversed 顺序）。

    Args:
        f: 系数表示，shape (n,)，值域 [0, q)
        n: 多项式维度 (默认 256)

    Returns:
        NTT 域表示 f_hat，shape (n,)
    """
    assert n > 0 and n & (n - 1) == 0, f"n must be a power of 2, got {n}"
    assert n <= N_REF, f"n must be <= {N_REF}, got {n}"
    if n != N_REF:
        # 对于非标准维度，使用直接 NTT（dilithium-py 只支持 n=256）
        return _direct_ntt(f, n)

    coeffs = f.astype(np.int64) % Q
    k, l = 0, 128
    zetas = _NTT_ZETAS

    while l > 0:
        start = 0
        while start < 256:
            k = k + 1
            zeta = int(zetas[k])
            for j in range(start, start + l):
                t = zeta * int(coeffs[j + l]) % Q
                coeffs[j + l] = (int(coeffs[j]) - t) % Q
                coeffs[j] = (int(coeffs[j]) + t) % Q
            start = l + start + l
        l >>= 1

    return np.array(coeffs, dtype=np.int64)


# ── Inverse NTT (来自 dilithium-py PolynomialNTT.from_ntt) ───────────────────

def intt(f_hat: np.ndarray, n: int = N_REF) -> np.ndarray:
    """Inverse negacyclic NTT。

    移植自 dilithium-py PolynomialNTT.from_ntt()。
    输入 NTT 域表示（bit-reversed 顺序），输出系数表示。

    Args:
        f_hat: NTT 域表示，shape (n,)
        n: 多项式维度 (默认 256)

    Returns:
        系数表示 f，shape (n,)，值域 [0, q)
    """
    assert n > 0 and n & (n - 1) == 0, f"n must be a power of 2, got {n}"
    assert n <= N_REF, f"n must be <= {N_REF}, got {n}"
    if n != N_REF:
        return _direct_intt(f_hat, n)

    coeffs = f_hat.astype(np.int64) % Q
    l, k_idx = 1, 256
    zetas = _NTT_ZETAS

    while l < 256:
        start = 0
        while start < 256:
            k_idx = k_idx - 1
            zeta = -int(zetas[k_idx])
            for j in range(start, start + l):
                t = int(coeffs[j])
                coeffs[j] = (t + int(coeffs[j + l])) % Q
                coeffs[j + l] = zeta * (t - int(coeffs[j + l])) % Q
            start = start + l + l
        l = l << 1

    for j in range(256):
        coeffs[j] = int(coeffs[j]) * _N_INV % Q

    return np.array(coeffs, dtype=np.int64)


# ── Negacyclic 卷积 ──────────────────────────────────────────────────────────

def ntt_mul(a: np.ndarray, b: np.ndarray, q: int = Q, n: int = N_REF) -> np.ndarray:
    """NTT 域多项式乘法 (negacyclic, mod x^n + 1)。"""
    a_hat = ntt(a, n)
    b_hat = ntt(b, n)
    c_hat = a_hat * b_hat % q
    return intt(c_hat, n)


def ntt_mat_vec_mul(A: np.ndarray, s: np.ndarray, q: int = Q,
                    n: int = N_REF) -> np.ndarray:
    """NTT 域矩阵-向量乘法。"""
    k, l_dim, _ = A.shape
    t = np.zeros((k, n), dtype=np.int64)
    s_hat = np.array([ntt(s[j], n) for j in range(l_dim)], dtype=np.int64)
    for i in range(k):
        acc_hat = np.zeros(n, dtype=np.int64)
        for j in range(l_dim):
            a_hat = ntt(A[i, j], n)
            acc_hat = (acc_hat + a_hat * s_hat[j]) % q
        t[i] = intt(acc_hat, n)
    return t


# ── 直接 NTT (用于非标准维度 n ≠ 256 的验证) ─────────────────────────────────

def _omega_2n(n: int) -> int:
    """返回 primitive 2n-th root of unity mod q。"""
    return pow(_ROOT_OF_UNITY, N_REF // n, Q)


def _direct_ntt(f: np.ndarray, n: int) -> np.ndarray:
    """直接 NTT (O(n²)，用于非标准维度)。"""
    w2n = _omega_2n(n)
    f_hat = np.zeros(n, dtype=np.int64)
    for k in range(n):
        val = 0
        for j in range(n):
            val = (val + int(f[j]) * pow(w2n, (2 * k + 1) * j, Q)) % Q
        f_hat[k] = val
    return f_hat


def _direct_intt(f_hat: np.ndarray, n: int) -> np.ndarray:
    """直接 INTT (O(n²)，用于非标准维度)。"""
    w2n = _omega_2n(n)
    w2n_inv = pow(w2n, Q - 2, Q)
    n_inv = pow(n, Q - 2, Q)
    f = np.zeros(n, dtype=np.int64)
    for j in range(n):
        val = 0
        for k in range(n):
            val = (val + int(f_hat[k]) * pow(w2n_inv, (2 * k + 1) * j, Q)) % Q
        f[j] = val * n_inv % Q
    return f


# ── 验证工具 ──────────────────────────────────────────────────────────────────

def verify_ntt_roundtrip(n: int = N_REF) -> bool:
    """验证 NTT → INTT 恒等变换。"""
    rng = np.random.default_rng(42)
    f = rng.integers(0, Q, size=n, dtype=np.int64)
    f_hat = ntt(f, n)
    f_back = intt(f_hat, n)
    return bool(np.array_equal(f, f_back))


def verify_vs_direct(n: int = N_REF) -> bool:
    """验证快速 NTT 与直接 NTT 一致（考虑 bit-reversal）。"""
    rng = np.random.default_rng(123)
    f = rng.integers(0, Q, size=n, dtype=np.int64)
    fast = ntt(f, n)
    direct = _direct_ntt(f, n)
    # dilithium-py NTT 输出是 bit-reversed 顺序
    if n == N_REF:
        return all(fast[i] == direct[_br(i, 8)] for i in range(n))
    return bool(np.array_equal(fast, direct))


def verify_negacyclic(n: int = 32) -> bool:
    """验证 NTT 乘法与 school-book negacyclic 卷积一致。"""
    rng = np.random.default_rng(123)
    a = rng.integers(0, Q, size=n, dtype=np.int64)
    b = rng.integers(0, Q, size=n, dtype=np.int64)

    c_ntt = ntt_mul(a, b, Q, n)

    c_ref = np.zeros(n, dtype=np.int64)
    for i in range(n):
        for j in range(n):
            idx = i + j
            if idx < n:
                c_ref[idx] = (c_ref[idx] + int(a[i]) * int(b[j])) % Q
            else:
                c_ref[idx - n] = (c_ref[idx - n] - int(a[i]) * int(b[j])) % Q
    c_ref = c_ref % Q

    return bool(np.array_equal(c_ntt, c_ref))


if __name__ == "__main__":
    zeta = _ROOT_OF_UNITY
    print(f"q = {Q}")
    print(f"ζ = {zeta}")
    print(f"ζ^256 = {pow(zeta, 256, Q)} (should be {Q-1})")
    print(f"ζ^512 = {pow(zeta, 512, Q)} (should be 1)")
    print(f"n_inv = {_N_INV}")
    print()

    # 验证标准维度 n=256
    print("=== n=256 (标准 ML-DSA) ===")
    rt = verify_ntt_roundtrip(256)
    vd = verify_vs_direct(256)
    print(f"  roundtrip={rt}, vs_direct={vd}  {'✓' if (rt and vd) else '✗'}")

    # 验证非标准维度（使用直接 NTT）
    print("\n=== 非标准维度 (直接 NTT) ===")
    for test_n in [4, 8, 16, 32, 64, 128]:
        rt = verify_ntt_roundtrip(test_n)
        vd = verify_vs_direct(test_n)
        status = "✓" if (rt and vd) else "✗"
        print(f"  n={test_n:>3}: roundtrip={rt}, vs_direct={vd}  {status}")

    print()
    print(f"  Negacyclic (n=32): {verify_negacyclic(32)}")
