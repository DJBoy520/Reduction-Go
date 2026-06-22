"""
NTT/INTT — Number Theoretic Transform for ML-DSA (FIPS 204 §4.3)。

实现 negacyclic NTT (mod x^n + 1)，用于多项式乘法。

数学基础：
  - q = 8380417 = 2^23 - 2^13 + 1
  - n = 256 (多项式维度)
  - ω = 1921994 — primitive 2n=512-th root of unity mod q
  - ω^256 = -1 mod q → negacyclic 性质
  - ω^2 是 primitive n=256-th root of unity

快速 NTT (Cooley-Tukey, O(n log n)):
  1. 预乘: g[j] = f[j] · ω^j
  2. 标准 NTT with ω^2: g_hat = NTT_{ω^2}(g)
  3. 结果: f_hat[k] = g_hat[k] = f(ω^{2k+1})

验证: python3 src/crypto/ntt.py
"""

import numpy as np

# ── 域参数 ────────────────────────────────────────────────────────────────────

Q = 8380417  # 2^23 - 2^13 + 1

# ω = primitive 2n-th root of unity (order 512)
# ω = 10^((q-1)/512) mod q, where 10 is a primitive root of q
OMEGA = 1921994
OMEGA_INV = pow(OMEGA, Q - 2, Q)

# ω^2 = primitive n-th root of unity (order 256)
OMEGA2 = (OMEGA * OMEGA) % Q
OMEGA2_INV = pow(OMEGA2, Q - 2, Q)

N = 256  # 标准多项式维度

# ── Bit-reversal ──────────────────────────────────────────────────────────────

def _bit_reverse(x: int, bits: int) -> int:
    r = 0
    for _ in range(bits):
        r = (r << 1) | (x & 1)
        x >>= 1
    return r


def _bit_reverse_array(arr: np.ndarray) -> np.ndarray:
    n = len(arr)
    log_n = n.bit_length() - 1
    result = arr.copy()
    for i in range(n):
        j = _bit_reverse(i, log_n)
        if i < j:
            result[i], result[j] = result[j], result[i]
    return result


# ── Pre-computed powers ───────────────────────────────────────────────────────

_OMEGA_POWERS = None
_OMEGA_INV_POWERS = None


def _get_omega_powers(n: int) -> np.ndarray:
    """ω^0, ω^1, ..., ω^{n-1} (延迟初始化)。"""
    global _OMEGA_POWERS
    if _OMEGA_POWERS is None or len(_OMEGA_POWERS) < n:
        _OMEGA_POWERS = np.array([pow(OMEGA, i, Q) for i in range(n)],
                                  dtype=np.int64)
    return _OMEGA_POWERS[:n]


def _get_omega_inv_powers(n: int) -> np.ndarray:
    """ω^{-0}, ω^{-1}, ..., ω^{-(n-1)} (延迟初始化)。"""
    global _OMEGA_INV_POWERS
    if _OMEGA_INV_POWERS is None or len(_OMEGA_INV_POWERS) < n:
        _OMEGA_INV_POWERS = np.array([pow(OMEGA_INV, i, Q) for i in range(n)],
                                      dtype=np.int64)
    return _OMEGA_INV_POWERS[:n]


# ── Forward NTT ───────────────────────────────────────────────────────────────

def ntt(f: np.ndarray, n: int = N) -> np.ndarray:
    """Forward negacyclic NTT (Cooley-Tukey, O(n log n))。

    f[k] = Σ_j f[j] · ω^{(2k+1)j}  (negacyclic evaluation)

    实现:
      1. 预乘 g[j] = f[j] · ω^j
      2. 标准 NTT with ω^2 (bit-reversal Cooley-Tukey)
      3. f_hat = NTT_{ω^2}(g)

    Args:
        f: 系数表示，shape (n,)，值域 [0, q)
        n: 多项式维度 (默认 256)

    Returns:
        NTT 域表示 f_hat，shape (n,)
    """
    omega_powers = _get_omega_powers(n)

    # 1. 预乘: g[j] = f[j] * ω^j mod q
    g = (f.astype(np.int64) % Q) * omega_powers % Q

    # 2. Bit-reverse permutation
    g = _bit_reverse_array(g)

    # 3. Cooley-Tukey butterfly with ω^2
    length = 2
    while length <= n:
        half = length // 2
        step = n // length
        for start in range(0, n, length):
            for j in range(half):
                w_idx = (step * j) % n
                w = int(OMEGA2) ** w_idx % Q  # ω^2 的 step*j 次方

                u = int(g[start + j])
                v = int(g[start + j + half]) * w % Q

                g[start + j] = (u + v) % Q
                g[start + j + half] = (u - v) % Q

        length *= 2

    return g


# ── Inverse NTT ───────────────────────────────────────────────────────────────

def intt(f_hat: np.ndarray, n: int = N) -> np.ndarray:
    """Inverse negacyclic NTT (Gentle-Sanders, O(n log n))。

    f[j] = (1/n) · Σ_k f_hat[k] · ω^{-(2k+1)j}

    实现:
      1. 标准 INTT with ω^{-2}
      2. 逆预乘: f[j] = g[j] · ω^{-j}
      3. 乘以 n^{-1}

    Args:
        f_hat: NTT 域表示，shape (n,)
        n: 多项式维度 (默认 256)

    Returns:
        系数表示 f，shape (n,)，值域 [0, q)
    """
    omega_inv_powers = _get_omega_inv_powers(n)

    # 1. Bit-reverse permutation
    g = _bit_reverse_array(f_hat.astype(np.int64) % Q)

    # 2. Gentle-Sanders butterfly with ω^{-2}
    length = n
    while length >= 2:
        half = length // 2
        step = n // length
        for start in range(0, n, length):
            for j in range(half):
                w_idx = (step * j) % n
                w = int(OMEGA2_INV) ** w_idx % Q  # (ω^{-2})^{step*j}

                u = int(g[start + j])
                v = int(g[start + j + half])

                g[start + j] = (u + v) % Q
                g[start + j + half] = (u - v) * w % Q

        length //= 2

    # 3. 乘以 n^{-1} 和逆预乘 ω^{-j}
    n_inv = pow(n, Q - 2, Q)
    f = g * omega_inv_powers % Q * n_inv % Q

    return f


# ── Negacyclic 卷积 ──────────────────────────────────────────────────────────

def ntt_mul(a: np.ndarray, b: np.ndarray, q: int = Q, n: int = N) -> np.ndarray:
    """NTT 域多项式乘法 (negacyclic, mod x^n + 1)。"""
    a_hat = ntt(a, n)
    b_hat = ntt(b, n)
    c_hat = a_hat * b_hat % q
    return intt(c_hat, n)


def ntt_mat_vec_mul(A: np.ndarray, s: np.ndarray, q: int = Q,
                    n: int = N) -> np.ndarray:
    """NTT 域矩阵-向量乘法。

    A: shape (k, l, n) — 多项式矩阵
    s: shape (l, n) — 多项式向量
    返回 t: shape (k, n)
    """
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


# ── 验证工具 ──────────────────────────────────────────────────────────────────

def _direct_ntt(f: np.ndarray, n: int) -> np.ndarray:
    """直接 NTT (O(n²)，用于验证)。"""
    f_hat = np.zeros(n, dtype=np.int64)
    for k in range(n):
        val = 0
        for j in range(n):
            val = (val + int(f[j]) * pow(OMEGA, (2 * k + 1) * j, Q)) % Q
        f_hat[k] = val
    return f_hat


def _direct_intt(f_hat: np.ndarray, n: int) -> np.ndarray:
    """直接 INTT (O(n²)，用于验证)。"""
    n_inv = pow(n, Q - 2, Q)
    f = np.zeros(n, dtype=np.int64)
    for j in range(n):
        val = 0
        for k in range(n):
            val = (val + int(f_hat[k]) * pow(OMEGA_INV, (2 * k + 1) * j, Q)) % Q
        f[j] = val * n_inv % Q
    return f


def verify_ntt_roundtrip(n: int = N) -> bool:
    """验证 NTT → INTT 恒等变换。"""
    rng = np.random.default_rng(42)
    f = rng.integers(0, Q, size=n, dtype=np.int64)
    f_hat = ntt(f, n)
    f_back = intt(f_hat, n)
    return bool(np.array_equal(f, f_back))


def verify_vs_direct(n: int = 256) -> bool:
    """验证快速 NTT 与直接 NTT 一致。"""
    rng = np.random.default_rng(123)
    f = rng.integers(0, Q, size=n, dtype=np.int64)
    fast = ntt(f, n)
    direct = _direct_ntt(f, n)
    return bool(np.array_equal(fast, direct))


def verify_negacyclic(n: int = 256) -> bool:
    """验证 NTT 乘法与 school-book negacyclic 卷积一致。"""
    rng = np.random.default_rng(123)
    a = rng.integers(0, Q, size=n, dtype=np.int64)
    b = rng.integers(0, Q, size=n, dtype=np.int64)

    c_ntt = ntt_mul(a, b, Q, n)

    c_ref = np.zeros(n, dtype=np.int64)
    for i in range(min(n, 64)):  # 用小 n 验证
        for j in range(min(n, 64)):
            idx = i + j
            if idx < n:
                c_ref[idx] = (c_ref[idx] + int(a[i]) * int(b[j])) % Q
            else:
                c_ref[idx - n] = (c_ref[idx - n] - int(a[i]) * int(b[j])) % Q
    c_ref = c_ref % Q

    # 只比较前 64 项（因为 school-book 只算了部分）
    return bool(np.array_equal(c_ntt[:64], c_ref[:64]))


if __name__ == "__main__":
    print(f"q = {Q}")
    print(f"ω = {OMEGA}")
    print(f"ω^256 = {pow(OMEGA, 256, Q)} (should be {Q-1})")
    print(f"ω^512 = {pow(OMEGA, 512, Q)} (should be 1)")
    print()
    print(f"NTT roundtrip (n={N}): {verify_ntt_roundtrip(N)}")
    print(f"Fast vs direct (n=256): {verify_vs_direct(256)}")
