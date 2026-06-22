"""
NTT/INTT — Number Theoretic Transform for ML-DSA (FIPS 204 §4.3)。

实现 negacyclic NTT (mod x^n + 1)，用于多项式乘法。

数学基础：
  - q = 8380417 = 2^23 - 2^13 + 1
  - N_REF = 256 (标准多项式维度)
  - ω = 1921994 — primitive 2·N_REF=512-th root of unity mod q
  - ω^256 = -1 mod q → negacyclic 性质
  - 对于任意 n，ω_{2n} = ω^{N_REF/n} 是 primitive 2n-th root of unity

快速 NTT (Cooley-Tukey, O(n log n)):
  1. 预乘: g[j] = f[j] · ω_{2n}^j
  2. 标准 NTT with ω_{2n}^2: g_hat = NTT_{ω_{2n}^2}(g)
  3. 结果: f_hat[k] = g_hat[k] = f(ω_{2n}^{2k+1})

验证: python3 src/crypto/ntt.py
"""

import numpy as np

# ── 域参数 ────────────────────────────────────────────────────────────────────

Q = 8380417  # 2^23 - 2^13 + 1
N_REF = 256  # 参考维度

# ω = primitive 2·N_REF=512-th root of unity
# ω = 10^((q-1)/512) mod q, where 10 is a primitive root of q
OMEGA = 1921994

# ── 维度自适应根 ─────────────────────────────────────────────────────────────

# 缓存: {(n, kind): np.array}
_POWERS_CACHE = {}


def _omega_2n(n: int) -> int:
    """返回 primitive 2n-th root of unity mod q。

    ω_{2n} = ω^{N_REF/n}，其中 ω 是 primitive 2·N_REF-th root。
    """
    return pow(OMEGA, N_REF // n, Q)


def _get_powers(n: int, kind: str = "premul") -> np.ndarray:
    """获取预计算的幂次表（带缓存）。

    kind:
      - "premul": ω_{2n}^0, ω_{2n}^1, ..., ω_{2n}^{n-1} (预乘用)
      - "premul_inv": ω_{2n}^{-0}, ..., ω_{2n}^{-(n-1)} (逆预乘用)
      - "twiddle": (ω_{2n}^2)^0, ..., (ω_{2n}^2)^{n-1} (蝶形 twiddle)
      - "twiddle_inv": (ω_{2n}^{-2})^0, ..., (ω_{2n}^{-2})^{n-1}
    """
    cache_key = (n, kind)
    if cache_key in _POWERS_CACHE:
        return _POWERS_CACHE[cache_key]

    w2n = _omega_2n(n)
    w2n_sq = (w2n * w2n) % Q
    w2n_inv = pow(w2n, Q - 2, Q)
    w2n_sq_inv = pow(w2n_sq, Q - 2, Q)

    if kind == "premul":
        powers = np.array([pow(w2n, i, Q) for i in range(n)], dtype=np.int64)
    elif kind == "premul_inv":
        powers = np.array([pow(w2n_inv, i, Q) for i in range(n)], dtype=np.int64)
    elif kind == "twiddle":
        powers = np.array([pow(w2n_sq, i, Q) for i in range(n)], dtype=np.int64)
    elif kind == "twiddle_inv":
        powers = np.array([pow(w2n_sq_inv, i, Q) for i in range(n)], dtype=np.int64)
    else:
        raise ValueError(f"Unknown kind: {kind}")

    _POWERS_CACHE[cache_key] = powers
    return powers


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


# ── Forward NTT (Cooley-Tukey DIT) ───────────────────────────────────────────

def ntt(f: np.ndarray, n: int = N_REF) -> np.ndarray:
    """Forward negacyclic NTT (Cooley-Tukey DIT, O(n log n))。

    f_hat[k] = Σ_j f[j] · ω_{2n}^{(2k+1)j}

    实现:
      1. 预乘 g[j] = f[j] · ω_{2n}^j
      2. Bit-reverse g
      3. Cooley-Tukey butterfly with ω_{2n}^2

    Args:
        f: 系数表示，shape (n,)，值域 [0, q)
        n: 多项式维度 (默认 256)

    Returns:
        NTT 域表示 f_hat，shape (n,)
    """
    premul = _get_powers(n, "premul")
    twiddle = _get_powers(n, "twiddle")

    # 1. 预乘
    g = (f.astype(np.int64) % Q) * premul % Q

    # 2. Bit-reverse
    g = _bit_reverse_array(g)

    # 3. Cooley-Tukey butterfly
    length = 2
    while length <= n:
        half = length // 2
        step = n // length
        for start in range(0, n, length):
            for j in range(half):
                w = int(twiddle[(step * j) % n])

                u = int(g[start + j])
                v = int(g[start + j + half]) * w % Q

                g[start + j] = (u + v) % Q
                g[start + j + half] = (u - v) % Q

        length *= 2

    return g


# ── Inverse NTT (Gentle-Sanders DIF) ─────────────────────────────────────────

def intt(f_hat: np.ndarray, n: int = N_REF) -> np.ndarray:
    """Inverse negacyclic NTT (Gentle-Sanders DIF, O(n log n))。

    f[j] = (1/n) · Σ_k f_hat[k] · ω_{2n}^{-(2k+1)j}

    实现 (DIF: 蝶形 → bit-reverse → 逆预乘 → 缩放):
      1. Gentle-Sanders butterfly with ω_{2n}^{-2}
      2. Bit-reverse (DIF 输出是 bit-reversed 的)
      3. 逆预乘: f[j] = g[j] · ω_{2n}^{-j}
      4. 乘以 n^{-1}

    Args:
        f_hat: NTT 域表示，shape (n,)
        n: 多项式维度 (默认 256)

    Returns:
        系数表示 f，shape (n,)，值域 [0, q)
    """
    twiddle_inv = _get_powers(n, "twiddle_inv")
    premul_inv = _get_powers(n, "premul_inv")

    # 1. Gentle-Sanders butterfly (DIF: large → small)
    g = f_hat.astype(np.int64) % Q

    length = n
    while length >= 2:
        half = length // 2
        step = n // length
        for start in range(0, n, length):
            for j in range(half):
                w = int(twiddle_inv[(step * j) % n])

                u = int(g[start + j])
                v = int(g[start + j + half])

                g[start + j] = (u + v) % Q
                g[start + j + half] = (u - v) * w % Q

        length //= 2

    # 2. Bit-reverse (DIF 输出是 bit-reversed)
    g = _bit_reverse_array(g)

    # 3. 逆预乘 + 缩放
    n_inv = pow(n, Q - 2, Q)
    f = g * premul_inv % Q * n_inv % Q

    return f


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


# ── 验证工具 ──────────────────────────────────────────────────────────────────

def _direct_ntt(f: np.ndarray, n: int) -> np.ndarray:
    """直接 NTT (O(n²)，用于验证)。"""
    w2n = _omega_2n(n)
    f_hat = np.zeros(n, dtype=np.int64)
    for k in range(n):
        val = 0
        for j in range(n):
            val = (val + int(f[j]) * pow(w2n, (2 * k + 1) * j, Q)) % Q
        f_hat[k] = val
    return f_hat


def _direct_intt(f_hat: np.ndarray, n: int) -> np.ndarray:
    """直接 INTT (O(n²)，用于验证)。"""
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


def verify_ntt_roundtrip(n: int = N_REF) -> bool:
    """验证 NTT → INTT 恒等变换。"""
    rng = np.random.default_rng(42)
    f = rng.integers(0, Q, size=n, dtype=np.int64)
    f_hat = ntt(f, n)
    f_back = intt(f_hat, n)
    return bool(np.array_equal(f, f_back))


def verify_vs_direct(n: int = N_REF) -> bool:
    """验证快速 NTT 与直接 NTT 一致。"""
    rng = np.random.default_rng(123)
    f = rng.integers(0, Q, size=n, dtype=np.int64)
    fast = ntt(f, n)
    direct = _direct_ntt(f, n)
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
    w2n = _omega_2n(N_REF)
    print(f"q = {Q}")
    print(f"ω = {OMEGA}")
    print(f"ω_{2*N_REF} = {w2n}")
    print(f"ω^{N_REF} = {pow(w2n, N_REF, Q)} (should be {Q-1})")
    print(f"ω^{2*N_REF} = {pow(w2n, 2*N_REF, Q)} (should be 1)")
    print()

    for test_n in [4, 8, 16, 32, 64, 128, 256]:
        rt = verify_ntt_roundtrip(test_n)
        vd = verify_vs_direct(test_n)
        status = "✓" if (rt and vd) else "✗"
        print(f"  n={test_n:>3}: roundtrip={rt}, vs_direct={vd}  {status}")

    print()
    print(f"  Negacyclic (n=32): {verify_negacyclic(32)}")
