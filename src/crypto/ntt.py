"""
NTT/INTT — Number Theoretic Transform for ML-DSA (FIPS 204 §4.3)。

实现 negacyclic NTT (mod x^n + 1)，用于多项式乘法。

数学基础：
  - q = 8380417 = 2^23 - 2^13 + 1
  - N_REF = 256 (参考多项式维度)
  - ω = 1921994 — primitive 2·N_REF = 512-th root of unity mod q
  - ω^{256} = -1 mod q → negacyclic 性质

对于任意维度 n (power of 2, n ≤ N_REF):
  - ω_{2n} = ω^{N_REF/n} — 适配维度的 primitive 2n-th root
  - premult:  g[j] = f[j] · ω_{2n}^j
  - butterfly: 用 ω_{2n}^2 (primitive n-th root)

快速 NTT (Cooley-Tukey DIT, O(n log n)):
  1. 预乘: g[j] = f[j] · ω_{2n}^j
  2. Bit-reverse 置换
  3. Cooley-Tukey 蝶形 with ω_{2n}^2
  4. 结果: f_hat[k] = g_hat[k] = f(ω_{2n}^{2k+1})

快速 INTT (Gentle-Sanders DIF + bit-reverse, O(n log n)):
  1. Gentle-Sanders 蝶形 with ω_{2n}^{-2} (输入正常序，输出 bit-reversed)
  2. Bit-reverse 置换
  3. 逆预乘: f[j] = g[j] · ω_{2n}^{-j}
  4. 乘以 n^{-1}

验证: python3 src/crypto/ntt.py
"""

import numpy as np

# ── 域参数 ────────────────────────────────────────────────────────────────────

Q = 8380417  # 2^23 - 2^13 + 1

# ω = primitive 2·N_REF-th root of unity (order 512)
# ω = 10^((q-1)/512) mod q, where 10 is a primitive root of q
OMEGA = 1921994
OMEGA_INV = pow(OMEGA, Q - 2, Q)

# 参考维度 N_REF=256 时的 butterfly root (order-N_REF)
# ω^2 = primitive N_REF=256-th root of unity
OMEGA_N_REF = (OMEGA * OMEGA) % Q

N_REF = 256  # 参考多项式维度（默认 n）

# ── ω 缩放缓存 ────────────────────────────────────────────────────────────────

def _omega_2n(n: int) -> int:
    """返回适配维度 n 的 primitive 2n-th root of unity。

    ω_{2n} = ω^{N_REF/n} mod q，其中 ω 是 primitive 2·N_REF-th root。
    对于 n=256: ω_{2n} = ω  (order 512)
    对于 n=128: ω_{2n} = ω^2  (order 256)
    """
    assert n > 0 and (n & (n - 1)) == 0, f"n={n} must be a power of 2"
    assert n <= N_REF, f"n={n} must be ≤ {N_REF}"
    return pow(OMEGA, N_REF // n, Q)


def _omega_2n_inv(n: int) -> int:
    """返回适配维度 n 的 primitive 2n-th root 的逆。"""
    return pow(_omega_2n(n), Q - 2, Q)


def _omega_n(n: int) -> int:
    """返回适配维度 n 的 primitive n-th root of unity (= ω_{2n}^2)。"""
    w = _omega_2n(n)
    return (w * w) % Q


def _omega_n_inv(n: int) -> int:
    """返回适配维度 n 的 primitive n-th root 的逆 (= ω_{2n}^{-2})。"""
    return pow(_omega_n(n), Q - 2, Q)


# ── Pre-computed powers (per-n 缓存) ──────────────────────────────────────────

_POWERS_CACHE: dict = {}           # (n, kind) → np.ndarray


def _get_omega_2n_powers(n: int) -> np.ndarray:
    """ω_{2n}^0, ω_{2n}^1, ..., ω_{2n}^{n-1} (per-n 缓存)。"""
    key = (n, "omega")
    if key not in _POWERS_CACHE:
        w = _omega_2n(n)
        _POWERS_CACHE[key] = np.array([pow(w, i, Q) for i in range(n)],
                                       dtype=np.int64)
    return _POWERS_CACHE[key]


def _get_omega_2n_inv_powers(n: int) -> np.ndarray:
    """ω_{2n}^{-0}, ω_{2n}^{-1}, ..., ω_{2n}^{-(n-1)} (per-n 缓存)。"""
    key = (n, "omega_inv")
    if key not in _POWERS_CACHE:
        w_inv = _omega_2n_inv(n)
        _POWERS_CACHE[key] = np.array([pow(w_inv, i, Q) for i in range(n)],
                                       dtype=np.int64)
    return _POWERS_CACHE[key]


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


# ── Forward NTT ───────────────────────────────────────────────────────────────

def ntt(f: np.ndarray, n: int = N_REF) -> np.ndarray:
    """Forward negacyclic NTT (Cooley-Tukey DIT, O(n log n))。

    f_hat[k] = Σ_j f[j] · ω_{2n}^{(2k+1)j}  (negacyclic evaluation)

    实现 (DIT Cooley-Tukey):
      1. 预乘: g[j] = f[j] · ω_{2n}^j
      2. Bit-reverse 置换
      3. DIT 蝶形 with ω_{2n}^2

    Args:
        f: 系数表示，shape (n,)，值域 [0, q)
        n: 多项式维度 (默认 256, must be power of 2 and ≤ 256)

    Returns:
        NTT 域表示 f_hat，shape (n,)
    """
    omega_2n_powers = _get_omega_2n_powers(n)
    omega_n = _omega_n(n)

    # 1. 预乘: g[j] = f[j] * ω_{2n}^j mod q
    g = (f.astype(np.int64) % Q) * omega_2n_powers % Q

    # 2. Bit-reverse permutation (DIT 要求输入 bit-reversed)
    g = _bit_reverse_array(g)

    # 3. Cooley-Tukey DIT butterfly with ω_{2n}^2
    length = 2
    while length <= n:
        half = length // 2
        step = n // length
        for start in range(0, n, length):
            for j in range(half):
                w = pow(omega_n, (step * j) % n, Q)

                u = int(g[start + j])
                v = int(g[start + j + half]) * w % Q

                g[start + j] = (u + v) % Q
                g[start + j + half] = (u - v) % Q

        length *= 2

    return g


# ── Inverse NTT ───────────────────────────────────────────────────────────────

def intt(f_hat: np.ndarray, n: int = N_REF) -> np.ndarray:
    """Inverse negacyclic NTT (Gentle-Sanders DIF, O(n log n))。

    f[j] = (1/n) · Σ_k f_hat[k] · ω_{2n}^{-(2k+1)j}

    实现 (Gentle-Sanders DIF + bit-reverse):
      1. Gentle-Sanders DIF 蝶形 with ω_{2n}^{-2} (输入正常序 → 输出 bit-reversed)
      2. Bit-reverse 置换
      3. 逆预乘: f[j] = g[j] · ω_{2n}^{-j}
      4. 乘以 n^{-1}

    Args:
        f_hat: NTT 域表示，shape (n,)
        n: 多项式维度 (默认 256)

    Returns:
        系数表示 f，shape (n,)，值域 [0, q)
    """
    omega_2n_inv_powers = _get_omega_2n_inv_powers(n)
    omega_n_inv = _omega_n_inv(n)

    # 1. Gentle-Sanders DIF butterfly with ω_{2n}^{-2}
    #    DIF: 输入正常序 → 输出 bit-reversed 序
    g = f_hat.astype(np.int64) % Q
    length = n
    while length >= 2:
        half = length // 2
        step = n // length
        for start in range(0, n, length):
            for j in range(half):
                w = pow(omega_n_inv, (step * j) % n, Q)

                u = int(g[start + j])
                v = int(g[start + j + half])

                g[start + j] = (u + v) % Q
                g[start + j + half] = (u - v) * w % Q

        length //= 2

    # 2. Bit-reverse 置换 (DIF 输出在 bit-reversed 序)
    g = _bit_reverse_array(g)

    # 3. 逆预乘 ω_{2n}^{-j} 和乘以 n^{-1}
    n_inv = pow(n, Q - 2, Q)
    f = g * omega_2n_inv_powers % Q * n_inv % Q

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
    """直接 NTT (O(n²)，用于验证)。

    使用适配维度 n 的 ω_{2n} = ω^{N_REF/n}。
    """
    omega_2n = _omega_2n(n)
    f_hat = np.zeros(n, dtype=np.int64)
    for k in range(n):
        val = 0
        for j in range(n):
            val = (val + int(f[j]) * pow(omega_2n, (2 * k + 1) * j, Q)) % Q
        f_hat[k] = val
    return f_hat


def _direct_intt(f_hat: np.ndarray, n: int) -> np.ndarray:
    """直接 INTT (O(n²)，用于验证)。

    使用适配维度 n 的 ω_{2n} = ω^{N_REF/n}。
    """
    omega_2n_inv = _omega_2n_inv(n)
    n_inv = pow(n, Q - 2, Q)
    f = np.zeros(n, dtype=np.int64)
    for j in range(n):
        val = 0
        for k in range(n):
            val = (val + int(f_hat[k]) * pow(omega_2n_inv, (2 * k + 1) * j, Q)) % Q
        f[j] = val * n_inv % Q
    return f


def verify_ntt_roundtrip(n: int = N_REF) -> bool:
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


def verify_negacyclic(n: int = 32) -> bool:
    """验证 NTT 乘法与 school-book negacyclic 卷积一致。"""
    rng = np.random.default_rng(123)
    a = rng.integers(0, Q, size=n, dtype=np.int64)
    b = rng.integers(0, Q, size=n, dtype=np.int64)

    c_ntt = ntt_mul(a, b, Q, n)

    # Full schoolbook negacyclic: c(x) = a(x)·b(x) mod (x^n + 1)
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
    print(f"q = {Q}")
    print(f"ω = {OMEGA} (primitive 512-th root)")
    print(f"ω^256 mod q = {pow(OMEGA, 256, Q)} (should be q-1 = {Q - 1})")
    print(f"ω^512 mod q = {pow(OMEGA, 512, Q)} (should be 1)")
    print()

    print(f"--- n = {N_REF} (default) ---")
    print(f"NTT roundtrip: {verify_ntt_roundtrip(N_REF)}")
    print(f"Fast vs direct: {verify_vs_direct(N_REF)}")
    print(f"Negacyclic conv: {verify_negacyclic(32)}")
    print()

    print("--- Various n (power of 2) ---")
    for t_n in [4, 8, 16, 32, 64, 128, 256]:
        rt = verify_ntt_roundtrip(t_n)
        fd = verify_vs_direct(t_n)
        print(f"  n={t_n:3d}: roundtrip={rt}, vs_direct={fd}")
