"""
ML-DSA key generation module (FIPS 204 §4.2)。

实现 FIPS 204 规范的：
  - ExpandA (§4.2.2): ρ' = ρ || j || i, SHAKE128 采样
  - CBD (§4.2.3): SHAKE256 生成随机位，popcount 差值
  - KeyGen (§4.2): 完整密钥生成流程

域分离顺序: ρ || j || i (j=列, i=行)
"""

import hashlib
import os

import numpy as np

from .params import get_params


# ── ExpandA (FIPS 204 §4.2.2) ────────────────────────────────────────────────

def rej_ntt_poly(seed: bytes, j: int, i: int, n: int, q: int) -> np.ndarray:
    """RejNTTPoly: SHAKE128 采样多项式系数 (FIPS 204 §4.2.2 Algorithm 6)。

    域分离: ρ' = ρ || bytes(j) || bytes(i)
    每次从 SHAKE128 输出中读 3 字节，取低 18 位，若 < q 则接受。

    Args:
        seed: ρ (32 字节种子)
        j: 列索引 (0 ≤ j < l)
        i: 行索引 (0 ≤ i < k)
        n: 多项式维度
        q: 模数
    """
    # FIPS 204 §4.2.2: ρ' = ρ || j || i
    ctx = seed + bytes([j, i])
    out = hashlib.shake_128(ctx)

    accept_rate = q / (2 ** 18)
    buf_size = max(n * 6, int(n * 3 / max(accept_rate, 0.001)) + 100)
    buf = out.digest(buf_size)

    coeffs = []
    pos = 0
    while len(coeffs) < n and pos + 3 <= len(buf):
        val = int.from_bytes(buf[pos:pos + 3], "little") & 0x3FFFF
        pos += 3
        if val < q:
            coeffs.append(val)

    # Buffer exhausted (极低概率)
    while len(coeffs) < n:
        extra = out.digest(3 * n)
        pos2 = 0
        while len(coeffs) < n and pos2 + 3 <= len(extra):
            val = int.from_bytes(extra[pos2:pos2 + 3], "little") & 0x3FFFF
            pos2 += 3
            if val < q:
                coeffs.append(val)

    return np.array(coeffs, dtype=np.int64)


def expand_a(rho: bytes, k: int, l: int, n: int, q: int) -> np.ndarray:
    """ExpandA: 从种子 ρ 生成公钥矩阵 A ∈ R_q^{k×l} (FIPS 204 Algorithm 6)。

    A[i,j] = RejNTTPoly(ρ, j, i) — 注意顺序是 j,i (列,行)。
    """
    A = np.empty((k, l, n), dtype=np.int64)
    for i in range(k):
        for j in range(l):
            A[i, j] = rej_ntt_poly(rho, j, i, n, q)
    return A


# ── CBD (FIPS 204 §4.2.3) ────────────────────────────────────────────────────

def cbd(seed: bytes, eta: int, n: int, idx: int) -> np.ndarray:
    """SamplePolyCBD: 从 CBD_η 分布采样多项式 (FIPS 204 Algorithm 7)。

    使用 SHAKE256(seed || idx) 生成随机位流。
    对每个系数: 取 2η 位，a = popcount(前 η 位)，b = popcount(后 η 位)，系数 = a - b。

    Args:
        seed: 种子 (通常为 K' 的一部分)
        eta: CBD 参数 (η)
        n: 多项式维度
        idx: 多项式索引 (用于域分离)
    """
    # SHAKE256(seed || uint16_le(idx)) — FIPS 204 Algorithm 7
    ctx = seed + idx.to_bytes(2, "little")
    out = hashlib.shake_256(ctx)

    # 需要 2η 位/系数 → 2η*n 位 → ceil(2η*n/8) 字节
    total_bits = 2 * eta * n
    buf = out.digest((total_bits + 7) // 8)

    coeffs = np.zeros(n, dtype=np.int64)
    bit_pos = 0
    for m in range(n):
        a = 0
        b = 0
        # 前 η 位
        for _ in range(eta):
            byte_idx = bit_pos // 8
            bit_idx = bit_pos % 8
            if byte_idx < len(buf):
                a += (buf[byte_idx] >> bit_idx) & 1
            bit_pos += 1
        # 后 η 位
        for _ in range(eta):
            byte_idx = bit_pos // 8
            bit_idx = bit_pos % 8
            if byte_idx < len(buf):
                b += (buf[byte_idx] >> bit_idx) & 1
            bit_pos += 1
        coeffs[m] = a - b

    return coeffs


# ── KeyGen (FIPS 204 Algorithm 1) ─────────────────────────────────────────────

def keygen(params_name: str = "toy", seed: bytes | None = None,
           params: dict | None = None):
    """ML-DSA 密钥生成 (FIPS 204 Algorithm 1)。

    输入: seed (32 字节，若为 None 则随机生成)
    输出: (rho, s1, s2, t, A)

    流程 (严格按 FIPS 204 Algorithm 6):
      1. ξ = seed (32 字节)
      2. (ρ, ρ', K) = H(ξ || k || l, 128)  — SHAKE256 输出 128 字节
         - ρ  = output[:32]    (ExpandA 种子, 32 字节)
         - ρ' = output[32:96]  (ExpandS 种子, 64 字节)
         - K  = output[96:128] (存入 SK, KeyGen 中不使用)
      3. A = ExpandA(ρ)
      4. s1 = ExpandS(ρ', 0..l-1)  — CBD(η) with ρ'
      5. s2 = ExpandS(ρ', l..l+k-1) — CBD(η) with ρ'
      6. t = A·s1 + s2
    """
    p = params if params is not None else get_params(params_name)
    k, l, n, q, eta = p["k"], p["l"], p["n"], p["q"], p["eta"]

    # 1. 种子
    if seed is None:
        xi = os.urandom(32)
    else:
        xi = seed[:32] if len(seed) >= 32 else seed.ljust(32, b"\x00")

    # 2. FIPS 204 Algorithm 6: (ρ, ρ', K) = H(ξ || k || l, 128)
    #    H = SHAKE256, 域分离: ξ || byte(k) || byte(l)
    #    ρ' = 64 字节 (用于 CBD 采样)
    seed_domain_sep = xi + bytes([k]) + bytes([l])
    h_out = hashlib.shake_256(seed_domain_sep).digest(128)
    rho = h_out[:32]        # 用于 ExpandA (32 bytes)
    rho_prime = h_out[32:96]  # 用于 ExpandS (64 bytes)
    # K = h_out[96:128]       # 存入 SK, KeyGen 中不使用

    # 3. ExpandA
    A = expand_a(rho, k, l, n, q)

    # 4. ExpandS: s1 = CBD(ρ', η, j) for j in 0..l-1
    s1 = np.zeros((l, n), dtype=np.int64)
    for j in range(l):
        s1[j] = cbd(rho_prime, eta, n, j)

    # 5. ExpandS: s2 = CBD(ρ', η, l+i) for i in 0..k-1
    s2 = np.zeros((k, n), dtype=np.int64)
    for i in range(k):
        s2[i] = cbd(rho_prime, eta, n, l + i)

    # 6. t = A·s1 + s2 (多项式乘法 mod x^n+1)
    from .poly_math import mat_vec_mul, vec_add_mod
    t = vec_add_mod(mat_vec_mul(A, s1, q), s2, q)

    return rho, s1, s2, t, A
