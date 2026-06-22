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
    # SHAKE256(seed || bytes(idx))
    ctx = seed + bytes([idx])
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


# ── KeyGen (FIPS 204 §4.2) ───────────────────────────────────────────────────

def keygen(params_name: str = "toy", seed: bytes | None = None,
           params: dict | None = None):
    """ML-DSA 密钥生成 (FIPS 204 Algorithm 1)。

    输入: seed (32 字节，若为 None 则随机生成)
    输出: (rho, s1, s2, t, A)

    流程:
      1. ξ = seed (32 字节)
      2. ρ = H(ξ, 0, 64)[:32]   (ExpandA 种子)
      3. ρ' = H(ξ, 1, 64)[:32]  (CBD 种子的前半)
      4. K = H(ξ, 2, 64)[:32]   (CBD 种子的后半)
      5. A = ExpandA(ρ)
      6. s1, s2 = CBD(ρ' || K)
      7. t = A·s1 + s2

    注意: 当前实现简化了步骤 2-4，直接用 seed 作为 ρ，
    独立随机采样 s1/s2。这不影响格攻击框架的正确性，
    但与 NIST KAT 不一致（KAT 验证需要完整流程）。
    """
    p = params if params is not None else get_params(params_name)
    k, l, n, q, eta = p["k"], p["l"], p["n"], p["q"], p["eta"]

    # 1. 种子
    if seed is None:
        rho = os.urandom(32)
    else:
        rho = seed[:32] if len(seed) >= 32 else seed.ljust(32, b"\x00")

    # 2. ExpandA
    A = expand_a(rho, k, l, n, q)

    # 3. CBD 采样 (使用 SHAKE256)
    # 用 ρ 作为 CBD 种子的简化版本
    # 完整 FIPS 204 需要 ρ' 和 K，此处用 ρ 的不同偏移
    cbd_seed = hashlib.shake_256(rho + b"CBD").digest(64)

    s1 = np.zeros((l, n), dtype=np.int64)
    s2 = np.zeros((k, n), dtype=np.int64)
    for j in range(l):
        s1[j] = cbd(cbd_seed, eta, n, j)
    for i in range(k):
        s2[i] = cbd(cbd_seed, eta, n, l + i)

    # 4. t = A·s1 + s2 (多项式乘法 mod x^n+1)
    from .poly_math import mat_vec_mul, vec_add_mod
    t = vec_add_mod(mat_vec_mul(A, s1, q), s2, q)

    return rho, s1, s2, t, A
