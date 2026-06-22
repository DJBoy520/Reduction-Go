"""
ML-DSA key generation module.

Implements ExpandA (FIPS 204 §4.2.2), CBD (§4.2.3), and key generation.
"""

import hashlib
import os
import struct

import numpy as np

from .params import get_params
from .poly_math import mat_vec_mul, vec_add_mod


# ── ExpandA ──────────────────────────────────────────────────────────────────

def rej_ntt_poly(seed: bytes, k_idx: int, l_idx: int, n: int, q: int) -> np.ndarray:
    """RejNTTPoly: sample a polynomial via rejection sampling (FIPS 204 §4.2.2).

    Each candidate coefficient is the low 18 bits of 3 bytes; accept if < q.
    Uses buffered hashing for efficiency.
    """
    # Domain separation: ρ' ‖ bytes([j, i]) — j first per FIPS 204 §4.2.2
    ctx = seed + bytes([l_idx, k_idx])
    out = hashlib.shake_128(ctx)
    # Pre-generate enough bytes: expect ~n/accept_rate candidates
    accept_rate = q / (2 ** 18)
    buf_size = max(n * 6, int(n * 3 / max(accept_rate, 0.001)) + 100)
    buf = out.digest(buf_size)
    coeffs = []
    pos = 0
    while len(coeffs) < n and pos + 3 <= len(buf):
        val = int.from_bytes(buf[pos:pos + 3], "little") & 0x3FFFF  # low 18 bits
        pos += 3
        if val < q:
            coeffs.append(val)
    # Extremely unlikely: buffer exhausted. Refill if needed.
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
    """ExpandA: generate the public matrix A ∈ R_q^{k×l} from seed ρ.

    Returns A as np.ndarray of shape (k, l, n).
    """
    A = np.empty((k, l, n), dtype=np.int64)
    for i in range(k):
        for j in range(l):
            A[i, j] = rej_ntt_poly(rho, i, j, n, q)
    return A


# ── CBD (Centered Binomial Distribution) ─────────────────────────────────────

def cbd(eta: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample a polynomial from the centered binomial distribution CBD_η.

    FIPS 204 §4.2.3:
      For each coefficient, draw 2η random bits.
      Let a = number of 1-bits in first η bits.
      Let b = number of 1-bits in second η bits.
      Coefficient = a − b, range [−η, η].
    """
    # Generate 2η random bits per coefficient → shape (n, 2*eta)
    bits = rng.integers(0, 2, size=(n, 2 * eta), dtype=np.int8)
    a = bits[:, :eta].sum(axis=1)
    b = bits[:, eta:].sum(axis=1)
    return (a - b).astype(np.int64)


# ── Key generation ───────────────────────────────────────────────────────────

def keygen(params_name: str = "toy", seed: bytes | None = None,
           params: dict | None = None):
    """Generate an ML-DSA keypair.

    If params is provided, it overrides the named parameter set.
    Returns (rho, s1, s2, t, A).
    """
    p = params if params is not None else get_params(params_name)
    k, l, n, q, eta = p["k"], p["l"], p["n"], p["q"], p["eta"]

    rng = np.random.default_rng()

    # 1. Random seed
    if seed is None:
        rho = os.urandom(32)
    else:
        rho = seed

    # 2. ExpandA
    A = expand_a(rho, k, l, n, q)

    # 3. Sample secret vectors via CBD
    s1 = np.zeros((l, n), dtype=np.int64)
    s2 = np.zeros((k, n), dtype=np.int64)
    for j in range(l):
        s1[j] = cbd(eta, n, rng)
    for i in range(k):
        s2[i] = cbd(eta, n, rng)

    # 4. t = A·s1 + s2  (polynomial multiplication mod x^n+1)
    t = vec_add_mod(mat_vec_mul(A, s1, q), s2, q)

    return rho, s1, s2, t, A
