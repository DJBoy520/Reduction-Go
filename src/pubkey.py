"""
Public key encoding/decoding module.

Encodes (ρ, t1) into ASN.1 DER format and parses it back.

FIPS 204 §6.2: 公钥 PK = (ρ, t1_enc)，其中 t1 是 Power2Round 高位，
每系数比特数 = ⌈log₂(⌊(q-1)/2ᵈ⌋+1)⌉。
"""

import math
import numpy as np
from asn1crypto.core import Sequence, OctetString, Integer


# ── ASN.1 schema ─────────────────────────────────────────────────────────────

class MLDSAPublicKey(Sequence):
    _fields = [
        ("rho", OctetString),
        ("t1", OctetString),
    ]


# ── Coefficient bit width (FIPS 204 §6.2) ───────────────────────────────────

def t1_coeff_bits(d: int, q: int = 8380417) -> int:
    """FIPS 204 §6.2: t1 每系数比特数 = ⌈log₂(⌊(q-1)/2ᵈ⌋+1)⌉。"""
    max_val = (q - 1) >> d
    return math.ceil(math.log2(max_val + 1))


# ── Coefficient packing ─────────────────────────────────────────────────────

def pack_t1(t1: np.ndarray, d: int, q: int = 8380417) -> bytes:
    """Pack t1 coefficients (k, n) into a bit string.

    FIPS 204 §6.2: 每系数使用 ⌈log₂(⌊(q-1)/2ᵈ⌋+1)⌉ 位。
    """
    k, n = t1.shape
    n_coeff_bits = t1_coeff_bits(d, q)
    total_bits = k * n * n_coeff_bits
    bit_str = 0
    idx = 0
    for i in range(k):
        for j in range(n):
            coeff = int(t1[i, j]) & ((1 << n_coeff_bits) - 1)
            bit_str |= coeff << (total_bits - (idx + 1) * n_coeff_bits)
            idx += 1
    byte_len = (total_bits + 7) // 8
    return bit_str.to_bytes(byte_len, "big")


def unpack_t1(data: bytes, k: int, n: int, d: int, q: int = 8380417,
              n_coeff_bits: int = None) -> np.ndarray:
    """Unpack a bit string back into the t1 matrix (k, n)。"""
    if n_coeff_bits is None:
        n_coeff_bits = t1_coeff_bits(d, q)
    bit_str = int.from_bytes(data, "big")
    total_bits = k * n * n_coeff_bits
    t1 = np.zeros((k, n), dtype=np.int64)
    idx = 0
    for i in range(k):
        for j in range(n):
            shift = total_bits - (idx + 1) * n_coeff_bits
            t1[i, j] = (bit_str >> shift) & ((1 << n_coeff_bits) - 1)
            idx += 1
    return t1


# ── ML-DSA 参数映射 ─────────────────────────────────────────────────────────

_MLDSA_D = {"ML-DSA-44": 13, "ML-DSA-65": 13, "ML-DSA-87": 13}


# ── Encode / Decode ──────────────────────────────────────────────────────────

def encode_public_key(rho: bytes, t1: np.ndarray, d: int, q: int = 8380417) -> bytes:
    """Encode public key (ρ, t1) into ASN.1 DER bytes (FIPS 204 §6.2)。"""
    t1_bytes = pack_t1(t1, d, q)
    pk = MLDSAPublicKey()
    pk["rho"] = rho
    pk["t1"] = t1_bytes
    return pk.dump()


def decode_public_key(der_data: bytes, k: int, n: int, d: int, q: int = 8380417):
    """Decode ASN.1 DER public key, return (rho, t1_matrix)。

    Does NOT rebuild A here — caller must invoke ExpandA separately.
    """
    pk = MLDSAPublicKey.load(der_data)
    rho = pk["rho"].native
    t1 = unpack_t1(pk["t1"].native, k, n, d, q)
    return rho, t1


def save_public_key(path: str, rho: bytes, t1: np.ndarray, d: int, q: int = 8380417):
    """Save public key (ρ, t1) to a DER file。"""
    der = encode_public_key(rho, t1, d, q)
    with open(path, "wb") as f:
        f.write(der)
    return len(der)


def load_public_key(path: str, k: int, n: int, d: int, q: int = 8380417):
    """Load public key from a DER file, return (rho, t1)。"""
    with open(path, "rb") as f:
        der = f.read()
    return decode_public_key(der, k, n, d, q)
