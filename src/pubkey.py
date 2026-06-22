"""
Public key encoding/decoding module.

Encodes (ρ, t) into ASN.1 DER format and parses it back.
"""

import numpy as np
from asn1crypto.core import Sequence, OctetString, Integer


# ── ASN.1 schema ─────────────────────────────────────────────────────────────

class MLDSAPublicKey(Sequence):
    _fields = [
        ("rho", OctetString),
        ("t", OctetString),
    ]


# ── Coefficient packing ─────────────────────────────────────────────────────

def pack_t(t: np.ndarray, n_coeff_bits: int = 23) -> bytes:
    """Pack the t vector (k, n) into a bit string, each coefficient using n_coeff_bits bits.

    Returns bytes (big-endian bit string).
    """
    k, n = t.shape
    total_bits = k * n * n_coeff_bits
    bit_str = 0
    idx = 0
    for i in range(k):
        for j in range(n):
            coeff = int(t[i, j]) & ((1 << n_coeff_bits) - 1)
            bit_str |= coeff << (total_bits - (idx + 1) * n_coeff_bits)
            idx += 1
    byte_len = (total_bits + 7) // 8
    return bit_str.to_bytes(byte_len, "big")


def unpack_t(data: bytes, k: int, n: int, n_coeff_bits: int = 23) -> np.ndarray:
    """Unpack a bit string back into the t matrix (k, n)."""
    bit_str = int.from_bytes(data, "big")
    total_bits = k * n * n_coeff_bits
    t = np.zeros((k, n), dtype=np.int64)
    idx = 0
    for i in range(k):
        for j in range(n):
            shift = total_bits - (idx + 1) * n_coeff_bits
            t[i, j] = (bit_str >> shift) & ((1 << n_coeff_bits) - 1)
            idx += 1
    return t


# ── Encode / Decode ──────────────────────────────────────────────────────────

def encode_public_key(rho: bytes, t: np.ndarray) -> bytes:
    """Encode public key (ρ, t) into ASN.1 DER bytes."""
    t_bytes = pack_t(t)
    pk = MLDSAPublicKey()
    pk["rho"] = rho
    pk["t"] = t_bytes
    return pk.dump()


def decode_public_key(der_data: bytes, k: int, n: int):
    """Decode ASN.1 DER public key, return (rho, t_matrix).

    Does NOT rebuild A here — caller must invoke ExpandA separately.
    """
    pk = MLDSAPublicKey.load(der_data)
    rho = pk["rho"].native
    t = unpack_t(pk["t"].native, k, n)
    return rho, t


def save_public_key(path: str, rho: bytes, t: np.ndarray):
    """Save public key to a DER file."""
    der = encode_public_key(rho, t)
    with open(path, "wb") as f:
        f.write(der)
    return len(der)


def load_public_key(path: str, k: int, n: int):
    """Load public key from a DER file, return (rho, t)."""
    with open(path, "rb") as f:
        der = f.read()
    return decode_public_key(der, k, n)
