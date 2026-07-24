"""
Public key encoding/decoding module (内部测试格式)。

使用自定义 ASN.1 结构 (rho, t1) 编解码公钥，用于测试闭环。
与 spki.py 的 X.509 SPKI 格式不同，但共享 t1 打包/解包逻辑。

FIPS 204 §6.2: 公钥 PK = (ρ, t1_enc)，其中 t1 是 Power2Round 高位，
每系数比特数 = ⌈log₂(⌊(q-1)/2ᵈ⌋+1)⌉。
"""

import numpy as np
from asn1crypto.core import Sequence, OctetString

# t1 打包/解包逻辑从 spki.py 引用（消除重复）
from .spki import pack_t1, unpack_t1


# ── ASN.1 schema (内部测试格式，非 X.509) ───────────────────────────────────

class MLDSAPublicKey(Sequence):
    _fields = [
        ("rho", OctetString),
        ("t1", OctetString),
    ]


# ── Encode / Decode ──────────────────────────────────────────────────────────

def encode_public_key(rho: bytes, t1: np.ndarray, d: int, q: int = 8380417) -> bytes:
    """Encode public key (ρ, t1) into ASN.1 DER bytes (内部测试格式)。"""
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
