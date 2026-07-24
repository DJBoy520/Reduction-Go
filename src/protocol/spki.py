"""
ML-DSA ASN.1 结构定义 — X.509 / SPKI 标准格式。

FIPS 204 §6.2 (Public Key Encoding)
====================================
公钥 PK = (rho, t1)，编码为:
    PK_encoded = rho || t1_enc
    - rho: 32 字节种子
    - t1_enc: t1 系数按 10-bit 打包 (所有 ML-DSA 变体, d=13)
      ⌈log₂(⌊(q-1)/2¹³⌋+1)⌉ = ⌈log₂(1024)⌉ = 10 bits/系数

X.509 SubjectPublicKeyInfo 结构:
    SubjectPublicKeyInfo ::= SEQUENCE {
        algorithm  AlgorithmIdentifier,
        subjectPublicKey  BIT STRING
    }

    AlgorithmIdentifier ::= SEQUENCE {
        algorithm   OBJECT IDENTIFIER,
        parameters  NULL
    }
"""

import math
import numpy as np
from asn1crypto import pem as asn1pem
from asn1crypto.core import Sequence, ObjectIdentifier, Null, BitString

# OID 和 d 值统一从 params.py 引用
from ..utils.params import MLDSA_OIDS, OID_TO_MLDSA, MLDSA_REGISTRY
# DER 构建工具从 der_utils.py 引用
from .der_utils import (
    build_der_sequence as _build_der_sequence,
    build_der_bitstring as _build_der_bitstring,
    build_der_oid as _build_der_oid,
    build_der_null as _build_der_null,
)


# ── t1 打包/解包 ────────────────────────────────────────────────────────────

def t1_coeff_bits(d: int, q: int = 8380417) -> int:
    """FIPS 204 §6.2: t1 每系数比特数 = ⌈log₂(⌊(q-1)/2ᵈ⌋+1)⌉。"""
    max_val = (q - 1) >> d
    return math.ceil(math.log2(max_val + 1))


def pack_t1(t1: np.ndarray, d: int, q: int = 8380417) -> bytes:
    """将 t1 系数打包为比特串 (FIPS 204 §6.2)。"""
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
    """从比特串解包 t1 系数。

    n_coeff_bits: 直接指定每系数位宽（优先于 d 的推算）。
    """
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


# ── SPKI ASN.1 类定义 ───────────────────────────────────────────────────────

class _AlgorithmIdentifier(Sequence):
    _fields = [
        ("algorithm", ObjectIdentifier),
        ("parameters", Null),
    ]


class _SubjectPublicKeyInfo(Sequence):
    _fields = [
        ("algorithm", _AlgorithmIdentifier),
        ("subjectPublicKey", BitString),
    ]


# ── d 值从 MLDSA_REGISTRY 获取 ──────────────────────────────────────────────

def _get_d_for(name: str) -> int:
    """从注册表获取 d 值。"""
    return MLDSA_REGISTRY.get(name, {}).get("d", 13)


# ── 编码/解码 ────────────────────────────────────────────────────────────────

def encode_spki(rho: bytes, t1: np.ndarray, mldsa_name: str = "ML-DSA-65", d: int = None) -> bytes:
    """将公钥 (rho, t1) 编码为 X.509 SubjectPublicKeyInfo DER。

    FIPS 204 §6.2: t1 使用 d 相关的 bit width 打包。
    """
    if len(rho) != 32:
        raise ValueError(f"rho 必须 32 字节，实际 {len(rho)} 字节")
    oid = MLDSA_OIDS[mldsa_name]
    if d is None:
        d = _get_d_for(mldsa_name)
    t1_bytes = pack_t1(t1, d=d)
    pk_encoded = rho + t1_bytes

    alg_id = _build_der_sequence(_build_der_oid(oid), _build_der_null())
    bitstring = _build_der_bitstring(pk_encoded)
    spki_der = _build_der_sequence(alg_id, bitstring)
    return spki_der


def decode_spki(der_data: bytes, k: int, n: int, d: int = None) -> tuple:
    """从 SubjectPublicKeyInfo DER 解码公钥。

    d 默认从 OID 推断 (所有变体 d=13)。
    Returns: (rho, t1, mldsa_name)
    """
    spki = _SubjectPublicKeyInfo.load(der_data)
    oid = spki["algorithm"]["algorithm"].dotted
    mldsa_name = OID_TO_MLDSA.get(oid, f"unknown-{oid}")

    if d is None:
        d = _get_d_for(mldsa_name)

    bs = spki["subjectPublicKey"]
    if bs.unused_bits:
        raise ValueError("BIT STRING unused_bits 非零，数据损坏")
    bs_contents = bs.contents
    pk_encoded = bs_contents[1:]  # 跳过 unused_bits 前缀

    n_coeff_bits = t1_coeff_bits(d)
    expected_len = 32 + (k * n * n_coeff_bits + 7) // 8
    if len(pk_encoded) < expected_len:
        raise ValueError(
            f"BIT STRING 数据不足: 需要 {expected_len} 字节，"
            f"实际 {len(pk_encoded)} 字节"
        )

    rho = pk_encoded[:32]
    t1_byte_len = (k * n * n_coeff_bits + 7) // 8
    t1_bytes = pk_encoded[32:32 + t1_byte_len]
    t1 = unpack_t1(t1_bytes, k, n, d=d)
    return rho, t1, mldsa_name


def save_spki_pem(path: str, rho: bytes, t1: np.ndarray, mldsa_name: str = "ML-DSA-65", d: int = None):
    """将公钥保存为 PEM 格式。"""
    der = encode_spki(rho, t1, mldsa_name, d=d)
    pem_data = asn1pem.armor("PUBLIC KEY", der)
    with open(path, "wb") as f:
        f.write(pem_data)
    return len(der)


def load_spki_pem(path: str, k: int, n: int, d: int = None) -> tuple:
    """从 PEM 文件加载公钥。Returns: (rho, t1, mldsa_name)"""
    with open(path, "rb") as f:
        pem_data = f.read()
    _, _, der = asn1pem.unarmor(pem_data)
    return decode_spki(der, k, n, d=d)


def save_spki_der(path: str, rho: bytes, t1: np.ndarray, mldsa_name: str = "ML-DSA-65", d: int = None):
    """将公钥保存为 DER 格式。"""
    der = encode_spki(rho, t1, mldsa_name, d=d)
    with open(path, "wb") as f:
        f.write(der)
    return len(der)


def load_spki_der(path: str, k: int, n: int, d: int = None) -> tuple:
    """从 DER 文件加载公钥。"""
    with open(path, "rb") as f:
        der = f.read()
    return decode_spki(der, k, n, d=d)
