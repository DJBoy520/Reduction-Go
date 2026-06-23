"""
X.509 证书解析器 — 从标准 ML-DSA 证书中提取公钥。

支持输入格式:
  - PEM 编码的 X.509 证书 (.pem, .crt)
  - DER 编码的 X.509 证书 (.der)
  - PEM 编码的 SubjectPublicKeyInfo (.pem, _pub.pem)
  - DER 编码的 SubjectPublicKeyInfo (.der)

解析流程:
  1. 检测输入是完整 X.509 证书还是裸 SPKI
  2. 若是证书: 解析 TBSCertificate → SubjectPublicKeyInfo
  3. 从 SPKI 中提取 OID、rho、t1
  4. 根据 OID 自动确定参数集 (k, l, n, d)
"""

import logging
import os

import numpy as np
from asn1crypto import pem as asn1pem
from asn1crypto.core import Sequence, ObjectIdentifier, Null, BitString

from .spki import (
    MLDSA_OIDS, OID_TO_MLDSA,
    unpack_t1, t1_coeff_bits,
)

logger = logging.getLogger(__name__)

# ── OID → 参数集映射 ────────────────────────────────────────────────────────

OID_PARAMS = {
    "2.16.840.1.101.3.4.3.17": {"name": "ML-DSA-44", "k": 4, "l": 4, "n": 256, "d": 10},
    "2.16.840.1.101.3.4.3.18": {"name": "ML-DSA-65", "k": 6, "l": 6, "n": 256, "d": 13},
    "2.16.840.1.101.3.4.3.19": {"name": "ML-DSA-87", "k": 8, "l": 8, "n": 256, "d": 13},
}

# toy 参数集的 OID 映射 (用于测试)
TOY_OID_PARAMS = {
    "2.16.840.1.101.3.4.3.17": {"name": "ML-DSA-44", "k": 2, "l": 2, "n": 30, "d": 10},
    "2.16.840.1.101.3.4.3.18": {"name": "ML-DSA-65", "k": 2, "l": 2, "n": 30, "d": 10},
    "2.16.840.1.101.3.4.3.19": {"name": "ML-DSA-87", "k": 2, "l": 2, "n": 30, "d": 10},
}


# ── X.509 ASN.1 类定义 ──────────────────────────────────────────────────────

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


# ── 核心解析函数 ────────────────────────────────────────────────────────────

def _detect_format(filepath: str) -> tuple[str, str]:
    """检测文件格式: (编码类型, 内容类型)。

    Returns:
        (encoding, content_type):
          encoding: "pem" 或 "der"
          content_type: "certificate" 或 "spki"
    """
    with open(filepath, "rb") as f:
        data = f.read()

    if b"-----BEGIN" in data:
        # PEM 格式
        if b"-----BEGIN CERTIFICATE-----" in data:
            return "pem", "certificate"
        elif b"-----BEGIN PUBLIC KEY-----" in data:
            return "pem", "spki"
        else:
            # 尝试当作 PEM 解码
            try:
                label, _, _ = asn1pem.unarmor(data)
                return "pem", label.lower().replace(" ", "_")
            except Exception:
                raise ValueError(f"无法识别 PEM 格式: {filepath}")
    else:
        # DER 格式 — 验证性解析，确认结构正确
        try:
            spki = _SubjectPublicKeyInfo.load(data)
            # 验证确实是 SPKI: algorithm 字段应包含有效 OID
            spki["algorithm"]["algorithm"].dotted
            spki["subjectPublicKey"]
            return "der", "spki"
        except Exception:
            pass
        try:
            cert = Sequence.load(data)
            # 验证是证书: 至少 3 个子元素 (TBS, sigAlg, sigValue)
            if len(cert) == 3 and isinstance(cert[2], BitString):
                return "der", "certificate"
        except Exception:
            pass
        raise ValueError(f"无法识别 DER 格式: {filepath}")


def _extract_spki_from_cert(cert_der: bytes) -> bytes:
    """从 X.509 证书 DER 中提取 SubjectPublicKeyInfo DER。

    X.509 结构:
        Certificate ::= SEQUENCE {
            tbsCertificate       TBSCertificate,
            signatureAlgorithm   AlgorithmIdentifier,
            signatureValue       BIT STRING
        }

        TBSCertificate ::= SEQUENCE {
            version [0] EXPLICIT INTEGER DEFAULT v1,
            serialNumber         CertificateSerialNumber,
            signature            AlgorithmIdentifier,
            issuer               Name,
            validity             Validity,
            subject              Name,
            subjectPublicKeyInfo SubjectPublicKeyInfo,
            ...
        }
    """
    cert = Sequence.load(cert_der)
    # cert[0] = TBSCertificate (SEQUENCE)
    tbs = cert[0]
    if not isinstance(tbs, Sequence):
        raise ValueError("无法解析 TBSCertificate")

    # TBSCertificate 的第 7 个元素 (index 6, 0-indexed) 是 SubjectPublicKeyInfo
    # 但 version 字段可能是 ContextSpecific [0]，导致偏移
    # 安全方法: 遍历查找包含 algorithm 字段的 SEQUENCE
    for child in tbs:
        if isinstance(child, Sequence) and len(child) >= 2:
            first_elem = child[0]
            if isinstance(first_elem, Sequence):
                # 可能是 AlgorithmIdentifier (在 SPKI 中)
                try:
                    oid = first_elem[0]
                    if isinstance(oid, ObjectIdentifier) and oid.dotted in OID_TO_MLDSA:
                        return child.dump()
                except Exception:
                    continue

    # 回退: 假设第 7 个元素是 SPKI (标准 X.509 编排)
    try:
        spki_candidate = tbs[6]
        if isinstance(spki_candidate, Sequence):
            return spki_candidate.dump()
    except (IndexError, Exception):
        pass

    raise ValueError("无法从证书中提取 SubjectPublicKeyInfo")


def _decode_spki_with_params(spki_der: bytes, use_toy: bool = False) -> tuple:
    """从 SPKI DER 解码公钥，返回 (rho, t1, params)。

    Args:
        spki_der: SPKI DER 字节
        use_toy: 是否使用 toy 参数 (k=l=2, n=30)

    Returns:
        (rho, t1, params_dict):
          rho: 32 字节种子
          t1: (k, n) 高位系数
          params_dict: {"name", "k", "l", "n", "d", "oid"}
    """
    spki = _SubjectPublicKeyInfo.load(spki_der)
    oid = spki["algorithm"]["algorithm"].dotted

    # 查找参数集
    params_table = TOY_OID_PARAMS if use_toy else OID_PARAMS
    if oid not in params_table:
        raise ValueError(f"未知 ML-DSA OID: {oid}")

    params = params_table[oid].copy()
    params["oid"] = oid
    k, n = params["k"], params["n"]
    d = params.get("d", 10)

    # 提取公钥字节
    bs = spki["subjectPublicKey"]
    if bs.unused_bits:
        raise ValueError("BIT STRING unused_bits 非零")
    bs_contents = bs.contents
    pk_encoded = bs_contents[1:]  # 跳过 unused_bits 前缀

    n_coeff_bits = t1_coeff_bits(d, params.get("q", 8380417))
    expected_len = 32 + (k * n * n_coeff_bits + 7) // 8
    if len(pk_encoded) != expected_len:
        # liboqs 统一使用 10 bits/coeff 编码 t1，与 FIPS 204 不同
        # 尝试用 10 bits 解码
        n_coeff_bits_alt = 10
        alt_len = 32 + (k * n * n_coeff_bits_alt + 7) // 8
        if len(pk_encoded) == alt_len:
            n_coeff_bits = n_coeff_bits_alt
        else:
            # 尝试从数据长度反推 bits/coeff
            t1_data_len = len(pk_encoded) - 32
            if t1_data_len > 0 and (t1_data_len * 8) % (k * n) == 0:
                n_coeff_bits = (t1_data_len * 8) // (k * n)
            else:
                raise ValueError(
                    f"公钥数据长度不匹配: 期望 {expected_len} 字节, "
                    f"实际 {len(pk_encoded)} 字节"
                )

    rho = pk_encoded[:32]
    t1_byte_len = (k * n * n_coeff_bits + 7) // 8
    t1_bytes = pk_encoded[32:32 + t1_byte_len]
    t1 = unpack_t1(t1_bytes, k, n, d=d, q=params.get("q", 8380417),
                    n_coeff_bits=n_coeff_bits)

    params["q"] = 8380417  # ML-DSA 标准 q
    return rho, t1, params


def parse_certificate(filepath: str, use_toy: bool = False) -> tuple:
    """从 X.509 证书或 SPKI 文件中提取 ML-DSA 公钥。

    支持:
      - 完整 X.509 证书 (PEM/DER)
      - 裸 SubjectPublicKeyInfo (PEM/DER)

    Args:
        filepath: 证书文件路径
        use_toy: 使用 toy 参数集 (k=l=2, n=30)，用于测试

    Returns:
        (rho, t1, params):
          rho: 32 字节种子
          t1: (k, n) 高位系数
          params: {"name", "k", "l", "n", "d", "oid"}
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"证书文件不存在: {filepath}")

    encoding, content_type = _detect_format(filepath)
    logger.info(f"解析证书: {filepath} ({encoding.upper()}, {content_type})")

    with open(filepath, "rb") as f:
        data = f.read()

    # 提取 SPKI DER
    if encoding == "pem":
        label, _, der = asn1pem.unarmor(data)
        if content_type == "certificate":
            spki_der = _extract_spki_from_cert(der)
            logger.info("  从 X.509 证书中提取 SPKI")
        else:
            spki_der = der
            logger.info("  直接读取 SPKI")
    else:
        if content_type == "certificate":
            spki_der = _extract_spki_from_cert(data)
            logger.info("  从 X.509 证书中提取 SPKI")
        else:
            spki_der = data
            logger.info("  直接读取 SPKI")

    # 解码
    rho, t1, params = _decode_spki_with_params(spki_der, use_toy)

    logger.info(f"  ML-DSA 变体: {params['name']} (OID {params['oid']})")
    logger.info(f"  参数: k={params['k']}, l={params['l']}, n={params['n']}, d={params['d']}")
    logger.info(f"  ρ: {rho.hex()[:16]}...")
    logger.info(f"  t1: shape={t1.shape}, range=[{t1.min()}, {t1.max()}]")

    return rho, t1, params
