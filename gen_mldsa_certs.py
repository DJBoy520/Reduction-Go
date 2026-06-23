#!/usr/bin/env python3
"""
使用 liboqs-python 生成 ML-DSA X.509 自签名证书。

生成 ML-DSA-44/65/87 各 5 个证书，保存到 certs/ 目录。
"""

import os
import sys
import hashlib
import secrets
import datetime

import oqs
from asn1crypto import pem as asn1pem
from asn1crypto.core import (
    Sequence, ObjectIdentifier, Null, BitString, OctetString,
    Integer, UTCTime, SequenceOf,
)

# ── ML-DSA OID 映射 ──────────────────────────────────────────────────────────

MLDSA_PARAMS = {
    "ML-DSA-44": {
        "oid": "2.16.840.1.101.3.4.3.17",
        "sig_oid": "2.16.840.1.101.3.4.3.17",  # 签名算法 OID 与密钥 OID 相同
    },
    "ML-DSA-65": {
        "oid": "2.16.840.1.101.3.4.3.18",
        "sig_oid": "2.16.840.1.101.3.4.3.18",
    },
    "ML-DSA-87": {
        "oid": "2.16.840.1.101.3.4.3.19",
        "sig_oid": "2.16.840.1.101.3.4.3.19",
    },
}


# ── DER 构建工具 ─────────────────────────────────────────────────────────────

def _encode_der_length(length: int) -> bytes:
    if length < 0x80:
        return bytes([length])
    elif length < 0x100:
        return bytes([0x81, length])
    elif length < 0x10000:
        return bytes([0x82, length >> 8, length & 0xFF])
    else:
        raise ValueError(f"DER length {length} too large")


def _build_der_sequence(*items: bytes) -> bytes:
    content = b"".join(items)
    return b"\x30" + _encode_der_length(len(content)) + content


def _build_der_set(*items: bytes) -> bytes:
    content = b"".join(items)
    return b"\x31" + _encode_der_length(len(content)) + content


def _build_der_bitstring(data: bytes, unused_bits: int = 0) -> bytes:
    content = bytes([unused_bits]) + data
    return b"\x03" + _encode_der_length(len(content)) + content


def _build_der_octet_string(data: bytes) -> bytes:
    return b"\x04" + _encode_der_length(len(data)) + data


def _build_der_oid(oid_str: str) -> bytes:
    parts = [int(x) for x in oid_str.split(".")]
    first_byte = 40 * parts[0] + parts[1]
    encoded = bytes([first_byte])
    for part in parts[2:]:
        if part < 0x80:
            encoded += bytes([part])
        else:
            multi = []
            multi.append(part & 0x7F)
            part >>= 7
            while part > 0:
                multi.append(0x80 | (part & 0x7F))
                part >>= 7
            encoded += bytes(reversed(multi))
    return b"\x06" + _encode_der_length(len(encoded)) + encoded


def _build_der_null() -> bytes:
    return b"\x05\x00"


def _build_der_integer(value: int) -> bytes:
    if value == 0:
        return b"\x02\x01\x00"
    neg = value < 0
    if neg:
        value = -value
    byte_len = (value.bit_length() + 8) // 8
    raw = value.to_bytes(byte_len, "big")
    # 补零防止最高位为 1 被误解为负数
    if raw[0] & 0x80 and not neg:
        raw = b"\x00" + raw
    if neg:
        # 二补码
        int_val = int.from_bytes(raw, "big")
        int_val = (1 << (len(raw) * 8)) - int_val
        raw = int_val.to_bytes(len(raw), "big")
    return b"\x02" + _encode_der_length(len(raw)) + raw


def _build_der_utc_time(dt: datetime.datetime) -> bytes:
    # YYMMDDHHMMSSZ
    s = dt.strftime("%y%m%d%H%M%SZ")
    return b"\x17" + bytes([len(s)]) + s.encode("ascii")


def _build_der_context_specific(tag: int, content: bytes) -> bytes:
    return bytes([0xA0 | tag]) + _encode_der_length(len(content)) + content


def _build_name(common_name: str) -> bytes:
    """构建 X.509 Name (SEQUENCE OF SET OF AttributeTypeAndValue)。

    只设置 CN (commonName)。
    """
    # OID 2.5.4.3 = commonName
    cn_attr = _build_der_sequence(
        _build_der_oid("2.5.4.3"),
        _build_der_octet_string(common_name.encode("utf-8")),
    )
    # SET 包含一个 AttributeTypeAndValue
    attr_set = _build_der_set(cn_attr)
    # SEQUENCE 包含一个 SET
    return _build_der_sequence(attr_set)


# ── 证书生成 ──────────────────────────────────────────────────────────────────

def generate_self_signed_cert(
    variant: str,
    serial: int,
    validity_days: int = 365,
) -> tuple[bytes, bytes, bytes]:
    """生成 ML-DSA 自签名 X.509 证书。

    Returns: (cert_der, public_key_bytes, secret_key_bytes)
    """
    params = MLDSA_PARAMS[variant]
    oid = params["oid"]

    # 1. 生成密钥对
    sig = oqs.Signature(variant)
    pk_bytes = sig.generate_keypair()
    sk_bytes = sig.export_secret_key()

    # 2. 构建 SubjectPublicKeyInfo
    spki = _build_der_sequence(
        _build_der_sequence(_build_der_oid(oid), _build_der_null()),
        _build_der_bitstring(pk_bytes),
    )

    # 3. 构建时间
    now = datetime.datetime.utcnow()
    not_before = now - datetime.timedelta(days=1)
    not_after = now + datetime.timedelta(days=validity_days)

    # 4. 构建 TBSCertificate
    cn = f"{variant}-test-{serial:04d}"
    issuer_name = _build_name(cn)
    subject_name = _build_name(cn)  # 自签名: issuer == subject

    validity = _build_der_sequence(
        _build_der_utc_time(not_before),
        _build_der_utc_time(not_after),
    )

    tbs = _build_der_sequence(
        # version [0] EXPLICIT INTEGER v3 (2)
        _build_der_context_specific(0, _build_der_integer(2)),
        # serialNumber
        _build_der_integer(serial),
        # signature (AlgorithmIdentifier)
        _build_der_sequence(_build_der_oid(oid), _build_der_null()),
        # issuer
        issuer_name,
        # validity
        validity,
        # subject
        subject_name,
        # subjectPublicKeyInfo
        spki,
    )

    # 5. 签名
    sig_bytes = sig.sign(tbs)

    # 6. 组装完整证书
    cert = _build_der_sequence(
        tbs,
        # signatureAlgorithm
        _build_der_sequence(_build_der_oid(oid), _build_der_null()),
        # signatureValue
        _build_der_bitstring(sig_bytes),
    )

    return cert, pk_bytes, sk_bytes


def verify_cert_signature(cert_der: bytes, variant: str) -> bool:
    """验证证书签名。"""
    cert = Sequence.load(cert_der)
    tbs_der = cert[0].dump()
    sig_bitstring = cert[2]
    sig_bytes = sig_bitstring.contents[1:]  # 跳过 unused_bits 字节

    # 从 TBSCertificate 提取公钥
    tbs = cert[0]
    spki = tbs[6]  # SubjectPublicKeyInfo
    pk_bitstring = spki[1]
    pk_bytes = pk_bitstring.contents  # 原始字节（含 unused_bits 前缀）
    pk_bytes = pk_bytes[1:]  # 跳过 unused_bits 字节

    sig = oqs.Signature(variant)
    return sig.verify(tbs_der, sig_bytes, pk_bytes)


def main():
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "certs")
    os.makedirs(output_dir, exist_ok=True)

    variants = ["ML-DSA-44", "ML-DSA-65", "ML-DSA-87"]
    count = 5

    total = 0
    for variant in variants:
        print(f"\n{'='*50}")
        print(f"  生成 {variant} 证书 ({count} 个)")
        print(f"{'='*50}")

        for i in range(1, count + 1):
            serial = secrets.randbits(64) | 1  # 随机序列号
            cert_der, pk, sk = generate_self_signed_cert(variant, serial)

            # 验证签名
            ok = verify_cert_signature(cert_der, variant)

            # 保存
            basename = f"{variant.lower().replace('-', '_')}_{i:02d}"
            cert_path = os.path.join(output_dir, f"{basename}.der")
            with open(cert_path, "wb") as f:
                f.write(cert_der)

            # PEM 版本
            pem_path = os.path.join(output_dir, f"{basename}.pem")
            pem_data = asn1pem.armor("CERTIFICATE", cert_der)
            with open(pem_path, "wb") as f:
                f.write(pem_data)

            # 私钥（用于后续测试攻击）
            sk_path = os.path.join(output_dir, f"{basename}_sk.bin")
            with open(sk_path, "wb") as f:
                f.write(sk)

            print(f"  [{i}/{count}] {basename}  "
                  f"DER={len(cert_der)} bytes  "
                  f"pk={len(pk)}  sig_verify={'✓' if ok else '✗'}")

            total += 1

    print(f"\n共生成 {total} 个证书，保存在: {output_dir}/")


if __name__ == "__main__":
    main()
