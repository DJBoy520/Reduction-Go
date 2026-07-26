#!/usr/bin/env python3
"""
使用 liboqs-python 生成 ML-DSA X.509 自签名证书。

生成 ML-DSA-44/65/87 各 5 个证书，保存到 certs/ 目录。
"""

import os
import sys
import secrets
import datetime

import oqs
from asn1crypto import pem as asn1pem
from asn1crypto.core import (
    Sequence, ObjectIdentifier, Null, BitString, OctetString,
    Integer, UTCTime, SequenceOf,
)

# DER 构建工具和 OID 从公共模块引用
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.protocol.der_utils import (
    build_der_sequence, build_der_set,
    build_der_bitstring, build_der_octet_string,
    build_der_oid, build_der_null, build_der_integer,
    build_der_utc_time, build_der_context_specific,
    build_name,
)
from src.domain.params import MLDSA_OIDS


# ── 证书生成 ──────────────────────────────────────────────────────────────────

def generate_self_signed_cert(
    variant: str,
    serial: int,
    validity_days: int = 365,
) -> tuple[bytes, bytes, bytes]:
    """生成 ML-DSA 自签名 X.509 证书。

    Returns: (cert_der, public_key_bytes, secret_key_bytes)
    """
    oid = MLDSA_OIDS[variant]

    # 1. 生成密钥对
    sig = oqs.Signature(variant)
    pk_bytes = sig.generate_keypair()
    sk_bytes = sig.export_secret_key()

    # 2. 构建 SubjectPublicKeyInfo
    spki = build_der_sequence(
        build_der_sequence(build_der_oid(oid), build_der_null()),
        build_der_bitstring(pk_bytes),
    )

    # 3. 构建时间
    now = datetime.datetime.now(datetime.UTC)
    not_before = now - datetime.timedelta(days=1)
    not_after = now + datetime.timedelta(days=validity_days)

    # 4. 构建 TBSCertificate
    cn = f"{variant}-test-{serial:04d}"
    issuer_name = build_name(cn)
    subject_name = build_name(cn)  # 自签名: issuer == subject

    validity = build_der_sequence(
        build_der_utc_time(not_before),
        build_der_utc_time(not_after),
    )

    tbs = build_der_sequence(
        # version [0] EXPLICIT INTEGER v3 (2)
        build_der_context_specific(0, build_der_integer(2)),
        # serialNumber
        build_der_integer(serial),
        # signature (AlgorithmIdentifier)
        build_der_sequence(build_der_oid(oid), build_der_null()),
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
    cert = build_der_sequence(
        tbs,
        # signatureAlgorithm
        build_der_sequence(build_der_oid(oid), build_der_null()),
        # signatureValue
        build_der_bitstring(sig_bytes),
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
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "certs")
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
            os.chmod(sk_path, 0o600)

            print(f"  [{i}/{count}] {basename}  "
                  f"DER={len(cert_der)} bytes  "
                  f"pk={len(pk)}  sig_verify={'✓' if ok else '✗'}")

            total += 1

    print(f"\n共生成 {total} 个证书，保存在: {output_dir}/")


if __name__ == "__main__":
    main()
