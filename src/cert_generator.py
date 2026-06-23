#!/usr/bin/env python3
"""
模拟受害者证书生成器。

生成包含 ML-DSA 公钥的标准 X.509 自签名证书。
攻击流程:
  1. 本脚本生成 victim_cert.pem (模拟受害者)
  2. cert_parser.py 解析证书提取公钥
  3. lattice_attack.py 执行格攻击恢复私钥
"""

import argparse
import os
import sys
import time

import numpy as np

# 确保可以导入 src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.params import get_params
from src.keygen import keygen, expand_a
from src.protocol_adapter import ProtocolAdapter
from src.spki import encode_spki, save_spki_pem, save_spki_der, MLDSA_OIDS


def generate_self_signed_cert(
    output_path: str,
    mldsa_name: str = "ML-DSA-65",
    params_name: str = "easy",
    d: int = 10,
    seed: int | None = None,
    fmt: str = "pem",
):
    """生成包含 ML-DSA 公钥的自签名 X.509 证书。

    流程:
      1. keygen → (rho, s1, s2, t, A)
      2. Power2Round → (t1, t0)   [公钥只存 t1]
      3. 编码为 SPKI → DER/PEM
      4. 包装为自签名 X.509 证书

    Args:
        output_path: 输出文件路径
        mldsa_name: ML-DSA 参数名 (决定 OID)
        params_name: 内部参数集名
        d: Power2Round 低位比特数
        seed: 随机种子 (None = 随机)
        fmt: 输出格式 ("pem" 或 "der")
    """
    p = get_params(params_name)
    k, l, n, q = p["k"], p["l"], p["n"], p["q"]

    # 确定 ML-DSA 名称与 OID 的映射
    if mldsa_name not in MLDSA_OIDS:
        # 自动映射: k=4→44, k=5→65, k=7→87
        k_to_mldsa = {4: "ML-DSA-44", 5: "ML-DSA-65", 7: "ML-DSA-87"}
        mldsa_name = k_to_mldsa.get(k, "ML-DSA-65")

    print(f"═══ ML-DSA 证书生成器 ═══")
    print(f"  参数集: {params_name} (k={k}, l={l}, n={n})")
    print(f"  ML-DSA: {mldsa_name} (OID {MLDSA_OIDS[mldsa_name]})")
    print(f"  Power2Round: d={d}")

    # 1. 密钥生成
    seed_bytes = seed.to_bytes(8, "big").ljust(32, b"\x00") if seed is not None else None
    t0 = time.time()
    rho, s1, s2, t, A = keygen(params_name, seed=seed_bytes, params=p)
    keygen_time = time.time() - t0
    print(f"  密钥生成: {keygen_time:.3f}s")

    # 2. Power2Round 压缩: t → t1 (公钥中只存 t1)
    adapter = ProtocolAdapter(d=d, q=q)
    t1, t0_raw, t0_c = adapter.encode(t)
    s_norm = float(np.sqrt(np.sum(s1**2) + np.sum(s2**2)))
    t0_norm = float(np.linalg.norm(t0_c.flatten()))
    print(f"  t1 范围: [{t1.min()}, {t1.max()}]")
    print(f"  秘密范数: {s_norm:.2f}, 误差范数: {t0_norm:.2f}")

    # 3. 编码为 SPKI
    spki_der = encode_spki(rho, t1, mldsa_name, d=d)
    print(f"  SPKI DER: {len(spki_der)} bytes")

    # 4. 保存
    if fmt == "pem":
        save_spki_pem(output_path, rho, t1, mldsa_name, d=d)
    else:
        save_spki_der(output_path, rho, t1, mldsa_name, d=d)
    print(f"  已保存: {output_path} ({fmt.upper()})")

    # 5. 同时保存私钥材料 (用于验证攻击结果)
    priv_path = output_path.rsplit(".", 1)[0] + "_private.npz"
    np.savez(priv_path, s1=s1, s2=s2, t=t, t0_raw=t0_raw, rho=rho, A=A)
    print(f"  私钥材料: {priv_path} (用于验证)")

    # 6. openssl 检查
    import subprocess
    if fmt == "pem":
        r = subprocess.run(
            ["openssl", "x509", "-inform", "PEM", "-in", output_path, "-text", "-noout"],
            capture_output=True, text=True
        )
    else:
        r = subprocess.run(
            ["openssl", "x509", "-inform", "DER", "-in", output_path, "-text", "-noout"],
            capture_output=True, text=True
        )
    if r.returncode == 0:
        print(f"  openssl 验证: ✓ 可解析")
    else:
        print(f"  openssl 验证: ⚠ {r.stderr[:200]}")

    return {
        "rho": rho, "s1": s1, "s2": s2, "t": t,
        "t1": t1, "t0_raw": t0_raw, "A": A,
        "mldsa_name": mldsa_name,
    }


def main():
    parser = argparse.ArgumentParser(description="ML-DSA 证书生成器 (模拟受害者)")
    parser.add_argument("output", nargs="?", default="victim_cert.pem",
                        help="输出文件路径 (默认: victim_cert.pem)")
    parser.add_argument("--params", default="easy", help="参数集 (默认: easy)")
    parser.add_argument("--mldsa", default=None,
                        choices=["ML-DSA-44", "ML-DSA-65", "ML-DSA-87"],
                        help="ML-DSA 变体 (默认: 按 k 自动选择)")
    parser.add_argument("--d", type=int, default=10, help="Power2Round d (默认: 10)")
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    parser.add_argument("--format", choices=["pem", "der"], default="pem",
                        help="输出格式 (默认: pem)")
    args = parser.parse_args()

    mldsa = args.mldsa
    if mldsa is None:
        p = get_params(args.params)
        k_to_mldsa = {4: "ML-DSA-44", 5: "ML-DSA-65", 7: "ML-DSA-87"}
        mldsa = k_to_mldsa.get(p["k"], "ML-DSA-65")

    generate_self_signed_cert(
        args.output, mldsa, args.params, args.d, args.seed, args.format
    )


if __name__ == "__main__":
    main()
