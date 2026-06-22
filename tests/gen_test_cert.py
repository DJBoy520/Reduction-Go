#!/usr/bin/env python3
"""
测试证书生成器 — 用于验证解析器和攻击流程。

生成包含 ML-DSA 公钥的 SPKI PEM/DER 文件。

注意: 使用 --toy-params 攻击时，需要确保 n=30:
  python tests/gen_test_cert.py out.pem --n 30 --seed 12345
  python main.py --cert out.pem --toy-params ...
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.keygen import keygen
from src.params import get_params
from src.protocol_adapter import ProtocolAdapter
from src.spki import save_spki_pem, save_spki_der, MLDSA_OIDS


def main():
    parser = argparse.ArgumentParser(description="ML-DSA 测试证书生成器")
    parser.add_argument("output", nargs="?", default="test_cert.pem",
                        help="输出文件路径 (默认: test_cert.pem)")
    parser.add_argument("--params", default="easy", help="参数集 (默认: easy)")
    parser.add_argument("--mldsa", default=None,
                        choices=["ML-DSA-44", "ML-DSA-65", "ML-DSA-87"])
    parser.add_argument("--d", type=int, default=10, help="Power2Round d (默认: 10)")
    parser.add_argument("--n", type=int, default=None, help="覆盖多项式维度 n")
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    parser.add_argument("--format", choices=["pem", "der"], default="pem")
    parser.add_argument("--priv", default=None, help="同时保存私钥材料 .npz")
    args = parser.parse_args()

    p = get_params(args.params)
    k, l, n, q = p["k"], p["l"], p["n"], p["q"]
    if args.n is not None:
        p["n"] = args.n
        n = args.n

    # 自动选择 ML-DSA 变体
    k_to_mldsa = {4: "ML-DSA-44", 5: "ML-DSA-65", 6: "ML-DSA-65", 7: "ML-DSA-87", 8: "ML-DSA-87"}
    mldsa = args.mldsa or k_to_mldsa.get(k, "ML-DSA-65")

    print(f"═══ 测试证书生成器 ═══")
    print(f"  参数集: {args.params} (k={k}, l={l}, n={n})")
    print(f"  ML-DSA: {mldsa} (OID {MLDSA_OIDS[mldsa]})")
    print(f"  Power2Round: d={args.d}")

    # 密钥生成
    seed_bytes = args.seed.to_bytes(8, "big").ljust(32, b"\x00") if args.seed is not None else None
    rho, s1, s2, t, A = keygen(args.params, seed=seed_bytes, params=p)

    # Power2Round
    adapter = ProtocolAdapter(d=args.d, q=q)
    t1, t0_raw, t0_c = adapter.encode(t)
    print(f"  t1: range=[{t1.min()}, {t1.max()}]")
    print(f"  秘密范数: {np.sqrt(np.sum(s1**2) + np.sum(s2**2)):.2f}")

    # 保存
    if args.format == "pem":
        save_spki_pem(args.output, rho, t1, mldsa)
    else:
        save_spki_der(args.output, rho, t1, mldsa)
    print(f"  已保存: {args.output}")

    # 私钥材料
    if args.priv:
        np.savez(args.priv, s1=s1, s2=s2, t=t, t0_raw=t0_raw, rho=rho, A=A)
        print(f"  私钥: {args.priv}")


if __name__ == "__main__":
    main()
