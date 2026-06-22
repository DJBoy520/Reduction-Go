#!/usr/bin/env python3
"""
证书解析与攻击入口。

从 PEM/DER 证书中提取 ML-DSA 公钥，执行格攻击。
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.params import get_params
from src.keygen import expand_a
from src.protocol_adapter import ProtocolAdapter, D_BY_PARAMS
from src.spki import load_spki_pem, load_spki_der, MLDSA_OIDS
from src.lattice_attack import run_attack, classify_results, verify_basis
from src.poly_math import mat_vec_mul, vec_add_mod
from src.logger import setup_logging

import logging
logger = logging.getLogger(__name__)


def parse_and_attack(
    cert_path: str,
    params_name: str = "easy",
    d: int = 10,
    bkz_block_size: int = 25,
    bkz_max_loops: int = 8,
    lll_delta: float = 0.99,
    float_type: str = "mpfr",
    precision: int = 200,
    priv_path: str | None = None,
    no_bkz: bool = False,
):
    """从证书提取公钥并执行格攻击。

    Args:
        cert_path: PEM 或 DER 证书路径
        params_name: 内部参数集名
        d: Power2Round d 参数
        bkz_block_size: BKZ 块大小
        bkz_max_loops: BKZ 最大循环数
        lll_delta: LLL delta
        float_type: 浮点类型
        precision: MPFR 精度
        priv_path: 私钥材料路径 (用于验证，可选)
    """
    p = get_params(params_name)
    k, l, n, q = p["k"], p["l"], p["n"], p["q"]

    # 1. 解析证书
    logger.info(f"═══ ML-DSA 证书攻击 ═══")
    logger.info(f"证书: {cert_path}")
    logger.info(f"参数: k={k}, l={l}, n={n}, q={q}, d={d}")

    fmt = "pem" if cert_path.endswith(".pem") else "der"
    if fmt == "pem":
        rho, t1, mldsa_name = load_spki_pem(cert_path, k, n)
    else:
        rho, t1, mldsa_name = load_spki_der(cert_path, k, n)

    logger.info(f"  ML-DSA 变体: {mldsa_name}")
    logger.info(f"  ρ: {rho.hex()[:20]}...")
    logger.info(f"  t1 范围: [{t1.min()}, {t1.max()}]")

    # 2. 还原 A 矩阵
    A = expand_a(rho, k, l, n, q)
    logger.info(f"  A 矩阵: shape={A.shape}")

    # 3. Power2Round: t1 → t_recon
    adapter = ProtocolAdapter(d=d, q=q)
    t_recon = adapter.decode(t1)
    logger.info(f"  t_recon 范围: [{t_recon.min()}, {t_recon.max()}]")
    logger.info(f"  方程: A·s1 + s2' ≡ t_recon (mod q)")
    logger.info(f"  其中 s2' = s2 - t0 (合并误差)")

    # 4. 如果有私钥材料，加载用于验证
    priv_path_str = priv_path  # 保存用于日志
    s1_real = None
    s2_real = None
    if priv_path and os.path.exists(priv_path):
        data = np.load(priv_path)
        s1_real = data["s1"]
        s2_real = data["s2"]
        t0_raw = data["t0_raw"]
        s2_prime = (s2_real.astype(np.int64) - t0_raw.astype(np.int64)) % q
        s2_real = s2_prime.reshape(k, n)
        logger.info(f"  已加载私钥材料 (用于验证)")
    elif priv_path:
        logger.warning(f"  私钥文件不存在: {priv_path}，跳过验证")
        priv_path_str = None

    # 5. 执行格攻击
    if s1_real is None:
        # 没有私钥，创建占位
        s1_real = np.zeros((l, n), dtype=np.int64)
        s2_real = np.zeros((k, n), dtype=np.int64)

    attack_result = run_attack(
        A, t_recon, q, s1_real, s2_real,
        bkz_block_size=bkz_block_size,
        bkz_max_loops=bkz_max_loops,
        lll_delta=lll_delta,
        float_type=float_type,
        precision=precision,
        no_bkz=no_bkz,
    )

    # 6. 结果
    logger.info(f"═══ 攻击结果 ═══")
    logger.info(f"  LLL: {attack_result['lll_time']:.2f}s")
    logger.info(f"  BKZ: {attack_result['bkz_time']:.2f}s")
    logger.info(f"  候选数: {len(attack_result['candidates'])}")

    if priv_path:
        classified = classify_results(attack_result)
        perfect = sum(1 for c in classified if c["verdict"] == "完美恢复私钥")
        s1_only = sum(1 for c in classified if c["verdict"] == "s1 完美恢复 (s2 不匹配)")
        logger.info(f"  完美恢复: {perfect}, s1 完美: {s1_only}")
    else:
        # 无私钥验证，只看方程满足
        eq_count = sum(1 for c in attack_result["candidates"] if c["eq_holds"])
        logger.info(f"  方程满足: {eq_count}/{len(attack_result['candidates'])}")

    return attack_result


def main():
    parser = argparse.ArgumentParser(description="ML-DSA 证书解析与攻击")
    parser.add_argument("cert", help="PEM 或 DER 证书/公钥文件")
    parser.add_argument("--params", default="easy", help="参数集 (默认: easy)")
    parser.add_argument("--d", type=int, default=10, help="Power2Round d (默认: 10)")
    parser.add_argument("--bkz-block-size", type=int, default=25)
    parser.add_argument("--bkz-max-loops", type=int, default=8)
    parser.add_argument("--lll-delta", type=float, default=0.99)
    parser.add_argument("--float-type", default="mpfr",
                        choices=["mpfr", "double", "long double"])
    parser.add_argument("--precision", type=int, default=200)
    parser.add_argument("--priv", default=None, help="私钥材料 .npz (用于验证)")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--no-bkz", action="store_true", help="跳过 BKZ")
    args = parser.parse_args()

    setup_logging(console_level=logging.DEBUG if args.verbose else logging.INFO)

    parse_and_attack(
        args.cert, args.params, args.d,
        args.bkz_block_size, args.bkz_max_loops, args.lll_delta,
        args.float_type, args.precision, args.priv,
        no_bkz=args.no_bkz,
    )


if __name__ == "__main__":
    main()
