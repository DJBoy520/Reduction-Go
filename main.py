#!/usr/bin/env python3
"""
ML-DSA 格攻击闭环测试环境 — 主入口。

流程：
  [1/5] 生成公钥
  [2/5] 解析公钥
  [3/5] 构造格基
  [4/5] LLL 约减
  [5/5] BKZ 约减 + 三层验证
"""

import argparse
import logging
import os
import sys
import time

import numpy as np

try:
    from tqdm import tqdm as _tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

from src.params import get_params
from src.keygen import keygen, expand_a
from src.pubkey import save_public_key, load_public_key
from src.lattice_attack import run_attack, classify_results, verify_basis
from src.protocol_adapter import ProtocolAdapter, D_BY_PARAMS
from src.poly_math import mat_vec_mul, vec_add_mod
from src.cert_parser import parse_certificate
from src.logger import setup_logging
from src.progress import print_estimate

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="ML-DSA 格攻击闭环测试环境"
    )
    parser.add_argument(
        "params", nargs="?", default="toy",
        help="参数集名称 (默认: toy)"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="控制台输出 DEBUG 级别日志"
    )
    parser.add_argument(
        "--log-level", dest="log_level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="自定义日志级别 (覆盖 --verbose)"
    )
    parser.add_argument(
        "--no-bkz", dest="no_bkz", action="store_true",
        help="跳过 BKZ 步骤，只运行 LLL"
    )
    parser.add_argument(
        "--bkz-block-size", dest="bkz_block_size", type=int, default=None,
        help="BKZ 块大小 (覆盖配置文件)"
    )
    parser.add_argument(
        "--bkz-max-loops", dest="bkz_max_loops", type=int, default=None,
        help="BKZ 最大循环数 (覆盖配置文件)"
    )
    parser.add_argument(
        "--bkz-auto-abort", dest="bkz_auto_abort", action="store_true",
        help="BKZ 自动终止（检测到无改善时提前退出）"
    )
    parser.add_argument(
        "--k", type=int, default=None,
        help="矩阵 A 的行数 k (覆盖配置文件)"
    )
    parser.add_argument(
        "--l", type=int, default=None,
        help="矩阵 A 的列数 l (覆盖配置文件)"
    )
    parser.add_argument(
        "--n", type=int, default=None,
        help="多项式维度 n (覆盖配置文件)"
    )
    parser.add_argument(
        "--lll-delta", dest="lll_delta", type=float, default=0.999,
        help="LLL 约减质量参数 delta (默认: 0.999)"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="随机种子 (便于复现)"
    )
    parser.add_argument(
        "--float-type", dest="float_type",
        choices=["mpfr", "double", "long double"],
        default=None,
        help=("浮点精度类型 (覆盖配置文件)。"
              "mpfr=多精度浮点(推荐,精度由--precision控制), "
              "double=IEEE754双精度53bit(最快), "
              "long double=扩展精度80bit")
    )
    parser.add_argument(
        "--precision", type=int, default=None,
        help="MPFR 精度(比特数)，仅 --float-type=mpfr 时生效 (覆盖配置文件)"
    )
    parser.add_argument(
        "--slack", action="store_true",
        help="启用 Power2Round 合并误差模式: 公钥仅含 t1，用 t_recon=t1·2^d 攻击"
    )
    parser.add_argument(
        "--d", type=int, default=None,
        help="Power2Round 的 d 参数 (低位比特数)。ML-DSA-44:10, ML-DSA-65/87:13"
    )
    parser.add_argument(
        "--cert", type=str, default=None,
        help="证书文件路径 (PEM/DER)，直接从证书提取公钥进行攻击"
    )
    parser.add_argument(
        "--toy-params", dest="toy_params", action="store_true",
        help="使用 toy 参数集 (k=l=2, n=30) 解析证书，用于测试"
    )
    return parser.parse_args()


def _run_cert_attack(args):
    """从证书文件提取公钥并执行格攻击。"""
    cert_path = args.cert
    if not os.path.exists(cert_path):
        logger.error(f"证书文件不存在: {cert_path}")
        sys.exit(1)

    # 解析证书
    rho, t1, cert_params = parse_certificate(cert_path, use_toy=args.toy_params)
    k, l, n = cert_params["k"], cert_params["l"], cert_params["n"]
    d_param = cert_params["d"]
    mldsa_name = cert_params["name"]

    # 获取 q
    q = cert_params.get("q", 8380417)

    logger.info(f"═══ ML-DSA 证书攻击 ({mldsa_name}) ═══")
    logger.info(f"参数: k={k}, l={l}, n={n}, q={q}, d={d_param}")
    logger.info(f"格维度: {k*n + l*n + 1}")

    # 重建 A 矩阵
    A = expand_a(rho, k, l, n, q)
    logger.info(f"  A 矩阵: shape={A.shape}")

    # Power2Round: t1 → t_recon
    adapter = ProtocolAdapter(d=d_param, q=q)
    t_recon = adapter.decode(t1)
    t_recon_norm = float(np.linalg.norm(t_recon.astype(np.float64).flatten()))
    logger.info(f"  t_recon: range=[{t_recon.min()}, {t_recon.max()}], norm={t_recon_norm:.1f}")
    logger.info(f"  方程: A·s1 + s2' ≡ t_recon (mod q)")
    logger.info(f"  其中 s2' = s2 - t0 (合并误差)")

    # 攻击参数
    bkz_block_size = args.bkz_block_size if args.bkz_block_size is not None else 20
    bkz_max_loops = args.bkz_max_loops if args.bkz_max_loops is not None else 8
    no_bkz = args.no_bkz
    lll_delta = args.lll_delta
    float_type = args.float_type if args.float_type is not None else "mpfr"
    precision = args.precision if args.precision is not None else 200

    logger.info(f"BKZ: block_size={bkz_block_size}, max_loops={bkz_max_loops}")
    logger.info(f"LLL delta: {lll_delta}, float_type={float_type}, precision={precision}")

    # 无私钥验证: s1_real/s2_real 用零占位
    s1_dummy = np.zeros((l, n), dtype=np.int64)
    s2_dummy = np.zeros((k, n), dtype=np.int64)

    # 参数和耗时估算
    dim = k * n + l * n + 1
    print_estimate(dim, bkz_block_size, bkz_max_loops, float_type, precision)

    # 格攻击
    attack_result = run_attack(
        A, t_recon, q, s1_dummy, s2_dummy,
        bkz_block_size=bkz_block_size,
        bkz_max_loops=bkz_max_loops,
        no_bkz=no_bkz,
        lll_delta=lll_delta,
        float_type=float_type,
        precision=precision,
    )

    logger.info(f"═══ 攻击结果 ═══")
    logger.info(f"  LLL: {attack_result['lll_time']:.2f}s")
    logger.info(f"  BKZ: {attack_result['bkz_time']:.2f}s")
    logger.info(f"  候选数: {len(attack_result['candidates'])}")

    # 方程满足统计
    eq_count = sum(1 for c in attack_result["candidates"] if c["eq_holds"])
    short_count = sum(1 for c in attack_result["candidates"] if c["cand_norm"] < 1000)
    logger.info(f"  方程满足: {eq_count}")
    logger.info(f"  短向量 (norm<1000): {short_count}")

    # 最佳候选
    valid = [c for c in attack_result["candidates"] if c["eq_holds"]]
    if valid:
        best = min(valid, key=lambda c: c["cand_norm"])
        s1p = best["s1_prime"]
        s2p = best["s2_prime"]
        logger.info(f"═══ 最佳候选 ═══")
        logger.info(f"  s1': norm={np.linalg.norm(s1p.flatten()):.2f}, range=[{s1p.min()}, {s1p.max()}]")
        logger.info(f"  s2': norm={np.linalg.norm(s2p.flatten()):.2f}, range=[{s2p.min()}, {s2p.max()}]")

        # 方程验证
        lhs = vec_add_mod(mat_vec_mul(A, s1p, q), s2p, q) % q
        rhs = t_recon.reshape(k, n) % q
        if np.array_equal(lhs, rhs):
            logger.info(f"  方程验证: A·s1' + s2' ≡ t_recon (mod q) 通过 ✓")
        else:
            logger.warning(f"  方程验证: 失败 ✗")
    else:
        logger.info(f"═══ 最佳候选: 无 ═══")

    logger.info("完成。")


def _find_best_candidate(classified: list[dict]) -> dict | None:
    """从分类结果中找最佳候选。

    优先级：完美恢复 > s1 完美恢复 > 最短有效候选。
    返回 None 如果没有有效候选。
    """
    perfect_cands = [c for c in classified if c.get("perfect", False)]
    if perfect_cands:
        return perfect_cands[0]

    s1_perfect_cands = [c for c in classified if c.get("s1_perfect", False)]
    if s1_perfect_cands:
        return s1_perfect_cands[0]

    valid = [c for c in classified if c["eq_holds"]]
    if valid:
        return min(valid, key=lambda c: c["cand_norm"])
    return None


def main():
    args = parse_args()

    # ── 日志级别：CLI > 配置 > 默认 ──
    if args.log_level:
        console_level = getattr(logging, args.log_level)
    elif args.verbose:
        console_level = logging.DEBUG
    else:
        console_level = logging.INFO

    setup_logging(console_level=console_level)

    # ── 证书攻击模式 ──
    if args.cert:
        if args.slack:
            logger.warning("--cert 模式下 --slack 被忽略 (证书模式天然使用 Power2Round)")
        _run_cert_attack(args)
        return

    params_name = args.params
    logger.info(f"═══ ML-DSA 格攻击闭环测试 (参数集: {params_name}) ═══")

    p = get_params(params_name)

    # CLI 参数覆盖：命令行 > 配置文件 > 默认值
    if args.k is not None:
        p["k"] = args.k
    if args.l is not None:
        p["l"] = args.l
    if args.n is not None:
        p["n"] = args.n

    k, l, n, q, eta = p["k"], p["l"], p["n"], p["q"], p["eta"]

    bkz_block_size = args.bkz_block_size if args.bkz_block_size is not None else p.get("bkz_block_size", 20)
    bkz_max_loops = args.bkz_max_loops if args.bkz_max_loops is not None else p.get("bkz_max_loops", 8)
    bkz_threads = p.get("bkz_threads", 6)
    no_bkz = args.no_bkz or not p.get("use_bkz", True)
    lll_delta = args.lll_delta
    seed = args.seed.to_bytes(8, "big").ljust(32, b"\x00") if args.seed is not None else None
    bkz_auto_abort = args.bkz_auto_abort or p.get("auto_abort", False)
    float_type = args.float_type if args.float_type is not None else p.get("float_type", "mpfr")
    precision = args.precision if args.precision is not None else p.get("precision", 200)
    use_slack = args.slack
    d_param = args.d if args.d is not None else D_BY_PARAMS.get(params_name, 10)

    logger.info(f"参数: k={k}, l={l}, n={n}, q={q}, η={eta}")
    if use_slack:
        logger.info(f"格维度: {k*n + l*n + 1} (Power2Round 合并误差, d={d_param})")
    else:
        logger.info(f"格维度: {k*n + l*n + 1} (Kannan 嵌入)")
    if seed:
        logger.info(f"随机种子: {args.seed}")
    if no_bkz:
        logger.info("模式: 仅 LLL（BKZ 已跳过）")
    else:
        logger.info(f"BKZ: block_size={bkz_block_size}, max_loops={bkz_max_loops}, omp_threads={bkz_threads}, auto_abort={bkz_auto_abort}")
    logger.info(f"LLL delta: {lll_delta}")
    if float_type == "mpfr":
        logger.info(f"浮点精度: {float_type}, precision={precision} bit")
    else:
        logger.info(f"浮点精度: {float_type}")

    out_dir = os.path.dirname(os.path.abspath(__file__))
    pub_path = os.path.join(out_dir, "toy_pub.der")

    # ── 进度条 ──
    pbar = _tqdm(total=5, desc="进度", unit="步") if HAS_TQDM else None

    def step_done():
        if pbar:
            pbar.update(1)

    # ── [1/5] 生成公钥 ──
    logger.info("[1/5] 生成公钥...")
    t0 = time.time()
    rho, s1, s2, t, A = keygen(params_name, seed=seed, params=p)
    keygen_time = time.time() - t0
    logger.info(f"  ρ = {rho.hex()}")
    logger.info(f"  A: shape={A.shape}, min={A.min()}, max={A.max()}, mean={A.mean():.1f}")
    logger.info(f"  t: shape={t.shape}, min={t.min()}, max={t.max()}, norm={np.linalg.norm(t.flatten()):.1f}")
    logger.info(f"  s1: shape={s1.shape}, min={s1.min()}, max={s1.max()}, norm={np.linalg.norm(s1.flatten()):.2f}")
    logger.info(f"  s2: shape={s2.shape}, min={s2.min()}, max={s2.max()}, norm={np.linalg.norm(s2.flatten()):.2f}")
    logger.info(f"  密钥生成耗时: {keygen_time:.3f}s")

    # Save to DER
    der_size = save_public_key(pub_path, rho, t)
    logger.info(f"  公钥已写入: {pub_path} ({der_size} bytes)")
    step_done()

    # ── [2/5] 解析公钥 ──
    logger.info("[2/5] 解析公钥...")
    t0 = time.time()
    rho_parsed, t_parsed = load_public_key(pub_path, k, n)
    parse_time = time.time() - t0
    assert rho_parsed == rho, "ρ 不匹配!"
    assert np.array_equal(t_parsed, t), "t 不匹配!"
    logger.info(f"  ρ 解析一致 ✓")
    logger.info(f"  t 解析一致 ✓")
    logger.info(f"  解析耗时: {parse_time:.4f}s")

    # Rebuild A from ρ
    A_rebuilt = expand_a(rho_parsed, k, l, n, q)
    assert np.array_equal(A_rebuilt, A), "A 重建不一致!"
    logger.info(f"  A 矩阵重建一致 ✓")

    # ── Power2Round (合并误差模式) ──
    t_attack = t       # 格基攻击目标：标准模式用 t，slack 模式用 t_recon
    s2_attack = s2     # 格基中的 s2 参考值：标准模式用 s2，slack 模式用 s2'
    if use_slack:
        adapter = ProtocolAdapter(d=d_param, q=q)
        logger.info(f"  [Power2Round] {adapter.describe()}")
        t1, t0_raw, t0_c = adapter.encode(t)
        t_recon = adapter.decode(t1)  # t1 * 2^d
        t0_norm = float(np.linalg.norm(t0_c.flatten()))
        s_norm = float(np.sqrt(np.sum(s1**2) + np.sum(s2**2)))
        logger.info(f"  [Power2Round] t1 范围: [{t1.min()}, {t1.max()}]")
        logger.info(f"  [Power2Round] t0 范围: [{t0_c.min()}, {t0_c.max()}], 范数={t0_norm:.2f}")
        logger.info(f"  [Power2Round] 秘密范数={s_norm:.2f}, 误差/秘密={t0_norm/s_norm:.1f}x")

        # 合并误差: s2' = s2 - t0_raw (mod q)
        s2_prime = (s2.astype(np.int64) - t0_raw.astype(np.int64)) % q
        s2_prime_c = s2_prime.copy()
        s2_prime_c[s2_prime_c >= q // 2] -= q
        s2_prime_norm = float(np.linalg.norm(s2_prime_c.flatten()))
        logger.info(f"  [Power2Round] s2' = s2 - t0, 范数={s2_prime_norm:.2f}")

        # 验证合并方程: A·s1 + s2' ≡ t_recon (mod q)
        lhs_merge = vec_add_mod(mat_vec_mul(A, s1, q), s2_prime.reshape(k, n), q)
        if np.array_equal(lhs_merge % q, t_recon.reshape(k, n) % q):
            logger.info("  [Power2Round] 方程 A·s1 + s2' ≡ t_recon (mod q) ✓")
        else:
            logger.error("  [Power2Round] 合并方程验证失败!")
            sys.exit(1)

        t_attack = t_recon
        s2_attack = s2_prime.reshape(k, n)

    # Verify lattice basis (用攻击目标向量验证)
    # slack 模式: 验证 A·s1 + s2' ≡ t_recon (mod q) — 已在上面验证
    # 标准模式: 验证 A·s1 + s2 ≡ t (mod q)
    basis_ok = verify_basis(A, t_attack, q, s1, s2_attack)
    logger.info(f"  格基验证: {'✓ v_target 在格中' if basis_ok else '✗ 验证失败!'}")
    if not basis_ok:
        logger.error("格基验证失败，目标向量不在格中，构造有误！")
        if pbar:
            pbar.close()
        sys.exit(1)
    step_done()

    # ── [3/5]–[5/5] 格攻击 ──
    # 参数和耗时估算
    dim = k * n + l * n + 1
    print_estimate(dim, bkz_block_size, bkz_max_loops, float_type, precision)

    # s1_real, s2_real 用于对比恢复结果
    # slack 模式下 s2_real = s2' (合并误差后的值)
    attack_result = run_attack(A, t_attack, q, s1, s2_attack,
                               bkz_block_size=bkz_block_size,
                               bkz_max_loops=bkz_max_loops,
                               bkz_threads=bkz_threads,
                               no_bkz=no_bkz,
                               lll_delta=lll_delta,
                               bkz_auto_abort=bkz_auto_abort,
                               float_type=float_type,
                               precision=precision)

    logger.info(f"  格基构造耗时: {attack_result['build_time']:.3f}s")
    logger.info(f"  LLL 耗时:     {attack_result['lll_time']:.3f}s")
    logger.info(f"  BKZ 耗时:     {attack_result['bkz_time']:.3f}s")
    if not no_bkz:
        logger.info(f"  BKZ 实际循环: {attack_result.get('bkz_loops', '?')} / {bkz_max_loops}")
    logger.info(f"  真实私钥范数: {attack_result['real_norm']:.4f}")
    step_done()
    step_done()

    # ── 三层验证 ──
    classified = classify_results(attack_result)

    logger.info(f"═══ 三层验证结果 (共 {len(classified)} 个候选) ═══")

    perfect = sum(1 for c in classified if c["verdict"] == "完美恢复私钥")
    s1_only = sum(1 for c in classified if c["verdict"] == "s1 完美恢复 (s2 不匹配)")
    alt = sum(1 for c in classified if c["verdict"] == "攻击成功，找到替代短向量")
    invalid = sum(1 for c in classified if c["verdict"] == "无效解")
    long_vec = sum(1 for c in classified if c["verdict"] == "满足方程但向量过长")

    logger.info("═══ 统计 ═══")
    logger.info(f"  完美恢复 (s1+s2): {perfect}")
    logger.info(f"  s1 完美恢复 (s2 不匹配): {s1_only}")
    logger.info(f"  替代短向量: {alt}")
    logger.info(f"  满足方程但过长: {long_vec}")
    logger.info(f"  无效解: {invalid}")
    total_time = keygen_time + parse_time + attack_result["build_time"] + attack_result["lll_time"] + attack_result["bkz_time"]
    logger.info(f"  总耗时: {total_time:.3f}s")

    # ── 最佳候选验证 ──
    best = _find_best_candidate(classified)

    if best:
        logger.info("═══ 最佳候选 ═══")
        s1p = best["s1_prime"]
        s2p = best["s2_prime"]
        logger.info(f"  s1': norm={np.linalg.norm(s1p.flatten()):.2f}, min={s1p.min()}, max={s1p.max()}")
        logger.info(f"  s2': norm={np.linalg.norm(s2p.flatten()):.2f}, min={s2p.min()}, max={s2p.max()}")
        if best.get("perfect", False):
            logger.info("  ✓ 完美恢复私钥 (s1 + s2 均匹配)")
        elif best.get("s1_perfect", False):
            logger.info("  ✓ s1 完美恢复 (签名伪造所需)")
            if use_slack:
                logger.info("  注意: s2' = s2 - t0 (非原始 s2，但不影响签名伪造)")

        # 方程验证: A·s1' + s2' ≡ t_attack (mod q)
        lhs = vec_add_mod(mat_vec_mul(A, s1p, q), s2p, q) % q
        rhs = t_attack.reshape(k, n) % q
        if np.array_equal(lhs, rhs):
            logger.info("  方程验证: A·s1' + s2' ≡ t_target (mod q) 通过 ✓")
        else:
            logger.warning("  方程验证: 失败 ✗")
    else:
        logger.info("═══ 最佳候选: 无 ═══")

    # ── 生成运行摘要 ──
    summary_path = os.path.join(out_dir, "logs", "summary.txt")
    best = _find_best_candidate(classified)

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("ML-DSA 格攻击运行摘要\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"参数集: {params_name}\n")
        f.write(f"k={k}, l={l}, n={n}, q={q}, η={eta}\n")
        if use_slack:
            f.write(f"模式: Power2Round 合并误差 (d={d_param})\n")
            f.write(f"格维度: {k*n + l*n + 1}\n")
        else:
            f.write(f"模式: 标准 Kannan 嵌入\n")
            f.write(f"格维度: {k*n + l*n + 1}\n")
        f.write(f"BKZ: block_size={bkz_block_size}, max_loops={bkz_max_loops}, threads={bkz_threads}\n")
        f.write(f"BKZ 跳过: {no_bkz}\n")
        if float_type == "mpfr":
            f.write(f"浮点精度: {float_type}, precision={precision} bit\n\n")
        else:
            f.write(f"浮点精度: {float_type}\n\n")
        f.write("耗时:\n")
        f.write(f"  密钥生成: {keygen_time:.3f}s\n")
        f.write(f"  公钥解析: {parse_time:.4f}s\n")
        f.write(f"  格基构造: {attack_result['build_time']:.3f}s\n")
        f.write(f"  LLL:      {attack_result['lll_time']:.3f}s\n")
        f.write(f"  BKZ:      {attack_result['bkz_time']:.3f}s\n")
        f.write(f"  总耗时:   {total_time:.3f}s\n\n")
        f.write("候选统计:\n")
        f.write(f"  完美恢复 (s1+s2): {perfect}\n")
        f.write(f"  s1 完美恢复:      {s1_only}\n")
        f.write(f"  替代短向量:     {alt}\n")
        f.write(f"  满足方程但过长: {long_vec}\n")
        f.write(f"  无效解:         {invalid}\n\n")
        f.write(f"真实私钥范数: {attack_result['real_norm']:.4f}\n")
        if best:
            f.write(f"最佳候选范数: {best['cand_norm']:.4f} (完美: {best['perfect']})\n")
        else:
            f.write("最佳候选: 无\n")
    logger.info(f"  摘要已写入: {summary_path}")
    logger.info("完成。")
    if pbar:
        pbar.close()


if __name__ == "__main__":
    main()
