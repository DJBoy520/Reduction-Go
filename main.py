#!/usr/bin/env python3
"""
ML-DSA 格攻击闭环测试环境 — CLI 入口。

流程：
  [1/5] 生成公钥
  [2/5] 解析公钥
  [3/5] 构造格基
  [4/5] LLL 约减
  [5/5] BKZ 约减 + 三层验证

本文件只负责 CLI 参数解析、日志配置和结果输出。
核心逻辑在 src/api.py 中，可供外部 UI 程序化调用。
"""

import argparse
import logging
import os
import sys

import numpy as np

from src.api import AttackConfig, run_attack
from src.utils.logger import setup_logging

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
        "--lll-delta", dest="lll_delta", type=float, default=0.79,
        help="LLL 约减质量参数 delta (默认: 0.79)"
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
        help="Power2Round 的 d 参数 (低位比特数)。ML-DSA-44/65/87:13"
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


def _print_result(result):
    """打印结构化攻击结果。"""
    logger.info(f"═══ 攻击结果汇总 ═══")
    logger.info(f"  参数: {result.params_name}, k={result.k}, l={result.l}, n={result.n}")
    logger.info(f"  格维度: {result.dim}")
    logger.info(f"  耗时: keygen={result.keygen_time:.3f}s, parse={result.parse_time:.3f}s, "
                f"build={result.build_time:.3f}s, LLL={result.lll_time:.3f}s, BKZ={result.bkz_time:.3f}s")
    logger.info(f"  总耗时: {result.total_time:.3f}s")
    logger.info(f"  候选统计: 完美={result.perfect_count}, s1完美={result.s1_perfect_count}, "
                f"替代={result.alt_count}, 过长={result.long_count}, 无效={result.invalid_count}")

    if result.best:
        s1p = result.best["s1_prime"]
        s2p = result.best["s2_prime"]
        logger.info(f"  最佳候选: s1 norm={np.linalg.norm(s1p.flatten()):.2f}, "
                    f"s2 norm={np.linalg.norm(s2p.flatten()):.2f}, "
                    f"perfect={result.best.get('perfect', False)}")
    else:
        logger.info(f"  最佳候选: 无")


def _write_summary(result, out_dir):
    """将攻击摘要写入 logs/summary.txt。"""
    summary_path = os.path.join(out_dir, "logs", "summary.txt")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("ML-DSA 格攻击运行摘要\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"参数集: {result.params_name}\n")
        f.write(f"k={result.k}, l={result.l}, n={result.n}, q={result.q}, η={result.eta}\n")
        if result.use_slack:
            f.write(f"模式: Power2Round 合并误差 (d={result.d})\n")
        else:
            f.write(f"模式: 标准 Kannan 嵌入\n")
        f.write(f"格维度: {result.dim}\n")
        f.write(f"BKZ: block_size={result.bkz_block_size}, max_loops={result.bkz_max_loops}, threads={result.bkz_threads}\n")
        f.write(f"BKZ 跳过: {result.no_bkz}\n")
        if result.float_type == "mpfr":
            f.write(f"浮点精度: {result.float_type}, precision={result.precision} bit\n\n")
        else:
            f.write(f"浮点精度: {result.float_type}\n\n")
        f.write("耗时:\n")
        f.write(f"  密钥生成: {result.keygen_time:.3f}s\n")
        f.write(f"  公钥解析: {result.parse_time:.4f}s\n")
        f.write(f"  格基构造: {result.build_time:.3f}s\n")
        f.write(f"  LLL:      {result.lll_time:.3f}s\n")
        f.write(f"  BKZ:      {result.bkz_time:.3f}s\n")
        f.write(f"  总耗时:   {result.total_time:.3f}s\n\n")
        f.write("候选统计:\n")
        f.write(f"  完美恢复 (s1+s2): {result.perfect_count}\n")
        f.write(f"  s1 完美恢复:      {result.s1_perfect_count}\n")
        f.write(f"  替代短向量:     {result.alt_count}\n")
        f.write(f"  满足方程但过长: {result.long_count}\n")
        f.write(f"  无效解:         {result.invalid_count}\n\n")
        f.write(f"真实私钥范数: {result.real_norm:.4f}\n")
        if result.best:
            f.write(f"最佳候选范数: {result.best['cand_norm']:.4f} (完美: {result.best['perfect']})\n")
        else:
            f.write("最佳候选: 无\n")

    logger.info(f"  摘要已写入: {summary_path}")


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

    # ── 证书攻击模式提示 ──
    if args.cert and args.slack:
        logger.warning("--cert 模式下 --slack 被忽略 (证书模式天然使用 Power2Round)")

    # ── 构建 AttackConfig ──
    out_dir = os.path.dirname(os.path.abspath(__file__))

    config = AttackConfig(
        params_name=args.params,
        k=args.k, l=args.l, n=args.n,
        no_bkz=args.no_bkz,
        bkz_block_size=args.bkz_block_size,
        bkz_max_loops=args.bkz_max_loops,
        bkz_auto_abort=args.bkz_auto_abort,
        lll_delta=args.lll_delta,
        seed=args.seed,
        float_type=args.float_type,
        precision=args.precision,
        use_slack=args.slack,
        d=args.d,
        cert_path=args.cert,
        toy_params=args.toy_params,
        output_dir=out_dir,
    )

    # ── 执行攻击 ──
    try:
        result = run_attack(config)
    except Exception as e:
        logger.error(f"攻击失败: {e}")
        sys.exit(1)

    # ── 输出结果 ──
    _print_result(result)
    _write_summary(result, out_dir)

    logger.info("完成。")


if __name__ == "__main__":
    main()
