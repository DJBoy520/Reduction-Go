#!/usr/bin/env python3
"""ML-DSA 格攻击闭环测试环境 — CLI 入口。"""

import argparse
import logging
import os
import sys

import numpy as np

from src.api import AttackConfig, run_attack
from src.utils.logger import setup_logging

logger = logging.getLogger(__name__)

EXAMPLES = """\
示例:
  # 秒级验证（最小参数）
  python3 main.py toy --no-bkz --n 10 --seed 42

  # 标准 toy 测试（LLL）
  python3 main.py toy --no-bkz

  # 带 BKZ 的完整攻击
  python3 main.py toy --bkz-block-size 5 --bkz-max-loops 1 --seed 42

  # 强制指定 mpmath 精度
  python3 main.py toy --no-bkz --mp-dps 150

  # 从证书攻击
  python3 main.py --cert certs/test.pem --toy-params --no-bkz

  # 自定义维度
  python3 main.py toy --k 3 --l 3 --n 20 --no-bkz --seed 42
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description="ML-DSA 格攻击闭环测试环境",
        epilog=EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("params", nargs="?", default="toy",
                        help="参数集名称 (默认: toy)")
    parser.add_argument("--verbose", action="store_true",
                        help="控制台输出 DEBUG 级别日志")
    parser.add_argument("--log-level", dest="log_level",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="自定义日志级别 (覆盖 --verbose)")
    parser.add_argument("--no-bkz", dest="no_bkz", action="store_true",
                        help="跳过 BKZ，只运行 LLL")
    parser.add_argument("--bkz-block-size", dest="bkz_block_size", type=int, default=None,
                        help="BKZ 块大小")
    parser.add_argument("--bkz-max-loops", dest="bkz_max_loops", type=int, default=None,
                        help="BKZ 最大循环数")
    parser.add_argument("--bkz-auto-abort", dest="bkz_auto_abort", action="store_true",
                        help="BKZ 连续无改善时提前终止")
    parser.add_argument("--k", type=int, default=None, help="矩阵 A 行数 k")
    parser.add_argument("--l", type=int, default=None, help="矩阵 A 列数 l")
    parser.add_argument("--n", type=int, default=None, help="多项式维度 n")
    parser.add_argument("--lll-delta", dest="lll_delta", type=float, default=0.79,
                        help="LLL δ 参数 (默认: 0.79)")
    parser.add_argument("--seed", type=int, default=None,
                        help="随机种子（便于复现）")
    parser.add_argument("--slack", action="store_true",
                        help="Power2Round 合并误差模式")
    parser.add_argument("--d", type=int, default=None,
                        help="Power2Round d 参数 (默认: 13)")
    parser.add_argument("--cert", type=str, default=None,
                        help="证书文件路径 (PEM/DER)")
    parser.add_argument("--toy-params", dest="toy_params", action="store_true",
                        help="用 toy 参数集解析证书")
    parser.add_argument("--mp-dps", dest="mp_dps", type=int, default=None,
                        help="强制指定 mpmath 精度位数（覆盖自动分层）")
    parser.add_argument("--no-auto-precision", dest="auto_precision",
                        action="store_false", default=True,
                        help="关闭自适应精度升级，失败直接报错")
    return parser.parse_args()


def _print_result(result):
    """打印攻击结果。"""
    logger.info("═══ 攻击结果汇总 ═══")
    logger.info(f"  参数: {result.params_name}, k={result.k}, l={result.l}, n={result.n}")
    logger.info(f"  格维度: {result.dim}")
    logger.info(f"  耗时: keygen={result.keygen_time:.3f}s, parse={result.parse_time:.3f}s, "
                f"build={result.build_time:.3f}s, LLL={result.lll_time:.3f}s, BKZ={result.bkz_time:.3f}s")
    logger.info(f"  总耗时: {result.total_time:.3f}s")
    logger.info(f"  候选: 完美={result.perfect_count}, s1完美={result.s1_perfect_count}, "
                f"替代={result.alt_count}, 过长={result.long_count}, 无效={result.invalid_count}")
    if result.best:
        s1p = result.best["s1_prime"]
        s2p = result.best["s2_prime"]
        logger.info(f"  最佳: s1 norm={np.linalg.norm(s1p.flatten()):.2f}, "
                    f"s2 norm={np.linalg.norm(s2p.flatten()):.2f}, "
                    f"perfect={result.best.get('perfect', False)}")
    else:
        logger.info("  最佳候选: 无")


def _write_summary(result, out_dir):
    """写摘要到 logs/summary.txt。"""
    summary_path = os.path.join(out_dir, "logs", "summary.txt")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("ML-DSA 格攻击运行摘要\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"参数集: {result.params_name}\n")
        f.write(f"k={result.k}, l={result.l}, n={result.n}, q={result.q}, η={result.eta}\n")
        mode = f"Power2Round (d={result.d})" if result.use_slack else "标准 Kannan 嵌入"
        f.write(f"模式: {mode}\n")
        f.write(f"格维度: {result.dim}\n")
        f.write(f"BKZ: block={result.bkz_block_size}, loops={result.bkz_max_loops}\n")
        f.write(f"精度引擎: mpmath (dps={result.precision})\n\n")
        f.write("耗时:\n")
        f.write(f"  密钥生成: {result.keygen_time:.3f}s\n")
        f.write(f"  公钥解析: {result.parse_time:.4f}s\n")
        f.write(f"  格基构造: {result.build_time:.3f}s\n")
        f.write(f"  LLL:      {result.lll_time:.3f}s\n")
        f.write(f"  BKZ:      {result.bkz_time:.3f}s\n")
        f.write(f"  总耗时:   {result.total_time:.3f}s\n\n")
        f.write("候选统计:\n")
        f.write(f"  完美恢复: {result.perfect_count}\n")
        f.write(f"  s1 完美:  {result.s1_perfect_count}\n")
        f.write(f"  替代:     {result.alt_count}\n")
        f.write(f"  过长:     {result.long_count}\n")
        f.write(f"  无效:     {result.invalid_count}\n\n")
        f.write(f"真实私钥范数: {result.real_norm:.4f}\n")
        if result.best:
            f.write(f"最佳候选范数: {result.best['cand_norm']:.4f} (完美: {result.best['perfect']})\n")
        else:
            f.write("最佳候选: 无\n")
    logger.info(f"  摘要已写入: {summary_path}")


def main():
    args = parse_args()

    if args.log_level:
        console_level = getattr(logging, args.log_level)
    elif args.verbose:
        console_level = logging.DEBUG
    else:
        console_level = logging.INFO

    setup_logging(console_level=console_level)

    if args.cert and args.slack:
        logger.warning("--cert 模式下 --slack 被忽略")

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
        use_slack=args.slack,
        d=args.d,
        cert_path=args.cert,
        toy_params=args.toy_params,
        output_dir=out_dir,
        auto_precision=args.auto_precision,
        mp_dps=args.mp_dps,
    )

    try:
        result = run_attack(config)
    except Exception as e:
        logger.error(f"攻击失败: {e}")
        sys.exit(1)

    _print_result(result)
    _write_summary(result, out_dir)
    logger.info("完成。")


if __name__ == "__main__":
    main()
