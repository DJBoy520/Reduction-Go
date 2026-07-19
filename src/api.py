"""
ML-DSA 格攻击 API 层。

提供程序化接口，供 CLI (main.py) 和外部 UI 共用。

设计原则:
  - 接收 dict 参数，不依赖 argparse
  - 返回结构化结果 (dict)，不只写日志
  - 用异常替代 sys.exit()
  - 支持可选的进度回调函数
  - 库代码不做日志全局配置
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

from .utils.params import get_params, get_d
from .keys.keygen import keygen, expand_a
from .keys.pubkey import save_public_key, load_public_key
from .lattice_attack import run_attack as _lattice_run_attack, classify_results, verify_basis
from .protocol.power2round import ProtocolAdapter
from .poly_math import mat_vec_mul, vec_add_mod
from .keys.cert_parser import parse_certificate
from .progress import print_estimate

logger = logging.getLogger(__name__)

# ── 类型定义 ─────────────────────────────────────────────────────────────────

ProgressFn = Optional[Callable[[str, dict], None]]
"""进度回调函数签名: (step_name, details_dict) -> None"""


@dataclass
class AttackConfig:
    """格攻击配置参数。"""
    # 基本参数
    params_name: str = "toy"
    k: Optional[int] = None
    l: Optional[int] = None
    n: Optional[int] = None

    # LLL / BKZ
    no_bkz: bool = False
    bkz_block_size: Optional[int] = None
    bkz_max_loops: Optional[int] = None
    bkz_threads: Optional[int] = None  # 预留：纯 Python BKZ 暂不支持多线程
    bkz_auto_abort: bool = False
    lll_delta: float = 0.79

    # 随机种子
    seed: Optional[int] = None

    # 浮点精度
    float_type: Optional[str] = None
    precision: Optional[int] = None

    # Power2Round
    use_slack: bool = False
    d: Optional[int] = None

    # 证书模式
    cert_path: Optional[str] = None
    toy_params: bool = False

    # 输出目录
    output_dir: Optional[str] = None

    # 精度控制
    auto_precision: bool = True
    mp_dps: Optional[int] = None

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class AttackResult:
    """格攻击结构化结果。"""
    # 基本信息
    params_name: str
    k: int
    l: int
    n: int
    q: int
    eta: Optional[int] = None
    d: int = 13
    dim: int = 0
    use_slack: bool = False

    # 耗时
    keygen_time: float = 0.0
    parse_time: float = 0.0
    build_time: float = 0.0
    lll_time: float = 0.0
    bkz_time: float = 0.0
    total_time: float = 0.0

    # 攻击结果
    candidates: list[dict] = field(default_factory=list)
    classified: list[dict] = field(default_factory=list)
    real_norm: float = 0.0

    # 统计
    perfect_count: int = 0
    s1_perfect_count: int = 0
    alt_count: int = 0
    long_count: int = 0
    invalid_count: int = 0

    # 最佳候选
    best: Optional[dict] = None

    # BKZ 信息
    bkz_loops: int = 0
    no_bkz: bool = False
    bkz_block_size: int = 0
    bkz_max_loops: int = 0
    bkz_threads: int = 6

    # 浮点精度
    float_type: str = ""
    precision: int = 0

    # 证书模式额外信息
    cert_name: Optional[str] = None

    def to_dict(self) -> dict:
        """转为可 JSON 序列化的 dict。"""
        d = {}
        for k, v in self.__dict__.items():
            if isinstance(v, np.ndarray):
                d[k] = v.tolist()
            elif isinstance(v, list) and v and isinstance(v[0], dict):
                d[k] = []
                for item in v:
                    item_d = {}
                    for ik, iv in item.items():
                        if isinstance(iv, np.ndarray):
                            item_d[ik] = iv.tolist()
                        else:
                            item_d[ik] = iv
                    d[k].append(item_d)
            else:
                d[k] = v
        return d


# ── 辅助函数 ─────────────────────────────────────────────────────────────────

def find_best_candidate(classified: list[dict]) -> dict | None:
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


def _resolve_config(config: AttackConfig) -> dict:
    """将 AttackConfig 解析为完整的参数 dict（合并默认值和覆盖值）。"""
    p = get_params(config.params_name)

    # CLI 覆盖
    if config.k is not None:
        p["k"] = config.k
    if config.l is not None:
        p["l"] = config.l
    if config.n is not None:
        p["n"] = config.n

    # 攻击参数（CLI > 配置 > 默认值）
    bkz_block_size = config.bkz_block_size if config.bkz_block_size is not None else p.get("bkz_block_size", 20)
    bkz_max_loops = config.bkz_max_loops if config.bkz_max_loops is not None else p.get("bkz_max_loops", 8)
    bkz_threads = config.bkz_threads if config.bkz_threads is not None else p.get("bkz_threads", 6)
    no_bkz = config.no_bkz or not p.get("use_bkz", True)
    bkz_auto_abort = config.bkz_auto_abort or p.get("auto_abort", False)
    float_type = config.float_type if config.float_type is not None else p.get("float_type", "mpfr")
    precision = config.precision if config.precision is not None else p.get("precision", 200)
    d_param = config.d if config.d is not None else get_d(config.params_name)
    mp_dps = config.mp_dps if config.mp_dps is not None else p.get("mp_dps_default", 100)

    seed_bytes = None
    if config.seed is not None:
        seed_bytes = config.seed.to_bytes(8, "big").ljust(32, b"\x00")

    return {
        "params": p,
        "k": p["k"], "l": p["l"], "n": p["n"],
        "q": p["q"], "eta": p["eta"],
        "bkz_block_size": bkz_block_size,
        "bkz_max_loops": bkz_max_loops,
        "bkz_threads": bkz_threads,
        "no_bkz": no_bkz,
        "bkz_auto_abort": bkz_auto_abort,
        "lll_delta": config.lll_delta,
        "float_type": float_type,
        "precision": precision,
        "d": d_param,
        "use_slack": config.use_slack,
        "seed_bytes": seed_bytes,
        "mp_dps": mp_dps,
    }


# ── 合成密钥攻击 ─────────────────────────────────────────────────────────────

def run_synthetic_attack(
    config: AttackConfig,
    progress: ProgressFn = None,
) -> AttackResult:
    """执行合成密钥格攻击（从 keygen 开始的完整闭环）。

    Args:
        config: 攻击配置
        progress: 可选的进度回调 (step_name, details) -> None

    Returns:
        AttackResult 结构化结果

    Raises:
        ValueError: 参数错误或格基验证失败
    """
    cfg = _resolve_config(config)
    k, l, n, q, eta = cfg["k"], cfg["l"], cfg["n"], cfg["q"], cfg["eta"]
    d_param = cfg["d"]
    use_slack = cfg["use_slack"]

    logger.info(f"═══ ML-DSA 格攻击闭环测试 (参数集: {config.params_name}) ═══")
    logger.info(f"参数: k={k}, l={l}, n={n}, q={q}, η={eta}")
    logger.info(f"格维度: {k*n + l*n + 1}")
    if progress:
        progress("init", {"k": k, "l": l, "n": n, "q": q, "dim": k*n + l*n + 1})

    out_dir = config.output_dir or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # api.py 在 src/ 下，上两级 = 项目根目录
    # 确保和 main.py 的 os.path.dirname(os.path.abspath(__file__)) 一致
    pub_path = os.path.join(out_dir, "certs", "toy_pub.der")

    # ── [1/5] 生成公钥 ──
    logger.info("[1/5] 生成公钥...")
    t0 = time.time()
    rho, s1, s2, t, A = keygen(config.params_name, seed=cfg["seed_bytes"], params=cfg["params"])
    keygen_time = time.time() - t0
    logger.info(f"  密钥生成耗时: {keygen_time:.3f}s")
    if progress:
        progress("keygen", {"time": keygen_time, "rho": rho.hex()})

    # Power2Round: 保存 t1
    adapter_save = ProtocolAdapter(d=d_param, q=q)
    t1_save, _, _ = adapter_save.encode(t)
    save_public_key(pub_path, rho, t1_save, d=d_param, q=q)
    logger.info(f"  公钥已写入: {pub_path}")

    # ── [2/5] 解析公钥 ──
    logger.info("[2/5] 解析公钥...")
    t0 = time.time()
    rho_parsed, t1_parsed = load_public_key(pub_path, k, n, d=d_param, q=q)
    parse_time = time.time() - t0

    assert rho_parsed == rho, "ρ 不匹配!"
    assert np.array_equal(t1_parsed, t1_save), "t1 不匹配!"
    logger.info(f"  ρ, t1 解析一致 ✓")

    A_rebuilt = expand_a(rho_parsed, k, l, n, q)
    assert np.array_equal(A_rebuilt, A), "A 重建不一致!"
    logger.info(f"  A 矩阵重建一致 ✓")
    if progress:
        progress("parse", {"time": parse_time})

    # ── Power2Round (合并误差模式) ──
    t_attack = t
    s2_attack = s2
    if use_slack:
        adapter = ProtocolAdapter(d=d_param, q=q)
        logger.info(f"  [Power2Round] {adapter.describe()}")
        t1, t0_raw, t0_c = adapter.encode(t)
        t_recon = adapter.decode(t1)
        t0_norm = float(np.linalg.norm(t0_c.flatten()))
        s_norm = float(np.sqrt(np.sum(s1**2) + np.sum(s2**2)))
        logger.info(f"  [Power2Round] 秘密范数={s_norm:.2f}, 误差/秘密={t0_norm/s_norm:.1f}x")

        # 合并误差: s2' = s2 - t0_raw (mod q)
        s2_prime = (s2.astype(np.int64) - t0_raw.astype(np.int64)) % q
        s2_prime_c = s2_prime.copy()
        s2_prime_c[s2_prime_c >= q // 2] -= q
        s2_prime_norm = float(np.linalg.norm(s2_prime_c.flatten()))
        logger.info(f"  [Power2Round] s2' = s2 - t0, 范数={s2_prime_norm:.2f}")

        # 验证合并方程
        lhs_merge = vec_add_mod(mat_vec_mul(A, s1, q), s2_prime.reshape(k, n), q)
        if not np.array_equal(lhs_merge % q, t_recon.reshape(k, n) % q):
            raise ValueError("Power2Round 合并方程验证失败: A·s1 + s2' ≡ t_recon (mod q) 不成立")
        logger.info("  [Power2Round] 合并方程验证 ✓")

        t_attack = t_recon
        s2_attack = s2_prime.reshape(k, n)

    # 格基验证
    basis_result = verify_basis(A, t_attack, q, s1, s2_attack)
    if not basis_result.passed:
        raise ValueError(
            f"格基验证失败: {basis_result.error or '目标向量不在格中'}"
        )
    logger.info(f"  格基验证: ✓ v_target 在格中")

    # ── [3/5]–[5/5] 格攻击 ──
    dim = k * n + l * n + 1
    print_estimate(dim, cfg["bkz_block_size"], cfg["bkz_max_loops"],
                   cfg["float_type"], cfg["precision"], cfg["mp_dps"])
    if progress:
        progress("attack_start", {"dim": dim})

    attack_result = _lattice_run_attack(
        A, t_attack, q, s1, s2_attack,
        bkz_block_size=cfg["bkz_block_size"],
        bkz_max_loops=cfg["bkz_max_loops"],
        bkz_threads=cfg["bkz_threads"],
        no_bkz=cfg["no_bkz"],
        lll_delta=cfg["lll_delta"],
        bkz_auto_abort=cfg["bkz_auto_abort"],
        float_type=cfg["float_type"],
        precision=cfg["precision"],
        auto_precision=config.auto_precision,
        mp_dps=cfg["mp_dps"],
    )

    logger.info(f"  格基构造: {attack_result['build_time']:.3f}s")
    logger.info(f"  LLL:      {attack_result['lll_time']:.3f}s")
    logger.info(f"  BKZ:      {attack_result['bkz_time']:.3f}s")
    logger.info(f"  真实私钥范数: {attack_result['real_norm']:.4f}")
    if progress:
        progress("attack_done", {
            "build_time": attack_result["build_time"],
            "lll_time": attack_result["lll_time"],
            "bkz_time": attack_result["bkz_time"],
            "real_norm": attack_result["real_norm"],
        })

    # ── 三层验证 ──
    classified = classify_results(attack_result)
    best = find_best_candidate(classified)

    perfect = sum(1 for c in classified if c["verdict"] == "完美恢复私钥")
    s1_only = sum(1 for c in classified if c["verdict"] == "s1 完美恢复 (s2 不匹配)")
    alt = sum(1 for c in classified if c["verdict"] == "攻击成功，找到替代短向量")
    invalid = sum(1 for c in classified if c["verdict"] == "无效解")
    long_vec = sum(1 for c in classified if c["verdict"] == "满足方程但向量过长")

    total_time = keygen_time + parse_time + attack_result["build_time"] + attack_result["lll_time"] + attack_result["bkz_time"]

    logger.info(f"═══ 三层验证结果 (共 {len(classified)} 个候选) ═══")
    logger.info(f"  完美恢复: {perfect}, s1完美: {s1_only}, 替代: {alt}, 过长: {long_vec}, 无效: {invalid}")
    logger.info(f"  总耗时: {total_time:.3f}s")

    if best:
        s1p = best["s1_prime"]
        s2p = best["s2_prime"]
        logger.info(f"═══ 最佳候选 ═══")
        logger.info(f"  s1': norm={np.linalg.norm(s1p.flatten()):.2f}")
        logger.info(f"  s2': norm={np.linalg.norm(s2p.flatten()):.2f}")
        if best.get("perfect", False):
            logger.info("  ✓ 完美恢复私钥")
        elif best.get("s1_perfect", False):
            logger.info("  ✓ s1 完美恢复")
    else:
        logger.info("═══ 最佳候选: 无 ═══")

    if progress:
        progress("done", {"total_time": total_time})

    return AttackResult(
        params_name=config.params_name,
        k=k, l=l, n=n, q=q, eta=eta, d=d_param,
        dim=dim, use_slack=use_slack,
        keygen_time=keygen_time, parse_time=parse_time,
        build_time=attack_result["build_time"],
        lll_time=attack_result["lll_time"],
        bkz_time=attack_result["bkz_time"],
        total_time=total_time,
        candidates=attack_result["candidates"],
        classified=classified,
        real_norm=attack_result["real_norm"],
        perfect_count=perfect, s1_perfect_count=s1_only,
        alt_count=alt, long_count=long_vec, invalid_count=invalid,
        best=best,
        bkz_loops=attack_result.get("bkz_loops", 0),
        no_bkz=cfg["no_bkz"],
        bkz_block_size=cfg["bkz_block_size"],
        bkz_max_loops=cfg["bkz_max_loops"],
        bkz_threads=cfg["bkz_threads"],
        float_type=cfg["float_type"],
        precision=cfg["precision"],
    )


# ── 证书攻击 ─────────────────────────────────────────────────────────────────

def run_cert_attack(
    config: AttackConfig,
    progress: ProgressFn = None,
) -> AttackResult:
    """从证书文件提取公钥并执行格攻击。

    Args:
        config: 攻击配置 (cert_path 必须设置)
        progress: 可选的进度回调

    Returns:
        AttackResult 结构化结果

    Raises:
        FileNotFoundError: 证书文件不存在
        ValueError: 解析失败或参数错误
    """
    if not config.cert_path:
        raise ValueError("cert_path 未设置")

    cert_path = config.cert_path
    if not os.path.exists(cert_path):
        raise FileNotFoundError(f"证书文件不存在: {cert_path}")

    # 解析证书
    rho, t1, cert_params = parse_certificate(cert_path, use_toy=config.toy_params)
    k, l, n = cert_params["k"], cert_params["l"], cert_params["n"]
    d_param = cert_params["d"]
    mldsa_name = cert_params["name"]
    q = cert_params.get("q", 8380417)

    logger.info(f"═══ ML-DSA 证书攻击 ({mldsa_name}) ═══")
    logger.info(f"参数: k={k}, l={l}, n={n}, q={q}, d={d_param}")
    logger.info(f"格维度: {k*n + l*n + 1}")
    if progress:
        progress("cert_parsed", {"name": mldsa_name, "k": k, "l": l, "n": n, "dim": k*n + l*n + 1})

    # 重建 A 矩阵
    A = expand_a(rho, k, l, n, q)
    logger.info(f"  A 矩阵: shape={A.shape}")

    # Power2Round: t1 → t_recon
    adapter = ProtocolAdapter(d=d_param, q=q)
    t_recon = adapter.decode(t1)
    logger.info(f"  t_recon: norm={float(np.linalg.norm(t_recon.astype(np.float64).flatten())):.1f}")

    # 攻击参数
    p = get_params(config.params_name)
    bkz_block_size = config.bkz_block_size if config.bkz_block_size is not None else p.get("bkz_block_size", 20)
    bkz_max_loops = config.bkz_max_loops if config.bkz_max_loops is not None else p.get("bkz_max_loops", 8)
    bkz_threads = config.bkz_threads if config.bkz_threads is not None else p.get("bkz_threads", 6)
    no_bkz = config.no_bkz
    bkz_auto_abort = config.bkz_auto_abort
    float_type = config.float_type if config.float_type is not None else p.get("float_type", "mpfr")
    precision = config.precision if config.precision is not None else p.get("precision", 200)
    mp_dps = config.mp_dps if config.mp_dps is not None else p.get("mp_dps_default", 100)

    logger.info(f"BKZ: block_size={bkz_block_size}, max_loops={bkz_max_loops}")

    # 无私钥验证: 用零占位
    s1_dummy = np.zeros((l, n), dtype=np.int64)
    s2_dummy = np.zeros((k, n), dtype=np.int64)

    dim = k * n + l * n + 1
    print_estimate(dim, bkz_block_size, bkz_max_loops, float_type, precision, mp_dps)
    if progress:
        progress("attack_start", {"dim": dim})

    attack_result = _lattice_run_attack(
        A, t_recon, q, s1_dummy, s2_dummy,
        bkz_block_size=bkz_block_size,
        bkz_max_loops=bkz_max_loops,
        bkz_threads=bkz_threads,
        no_bkz=no_bkz,
        lll_delta=config.lll_delta,
        bkz_auto_abort=bkz_auto_abort,
        float_type=float_type,
        precision=precision,
        auto_precision=config.auto_precision,
        mp_dps=mp_dps,
    )

    logger.info(f"  LLL: {attack_result['lll_time']:.2f}s, BKZ: {attack_result['bkz_time']:.2f}s")
    logger.info(f"  候选数: {len(attack_result['candidates'])}")

    # 方程满足统计
    eq_count = sum(1 for c in attack_result["candidates"] if c["eq_holds"])
    short_count = sum(1 for c in attack_result["candidates"] if c["cand_norm"] < 1000)
    logger.info(f"  方程满足: {eq_count}, 短向量: {short_count}")

    # 分类
    classified = classify_results(attack_result)
    best = find_best_candidate(classified)

    total_time = attack_result["build_time"] + attack_result["lll_time"] + attack_result["bkz_time"]

    if best:
        s1p = best["s1_prime"]
        s2p = best["s2_prime"]
        logger.info(f"═══ 最佳候选 ═══")
        logger.info(f"  s1': norm={np.linalg.norm(s1p.flatten()):.2f}")
        logger.info(f"  s2': norm={np.linalg.norm(s2p.flatten()):.2f}")

        # 方程验证
        lhs = vec_add_mod(mat_vec_mul(A, s1p, q), s2p, q) % q
        rhs = t_recon.reshape(k, n) % q
        if np.array_equal(lhs, rhs):
            logger.info("  方程验证: ✓")
        else:
            logger.warning("  方程验证: 失败 ✗")
    else:
        logger.info("═══ 最佳候选: 无 ═══")

    if progress:
        progress("done", {"total_time": total_time})

    perfect = sum(1 for c in classified if c["verdict"] == "完美恢复私钥")
    s1_only = sum(1 for c in classified if c["verdict"] == "s1 完美恢复 (s2 不匹配)")
    alt = sum(1 for c in classified if c["verdict"] == "攻击成功，找到替代短向量")
    invalid = sum(1 for c in classified if c["verdict"] == "无效解")
    long_vec = sum(1 for c in classified if c["verdict"] == "满足方程但向量过长")

    return AttackResult(
        params_name=mldsa_name,
        k=k, l=l, n=n, q=q, eta=None, d=d_param,
        dim=dim, use_slack=True,
        build_time=attack_result["build_time"],
        lll_time=attack_result["lll_time"],
        bkz_time=attack_result["bkz_time"],
        total_time=total_time,
        candidates=attack_result["candidates"],
        classified=classified,
        real_norm=attack_result["real_norm"],
        perfect_count=perfect, s1_perfect_count=s1_only,
        alt_count=alt, long_count=long_vec, invalid_count=invalid,
        best=best,
        bkz_loops=attack_result.get("bkz_loops", 0),
        no_bkz=no_bkz,
        bkz_block_size=bkz_block_size,
        bkz_max_loops=bkz_max_loops,
        bkz_threads=bkz_threads,
        float_type=float_type,
        precision=precision,
        cert_name=mldsa_name,
    )


# ── 统一入口 ─────────────────────────────────────────────────────────────────

def run_attack(
    config: AttackConfig,
    progress: ProgressFn = None,
) -> AttackResult:
    """统一入口：根据 config.cert_path 自动选择攻击模式。

    Args:
        config: 攻击配置
        progress: 可选的进度回调

    Returns:
        AttackResult 结构化结果
    """
    if config.cert_path:
        return run_cert_attack(config, progress)
    return run_synthetic_attack(config, progress)
