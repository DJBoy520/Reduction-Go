"""ML-DSA 参数配置模块 — 唯一定义源。

本模块是项目的最底层模块，零外部依赖。
所有 ML-DSA 元数据（k/l/n/q/d/eta/OID/BKZ 参数）从此模块引用。

合并来源:
  - src/utils/params.py (去除对 lattice.base 的循环依赖)
  - src/common/params.py
  - src/domain/constants.py 的常量

定义层次:
  1. MLDSA_REGISTRY — FIPS 204 标准参数集（含 OID、d、签名参数等完整元数据）
  2. PARAMS — 测试用参数集（easy/medium/hard/extreme，维度小，用于实验）
  3. get_params() — 统一入口，toy 是 easy 的别名
"""

from __future__ import annotations

from typing import Any

# ── 公共常量 ─────────────────────────────────────────────────────────────────

MLDSA_Q: int = 8380417  # 2^23 - 2^13 + 1

# ── mpmath 精度配置 ──────────────────────────────────────────────────────────
# 来源: 原 src/common/params.py 与 src/lattice/base/precision_constants.py
MP_DPS_LOW: int = 80      # 低维度默认精度（dim ≤ 150）
MP_DPS_HIGH: int = 200    # 高维度默认精度（dim > 250）
MP_STUCK_THRESHOLD: int = 100  # 连续无更新迭代上限
MP_MAX_RETRY: int = 1     # 精度升级最大重试次数

# ── 自动精度分层阈值 ─────────────────────────────────────────────────────────
# 来源: 原 src/lattice/base/precision_constants.py
AUTO_PRECISION_DIM_LOW: int = 150   # dim ≤ 此值用低精度
AUTO_PRECISION_DIM_HIGH: int = 250  # dim > 此值用高精度

# ── 精度模式枚举 ─────────────────────────────────────────────────────────────
PRECISION_MODE_LOW: int = 0   # 强制低精度
PRECISION_MODE_AUTO: int = 1  # 按维度自动选择
PRECISION_MODE_HIGH: int = 2  # 强制高精度

# ── LLL 参数 ─────────────────────────────────────────────────────────────────
LOVASZ_CONDITION_PARAM: float = 0.79  # δ ∈ (0.25, 1.0)
LLL_MP_DEFAULT_DPS: int = 50          # mpmath 默认精度（LLL 入口）


# ── FIPS 204 标准 ML-DSA 参数集 ──────────────────────────────────────────────

MLDSA_REGISTRY: dict[str, dict[str, Any]] = {
    "ML-DSA-44": {
        # 基本维度
        "k": 4, "l": 4, "n": 256,
        "q": MLDSA_Q, "eta": 2,
        # Power2Round
        "d": 13,
        # 签名参数 (FIPS 204)
        "tau": 39, "gamma1": (1 << 17), "gamma2": (MLDSA_Q - 1) // 32, "omega": 80,
        # X.509 OID
        "oid": "2.16.840.1.101.3.4.3.17",
        # BKZ 默认参数
        "bkz_block_size": 25, "bkz_max_loops": 8,
        "bkz_threads": 6, "use_bkz": True, "auto_abort": False,
        # 浮点精度
        "float_type": "mpfr", "precision": 200,
        "use_auto_precision": True,
        "mp_dps_default": MP_DPS_HIGH,
    },
    "ML-DSA-65": {
        "k": 6, "l": 6, "n": 256,
        "q": MLDSA_Q, "eta": 4,
        "d": 13,
        "tau": 49, "gamma1": (1 << 19), "gamma2": (MLDSA_Q - 1) // 32, "omega": 55,
        "oid": "2.16.840.1.101.3.4.3.18",
        "bkz_block_size": 30, "bkz_max_loops": 10,
        "bkz_threads": 6, "use_bkz": True, "auto_abort": False,
        "float_type": "mpfr", "precision": 200,
        "use_auto_precision": True,
        "mp_dps_default": MP_DPS_HIGH,
    },
    "ML-DSA-87": {
        "k": 8, "l": 8, "n": 256,
        "q": MLDSA_Q, "eta": 2,
        "d": 13,
        "tau": 60, "gamma1": (1 << 19), "gamma2": (MLDSA_Q - 1) // 32, "omega": 75,
        "oid": "2.16.840.1.101.3.4.3.19",
        "bkz_block_size": 35, "bkz_max_loops": 12,
        "bkz_threads": 6, "use_bkz": True, "auto_abort": False,
        "float_type": "mpfr", "precision": 200,
        "use_auto_precision": True,
        "mp_dps_default": MP_DPS_HIGH,
    },
}

# ── OID → 参数集名称映射 ─────────────────────────────────────────────────────

OID_TO_MLDSA: dict[str, str] = {
    v["oid"]: k for k, v in MLDSA_REGISTRY.items()
}

MLDSA_OIDS: dict[str, str] = dict(OID_TO_MLDSA)

# ── k 值 → ML-DSA 名称映射 ──────────────────────────────────────────────────

K_TO_MLDSA: dict[int, str] = {v["k"]: k for k, v in MLDSA_REGISTRY.items()}


# ── 测试用参数集（小维度，便于快速实验）──────────────────────────────────────

PARAMS: dict[str, dict[str, Any]] = {
    "easy": {
        "k": 4, "l": 4, "n": 32,
        "q": MLDSA_Q, "eta": 2,
        "d": 13,
        "tau": 8, "gamma1": (1 << 10), "gamma2": (MLDSA_Q - 1) // 32, "omega": 10,
        "bkz_block_size": 8, "bkz_max_loops": 4,
        "bkz_threads": 2, "use_bkz": True, "auto_abort": False,
        "float_type": "mpf", "precision": 80,
        "use_auto_precision": False,
        "mp_dps_default": MP_DPS_LOW,
    },
    "medium": {
        "k": 6, "l": 6, "n": 64,
        "q": MLDSA_Q, "eta": 4,
        "d": 13,
        "tau": 16, "gamma1": (1 << 12), "gamma2": (MLDSA_Q - 1) // 32, "omega": 20,
        "bkz_block_size": 12, "bkz_max_loops": 6,
        "bkz_threads": 4, "use_bkz": True, "auto_abort": False,
        "float_type": "mpf", "precision": 80,
        "use_auto_precision": False,
        "mp_dps_default": MP_DPS_LOW,
    },
    "hard": {
        "k": 8, "l": 8, "n": 128,
        "q": MLDSA_Q, "eta": 2,
        "d": 13,
        "tau": 24, "gamma1": (1 << 14), "gamma2": (MLDSA_Q - 1) // 32, "omega": 30,
        "bkz_block_size": 16, "bkz_max_loops": 8,
        "bkz_threads": 6, "use_bkz": True, "auto_abort": False,
        "float_type": "mpf", "precision": 100,
        "use_auto_precision": False,
        "mp_dps_default": MP_DPS_LOW,
    },
    "extreme": {
        "k": 8, "l": 8, "n": 256,
        "q": MLDSA_Q, "eta": 2,
        "d": 13,
        "tau": 60, "gamma1": (1 << 19), "gamma2": (MLDSA_Q - 1) // 32, "omega": 75,
        "bkz_block_size": 25, "bkz_max_loops": 10,
        "bkz_threads": 6, "use_bkz": True, "auto_abort": False,
        "float_type": "mpfr", "precision": 200,
        "use_auto_precision": True,
        "mp_dps_default": MP_DPS_HIGH,
    },
}


def get_params(name: str) -> dict[str, Any]:
    """获取指定名称的参数集。

    Args:
        name: 参数集名称，支持:
              - FIPS 204 标准名: "ML-DSA-44", "ML-DSA-65", "ML-DSA-87"
              - 测试集名: "easy", "medium", "hard", "extreme"
              - 兼容别名: "toy" → "easy"

    Returns:
        参数集字典的副本

    Raises:
        ValueError: 参数集名称不存在
    """
    # 兼容旧版测试脚本
    if name == "toy":
        name = "easy"

    if name in MLDSA_REGISTRY:
        return dict(MLDSA_REGISTRY[name])
    if name in PARAMS:
        return dict(PARAMS[name])

    available = list(MLDSA_REGISTRY.keys()) + list(PARAMS.keys())
    raise ValueError(f"Unknown parameter set: {name!r}. Available: {available}")


def mldsa_from_k(k: int) -> str:
    """从 k 值反推 ML-DSA 变体名称。

    Args:
        k: ML-DSA 参数 k（维度）

    Returns:
        ML-DSA 变体名称（如 "ML-DSA-44"）

    Raises:
        ValueError: 无对应 k 值的 ML-DSA 变体
    """
    if k not in K_TO_MLDSA:
        raise ValueError(f"No ML-DSA variant with k={k}")
    return K_TO_MLDSA[k]


def get_d(name: str | dict) -> int:
    """获取参数集的 Power2Round 低位比特数 d。

    Args:
        name: 参数集名称（str）或已解析的参数字典（dict）

    Returns:
        d 值（通常为 13）

    Raises:
        ValueError: 参数集名称不存在
    """
    if isinstance(name, dict):
        return name.get("d", 13)
    return get_params(name)["d"]


__all__ = [
    "MLDSA_Q",
    "MLDSA_REGISTRY",
    "OID_TO_MLDSA",
    "MLDSA_OIDS",
    "K_TO_MLDSA",
    "PARAMS",
    "MP_DPS_LOW",
    "MP_DPS_HIGH",
    "MP_STUCK_THRESHOLD",
    "MP_MAX_RETRY",
    "AUTO_PRECISION_DIM_LOW",
    "AUTO_PRECISION_DIM_HIGH",
    "PRECISION_MODE_LOW",
    "PRECISION_MODE_AUTO",
    "PRECISION_MODE_HIGH",
    "LOVASZ_CONDITION_PARAM",
    "LLL_MP_DEFAULT_DPS",
    "get_params",
    "get_d",
    "mldsa_from_k",
]
