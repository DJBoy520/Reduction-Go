"""ML-DSA 参数配置模块。

所有 ML-DSA 元数据（k/l/n/q/d/eta/OID/BKZ 参数）的唯一定义源。
其他模块一律从这里引用，不再各自维护重复字典。

定义层次:
  1. MLDSA_REGISTRY — FIPS 204 标准参数集（含 OID、d、签名参数等完整元数据）
  2. PARAMS — 测试用参数集（easy/medium/hard/extreme，维度小，用于实验）
  3. get_params() — 统一入口，toy 是 easy 的别名
"""

from __future__ import annotations

from typing import Any

# 精度常量 — 从 bkz_params 迁移过来，避免循环导入
MP_DPS_LOW = 80
MP_DPS_HIGH = 200

# ── 公共常量 ─────────────────────────────────────────────────────────────────

MLDSA_Q = 8380417  # 2^23 - 2^13 + 1


# ── FIPS 204 标准 ML-DSA 参数集 ──────────────────────────────────────────────

MLDSA_REGISTRY: dict[str, dict[str, Any]] = {
    "ML-DSA-44": {
        "k": 4, "l": 4, "n": 256,
        "q": MLDSA_Q, "eta": 2,
        "d": 13,
        "tau": 39, "gamma1": (1 << 17), "gamma2": (MLDSA_Q - 1) // 32, "omega": 80,
        "oid": "2.16.840.1.101.3.4.3.17",
        "bkz_block_size": 25, "bkz_max_loops": 8,
        "bkz_threads": 6, "use_bkz": True, "auto_abort": False,
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
        "bkz_threads": 8, "use_bkz": True, "auto_abort": False,
        "float_type": "mpfr", "precision": 250,
        "use_auto_precision": True,
        "mp_dps_default": MP_DPS_HIGH,
    },
    "ML-DSA-87": {
        "k": 8, "l": 7, "n": 256,
        "q": MLDSA_Q, "eta": 2,
        "d": 13,
        "tau": 60, "gamma1": (1 << 19), "gamma2": (MLDSA_Q - 1) // 32, "omega": 75,
        "oid": "2.16.840.1.101.3.4.3.19",
        "bkz_block_size": 40, "bkz_max_loops": 15,
        "bkz_threads": 12, "use_bkz": True, "auto_abort": False,
        "float_type": "mpfr", "precision": 300,
        "use_auto_precision": True,
        "mp_dps_default": MP_DPS_HIGH,
    },
}


# ── 测试用参数集（小维度，便于快速实验）──────────────────────────────────────

PARAMS: dict[str, dict[str, Any]] = {
    "easy": {
        "k": 2, "l": 2, "n": 16,
        "q": MLDSA_Q, "eta": 2, "d": 13,
        "tau": 5, "gamma1": (1 << 10), "gamma2": (MLDSA_Q - 1) // 32, "omega": 10,
        "bkz_block_size": 10, "bkz_max_loops": 2,
        "bkz_threads": 1, "use_bkz": True, "auto_abort": False,
        "float_type": "mpf", "precision": 80,
        "use_auto_precision": False,
        "mp_dps_default": MP_DPS_LOW,
    },
    "medium": {
        "k": 3, "l": 3, "n": 32,
        "q": MLDSA_Q, "eta": 3, "d": 13,
        "tau": 10, "gamma1": (1 << 12), "gamma2": (MLDSA_Q - 1) // 32, "omega": 20,
        "bkz_block_size": 15, "bkz_max_loops": 3,
        "bkz_threads": 2, "use_bkz": True, "auto_abort": False,
        "float_type": "mpf", "precision": 100,
        "use_auto_precision": False,
        "mp_dps_default": MP_DPS_LOW,
    },
    "hard": {
        "k": 4, "l": 4, "n": 64,
        "q": MLDSA_Q, "eta": 4, "d": 13,
        "tau": 15, "gamma1": (1 << 14), "gamma2": (MLDSA_Q - 1) // 32, "omega": 30,
        "bkz_block_size": 20, "bkz_max_loops": 5,
        "bkz_threads": 4, "use_bkz": True, "auto_abort": False,
        "float_type": "mpf", "precision": 150,
        "use_auto_precision": False,
        "mp_dps_default": MP_DPS_HIGH,
    },
    "extreme": {
        "k": 6, "l": 6, "n": 128,
        "q": MLDSA_Q, "eta": 4, "d": 13,
        "tau": 20, "gamma1": (1 << 16), "gamma2": (MLDSA_Q - 1) // 32, "omega": 50,
        "bkz_block_size": 25, "bkz_max_loops": 8,
        "bkz_threads": 6, "use_bkz": True, "auto_abort": False,
        "float_type": "mpf", "precision": 200,
        "use_auto_precision": False,
        "mp_dps_default": MP_DPS_HIGH,
    },
}


def get_params(name: str) -> dict[str, Any]:
    """获取参数集配置。

    Args:
        name: 参数集名称（ML-DSA-44/65/87, easy/medium/hard/extreme, toy）

    Returns:
        参数集字典的副本

    Raises:
        ValueError: 参数集名称不存在
    """
    # toy 是 easy 的别名
    if name == "toy":
        name = "easy"

    # 先查标准参数集
    if name in MLDSA_REGISTRY:
        return MLDSA_REGISTRY[name].copy()

    # 再查测试参数集
    if name in PARAMS:
        return PARAMS[name].copy()

    available = list(MLDSA_REGISTRY.keys()) + list(PARAMS.keys()) + ["toy"]
    raise ValueError(f"未知参数集 '{name}'，可用: {available}")


def get_d(params: dict[str, Any]) -> int:
    """获取 d 参数（Power2Round 位数），默认值 13。"""
    return params.get("d", 13)
