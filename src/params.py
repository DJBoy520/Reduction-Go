"""
ML-DSA parameter configuration module.

所有 ML-DSA 元数据（k/l/n/q/d/eta/OID/BKZ 参数）的唯一定义源。
其他模块一律从这里引用，不再各自维护重复字典。

定义层次:
  1. MLDSA_REGISTRY — FIPS 204 标准参数集（含 OID、d、签名参数等完整元数据）
  2. PARAMS — 测试用参数集（easy/medium/hard/extreme，维度小，用于实验）
  3. get_params() — 统一入口，toy 是 easy 的别名
"""

# ── 公共常量 ─────────────────────────────────────────────────────────────────

MLDSA_Q = 8380417  # 2^23 - 2^13 + 1


# ── FIPS 204 标准 ML-DSA 参数集 ──────────────────────────────────────────────

MLDSA_REGISTRY = {
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
    },
    "ML-DSA-65": {
        "k": 6, "l": 6, "n": 256,
        "q": MLDSA_Q, "eta": 4,
        "d": 13,
        "tau": 49, "gamma1": (1 << 19), "gamma2": (MLDSA_Q - 1) // 32, "omega": 55,
        "oid": "2.16.840.1.101.3.4.3.18",
        "bkz_block_size": 30, "bkz_max_loops": 8,
        "bkz_threads": 6, "use_bkz": True, "auto_abort": False,
        "float_type": "mpfr", "precision": 200,
    },
    "ML-DSA-87": {
        "k": 8, "l": 8, "n": 256,
        "q": MLDSA_Q, "eta": 2,
        "d": 13,
        "tau": 60, "gamma1": (1 << 19), "gamma2": (MLDSA_Q - 1) // 32, "omega": 75,
        "oid": "2.16.840.1.101.3.4.3.19",
        "bkz_block_size": 35, "bkz_max_loops": 8,
        "bkz_threads": 6, "use_bkz": True, "auto_abort": False,
        "float_type": "mpfr", "precision": 200,
    },
}

# OID → 名称的反查表
OID_TO_MLDSA = {v["oid"]: k for k, v in MLDSA_REGISTRY.items()}

# 名称 → OID 的正查表
MLDSA_OIDS = {k: v["oid"] for k, v in MLDSA_REGISTRY.items()}

# k 值 → ML-DSA 名称的映射（用于从维度反推变体）
K_TO_MLDSA = {v["k"]: k for k, v in MLDSA_REGISTRY.items()}


# ── 测试用参数集 ─────────────────────────────────────────────────────────────

PARAMS = {
    "easy": {
        "k": 2, "l": 2, "n": 50,
        "q": MLDSA_Q, "eta": 2,
        "d": 13,  # 与 FIPS 204 保持一致
        "bkz_block_size": 8, "bkz_max_loops": 8,
        "bkz_threads": 6, "use_bkz": True, "auto_abort": False,
        "float_type": "double", "precision": 53,
    },
    "medium": {
        "k": 3, "l": 3, "n": 80,
        "q": MLDSA_Q, "eta": 2,
        "d": 13,
        "bkz_block_size": 15, "bkz_max_loops": 8,
        "bkz_threads": 6, "use_bkz": True, "auto_abort": False,
        "float_type": "mpfr", "precision": 200,
    },
    "hard": {
        "k": 4, "l": 4, "n": 120,
        "q": MLDSA_Q, "eta": 2,
        "d": 13,
        "bkz_block_size": 20, "bkz_max_loops": 8,
        "bkz_threads": 6, "use_bkz": True, "auto_abort": False,
        "float_type": "mpfr", "precision": 200,
    },
    "extreme": {
        "k": 5, "l": 5, "n": 200,
        "q": MLDSA_Q, "eta": 2,
        "d": 13,
        "bkz_block_size": 25, "bkz_max_loops": 8,
        "bkz_threads": 6, "use_bkz": True, "auto_abort": False,
        "float_type": "mpfr", "precision": 200,
    },
    # Aliases
    "toy": None,  # maps to 'easy'
}


# ── 统一入口 ─────────────────────────────────────────────────────────────────

def get_params(name: str = "easy") -> dict:
    """Return parameter dict for the given parameter set name.

    'toy' is an alias for 'easy'.
    'ML-DSA-44/65/87' are standard FIPS 204 parameter sets.
    Raises ValueError if the parameter set is not implemented yet.
    """
    if name == "toy":
        name = "easy"
    if name in MLDSA_REGISTRY:
        return MLDSA_REGISTRY[name].copy()
    if name not in PARAMS:
        raise ValueError(
            f"Unknown parameter set: {name!r}. "
            f"Available: {list(PARAMS.keys()) + list(MLDSA_REGISTRY.keys())}"
        )
    if PARAMS[name] is None:
        raise ValueError(f"Parameter set {name!r} is reserved but not yet implemented")
    return PARAMS[name].copy()


def get_d(name: str) -> int:
    """返回参数集对应的 Power2Round d 值。

    所有 ML-DSA 变体和测试参数集统一 d=13。
    """
    if name == "toy":
        name = "easy"
    if name in MLDSA_REGISTRY:
        return MLDSA_REGISTRY[name]["d"]
    if name in PARAMS and PARAMS[name] is not None:
        return PARAMS[name]["d"]
    raise ValueError(f"Unknown parameter set: {name!r}")


def get_oid(name: str) -> str:
    """返回 ML-DSA 参数集对应的 X.509 OID。"""
    if name not in MLDSA_REGISTRY:
        raise ValueError(f"No OID for parameter set: {name!r}")
    return MLDSA_REGISTRY[name]["oid"]


def mldsa_from_k(k: int) -> str:
    """从 k 值反推 ML-DSA 变体名称。"""
    if k not in K_TO_MLDSA:
        raise ValueError(f"No ML-DSA variant with k={k}")
    return K_TO_MLDSA[k]
