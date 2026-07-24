"""项目全局常量 — 唯一定义源。

本模块是项目的最底层模块，不导入任何项目内其他模块（零依赖）。
所有标量常量、参数集注册表、OID 映射集中定义于此。

来源追溯:
  - MLDSA_Q, MLDSA_REGISTRY, OID_TO_MLDSA, MLDSA_OIDS  ← 原 src/utils/params.py, src/common/params.py
  - MP_DPS_LOW/HIGH 等精度常量                            ← 原 src/lattice/base/precision_constants.py, src/common/params.py
  - LOVASZ_CONDITION_PARAM, LLL_MP_DEFAULT_DPS            ← 原 src/lattice/algorithms/lll_params.py
"""

from __future__ import annotations

# ── ML-DSA 模数 ──────────────────────────────────────────────────────────────
# 来源: FIPS 204 §4.1 — q = 2^23 - 2^13 + 1
MLDSA_Q: int = 8380417

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
# 来源: 原 src/lattice/base/precision_constants.py
PRECISION_MODE_LOW: int = 0   # 强制低精度
PRECISION_MODE_AUTO: int = 1  # 按维度自动选择
PRECISION_MODE_HIGH: int = 2  # 强制高精度

# ── LLL 参数 ─────────────────────────────────────────────────────────────────
# 来源: FIPS 204 / 原 src/lattice/algorithms/lll_params.py
LOVASZ_CONDITION_PARAM: float = 0.79  # δ ∈ (0.25, 1.0)，Lovász 条件参数
LLL_MP_DEFAULT_DPS: int = 50          # mpmath 默认精度（LLL 入口）

# ── FIPS 204 标准 ML-DSA 参数集注册表 ────────────────────────────────────────
# 来源: 原 src/common/params.py / src/utils/params.py
MLDSA_REGISTRY: dict[str, dict] = {
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

# ── 测试用参数集 ─────────────────────────────────────────────────────────────
# 来源: 原 src/common/params.py / src/utils/params.py（小维度，便于快速实验）
PARAMS: dict[str, dict] = {
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
        "k": 4, "l": 4, "n": 120,
        "q": MLDSA_Q, "eta": 2, "d": 13,
        "bkz_block_size": 20, "bkz_max_loops": 8,
        "bkz_threads": 6, "use_bkz": True, "auto_abort": False,
        "float_type": "mpfr", "precision": 200,
        "use_auto_precision": True,
        "mp_dps_default": MP_DPS_LOW,
    },
    "extreme": {
        "k": 5, "l": 5, "n": 200,
        "q": MLDSA_Q, "eta": 2, "d": 13,
        "bkz_block_size": 25, "bkz_max_loops": 8,
        "bkz_threads": 6, "use_bkz": True, "auto_abort": False,
        "float_type": "mpfr", "precision": 200,
        "use_auto_precision": True,
        "mp_dps_default": MP_DPS_HIGH,
    },
    "toy": None,  # maps to 'easy'
}

# ── OID ↔ ML-DSA 名称映射 ───────────────────────────────────────────────────
# 来源: 原 src/utils/params.py（由 MLDSA_REGISTRY 派生）
OID_TO_MLDSA: dict[str, str] = {v["oid"]: k for k, v in MLDSA_REGISTRY.items()}
MLDSA_OIDS: dict[str, str] = {k: v["oid"] for k, v in MLDSA_REGISTRY.items()}
