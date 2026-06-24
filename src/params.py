"""
ML-DSA parameter configuration module.

Provides parameter sets for ML-DSA. Currently implements 'toy' for testing.
Switch parameter sets by passing a different key to get_params().
"""

PARAMS = {
    "easy": {
        "k": 2, "l": 2, "n": 50,
        "q": 8380417, "eta": 2,
        "bkz_block_size": 8, "bkz_max_loops": 8,
        "bkz_threads": 6, "use_bkz": True, "auto_abort": False,
        "float_type": "double", "precision": 53,
    },
    "medium": {
        "k": 3, "l": 3, "n": 80,
        "q": 8380417, "eta": 2,
        "bkz_block_size": 15, "bkz_max_loops": 8,
        "bkz_threads": 6, "use_bkz": True, "auto_abort": False,
        "float_type": "mpfr", "precision": 200,
    },
    "hard": {
        "k": 4, "l": 4, "n": 120,
        "q": 8380417, "eta": 2,
        "bkz_block_size": 20, "bkz_max_loops": 8,
        "bkz_threads": 6, "use_bkz": True, "auto_abort": False,
        "float_type": "mpfr", "precision": 200,
    },
    "extreme": {
        "k": 5, "l": 5, "n": 200,
        "q": 8380417, "eta": 2,
        "bkz_block_size": 25, "bkz_max_loops": 8,
        "bkz_threads": 6, "use_bkz": True, "auto_abort": False,
        "float_type": "mpfr", "precision": 200,
    },
    # Aliases
    "toy": None,  # maps to 'easy'
}


# FIPS 204 标准 ML-DSA 参数集
_MLDSA_Q = 8380417

MLDSA_PARAMS = {
    "ML-DSA-44": {
        "k": 4, "l": 4, "n": 256,
        "q": _MLDSA_Q, "eta": 2,
        "d": 13, "tau": 39, "gamma1": (1 << 17), "gamma2": (_MLDSA_Q - 1) // 32,
        "omega": 80,
        "bkz_block_size": 25, "bkz_max_loops": 8,
        "bkz_threads": 6, "use_bkz": True, "auto_abort": False,
        "float_type": "mpfr", "precision": 200,
    },
    "ML-DSA-65": {
        "k": 6, "l": 6, "n": 256,
        "q": _MLDSA_Q, "eta": 4,
        "d": 13, "tau": 49, "gamma1": (1 << 19), "gamma2": (_MLDSA_Q - 1) // 32,
        "omega": 55,
        "bkz_block_size": 30, "bkz_max_loops": 8,
        "bkz_threads": 6, "use_bkz": True, "auto_abort": False,
        "float_type": "mpfr", "precision": 200,
    },
    "ML-DSA-87": {
        "k": 8, "l": 8, "n": 256,
        "q": _MLDSA_Q, "eta": 2,
        "d": 13, "tau": 60, "gamma1": (1 << 19), "gamma2": (_MLDSA_Q - 1) // 32,
        "omega": 75,
        "bkz_block_size": 35, "bkz_max_loops": 8,
        "bkz_threads": 6, "use_bkz": True, "auto_abort": False,
        "float_type": "mpfr", "precision": 200,
    },
}


def get_params(name: str = "easy") -> dict:
    """Return parameter dict for the given parameter set name.

    'toy' is an alias for 'easy'.
    'ML-DSA-44/65/87' are standard FIPS 204 parameter sets.
    Raises ValueError if the parameter set is not implemented yet.
    """
    if name == "toy":
        name = "easy"
    if name in MLDSA_PARAMS:
        return MLDSA_PARAMS[name].copy()
    if name not in PARAMS:
        raise ValueError(f"Unknown parameter set: {name!r}. Available: {list(PARAMS.keys()) + list(MLDSA_PARAMS.keys())}")
    if PARAMS[name] is None:
        raise ValueError(f"Parameter set {name!r} is reserved but not yet implemented")
    return PARAMS[name].copy()
