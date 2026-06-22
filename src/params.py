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


# FIPS 204 ML-DSA OID → 参数集映射
MLDSA_OID_PARAMS = {
    "2.16.840.1.101.3.4.3.17": {"name": "ML-DSA-44", "k": 4, "l": 4, "n": 256, "d": 10},
    "2.16.840.1.101.3.4.3.18": {"name": "ML-DSA-65", "k": 6, "l": 6, "n": 256, "d": 13},
    "2.16.840.1.101.3.4.3.19": {"name": "ML-DSA-87", "k": 8, "l": 8, "n": 256, "d": 13},
}


def get_params(name: str = "easy") -> dict:
    """Return parameter dict for the given parameter set name.

    'toy' is an alias for 'easy'.
    Raises ValueError if the parameter set is not implemented yet.
    """
    if name == "toy":
        name = "easy"
    if name not in PARAMS:
        raise ValueError(f"Unknown parameter set: {name!r}. Available: {list(PARAMS.keys())}")
    if PARAMS[name] is None:
        raise ValueError(f"Parameter set {name!r} is reserved but not yet implemented")
    return PARAMS[name].copy()
