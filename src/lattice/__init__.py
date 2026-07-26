"""格基约减模块。

对外接口:
    lll_reduce(B, delta=0.999)  — LLL 约减（原地修改行向量基 B）
    bkz_reduce(B, ...)          — BKZ 约减（原地修改行向量基 B）
"""

from .adapter import (
    lll_reduce, lll_reduce_full, bkz_reduce,
    evaluate_basis_quality,
    LatticeReducer, ReductionResult,
)
from ..domain.exceptions import (
    LatticeReductionError, InvalidBasisError, ReductionFailedError,
    PrecisionFailureError,
)

__all__ = [
    "lll_reduce", "lll_reduce_full", "bkz_reduce",
    "evaluate_basis_quality",
    "LatticeReducer", "ReductionResult",
    "LatticeReductionError", "InvalidBasisError", "ReductionFailedError",
    "PrecisionFailureError",
]
