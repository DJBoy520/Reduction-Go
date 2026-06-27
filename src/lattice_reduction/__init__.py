"""
lattice_reduction — 格基约减算法模块。

替代 fpylll/fplll，提供纯 Python 格约减能力。

对外接口:
    lll_reduce(B, delta=0.999)         — LLL 约减（原地修改行向量基 B）
    bkz_reduce(B, block_size=20, ...)  — BKZ 约减（原地修改行向量基 B）
    evaluate_basis_quality(B)          — 格基质量评估
"""

from .adapter.lll_adapter import lll_reduce, lll_reduce_full
from .adapter.bkz_adapter import bkz_reduce
from .adapter.quality_adapter import evaluate_basis_quality
from .adapter.common_adapter import basis_to_numpy

__all__ = [
    "lll_reduce",
    "lll_reduce_full",
    "bkz_reduce",
    "evaluate_basis_quality",
    "basis_to_numpy",
]
