import sys
from pathlib import Path

import numpy as np
import mpmath

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.lattice_reduction._native.lll_mp import _size_reduction_lll


def test_size_reduction_uses_large_int_fallback_for_huge_mu():
    basis = np.array([[1, 0], [0, 1]], dtype=np.int64)
    gsc = [
        [mpmath.mpf(1), mpmath.mpf(2**55)],
        [mpmath.mpf(0), mpmath.mpf(1)],
    ]
    gsn = [mpmath.mpf(1), mpmath.mpf(1)]

    f_c, need_full_refresh, _ = _size_reduction_lll(1, gsc, gsn, basis, dps=50)

    assert f_c is False
    assert need_full_refresh is True
    assert basis[:, 1][0] == -(2**55)
    assert gsc[0][1] == mpmath.mpf(-(2**55))


def test_lll_mp_can_skip_gso_output():
    from src.lattice_reduction._native.lll_mp import lll_mp

    basis = np.array([
        [4, 1, 2],
        [1, 5, 3],
        [0, 2, 6],
    ], dtype=np.int64).T

    reduced = lll_mp(basis.copy(), Lovasz_cond_param=0.79, dps=50, return_gso=False)

    assert isinstance(reduced, np.ndarray)
    assert reduced.shape == basis.shape
