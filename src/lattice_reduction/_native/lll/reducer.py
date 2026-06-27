from .L3fp_params import SIZE_REDUCTION_CONDITION_PARAM, TAU
from ..jit_compat import jit_available
from ..jit_functions import size_reduction_jit


def size_reduction_loop(stage, gs_coeff_matrix, spanning_matrix, f_c):
    """Size reduction for column `stage`.

    Uses JIT-accelerated version when Numba is available.

    Args:
        stage: Current column index.
        gs_coeff_matrix: (m, m) GSO coefficients.
        spanning_matrix: (n, m) basis vectors (columns).
        f_c: Floating-point precision flag.

    Returns:
        (f_c, gs_coeff_matrix, spanning_matrix)
    """
    if jit_available:
        tau_limit = 2 ** (TAU / 2)
        gsc_col = gs_coeff_matrix[:, stage].copy()
        basis_col = spanning_matrix[:, stage].copy()
        f_c = size_reduction_jit(
            gsc_col, gs_coeff_matrix, basis_col, spanning_matrix,
            stage, tau_limit,
        )
        gs_coeff_matrix[:, stage] = gsc_col
        spanning_matrix[:, stage] = basis_col
    else:
        f_c, gs_coeff_matrix, spanning_matrix = _size_reduction_python(
            stage, gs_coeff_matrix, spanning_matrix, f_c
        )

    return f_c, gs_coeff_matrix, spanning_matrix


def _size_reduction_python(stage, gs_coeff_matrix, spanning_matrix, f_c):
    """Pure Python size reduction (fallback)."""
    gsc_col = gs_coeff_matrix[:, stage]
    threshold = SIZE_REDUCTION_CONDITION_PARAM
    tau_limit = 2 ** (TAU / 2)

    for i in range(stage - 1, -1, -1):
        if abs(gsc_col[i]) > threshold:
            mu = round(gsc_col[i])
            if abs(mu) > tau_limit:
                f_c = True
            gsc_col[:i] -= mu * gs_coeff_matrix[:i, i]
            gsc_col[i] -= mu
            spanning_matrix[:, stage] -= mu * spanning_matrix[:, i]

    return f_c, gs_coeff_matrix, spanning_matrix
