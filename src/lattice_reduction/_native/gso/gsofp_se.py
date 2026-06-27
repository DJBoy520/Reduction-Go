import numpy as np

from ..jit_compat import jit_available
from ..jit_functions import gso_step_jit


def gso_step(basis_slice, gs_coeff_matrix, gs_squared_norms, stage):
    """GSO single step with optional Numba JIT acceleration.

    Falls back to pure Python if Numba is not available.

    Args:
        basis_slice: (n, stage+1) — columns 0..stage of basis (float64)
        gs_coeff_matrix: (stage+1, stage+1) — GSO coefficients
        gs_squared_norms: (stage+1,) — GSO squared norms
        stage: current stage index

    Returns:
        (gs_squared_norms[:stage+1], gs_coeff_matrix[:, :stage+1])
    """
    if jit_available:
        # basis_slice is already float64 (from initialize()), no copy needed.
        # Numba handles non-contiguous slices via stride info.
        gso_step_jit(
            basis_slice,
            gs_coeff_matrix,
            gs_squared_norms,
            stage,
        )
    else:
        _gso_step_python(basis_slice, gs_coeff_matrix, gs_squared_norms, stage)

    return gs_squared_norms[: stage + 1], gs_coeff_matrix[:, : stage + 1]


def _gso_step_python(basis_slice, gsc, gs, stage):
    """Pure Python GSO step (fallback)."""
    if stage == 1:
        gs[0] = np.dot(basis_slice[:, 0], basis_slice[:, 0])

    b_k = basis_slice[:, stage]
    gs[stage] = np.dot(b_k, b_k)

    dots = basis_slice[:, :stage].T @ b_k

    for j in range(stage):
        if j > 0:
            correction_term = np.dot(gsc[:j, j] * gs[:j], gsc[:j, stage])
        else:
            correction_term = 0.0

        gsc[j, stage] = (dots[j] - correction_term) / gs[j]
        gs[stage] -= gsc[j, stage] * gsc[j, stage] * gs[j]

    gsc[stage, stage] = 1.0
