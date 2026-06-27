import numpy as np

from ..gso.gsofp_se import gso_step
from ..gso.initializer import initialize
from .L3fp_params import LOVASZ_CONDITION_PARAM
from .reducer import size_reduction_loop


def l3fp(
    basis_matrix,
    gs_coeff_matrix=None,
    gs_squared_norms=None,
    start_stage=0,
    Lovasz_cond_param=LOVASZ_CONDITION_PARAM,
    f_c=False,
):
    """Floating-point LLL reduction (Schnorr-Euchner 1994).

    Optimized: tqdm removed from hot path, local variable caching.

    Args:
        basis_matrix: (n, n) numpy array, columns are basis vectors.
        gs_coeff_matrix: (n, n) GSO coefficients, or None.
        gs_squared_norms: (n,) GSO squared norms, or None.
        start_stage: Starting stage index.
        Lovasz_cond_param: δ parameter in (0.25, 1.0).
        f_c: Floating-point precision flag.

    Returns:
        (basis_matrix, gs_coeff_matrix, gs_squared_norms)
    """
    basis_matrix, gs_coeff_matrix, gs_squared_norms, stage, end_stage = initialize(
        basis_matrix, gs_coeff_matrix, gs_squared_norms, start_stage
    )

    # Local variable cache for inner loop performance
    bm = basis_matrix
    gsc = gs_coeff_matrix
    gs = gs_squared_norms
    delta = Lovasz_cond_param

    while stage < end_stage:
        # Update GSO at current stage
        gs[: stage + 1], gsc[:, : stage + 1] = gso_step(
            bm[:, : stage + 1],
            gsc[:, : stage + 1],
            gs[: stage + 1],
            stage,
        )

        # Size reduction
        f_c, gsc, bm = size_reduction_loop(stage, gsc, bm, f_c)

        # Check for cumulated floating-point inaccuracies
        if f_c:
            f_c = False
            stage = max(stage - 1, 1)
            continue

        # Lovász condition: δ · ||b*_{k-1}||² ≤ ||b*_k||² + μ_{k,k-1}² · ||b*_{k-1}||²
        mu = gsc[stage - 1, stage]
        if delta * gs[stage - 1] > gs[stage] + mu * mu * gs[stage - 1]:
            # Swap columns
            bm[:, [stage - 1, stage]] = bm[:, [stage, stage - 1]]
            stage = max(stage - 1, 1)
        else:
            stage += 1

    return bm, gsc, gs
