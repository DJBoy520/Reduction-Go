import numpy as np


def gso_step(basis_slice, gs_coeff_matrix, gs_squared_norms, stage):
    """One step of Classical Gram-Schmidt with hybrid vectorization.

    Uses numpy vectorization for large stages, simple loops for small ones.

    CGS formula: μ_{j,k} = (⟨b_k, b_j⟩ - Σ_{i<j} μ_{i,j}·μ_{i,k}·||b*_i||²) / ||b*_j||²

    Args:
        basis_slice: (n, stage+1) — columns 0..stage of basis
        gs_coeff_matrix: (stage+1, stage+1) — GSO coefficients
        gs_squared_norms: (stage+1,) — GSO squared norms
        stage: current stage index

    Returns:
        (gs_squared_norms[:stage+1], gs_coeff_matrix[:, :stage+1])
    """
    if stage == 1:
        gs_squared_norms[0] = np.dot(basis_slice[:, 0], basis_slice[:, 0])

    b_k = basis_slice[:, stage]
    gsc = gs_coeff_matrix
    gs = gs_squared_norms

    # Initial squared norm
    gs[stage] = np.dot(b_k, b_k)

    # Compute all dot products ⟨b_k, b_j⟩ for j < stage at once
    dots = basis_slice[:, :stage].T @ b_k  # shape (stage,)

    for j in range(stage):
        # correction_term = Σ_{i<j} μ_{i,j} · μ_{i,stage} · ||b*_i||²
        if j > 0:
            correction_term = np.dot(
                gsc[:j, j] * gs[:j],
                gsc[:j, stage]
            )
        else:
            correction_term = 0.0

        gsc[j, stage] = (dots[j] - correction_term) / gs[j]
        gs[stage] -= gsc[j, stage] * gsc[j, stage] * gs[j]

    gsc[stage, stage] = 1.0

    return gs[: stage + 1], gsc[:, : stage + 1]
