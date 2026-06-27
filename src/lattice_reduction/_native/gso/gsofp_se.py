import numpy as np


def gso_step(basis_slice, gs_coeff_matrix, gs_squared_norms, stage):
    """Updates the Gram-Schmidt coefficient matrix and squared norms at a specific stage.
    This function performs one step of the Gram-Schmidt orthogonalization process,
    updating the entries in `gs_coeff_matrix` and `gs_squared_norms` corresponding
    to the given `stage`. It is assumed that all entries up to `stage - 1` are already
    correct and up to date. If `stage == 1`, the squared norm at index 0 is also updated.

    Uses the Classical Gram-Schmidt (CGS) formula with vectorized numpy operations.

    args:
        basis_slice (np.ndarray):
            2D NumPy array of shape (n, stage+1) — columns 0..stage of the basis.

        gs_coeff_matrix (np.ndarray):
            A 2D NumPy array of shape (stage+1, stage+1), representing the Gram-Schmidt
            coefficients. Values at column `stage` may not be up to date.

        gs_squared_norms (np.ndarray):
            A 1D NumPy array of shape (stage+1,), representing the squared lengths of
            the Gram-Schmidt vectors. The value at index `stage` may not be up to date.

        stage (int):
            The current stage index (0-based).

    returns:
        (tuple):
            - gs_squared_norms (np.ndarray): Updated squared norms.
            - gs_coeff_matrix (np.ndarray): Updated coefficient matrix.
    """
    if stage == 1:
        gs_squared_norms[0] = np.dot(basis_slice[:, 0], basis_slice[:, 0])

    b_k = basis_slice[:, stage]

    # Squared norm of b_k (will be reduced iteratively)
    gs_squared_norms[stage] = np.dot(b_k, b_k)

    # Cache references for inner loop
    gsc = gs_coeff_matrix
    gs = gs_squared_norms

    for j in range(stage):
        # dot_product = <b_k, b_j>
        dot_product = np.dot(b_k, basis_slice[:, j])

        # correction_term = Σ_{k<j} μ_{k,j} · μ_{k,stage} · ||b*_k||²
        # Vectorized: dot product of (μ_{:,j} · ||b*_||²)[0:j] with μ_{:,stage}[0:j]
        if j > 0:
            correction_term = np.dot(
                gsc[:j, j] * gs[:j],
                gsc[:j, stage]
            )
        else:
            correction_term = 0.0

        gsc[j, stage] = (dot_product - correction_term) / gs[j]

        # Update squared norm: ||b*_stage||² -= μ_{j,stage}² · ||b*_j||²
        gs[stage] -= gsc[j, stage] * gsc[j, stage] * gs[j]

    gsc[stage, stage] = 1.0

    return gs_squared_norms[: stage + 1], gs_coeff_matrix[:, : stage + 1]
