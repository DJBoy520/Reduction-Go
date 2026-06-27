import numpy as np

from .bkz_params import DELTA
from ..lll.L3fp import l3fp
from ..lll.L3fp_deep_insertion import l3fp_deep_insert
from ..enumeration import ENUM_ALGORITHMS


def structural_changes(gs_norms_before, gs_norms_after, block_size):
    """Detect whether a processed block experienced a *material* structural change
    in its Gram–Schmidt squared norms. The check compares `gs_norms_before` and `gs_norms_after` using `np.allclose`
    with a dimension-aware absolute tolerance: `atol = max(max(gs_norms_before), max(gs_norms_after), 1.0) * (block_size * 1e-12)`
    and zero relative tolerance `rtol = 0`. This normalization scales the tolerance
    to the magnitude of the block’s GS norms and the block size, making the check
    robust across dimensions and numeric ranges.

    Args:
        gs_norms_before (np.ndarray):
            1D array of GS squared norms for the block *before* the candidate injection and deep insertion; shape (block_size,).
        gs_norms_after (np.ndarray):
            1D array of GS squared norms for the block *after* deep insertion; shape (block_size,).
        block_size (int):
            The BKZ block size. Used to scale the absolute tolerance.

    Returns:
        (bool):
            - True  → no material structural change (within tolerance).
            - False → a material change occurred (outside tolerance).

    """
    scale = max(np.max(gs_norms_before), np.max(gs_norms_after), 1.0)
    tol = block_size * 1e-12 * scale
    return np.allclose(gs_norms_before, gs_norms_after, rtol=0, atol=tol)


def bkz_se_pc(basis_matrix, block_size, enum_algo):
    """Executes the BKZ reduction algorithm as presented in
    *Lattice Basis Reduction: Improved Practical Algorithms and Solving Subset Sum Problems*
    by C. P. Schnorr, M. Euchner (1994), with an additional progress tracking mechanism
    to monitor Gram-Schmidt vector lengths and prevent rare infinite loop scenarios. In high dimensions,
    rare cases can occur where the algorithm repeatedly injects a short vector `x`
    into the basis, only for `lll_deep_insert` to remove it immediately afterward. This creates a false
    sense of improvement in basis quality, causing the algorithm to run indefinitely. The implemented
    tracking mechanism detects and mitigates this behavior.

    Args:
        basis_matrix (np.ndarray):
            A 2D NumPy array of shape (n, n) representing a lattice basis, where each column is a basis vector.
        block_size (int):
            The BKZ block size, determining the width of the local enumeration window.
            Larger values improve reduction quality but increase runtime.
        enum_algo (string):
            A string key selecting the enumeration algorithm variant from `ENUM_ALGORITHMS`.

    Notes:
        - Our implementation uses 0-based indices (`0,...,n-1`) for basis and block boundaries,
        whereas the original Schnorr–Euchner paper uses 1-based indices (`1,...,n`).

    Returns:
        (tuple):
            - basis_matrix (np.ndarray):
                BKZ-reduced lattice basis of shape (n, n).
            - gs_coeff_matrix (np.ndarray):
                Gram-Schmidt coefficient matrix of shape (n, n).
            - gs_squared_norms (np.ndarray):
                Squared norms of Gram-Schmidt vectors, shape (n,).
    """
    svp_solver = ENUM_ALGORITHMS[enum_algo]
    m = len(basis_matrix[0]) - 1
    basis_matrix, gs_coeff_matrix, gs_squared_norms = l3fp(basis_matrix)
    z = 0
    j = -1
    while z < m:
        j += 1
        k = min(j + block_size - 1, m)
        if j == m:
            j = 0
            k = block_size

        candidate_proj_len, candidate_coeff_vec = svp_solver(
            basis_matrix[:, j : k + 1], gs_squared_norms[j : k + 1], gs_coeff_matrix[:, j : k + 1]
        )
        block_end = min(k + 1, m)
        if DELTA * gs_squared_norms[j] > candidate_proj_len:
            # Save block_gs_norms for progress tracking
            block_gs_norms_before = gs_squared_norms[j : k + 1].copy()
            b_new = np.dot(basis_matrix[:, j : k + 1], candidate_coeff_vec)
            injected_basis = np.insert(
                basis_matrix[:, : block_end + 1], j, np.transpose(b_new), axis=1
            )
            (
                di_basis,
                gs_coeff_matrix[: block_end + 1, : block_end + 1],
                gs_squared_norms[: block_end + 1],
            ) = l3fp_deep_insert(
                injected_basis_matrix=injected_basis,
                gs_coeff_matrix=gs_coeff_matrix[:j, :j],
                gs_squared_norms=gs_squared_norms[:j],
                start_stage=j,
                Lovasz_cond_param=DELTA,
                f_c=True,
            )
            # Round float64 basis back to int64 (LLL guarantees integer basis)
            basis_matrix[:, : block_end + 1] = np.round(di_basis).astype(np.int64)

            # Evaluate improvement
            block_gs_norms_after = gs_squared_norms[j : k + 1].copy()
            z = 0
            # structural_changes returns True = no change, False = changed.
            # Only skip z increment when a real change occurred (genuine progress).
            if not structural_changes(block_gs_norms_before, block_gs_norms_after, block_size):
                continue

        z += 1
        (
            basis_matrix[:, : block_end + 1],
            gs_coeff_matrix[: block_end + 1, : block_end + 1],
            gs_squared_norms[: block_end + 1],
        ) = l3fp(
            basis_matrix=basis_matrix[:, : block_end + 1],
            gs_coeff_matrix=gs_coeff_matrix[:block_end, :block_end],
            gs_squared_norms=gs_squared_norms[:block_end],
            start_stage=block_end - 1,
            Lovasz_cond_param=0.99,
        )

    return basis_matrix, gs_coeff_matrix, gs_squared_norms
