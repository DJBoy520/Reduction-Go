import numpy as np

from .bkz_params import DELTA
from ..lll.L3fp import l3fp
from ..lll.L3fp_deep_insertion import l3fp_deep_insert
from ..enumeration import ENUM_ALGORITHMS


def bkz_se(basis_matrix, block_size, enum_algo):
    """Executes the BKZ reduction algorithm as presented in
    *Lattice Basis Reduction: Improved Practical Algorithms and Solving Subset Sum Problems*
    by C. P. Schnorr, M. Euchner (1994).

    Args:
        basis_matrix (np.ndarray):
            A 2D NumPy array of shape (n, n) representing a lattice basis, where each column is a basis vector.
        block_size (int):
            An integer that determines the width of the search window for svp-solver.
        enum_algo (string):
            A string key selecting the enumeration algorithm variant from `ENUM_ALGORITHMS`.

    Notes:
        - Our implementation uses 0-based indices (`0,...,n-1`) for basis and block boundaries,
        whereas the original Schnorr–Euchner paper uses 1-based indices (`1,...,n`).

    Returns:
        (tuple):
            -basis_matrix (np.ndarray):
                A 2D Numpy array of shape (n, n) representing a BKZ-reduced lattice basis,
                where each column is a basis vector.

            -gs_coeff_matrix (np.ndarray):
                A 2D Numpy array of shape (n, n) representing the updated Gram-Schmidt coefficients.

            -gs_squared_norms (np.ndarray):
                A 1D Numpy array of shape (n,) representing the updated squared lengths of The Gram-Schmidt vectors.
    """
    svp_solver = ENUM_ALGORITHMS[enum_algo]
    m = len(basis_matrix[0]) - 1
    basis_matrix, gs_coeff_matrix, gs_squared_norms = l3fp(basis_matrix)
    z = 0
    j = -1  # Ensure that we start the first loop from j=0
    while z < m:
        j += 1
        k = min(j + block_size - 1, m)
        if j == m:
            j = 0
            k = block_size
        candidate_proj_len, candidate_coeff_vec = svp_solver(
            basis_matrix[:, j:k + 1], gs_squared_norms[j:k + 1], gs_coeff_matrix[:, j:k + 1]
        )
        block_end = min(k + 1, m)
        if DELTA * gs_squared_norms[j] > candidate_proj_len:
            b_new = np.dot(basis_matrix[:, j:k + 1], candidate_coeff_vec)
            injected_basis = np.insert(basis_matrix[:, :block_end + 1], j, np.transpose(b_new), axis=1)

            (
                di_basis,
                gs_coeff_matrix[:block_end + 1, :block_end + 1],
                gs_squared_norms[:block_end + 1],
            ) = l3fp_deep_insert(
                injected_basis_matrix=injected_basis,
                gs_coeff_matrix=gs_coeff_matrix[:j, :j],
                gs_squared_norms=gs_squared_norms[:j],
                start_stage=j,
                Lovasz_cond_param=DELTA,
                f_c=True,
            )
            basis_matrix[:, :block_end + 1] = np.round(di_basis).astype(np.int64)
            z = 0

        else:
            z += 1
            (
                basis_matrix[:, :block_end + 1],
                gs_coeff_matrix[:block_end + 1, :block_end + 1],
                gs_squared_norms[:block_end + 1],
            ) = l3fp(
                basis_matrix=basis_matrix[:, :block_end + 1],
                gs_coeff_matrix=gs_coeff_matrix[: block_end, : block_end],
                gs_squared_norms=gs_squared_norms[: block_end],
                start_stage=block_end - 1,
                Lovasz_cond_param=0.99,
            )

    return basis_matrix, gs_coeff_matrix, gs_squared_norms
