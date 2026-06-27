import numpy as np

from ..gso.delete_zero import delete_zero_vector
from ..gso.gsofp_se import gso_step
from ..gso.initializer import initialize
from .L3fp_params import LOVASZ_CONDITION_PARAM
from .reducer import size_reduction_loop


def l3fp_deep_insert(
    injected_basis_matrix,
    gs_coeff_matrix,
    gs_squared_norms,
    start_stage,
    Lovasz_cond_param=LOVASZ_CONDITION_PARAM,
    f_c=False,
):
    """Executes the floating-point LLL deep insertion algorithm as presented in:
    Lattice Basis Reduction: Improved Practical Algorithms and Solving Subset Sum Problems
    by C. P. Schnorr and M. Euchner.

    This variant of LLL reduction operates on a matrix that includes an injected short vector
    (e.g., from an SVP solver), which introduces linear dependencies. The algorithm performs
    deep insertion to optimally reorder the basis vectors and maintain reduction quality.

    During execution, one of the column vectors in `injected_basis_matrix` will become a zero vector
    as a result of the linear dependencies. This zero vector is detected and removed from `injected_basis_matrix`.


    Args:
        injected_basis_matrix (np.ndarray):
            A 2D NumPy array of shape (n, m), where n<=m, representing a modified lattice basis matrix
            that includes an injected short (column) vector.

        gs_coeff_matrix (np.ndarray):
            A 2D NumPy array of shape (n, m), where n<=m, representing the Gram-Schmidt coefficients
            of the injected_basis_matrix.

        gs_squared_norms (np.ndarray):
                    1D NumPy array of shape (m,) representing the squared lengths of the
            Gram-Schmidt vectors.

        start_stage (int):
            The index of the injected_basis_matrix column from which the reduction process begins.

        Lovasz_cond_param (float):
            The Lovasz condition parameter (typically in ]1/2, 1[) used to determine
            whether a re-order is necessary during the deep insertion loop at each stage.

        f_c (bool):
            A flag used to track floating-point precision issues. If set to True and a precision flaw
            is detected, the algorithm will backtrack one step or restart from stage 1.

    Returns:
        (tuple):
            -injected_basis_matrix (np.ndarray):
                A 2D NumPy array of shape (n, n) representing the reduced matrix after deep insertion, with the zero vector removed.

            -gs_coeff_matrix (np.ndarray):
                A 2D Numpy array of shape (n, n) representing the updated Gram-Schmidt coefficients.

            -gs_squared_norms (np.ndarray):
                A 1D Numpy array of shape (n,) representing the updated squared lengths of The Gram-Schmidt vectors.
    """
    injected_basis_matrix, gs_coeff_matrix, gs_squared_norms, stage, end_stage = initialize(
        injected_basis_matrix, gs_coeff_matrix, gs_squared_norms, start_stage
    )

    # Enter reduction loop
    while stage < end_stage:
        # Append / update Gram-Schmidt orthogonalization with current column
        gs_squared_norms[: stage + 1], gs_coeff_matrix[:, : stage + 1] = gso_step(
            injected_basis_matrix[:, : stage + 1],
            gs_coeff_matrix[:, : stage + 1],
            gs_squared_norms[: stage + 1],
            stage,
        )

        # Size reduction step
        f_c, gs_coeff_matrix, injected_basis_matrix = size_reduction_loop(
            stage, gs_coeff_matrix, injected_basis_matrix, f_c
        )

        # Check for cumulated floating-point inaccuracies
        if f_c:
            f_c = False
            stage = max(stage - 1, 1)
            continue

        # Zero vector check (appears at some point if spanning matrix has linear dependencies between columns)
        if np.all(injected_basis_matrix[:, stage] == 0):
            injected_basis_matrix, gs_squared_norms, gs_coeff_matrix = delete_zero_vector(
                injected_basis_matrix, gs_squared_norms, gs_coeff_matrix, stage
            )
            # After deleting zero vector we back up to stage 1 to ensure correct structure for GSO
            stage = 1
            end_stage -= 1
            continue

        # Deep insertion loop
        temp_gs_squared_norm = np.dot(
            injected_basis_matrix[:, stage], injected_basis_matrix[:, stage]
        )
        i = 0
        re_ordered = False
        while i < stage:
            if Lovasz_cond_param * gs_squared_norms[i] <= temp_gs_squared_norm:
                temp_gs_squared_norm -= (gs_coeff_matrix[i, stage] ** 2) * gs_squared_norms[i]
                i += 1
            else:
                # Shift all columns from i to stage one position right. We end up with [..., b_i-1, b_stage, b_i, ..., b_stage-1, b_stage+1, ...]
                injected_basis_matrix[:, i : stage + 1] = np.roll(
                    injected_basis_matrix[:, i : stage + 1], shift=1, axis=1
                )
                re_ordered = True
                stage = max(i - 1, 1)
                break

        if not re_ordered:
            stage += 1

    return injected_basis_matrix, gs_coeff_matrix, gs_squared_norms
