from tqdm import tqdm

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
    """Executes the Floating-point LLL reduction algorithm as presented in
    *Lattice Basis Reduction: Improved Practical Algorithms and Solving Subset Sum Problems*
    by C. P. Schnorr, M. Euchner (1994).

    Args:
        basis_matrix (np.ndarray):
            A 2D NumPy array of shape (n, n) representing a lattice basis, where each column is a basis vector.

        gs_coeff_matrix (np.ndarray):
            A 2D Numpy array of shape (n, n) representing
            the Gram-Schmidt coefficients of the basis_matrix.
            If None, it will be initialized internally.

        gs_squared_norms (np.ndarray):
            A 1D Numpy array of shape (n,) representing squared lengths of
            The Gram-Schmidt vectors of the basis_matrix.
            If None, it will be initialized internally.

        start_stage (int):
            The index of the basis vector from which the reduction process begins.

        Lovasz_cond_param (float):
            The Lovasz condition parameter (typically in ]1/2, 1[) used to determine
            whether a column swap is necessary at each stage.

        f_c (bool):
            A flag used to track floating-point precision issues. If set to True and a precision flaw is detected, the algorithm will backtrack one step or restart from stage 1.

    Returns:
        (tuple):
            -basis_matrix (np.ndarray):
                A 2D Numpy array of shape (n, n) representing a lll-reduced lattice basis,
                where each column is a basis vector.

            -gs_coeff_matrix (np.ndarray):
                A 2D Numpy array of shape (n, n) representing the updated Gram-Schmidt coefficients.

            -gs_squared_norms (np.ndarray):
                A 1D Numpy array of shape (n,) representing the updated squared lengths of The Gram-Schmidt vectors.
    """

    basis_matrix, gs_coeff_matrix, gs_squared_norms, stage, end_stage = initialize(
        basis_matrix, gs_coeff_matrix, gs_squared_norms, start_stage
    )

    pbar = tqdm(
        total=end_stage,
        desc="LLL reduction loop",
        leave=False,
        ascii="-##",
        colour="white",
        position=2,
    )
    # Enter reduction loop
    while stage < end_stage:
        # Append / update Gram-Schmidt orthogonalization with current column
        gs_squared_norms[: stage + 1], gs_coeff_matrix[:, : stage + 1] = gso_step(
            basis_matrix[:, : stage + 1],
            gs_coeff_matrix[:, : stage + 1],
            gs_squared_norms[: stage + 1],
            stage,
        )

        # Size reduction step
        f_c, gs_coeff_matrix, basis_matrix_matrix = size_reduction_loop(
            stage, gs_coeff_matrix, basis_matrix, f_c
        )

        # Check for cumulated floating-point inaccuracies
        if f_c:
            f_c = False
            stage = max(stage - 1, 1)
            continue

        # Lovaz condition check (Columns of spanning matrix correctly ordered?)
        if (
            Lovasz_cond_param * gs_squared_norms[stage - 1]
            > gs_squared_norms[stage]
            + gs_coeff_matrix[stage - 1, stage] ** 2 * gs_squared_norms[stage - 1]
        ):
            # If ordering incorrect:
            # Execute column swap
            basis_matrix[:, [stage - 1, stage]] = basis_matrix[:, [stage, stage - 1]]
            # step back
            stage = max(stage - 1, 1)
        else:
            # If ordering correct -> move to next step
            stage += 1
            pbar.update(1)

    pbar.close()

    return basis_matrix, gs_coeff_matrix, gs_squared_norms
