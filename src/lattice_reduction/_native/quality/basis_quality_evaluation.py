import numpy as np

from .basis_quality_characteristics import (
    compute_column_norms,
    compute_root_hermite_factor,
    compute_lattice_volume_log,
    compute_orthogonality_defect,
)


def compute_basis_quality_characteristics(basis, reduced):
    """Computes key quality metrics for a lattice basis.
    This function evaluates the shortest vector norm, Hermite factor, and
    orthogonality defect for a given lattice basis. If the basis is not reduced,
    column norms are sorted to provide a more meaningful comparison.

    Args:
        basis (np.ndarray):
            A 2D NumPy array of shape (n, n) representing the lattice basis.

        reduced (bool):
            Indicates whether the basis is already reduced. If False, column norms
            are sorted before computing metrics.

    Returns:
        (list): A list containing Shortest column norm (float), Hermite factor (float), Orthogonality defect (float).
    """
    lattice_dim = basis.shape[1]
    log_lattice_volume = compute_lattice_volume_log(basis)
    col_norms = compute_column_norms(basis)
    if not reduced:
        col_norms = np.sort(col_norms)
    root_hermite_factor = compute_root_hermite_factor(col_norms[0], log_lattice_volume, lattice_dim)
    orthogonality_defect = compute_orthogonality_defect(col_norms, log_lattice_volume, lattice_dim)

    return [col_norms[0], root_hermite_factor, orthogonality_defect]
