import numpy as np


def basis_gen(dim, entry_bound):
	"""Generates a random full-rank integer lattice basis.
	This function creates a square integer matrix of size `dim x dim` where each entry
	is sampled uniformly from `[0, entry_bound - 1]`. The matrix is constructed row by row,
	ensuring that the rank increases with each addition. If adding a new row does not
	increase the rank, that row is discarded. The final matrix is transposed so that
	basis vectors are represented as columns.

	Args:
	    dim (int):
	        Dimension of the lattice basis (number of basis vectors).
	        Defaults to `bkz.constants.N`.
	    entry_bound (int):
	        Upper bound for random integer entries (exclusive).
	        Defaults to `bkz.constants.ENTRY_BOUND`.

	Returns:
	    (np.ndarray):
	        A 2D NumPy array of shape `(dim, dim)` representing a full-rank lattice

	"""

	matrix = np.empty((0, dim), int)

	while matrix.shape[0] < dim:
		vec = np.random.randint(0, entry_bound, dim)
		matrix = np.vstack([matrix, vec])
		if np.linalg.matrix_rank(matrix) != matrix.shape[0]:
			matrix = matrix[:-1]

	return matrix.T
