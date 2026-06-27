import numpy as np


def delete_zero_vector(spanning_matrix, gs_squared_norms, gs_coeff_matrix, stage):
	"""Removes a zero column vector from the spanning matrix and updates associated Gram-Schmidt data.
	This function is called when a column in the spanning matrix becomes a zero vector due to
	linear dependencies introduced during deep insertion. It removes the zero vector at the
	specified `stage` index and updates the Gram-Schmidt squared norms and coefficient matrix
	accordingly by deleting the corresponding row and column.

	Args:
		spanning_matrix (np.ndarray):
			A 2D NumPy array of shape (n, m), where n=m-1, representing the current injected lattice basis.

		gs_squared_norms (np.ndarray):
			A 1D NumPy array of shape (m,), representing the squared norms of the Gram-Schmidt vectors.

		gs_coeff_matrix (np.ndarray):
			A 2D NumPy array of shape (m, m), representing the Gram-Schmidt coefficient matrix.

		stage (int):
			The index of the column to be removed due to it being a zero vector.

	Returns:
		(tuple):
			- spanning_matrix (np.ndarray):
				Updated lattice basis with the zero vector removed (shape becomes (n, n)).

			- gs_squared_norms (np.ndarray):
				Updated squared norms with the corresponding entry removed (shape becomes (n, )).

			- gs_coeff_matrix (np.ndarray):
				Updated Gram-Schmidt coefficient matrix with the corresponding row and column removed (shape becomes (n, n)).
	"""
	spanning_matrix = np.delete(spanning_matrix, stage, axis=1)
	gs_squared_norms = np.delete(gs_squared_norms, stage)
	gs_coeff_matrix = np.delete(gs_coeff_matrix, stage, axis=1)  # Delete consecutive column
	gs_coeff_matrix = np.delete(gs_coeff_matrix, stage, axis=0)  # Delete consecutive row

	return spanning_matrix, gs_squared_norms, gs_coeff_matrix
