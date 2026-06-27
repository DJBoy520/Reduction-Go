import numpy as np


def gso_step(basis_slice, gs_coeff_matrix, gs_squared_norms, stage):
	"""Updates the Gram-Schmidt coefficient matrix and squared norms at a specific stage.
	This function performs one step of the Gram-Schmidt orthogonalization process,
	updating the entries in `gs_coeff_matrix` and `gs_squared_norms` corresponding
	to the given `stage`. It is assumed that all entries up to `stage - 1` are already
	correct and up to date. If `stage == 1`, the squared norm at index 0 is also updated.


	args:
		basis_slice (np.ndarray):
			2D NumPy array of shape (stage, stage) corresponding the slice of
			basis_matrix or injected_basis_matrix.

		gs_coeff_matrix (np.ndarray):
			A 2D NumPy array of shape (stage, stage), representing the Gram-Schmidt
	        coefficients. Values at index `stage` may not be up to date.


		gs_squared_norms (np.ndarray):
			A 1D NumPy array of shape (stage,), representing the squared lengths of the
	        Gram-Schmidt vectors. The value at index `stage` may not be up to date.


	returns:
		(tuple):
			- gs_squared_norms (np.ndarray):
				Gram-Schmidt squared norms with updated value(s) at index `stage` (and index 0 if `stage == 1`).

			- gs_coeff_matrix (np.ndarray):
				Gram-Schmidt coefficient matrix with updated values in column `stage`.
	"""
	if stage == 1:
		gs_squared_norms[0] = np.dot(basis_slice[:, 0], basis_slice[:, 0])

	gs_squared_norms[stage] = np.dot(
		basis_slice[:, stage], basis_slice[:, stage]
	)  # Initial value is the squared norm of b_stage
	for j in range(stage):  # Compute mu[j, i] for j < i using the refined formula
		dot_product = np.dot(basis_slice[:, stage], basis_slice[:, j])  # <b_k, b_j>
		correction_term = sum(
			gs_coeff_matrix[k, j] * gs_coeff_matrix[k, stage] * gs_squared_norms[k]
			for k in range(j)
		)
		gs_coeff_matrix[j, stage] = (dot_product - correction_term) / gs_squared_norms[j]

		# Update squared norm using the iterative formula
		gs_squared_norms[stage] -= (gs_coeff_matrix[j, stage] ** 2) * gs_squared_norms[j]

	gs_coeff_matrix[stage, stage] = 1.0  # Diagonal elements should be 1 (by definition)

	return gs_squared_norms[: stage + 1], gs_coeff_matrix[:, : stage + 1]
