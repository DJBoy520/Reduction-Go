import numpy as np


def initialize(spanning_matrix, gs_coeff_matrix, gs_squared_norms, start_stage):
	"""L3fp and L3fp_deep_insert functions are called at different phases and on different stages during the bkz execution.
	This function is used to initialize the LLL-setup (Gram-Schmidt coefficients & Gram-Schmidt squared norms)
	based on the start_stage and spanning_matrix (which correspond to `basis_matrix` or `injected_basis_matrix`) dimensions. Basically, if start_stage is equal to zero
	it just initializes the NumPy arrays `gs_coeff_matrix` and `gs_squared_norms`. In other situations, it scales the current
	`gs_coeff_matrix` and `gs_squared_norms` to match `basis_matrix`/`injected_basis_matrix` dimensions.

	Args:
		spanning_matrix (np.ndarray):
			A 2D NumPy array of shape (n, m), where n<=m, representing a vector space
			(which correspond to basis_matrix or injected_basis_matrix).

		gs_coeff_matrix (np.ndarray):
			A 2D NumPy array of shape (start_stage, start_stage) representing the Gram-Schmidt coefficients
	        of the spanning_matrix. If start_stage=0, then has value None.

		gs_squared_norms (np.ndarray):
			A 1D NumPy array of shape (start_stage, ) representing the squared lengths of the
	        Gram-Schmidt vectors. If start_stage=0, then has value None.

		start_stage (int): The index of the basis vector from which the reduction process begins.

	Returns:
		(tuple):
			- spanning_matrix (np.ndarray)

			- gs_coeff_matrix (np.ndarray):
				A 2D NumPy array of shape (m, m) representing the Gram-Schmidt coefficients
				of the spanning_matrix.

			- gs_squared_norms (np.ndarray):
				A 1D NumPy array of shape (m, ) representing the squared lengths of the
				Gram-Schmidt vectors.

			- stage (int): The index of the basis vector that is currently under investigation.

			- end_stage (int): The index at which the current l3fp/l3fp_deep_insert loop will terminate.
	"""
	span_width = len(spanning_matrix[0])
	end_stage = span_width
	spanning_matrix = spanning_matrix.astype(np.float64)
	if start_stage == 0:
		stage = 1
		gs_coeff_matrix = np.zeros((span_width, span_width), dtype=np.float64)
		gs_coeff_matrix[0, 0] = 1.0
		gs_squared_norms = np.zeros(span_width, dtype=np.float64)

	else:
		stage = start_stage
		# Execute padding to match the number of columns in spanning_matrix
		pad_dimensions = spanning_matrix.shape[1] - gs_coeff_matrix.shape[1]
		gs_coeff_matrix = np.pad(
			gs_coeff_matrix,
			((0, pad_dimensions), (0, pad_dimensions)),
			mode="constant",
			constant_values=0,
		)
		gs_coeff_matrix = gs_coeff_matrix.astype(np.float64)
		gs_squared_norms = np.pad(
			gs_squared_norms, (0, pad_dimensions), mode="constant", constant_values=0
		)
		gs_squared_norms = gs_squared_norms.astype(np.float64)

	return spanning_matrix, gs_coeff_matrix, gs_squared_norms, stage, end_stage
