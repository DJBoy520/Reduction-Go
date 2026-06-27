from .L3fp_params import SIZE_REDUCTION_CONDITION_PARAM, TAU


def size_reduction_loop(stage, gs_coeff_matrix, spanning_matrix, f_c):
	"""Performs size reduction on the specified column of the Gram-Schmidt coefficient matrix.
	This function iterates over the Gram-Schmidt coefficients of the column indexed by `stage`,
	checking whether each coefficient satisfies the size reduction condition. If the absolute
	value of a coefficient exceeds the predefined threshold, the `reduce` function is invoked
	to update the basis and Gram-Schmidt data accordingly.

	Args:
	    stage (int): The index of the basis vector that is currently under investigation.

	    gs_coeff_matrix (np.ndarray):
	        A 2D NumPy array of shape (m, m) representing the Gram-Schmidt coefficients
	        of the spanning_matrix.

	    spanning_matrix (np.ndarray):
	        A 2D NumPy array of shape (n, m), where n<=m, representing a vector space
	        (which correspond to basis_matrix or injected_basis_matrix).

	    f_c (bool):
	        A flag used to track floating-point precision issues.

	Returns:
	    (tuple):
	        - f_c (bool): A flag used to track floating-point precision issues.

	        - gs_coeff_matrix (np.ndarray): Updated Gram-Schmidt coefficient matrix of shape (m, m).

	        - spanning_matrix (np.ndarray): Updated spanning matrix of shape (n, m).
	"""
	for i in range(stage - 1, -1, -1):
		if abs(gs_coeff_matrix[i, stage]) > SIZE_REDUCTION_CONDITION_PARAM:
			f_c, gs_coeff_matrix[:, stage], spanning_matrix[:, stage] = reduce(
				f_c,
				gs_coeff_matrix[:, stage],
				gs_coeff_matrix[:, i],
				spanning_matrix[:, stage],
				spanning_matrix[:, i],
				i,
			)
		# This part of the algorithm documentation is a bit unclear.
		# END if |gs_coeff_matrix[i, stage]|
		# if abs(gs_coeff_matrix[i, stage]) < 1e-10:
		#    break

	return f_c, gs_coeff_matrix, spanning_matrix


def reduce(f_c, gsc_k, gsc_l, spanning_vec_k, spanning_vec_l, l):
	"""Performs size reduction by subtracting a multiple of one basis vector from another, updating the corresponding Gram-Schmidt coefficients and spanning vector.
	This function computes `mu = round(gsc_k[l])` and, if `mu` exceeds a threshold, sets the
	floating-point precision flag `f_c` to `True`. It then updates `spanning_vec_k`
	by subtracting mu times `spanning_vec_l`, respectively. Values in `gsc_l` are also
	updated based on mu and the contents of `gsc_l`.

	Args:
		f_c (bool): A flag used to track floating-point precision issues. May be updated.
		gsc_k (np.ndarray): The k-th column of Gram-Schmidt coefficent matrix.
		gsc_l (np.ndarray): The l-th column of Gram-Schmidt coefficent matrix.
		spanning_vec_k (np.ndarray): The k-th column of spanning_matrix.
		spanning_vec_l (np.ndarray): The l-th column of spanning_matrix.
		l (int): Index representing the column position of gsc_l and spanning_vec_l.

	Returns:
		(tuple):
			- f_c (bool): Possibly updated flag for tracking floating-point precision issues.

			- gsc_k (np.ndarray): Updated k-th column of the Gram-Schmidt coefficent matrix.

			- spanning_vec_k (np.ndarray): Updated k-th column of the spanning_matrix.
	"""

	mu = round(gsc_k[l])
	if abs(mu) > 2 ** (TAU / 2):
		f_c = True
	for j in range(0, l):
		gsc_k[j] -= mu * gsc_l[j]

	gsc_k[l] -= mu
	spanning_vec_k -= mu * spanning_vec_l

	return f_c, gsc_k, spanning_vec_k
