import numpy as np

LOVASZ_CONDITION_PARAM = 3 / 4  # Determined range ]1/4, 1[

SIZE_REDUCTION_CONDITION_PARAM = 1 / 2

# EPSILON = 1e-10

TAU = 40


def update_tau(basis_matrix, precision_level="default"):
	"""Calls compute_tau and updates the module-global TAU.

	Args:
	    basis_matrix (np.ndarray): A 2D NumPy array of shape (n, m) representing a lattice basis, where each column is treated as a basis vector.
		precision_level (str): Determines the scaling of TAU.
	"""
	global TAU
	TAU = compute_tau(basis_matrix, precision_level)


def compute_tau(basis_matrix, precision_level="default"):
	"""Computes TAU parameter based on the average norm of the input basis matrix.

	Args:
	    basis_matrix (np.ndarray): A 2D NumPy array of shape (n, m) representing a lattice basis, where each column is treated as a basis vector.
	    precision_level (str): Determines the scaling of TAU. `high`: favors higher TAU values for increased precision. TAU in range [30,80]. `low`: favors lower TAU values for faster computation. TAU in range [10,40]. `default`: balanced TAU scaling. TAU in range [20,60].

	Returns:
	    (int): A dynamically computed TAU value, scaled based on the average norm of the basis vectors and the selected precision level.
	"""
	norms = np.linalg.norm(basis_matrix, axis=0)
	avg_norm = np.mean(norms)

	if precision_level == "high":
		return max(30, min(80, int(np.log2(avg_norm) * 2)))
	elif precision_level == "low":
		return max(10, min(40, int(np.log2(avg_norm))))
	else:  # default
		return max(20, min(60, int(np.log2(avg_norm) * 1.5)))
