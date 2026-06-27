import numpy as np


def compute_column_norms(basis):
	"""Computes the Euclidean norms of the columns of a basis matrix.

	Args:
	    basis (np.ndarray): A 2D NumPy array of shape (n, n) representing the lattice basis.

	Returns:
	    (np.ndarray): A 1D NumPy array of length n containing the norms of each column.

	"""
	return np.linalg.norm(basis, axis=0)


def compute_lattice_volume_log(basis):
	"""Compute log-volume of a full-rank square lattice basis B in a numerically stable way. For a square, full-rank basis B, `Vol(L) = |det(B)|`.

	Args:
	    basis (np.ndarray): A 2D NumPy array of shape (n, n) representing the lattice basis.

	Returns:
	    (float): The log-space value of the volume of the lattice.
	"""
	_, logdet = np.linalg.slogdet(basis)  # returns sign and log(abs(det(B)))
	return logdet


def compute_root_hermite_factor(shortest_vector_len, log_vol, dim):
	"""Computes the Root Hermite factor (RHF) for a lattice basis in a numerically stable way.
	The function receives standard (non-log) inputs, internally converts
	the values to log-space, performs the computation in log-space, and
	returns the result in normal numeric space.
	The Root Hermite factor is defined as `(||b_0|| / (Vol(L))^(1/n))^(1/n))`
	where `||b_0||` is the length of the shortest vector, `Vol(L)` is the lattice volume,
	and `n` is the dimension.

	Args:
	    shortest_vector_len (float): Length of the shortest vector in the basis.
	    vol (float): Volume of the lattice.
	    dim (int): Dimension of the lattice (n).

	Returns:
	    (float): The Root Hermite factor.
	"""
	# Convert inputs to log-space
	log_b0 = np.log(shortest_vector_len)
	# Compute RHF in log-space
	log_rhf = (log_b0 - log_vol / dim) / dim

	return np.exp(log_rhf)


def compute_hermite_factor(shortest_vector_len, log_vol, dim):
	"""Computes the Hermite factor for a lattice basis in a numerically stable way.
	The function receives standard (non-log) inputs, internally converts
	the values to log-space, performs the computation in log-space, and
	returns the result in normal numeric space.
	The Hermite factor is defined as `||b_0|| / (Vol(L))^(1/n)`
	where `||b_0||` is the length of the shortest vector, `Vol(L)` is the lattice volume,
	and `n` is the dimension.

	Args:
		shortest_vector_len (float): Length of the shortest vector in the basis.
		vol (float): Volume of the lattice.
		dim (int): Dimension of the lattice (n).

	Returns:
	    (float): The Hermite factor.
	"""
	# Convert inputs to log-space
	log_b0 = np.log(shortest_vector_len)
	# Compute HF in log-space
	log_rh = log_b0 - log_vol / dim
	return np.exp(log_rh)


def compute_orthogonality_defect(norms, log_vol, dim):
	"""Computes the *dimension-normalized* orthogonality defect of a lattice basis
	in a numerically stable way. This function calculates: `OD^(1/dim) = (prod(||b_i||) / Vol(L))^(1/dim)`
	where `prod(||b_i||)` is the product of the basis vector norms, `Vol(L)` is the lattice volume (absolute determinant), `dim` is the lattice dimension (number of basis vectors). Normalizing by dimension removes the exponential growth of `OD` with `n`, making the metric comparable across different lattice sizes.

	Args:
	    norms (np.ndarray): 1D NumPy array of basis vector norms `(||b_i||)`.
	    log_vol (float): Logarithm of the lattice volume, `log(|det(B)|)`.
	    dim (int): Lattice dimension (number of basis vectors).

	Returns:
	    (float): The per-dimension orthogonality defect, `OD^(1/dim)`. For an orthogonal basis, this equals 1. Larger values indicate less orthogonality.
	"""
	log_prod = np.sum(np.log(norms))
	log_defect = (log_prod - log_vol) / dim
	return np.exp(log_defect)
