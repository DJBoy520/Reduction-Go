import numpy as np


def enum_sh_solver(basis_block, gs_squared_norms, gs_coeffs):
	"""Performs a shortest vector enumeration within a given lattice block using
	Schnorr-Hörner's improved enumeration strategy for lattice reduction.

	This function explores integer coefficient combinations for the basis vectors
	in `basis_block` to find the shortest non-zero lattice vector in the sublattice
	spanned by the block. It uses Gram-Schmidt norms and coefficients for pruning
	and backtracking during the search.

	Args:
	    basis_block (np.ndarray):
	        A 2D array of shape (block_size, dimension) representing the lattice basis
	        vectors for the current block.
	    gs_squared_norms (np.ndarray):
	        A 1D array of length `block_size` containing the squared norms of the
	        Gram-Schmidt orthogonalized basis vectors. Used for pruning during enumeration.
	    gs_coeffs (np.ndarray):
	        A 2D array of shape (block_size, block_size) containing Gram-Schmidt
	        coefficients for projections between basis vectors.

	Returns:
	    (np.ndarray):
	        - If a shorter vector than the pruning bound is found, returns the new lattice
	          vector as a 1D NumPy array of length `dimension`.
	        - Otherwise, returns an empty list.

	Notes:
	    - Implements the recursive enumeration tree described in Schnorr & Hörner (1995),
	      "Attacking the Chor-Rivest cryptosystem by improved lattice reduction".
	    - The pruning condition is based on the current squared norm compared to the
	      best found so far (`search_radius`).
	    - Uses rounding of projections to select integer coefficients efficiently.

	Complexity:
	    Exponential in block size (k), but optimized by pruning using Gram-Schmidt norms.

	References:
	    Claus-Peter Schnorr and Horst Helmut Hörner,
	    *"Attacking the Chor-Rivest cryptosystem by improved lattice reduction"*,
	    EUROCRYPT 1995.
	"""
	# Number of columns (dimension of the sublattice) -> the current block size
	k = len(basis_block[0])
	# Squared norms of each Gram-Schmidt vectors (used for pruning)
	# c = gs_squared_norms (in original paper)
	# Initialize tilde_c, tilde_u, u, y, tri, v with zero entries
	tilde_c = np.zeros(k + 1)  # Partial squared norms during enumeration
	tilde_u = np.zeros(k + 1)  # Stores current coefficient vector
	u = np.zeros(k)  # Best coefficient vector found
	y = np.zeros(k)  # Stores intermediate projections
	# Initialize s, t as zero
	t_max, t = 0, 0  # s = max enumeration tree depth reached, t = current index in recursion
	# Stores the best/smallest squared norm found so far (notated as "barred_c" in the original paper)
	search_radius = gs_squared_norms[0]
	tilde_u[0], u[0] = 1, 1  # Initialize first coefficient

	# This loop explores all possible integer coefficients of the lattice basis vectors, backtracking if necessary.
	while t < k:
		#  Compute the squared length of the current enumerated vector using Gram-Schmidt norms.
		tilde_c[t] = (
			tilde_c[t + 1] + np.square(y[t] + tilde_u[t]) * gs_squared_norms[t]
		)  # Helps prune out long vectors early.
		# Check if the current vector is shorter than the best found so far (the pruning condition)
		# If true (tilde_c[t] shorter than previously found vectors) -> continue downward in the search tree
		if tilde_c[t] < search_radius:
			if t > 0:
				t -= 1  # Move deeper in enumeration
				# Compute projection
				y[t] = np.dot(
					tilde_u[t + 1 : t_max + 1], gs_coeffs[t, t + 1 : t_max + 1]
				)  # Sum_{i=t+1}^s tilde_u[i] * gs_coeffs[t, i]
				tilde_u[t] = round(-y[t])  # Round to the nearest integer
			else:
				search_radius = tilde_c[0]  # Update best squared norm found
				u[:k] = tilde_u[:k]  # Update best coefficient vector
		else:  # If the current vector is too long, execution moves the search back up the enumeration tree
			t += 1  # Move back up -> reconsider the next coefficient in the sequence
			t_max = max(t_max, t)  # Update max enumeration depth reached
			if t == t_max:
				tilde_u[t] += 1
			else:
				tilde_u[t] = next(tilde_u[t], -y[t])
	
	return search_radius, u[:k]

def next(a, r):
	if r > a:
		return a - 1
	else:
		return a + 1