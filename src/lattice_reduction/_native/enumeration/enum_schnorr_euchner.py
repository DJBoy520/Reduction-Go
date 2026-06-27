import numpy as np


def enum_se_solver(basis_block, gs_squared_norms, gs_coeffs):
    """Performs shortest vector enumeration using the Schnorr–Euchner strategy for
	lattice basis reduction within a given block.

	This algorithm enumerates integer coefficient combinations for the basis vectors
	in `basis_block` to find the shortest non-zero lattice vector in the sublattice
	spanned by the block. It improves upon naive enumeration by using controlled
	stepping and pruning based on Gram-Schmidt norms, as described in Schnorr &
	Euchner (1994).

	Args:
	    basis_block (np.ndarray):
	        A 2D array of shape (dimension, block_size) representing the lattice basis
	        vectors for the current block.
	    gs_squared_norms (np.ndarray):
	        A 1D array of length `block_size` containing the squared norms of the
	        Gram-Schmidt orthogonalized basis vectors. Used for pruning during enumeration.
	    gs_coeffs (np.ndarray):
	        A 2D array of shape (dimension, block_size) containing Gram-Schmidt
	        coefficients for projections between basis vectors.

	Returns:
	    (tuple):
	        - search_radius (float): The smallest squared norm found during enumeration.
	        - u (np.ndarray): A 1D array of length `block_size` representing the integer
	          coefficient vector corresponding to the shortest lattice vector found.

	Notes:
	    - Implements the Schnorr–Euchner enumeration algorithm, which uses a depth-first
	      search with controlled stepping (`delta`, `tri`) to efficiently explore the
	      enumeration tree.
	    - The pruning condition is based on comparing the partial squared norm
	      (`tilde_c[t]`) to the best found so far (`search_radius`).
	    - The algorithm uses rounding and adaptive stepping to minimize redundant search
	      paths and improve practical performance.

	References:
	    Claus-Peter Schnorr and Martin Euchner,
	    "Lattice basis reduction: Improved practical algorithms and solving subset sum problems",
	    Mathematical Programming, 1994.
    """
    k = len(basis_block[0]) - 1
    tilde_c = np.zeros(k + 2)  # Partial squared norms during enumeration
    tilde_u = np.zeros(k + 2)  # Stores current coefficient vector
    u = np.zeros(k +1)  # Best coefficient vector found
    y = np.zeros(k + 1)  # Stores intermediate projections
    tri = np.zeros(k + 2)  # Controls stepping during enumeration
    v = np.zeros(k + 2)  # Stores rounded values of tilde_u
    delta = np.ones(k + 2)  # Controls direction of stepping
    s, t = 0, 0  # s = max enumeration tree depth reached, t = current index in recursion
    min_squared_norm = gs_squared_norms[0]  # Start with max first squared norm !!!!HOX!!!
    tilde_u[0], u[0] = 1, 1  # Initialize first coefficient

    while t <= k:
        #  Compute the squared length of the current enumerated vector using Gram-Schmidt norms.
        # Helps prune out long vectors early.
        tilde_c[t] = tilde_c[t + 1] + np.square(y[t] + tilde_u[t]) * gs_squared_norms[t]
        # Check if the current vector is shorter than the best found so far (the pruning condition)
        # If true (tilde_c[t] shorter than previously found vectors) -> continue downward in the search tree
        alpha = np.minimum(1.05 * (k - t + 1) / k, 1)
        if tilde_c[t] < alpha * min_squared_norm:  # HOX alpha
            if t > 0:
                t -= 1  # Move deeper in enumeration
                # Compute projection Sum_{i=t+1}^s tilde_u[i] * gs_coeffs[t, i]
                y[t] = np.dot(tilde_u[t + 1 : s + 1], gs_coeffs[t, t + 1 : s + 1])
                tilde_u[t] = round(-y[t])  # Round to the nearest integer
                v[t] = tilde_u[t]  # Store rounded value
                tri[t] = 0  # Reset step controller
                # Adjust stepping direction depending on which one leads closer to y[t]
                if tilde_u[t] > (-y[t]):
                    delta[t] = -1  # Step left (decrease value)
                else:
                    delta[t] = 1  # Step right (increase value)
            else:
                min_squared_norm = tilde_c[0]  # Update best squared norm found
                u[:k + 1] = tilde_u[:k + 1]  # Update best coefficient vector
        else:  # If the current vector is too long, execution moves the search back up the enumeration tree
            t += 1  # Move back up -> reconsider the next coefficient in the sequence
            s = max(s, t)  # Update max enumeration depth reached
            if t < s:
                # Flipping the sign ensures that we switch directions in the search.
                tri[t] *= -1
            if tri[t] * delta[t] >= 0:
                # Adjust stepping size -> delta[t] determines whether to increase or decrease tri[t]
                tri[t] += delta[t]
            # Compute next coefficient
            # Starting from the initially rounded value v[t], adding tri[t] ensures controlled stepping
            tilde_u[t] = v[t] + tri[t]

    return min_squared_norm, u[:k + 1]

