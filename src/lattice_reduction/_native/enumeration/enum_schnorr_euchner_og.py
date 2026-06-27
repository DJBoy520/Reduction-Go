import numpy as np

def enum_se_og_solver(basis_block, gs_squared_norms, gs_coeffs):
    """Performs shortest vector enumeration using the *original* Schnorr–Euchner
    	(1991, FCT) strategy on a lattice block.

    	This routine explores integer coefficient vectors for the block basis in
    	`basis_block` to find a short lattice vector in the sublattice spanned by the
    	block. It uses Gram–Schmidt squared norms and coefficients for pruning and
    	computes the next candidate coefficient via a ceil-bound derived from the
    	remaining radius, rather than the controlled stepping refinement that appeared later.

    	Args:
    	    basis_block (np.ndarray):
    	        A 2D array of shape (dimension, block_size) representing the lattice
    	        basis vectors for the current block (columns are the basis
    	        vectors used to form the sublattice).
    	    gs_squared_norms (np.ndarray):
    	        A 1D array of length `block_size` with the squared lengths of the
    	        Gram–Schmidt orthogonalized vectors. Used as pruning weights
    	        in the partial norm accumulation.
    	    gs_coeffs (np.ndarray):
    	        A 2D array of shape (dimension, block_size) containing the Gram–Schmidt
    	        coefficients used to compute projections.

    	Returns:
    	    (tuple):
    	        - search_bound (float): The smallest accumulated squared norm found
    	          during enumeration for the current block.
    	        - u (np.ndarray): A 1D array of length `block_size` giving the integer
    	          coefficient vector that attains `search_bound`. This corresponds
    	          to the candidate shortest lattice vector `basis_block @ u`.

    	Notes:
    	    - This follows the early Schnorr–Euchner enumeration (FCT 1991), where the
    	      next coefficient for index `t` is chosen using:
    	          `tilde_u[t] = ceil( -y[t] - sqrt((min_squared_norm - tilde_c[t+1]) / c_t) )`,
    	      with `c_t = gs_squared_norms[t]` and `y[t]` computed from μ-coefficients.
    	    - The pruning condition compares the partial squared norm
    	      `tilde_c[t] = tilde_c[t+1] + (y[t] + tilde_u[t])² * c_t`
    	      against the current best `search_radius`. If it exceeds, the search
    	      backtracks by increasing `t`.
    	    - Unlike the later Schnorr–Euchner variant (1994), this version does not
    	      use the symmetric controlled stepping (`delta`, `tri`, `v`) around the
    	      rounded value; it uses a direct bound-based update via `ceil`.
    	    - `u[0] = 1` seeds the initial vector; the algorithm updates `search_bound`
    	      only when at least one coefficient is non-zero.
    	    - The final lattice vector can be recovered by `basis_block @ u` if needed.

    	References:
    	    - Claus-Peter Schnorr and Martin Euchner,
    	      "Lattice basis reduction: Improved practical algorithms and solving subset
    	      sum problems", *International Symposium on Fundamentals of Computation
    	      Theory (FCT)*, 1991, pp. 68–85.
    	"""
    # Step 1 (initiation)
    k = len(basis_block[0]) - 1 # Fixed for indexing that starts from 0.
    search_radius = gs_squared_norms[0]
    tilde_c = np.zeros(k + 2)
    tilde_u = np.zeros(k + 2)
    u = np.zeros(k + 1)
    y = np.zeros(k + 1)
    t = k
    u[0] = 1

    # (Initial) Step 2
    # y[t] = 0 for t=k
    y[t] = 0
    # For y[t]=0, tilde_c[t+1]=0
    # -> tilde_u[t] = np.ceil(-0 - np.sqrt((search_radius - 0) / gs_squared_norms[t]))
    tilde_u[t] = np.ceil(-np.sqrt(search_radius/gs_squared_norms[t]))

    while True:
        # Step 3
        tilde_c[t] = (tilde_c[t + 1] + np.square(y[t] + tilde_u[t]) * gs_squared_norms[t])
        if tilde_c[t] < search_radius:
            if t > 0:
                t -= 1  # go to step 2
                # step 2
                y[t] = np.dot(tilde_u[t + 1: k + 1], gs_coeffs[t, t + 1: k + 1])
                tilde_u[t] = np.ceil(-y[t] - np.sqrt((search_radius - tilde_c[t + 1]) / gs_squared_norms[t]))
                continue # go to step 3
            elif np.count_nonzero(tilde_u) != 0:
                search_radius = tilde_c[0]
                u[:k + 1] = tilde_u[:k + 1]
        else:
            t += 1
        # Step 4
        if t <= k:
            tilde_u[t] += 1
            # go to step 3
        else:
            break

    return search_radius, u[:k + 1]