from .L3fp_params import SIZE_REDUCTION_CONDITION_PARAM, TAU


def size_reduction_loop(stage, gs_coeff_matrix, spanning_matrix, f_c):
    """Size reduction for column `stage` of the GSO coefficient matrix.

    Iterates over predecessors in reverse order, reducing each coefficient
    that exceeds the size reduction threshold (1/2).

    Args:
        stage: Current column index.
        gs_coeff_matrix: (m, m) GSO coefficients.
        spanning_matrix: (n, m) basis vectors (columns).
        f_c: Floating-point precision flag.

    Returns:
        (f_c, gs_coeff_matrix, spanning_matrix)
    """
    gsc_col = gs_coeff_matrix[:, stage]
    threshold = SIZE_REDUCTION_CONDITION_PARAM
    tau_limit = 2 ** (TAU / 2)

    for i in range(stage - 1, -1, -1):
        if abs(gsc_col[i]) > threshold:
            mu = round(gsc_col[i])
            if abs(mu) > tau_limit:
                f_c = True

            # Vectorized: update all GSO coefficients at once
            gsc_col[:i] -= mu * gs_coeff_matrix[:i, i]
            gsc_col[i] -= mu

            # Update basis vector
            spanning_matrix[:, stage] -= mu * spanning_matrix[:, i]

    return f_c, gs_coeff_matrix, spanning_matrix
