"""SVP 枚举器：Schnorr-Euchner 1991 原始变体（ceil-bound）。"""

import numpy as np
import mpmath


def enum_se_og_solver(basis_block, gs_squared_norms, gs_coeffs):
    """Schnorr-Euchner SVP 枚举（1991，ceil-bound 方式）。"""
    k = len(basis_block[0]) - 1
    search_radius = gs_squared_norms[0]
    tilde_c = np.zeros(k + 2, dtype=object)
    tilde_u = np.zeros(k + 2, dtype=object)
    u = np.zeros(k + 1, dtype=object)
    y = np.zeros(k + 1, dtype=object)
    t = k
    u[0] = 1

    y[t] = 0
    tilde_u[t] = mpmath.ceil(-mpmath.sqrt(search_radius / gs_squared_norms[t]))

    while True:
        tilde_c[t] = tilde_c[t + 1] + (y[t] + tilde_u[t]) ** 2 * gs_squared_norms[t]
        if tilde_c[t] < search_radius:
            if t > 0:
                t -= 1
                y[t] = np.dot(tilde_u[t + 1: k + 1], gs_coeffs[t, t + 1: k + 1])
                tilde_u[t] = mpmath.ceil(-y[t] - mpmath.sqrt(
                    (search_radius - tilde_c[t + 1]) / gs_squared_norms[t]))
                continue
            elif sum(1 for x in tilde_u if x != 0) != 0:
                search_radius = tilde_c[0]
                u[:k + 1] = tilde_u[:k + 1]
        else:
            t += 1
        if t <= k:
            tilde_u[t] += 1
        else:
            break

    return search_radius, u[:k + 1]
