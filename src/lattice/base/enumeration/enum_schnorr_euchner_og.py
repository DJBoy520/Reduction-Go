"""SVP 枚举器：Schnorr-Euchner 1991 原始变体（ceil-bound）。"""

import numpy as np


def enum_se_og_solver(basis_block, gs_squared_norms, gs_coeffs):
    """Schnorr-Euchner SVP 枚举（1991，ceil-bound 方式）。"""
    k = len(basis_block[0]) - 1
    search_radius = gs_squared_norms[0]
    tilde_c = np.zeros(k + 2)
    tilde_u = np.zeros(k + 2)
    u = np.zeros(k + 1)
    y = np.zeros(k + 1)
    t = k
    u[0] = 1

    y[t] = 0
    tilde_u[t] = np.ceil(-np.sqrt(search_radius / gs_squared_norms[t]))

    while True:
        tilde_c[t] = tilde_c[t + 1] + np.square(y[t] + tilde_u[t]) * gs_squared_norms[t]
        if tilde_c[t] < search_radius:
            if t > 0:
                t -= 1
                y[t] = np.dot(tilde_u[t + 1: k + 1], gs_coeffs[t, t + 1: k + 1])
                tilde_u[t] = np.ceil(-y[t] - np.sqrt(
                    (search_radius - tilde_c[t + 1]) / gs_squared_norms[t]))
                continue
            elif np.count_nonzero(tilde_u) != 0:
                search_radius = tilde_c[0]
                u[:k + 1] = tilde_u[:k + 1]
        else:
            t += 1
        if t <= k:
            tilde_u[t] += 1
        else:
            break

    return search_radius, u[:k + 1]
