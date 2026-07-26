"""SVP 枚举器：Schnorr-Euchner 1994 变体。"""

import numpy as np


def enum_se_solver(basis_block, gs_squared_norms, gs_coeffs):
    """Schnorr-Euchner SVP 枚举（1994，controlled stepping）。"""
    k = len(basis_block[0]) - 1
    tilde_c = np.zeros(k + 2, dtype=object)
    tilde_u = np.zeros(k + 2, dtype=object)
    u = np.zeros(k + 1, dtype=object)
    y = np.zeros(k + 1, dtype=object)
    tri = np.zeros(k + 2, dtype=object)
    v = np.zeros(k + 2, dtype=object)
    delta = np.ones(k + 2, dtype=object)
    s, t = 0, 0
    min_squared_norm = gs_squared_norms[0]
    tilde_u[0], u[0] = 1, 1

    while t <= k:
        tilde_c[t] = tilde_c[t + 1] + (y[t] + tilde_u[t]) ** 2 * gs_squared_norms[t]
        alpha = min(1.05 * (k - t + 1) / k, 1)
        if tilde_c[t] < alpha * min_squared_norm:
            if t > 0:
                t -= 1
                y[t] = np.dot(tilde_u[t + 1: s + 1], gs_coeffs[t, t + 1: s + 1])
                tilde_u[t] = round(-y[t])
                v[t] = tilde_u[t]
                tri[t] = 0
                delta[t] = -1 if tilde_u[t] > (-y[t]) else 1
            else:
                min_squared_norm = tilde_c[0]
                u[:k + 1] = tilde_u[:k + 1]
        else:
            t += 1
            s = max(s, t)
            if t < s:
                tri[t] *= -1
            if tri[t] * delta[t] >= 0:
                tri[t] += delta[t]
            tilde_u[t] = v[t] + tri[t]

    return min_squared_norm, u[:k + 1]
