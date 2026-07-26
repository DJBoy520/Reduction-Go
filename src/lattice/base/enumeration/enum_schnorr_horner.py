"""SVP 枚举器：Schnorr-Hörner 1995 变体。"""

import numpy as np


def enum_sh_solver(basis_block, gs_squared_norms, gs_coeffs):
    """Schnorr-Hörner SVP 枚举（1995）。"""
    k = len(basis_block[0])
    tilde_c = np.zeros(k + 1, dtype=object)
    tilde_u = np.zeros(k + 1, dtype=object)
    u = np.zeros(k, dtype=object)
    y = np.zeros(k, dtype=object)
    t_max, t = 0, 0
    search_radius = gs_squared_norms[0]
    tilde_u[0], u[0] = 1, 1

    while t < k:
        tilde_c[t] = tilde_c[t + 1] + (y[t] + tilde_u[t]) ** 2 * gs_squared_norms[t]
        if tilde_c[t] < search_radius:
            if t > 0:
                t -= 1
                y[t] = np.dot(tilde_u[t + 1: t_max + 1], gs_coeffs[t, t + 1: t_max + 1])
                tilde_u[t] = round(-y[t])
            else:
                search_radius = tilde_c[0]
                u[:k] = tilde_u[:k]
        else:
            t += 1
            t_max = max(t_max, t)
            if t == t_max:
                tilde_u[t] += 1
            else:
                tilde_u[t] = _zigzag_next(tilde_u[t], -y[t])

    return search_radius, u[:k]


def _zigzag_next(a, r):
    """zigzag 步进（不完整，建议用 enum_se_solver 或 enum_se_og_solver 替代）。"""
    return a - 1 if r > a else a + 1
