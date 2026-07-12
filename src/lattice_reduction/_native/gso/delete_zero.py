"""删除零向量列。"""

import numpy as np


def delete_zero_vector(spanning_matrix, gs_squared_norms, gs_coeff_matrix, stage):
    """删除 spanning_matrix 的第 stage 列（零向量），同步截断 GSO 数组。"""
    spanning_matrix = np.delete(spanning_matrix, stage, axis=1)
    gs_squared_norms = np.delete(gs_squared_norms, stage)
    gs_coeff_matrix = np.delete(gs_coeff_matrix, stage, axis=1)
    gs_coeff_matrix = np.delete(gs_coeff_matrix, stage, axis=0)
    return spanning_matrix, gs_squared_norms, gs_coeff_matrix
