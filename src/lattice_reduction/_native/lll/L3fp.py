import numpy as np

from ..gso.gsofp_se import gso_step
from ..gso.initializer import initialize
from ..jit_compat import jit_available
from ..jit_functions import l3fp_jit, l3fp_jit_int
from .L3fp_params import LOVASZ_CONDITION_PARAM, TAU
from .reducer import size_reduction_loop


def _init_gso(basis_matrix, gs_coeff_matrix, gs_squared_norms, start_stage):
    """初始化 GSO 数组，不转换基的 dtype。

    与 initialize() 不同，保留 basis 的原始 dtype（如 int64），
    使精确整数点积成为可能。
    """
    m = basis_matrix.shape[1]

    if start_stage == 0:
        stage = 1
        gs_coeff_matrix = np.zeros((m, m), dtype=np.float64)
        gs_coeff_matrix[0, 0] = 1.0
        gs_squared_norms = np.zeros(m, dtype=np.float64)
    else:
        stage = start_stage
        pad = m - gs_coeff_matrix.shape[1]
        if pad > 0:
            gs_coeff_matrix = np.pad(
                gs_coeff_matrix, ((0, pad), (0, pad)), mode="constant"
            )
            gs_squared_norms = np.pad(
                gs_squared_norms, (0, pad), mode="constant"
            )
        gs_coeff_matrix = gs_coeff_matrix.astype(np.float64)
        gs_squared_norms = gs_squared_norms.astype(np.float64)

    return gs_coeff_matrix, gs_squared_norms, stage


def l3fp(
    basis_matrix,
    gs_coeff_matrix=None,
    gs_squared_norms=None,
    start_stage=0,
    Lovasz_cond_param=LOVASZ_CONDITION_PARAM,
    f_c=False,
):
    """Floating-point LLL reduction (Schnorr-Euchner 1994).

    策略（参考 fplll）:
      - int64 基 → 精确整数点积 + Kahan GSO + 定期完整刷新
      - float64 基 → float64 路径（deep insertion 兼容）
      - 无 Numba → 纯 Python 回退

    Args:
        basis_matrix: (n, m) numpy array, columns are basis vectors.
        gs_coeff_matrix: (m, m) GSO coefficients, or None.
        gs_squared_norms: (m,) GSO squared norms, or None.
        start_stage: Starting stage index.
        Lovasz_cond_param: δ parameter in (0.25, 1.0).
        f_c: Floating-point precision flag.

    Returns:
        (basis_matrix, gs_coeff_matrix, gs_squared_norms)
    """
    gs_coeff_matrix, gs_squared_norms, stage = _init_gso(
        basis_matrix, gs_coeff_matrix, gs_squared_norms, start_stage
    )
    end_stage = basis_matrix.shape[1]

    if jit_available:
        tau_limit = 2.0 ** (TAU / 2)
        if basis_matrix.dtype == np.int64:
            # 快速路径: 精确整数点积，基保持 int64
            l3fp_jit_int(
                basis_matrix,
                gs_coeff_matrix,
                gs_squared_norms,
                stage,
                Lovasz_cond_param,
                tau_limit,
            )
        elif basis_matrix.dtype == np.float64:
            # float64 路径 (deep insertion 返回 float64)
            l3fp_jit(
                basis_matrix,
                gs_coeff_matrix,
                gs_squared_norms,
                stage,
                Lovasz_cond_param,
                tau_limit,
            )
        else:
            # 其他类型 → 转 int64 用精确路径
            bm = basis_matrix.astype(np.int64)
            l3fp_jit_int(
                bm,
                gs_coeff_matrix,
                gs_squared_norms,
                stage,
                Lovasz_cond_param,
                tau_limit,
            )
            basis_matrix[:] = bm
    else:
        # 纯 Python 回退
        bm = (
            basis_matrix.astype(np.float64)
            if basis_matrix.dtype != np.float64
            else basis_matrix
        )
        gsc = gs_coeff_matrix
        gs = gs_squared_norms
        delta = Lovasz_cond_param

        while stage < end_stage:
            gs[: stage + 1], gsc[:, : stage + 1] = gso_step(
                bm[:, : stage + 1],
                gsc[:, : stage + 1],
                gs[: stage + 1],
                stage,
            )

            f_c, gsc, bm = size_reduction_loop(stage, gsc, bm, f_c)

            if f_c:
                f_c = False
                stage = max(stage - 1, 1)
                continue

            mu = gsc[stage - 1, stage]
            if delta * gs[stage - 1] > gs[stage] + mu * mu * gs[stage - 1]:
                bm[:, [stage - 1, stage]] = bm[:, [stage, stage - 1]]
                stage = max(stage - 1, 1)
            else:
                stage += 1

        if basis_matrix.dtype != np.float64:
            basis_matrix[:] = np.round(bm).astype(basis_matrix.dtype)

    return basis_matrix, gs_coeff_matrix, gs_squared_norms
