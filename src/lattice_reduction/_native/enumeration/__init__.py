from .enum_schnorr_euchner import enum_se_solver
from .enum_schnorr_euchner_og import enum_se_og_solver
from .enum_schnorr_horner import enum_sh_solver

ENUM_ALGORITHMS = {
    "1": enum_se_og_solver,
    "2": enum_se_solver,
    "3": enum_sh_solver,
}
