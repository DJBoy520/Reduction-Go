from .bkz_schnorr_euchner import bkz_se
from .bkz_schnorr_euchner_progress_check import bkz_se_pc

BKZ_ALGORITHMS = {
    "1": bkz_se_pc,
    "2": bkz_se,
}
