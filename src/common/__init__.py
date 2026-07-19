"""FPLL 通用基础模块。

提供统一的配置、异常、参数、日志和工具函数。
"""

from .config import FPLLConfig
from .exceptions import (
    FPLLError,
    ConfigError,
    ParamsError,
    LatticeError,
    GSOError,
    LLLError,
    BKZError,
    EnumerationError,
    PrecisionError,
    AttackError,
    BasisError,
    ClassificationError,
    ProtocolError,
    KeyGenError,
    CertParseError,
    CertGenError,
    DERError,
)
from .params import get_params, get_d, MLDSA_Q, PARAMS, MLDSA_REGISTRY
from .logger import setup_logging
from .utils import Timer, format_duration, ensure_list, flatten_dict

__all__ = [
    # Config
    "FPLLConfig",
    # Exceptions
    "FPLLError",
    "ConfigError",
    "ParamsError",
    "LatticeError",
    "GSOError",
    "LLLError",
    "BKZError",
    "EnumerationError",
    "PrecisionError",
    "AttackError",
    "BasisError",
    "ClassificationError",
    "ProtocolError",
    "KeyGenError",
    "CertParseError",
    "CertGenError",
    "DERError",
    # Params
    "get_params",
    "get_d",
    "MLDSA_Q",
    "PARAMS",
    "MLDSA_REGISTRY",
    # Logger
    "setup_logging",
    # Utils
    "Timer",
    "format_duration",
    "ensure_list",
    "flatten_dict",
]
