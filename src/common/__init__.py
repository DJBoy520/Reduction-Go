"""FPLL 通用基础模块。

提供统一的异常、参数、日志和工具函数。
所有领域定义统一从 domain 层重导出。
"""

from ..domain import (
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
    get_params,
    get_d,
    MLDSA_Q,
    PARAMS,
    MLDSA_REGISTRY,
    AttackConfig,
)
from .logger import setup_logging
from .utils import Timer, format_duration, ensure_list, flatten_dict

# 向后兼容别名
FPLLConfig = AttackConfig

__all__ = [
    # Config
    "FPLLConfig",
    "AttackConfig",
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
