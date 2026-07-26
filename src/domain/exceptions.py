"""FPLL 统一异常体系 — 唯一定义源。

本模块是项目的最底层异常定义，不导入任何项目内其他模块（零依赖）。
所有自定义异常均从此模块导出，便于上层统一捕获。

合并来源:
  - src/common/exceptions.py (FPLLError 体系)
  - src/lattice/base/exceptions.py (LatticeReductionError 体系)
"""


class FPLLError(Exception):
    """FPLL 项目基础异常。"""
    pass


# ── 配置与参数异常 ──────────────────────────────────────────────────────────

class ConfigError(FPLLError):
    """配置相关错误（参数缺失、类型不匹配等）。"""
    pass


class ParamsError(FPLLError):
    """ML-DSA 参数集错误（名称不存在、参数越界等）。"""
    pass


# ── 格约减异常 ──────────────────────────────────────────────────────────────

class LatticeError(FPLLError):
    """格约减算法错误。"""
    pass


class LatticeReductionError(LatticeError):
    """格约减模块统一异常基类（向后兼容别名）。"""
    pass


class InvalidBasisError(LatticeError):
    """格基格式或内容非法。"""
    pass


class ReductionFailedError(LatticeError):
    """约减过程失败。"""
    pass


class GSOError(LatticeError):
    """GSO 正交化计算错误。"""
    pass


class LLLError(LatticeError):
    """LLL 约减失败。"""
    pass


class BKZError(LatticeError):
    """BKZ 约减失败。"""
    pass


class EnumerationError(LatticeError):
    """格枚举失败。"""
    pass


class PrecisionError(LatticeError):
    """精度不足或溢出。"""
    pass


class PrecisionFailureError(PrecisionError):
    """精度不足，可通过升级 dps 重试。

    Attributes:
        reason: 失败原因描述
        current_dps: 当前精度（小数位数）
    """

    def __init__(self, reason: str, current_dps: int):
        self.reason = reason
        self.current_dps = current_dps
        super().__init__(f"Precision failure at dps={current_dps}: {reason}")


# ── 攻击流程异常 ─────────────────────────────────────────────────────────────

class AttackError(FPLLError):
    """攻击流程错误。"""
    pass


class BasisError(AttackError):
    """Kannan 嵌入格基构造错误。"""
    pass


class ClassificationError(AttackError):
    """候选分类与验证错误。"""
    pass


# ── 协议异常 ─────────────────────────────────────────────────────────────────

class ProtocolError(FPLLError):
    """协议/密钥/证书处理错误。"""
    pass


class KeyGenError(ProtocolError):
    """密钥生成错误。"""
    pass


class CertParseError(ProtocolError):
    """证书解析错误。"""
    pass


class CertGenError(ProtocolError):
    """证书生成错误。"""
    pass


class DERError(ProtocolError):
    """DER 编解码错误。"""
    pass


__all__ = [
    # 基础
    "FPLLError",
    # 配置与参数
    "ConfigError",
    "ParamsError",
    # 格约减
    "LatticeError",
    "LatticeReductionError",
    "InvalidBasisError",
    "ReductionFailedError",
    "GSOError",
    "LLLError",
    "BKZError",
    "EnumerationError",
    "PrecisionError",
    "PrecisionFailureError",
    # 攻击流程
    "AttackError",
    "BasisError",
    "ClassificationError",
    # 协议
    "ProtocolError",
    "KeyGenError",
    "CertParseError",
    "CertGenError",
    "DERError",
]
