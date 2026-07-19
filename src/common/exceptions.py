"""FPLL 统一异常体系。

所有自定义异常均从此模块导出，便于上层统一捕获。
"""


class FPLLError(Exception):
    """FPLL 项目基础异常。"""
    pass


class ConfigError(FPLLError):
    """配置相关错误（参数缺失、类型不匹配等）。"""
    pass


class ParamsError(FPLLError):
    """ML-DSA 参数集错误（名称不存在、参数越界等）。"""
    pass


class LatticeError(FPLLError):
    """格约减算法错误。"""
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


class AttackError(FPLLError):
    """攻击流程错误。"""
    pass


class BasisError(AttackError):
    """Kannan 嵌入格基构造错误。"""
    pass


class ClassificationError(AttackError):
    """候选分类与验证错误。"""
    pass


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
