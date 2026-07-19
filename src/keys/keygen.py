"""ML-DSA 密钥生成 — 重新导出 src.protocol.keygen。

原始实现: src/protocol/keygen.py (FIPS 204 §4.2)
"""
from ..protocol.keygen import keygen, expand_a

__all__ = ["keygen", "expand_a"]
