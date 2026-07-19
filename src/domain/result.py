"""格基验证结果数据类。

VerifyResult 用于 verify_basis 的返回值，取代原来的 bool 返回类型，
强制调用方显式处理验证结果，防止静默跳过验证。
"""

from dataclasses import dataclass


@dataclass
class VerifyResult:
    """格基验证的结构化结果。

    Attributes
    ----------
    valid_structure : bool
        结构是否合法（矩阵维度等基础检查）
    equation_checked : bool
        是否实际执行了方程验证（t_recon 提供时才执行）
    passed : bool
        方程是否成立（仅 equation_checked=True 时有意义）
    error : str
        失败原因（可选，用于人类可读诊断）
    """

    valid_structure: bool      # 结构是否合法（矩阵维度等）
    equation_checked: bool     # 是否实际执行了方程验证
    passed: bool               # 方程是否成立
    error: str = ""            # 失败原因（可选）

    def __bool__(self) -> bool:
        """兼容旧代码中 if verify_basis(...) 的写法。

        注意: 建议逐步替换为显式 .passed 检查。
        """
        return self.passed
