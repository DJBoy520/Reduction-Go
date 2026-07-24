"""FPLL 统一配置类。

集中管理所有可配置参数，提供类型安全的访问方式。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .exceptions import ConfigError


@dataclass
class FPLLConfig:
    """FPLL 项目统一配置。

    Attributes:
        params_name: ML-DSA 参数集名称（如 "toy", "ML-DSA-44"）
        k: 矩阵行维度（可选，覆盖参数集默认值）
        l: 矩阵列维度（可选）
        n: 多项式维度（可选）
        no_bkz: 禁用 BKZ，仅使用 LLL
        bkz_block_size: BKZ 块大小
        bkz_max_loops: BKZ 最大循环次数
        bkz_threads: BKZ 并行线程数
        mp_dps: mpmath 精度（小数位数）
        seed: 随机种子（可复现）
        verbose: 启用详细输出
        log_level: 日志级别
        cert_path: 证书文件路径（可选）
        use_toy_params: 使用 toy 参数（证书攻击时）
    """
    # 基本参数
    params_name: str = "toy"
    k: int | None = None
    l: int | None = None
    n: int | None = None

    # LLL / BKZ
    no_bkz: bool = False
    bkz_block_size: int | None = None
    bkz_max_loops: int | None = None
    bkz_threads: int | None = None

    # 精度
    mp_dps: int | None = None

    # 随机种子
    seed: int | None = None

    # 日志
    verbose: bool = False
    log_level: str | None = None

    # 证书
    cert_path: str | None = None
    use_toy_params: bool = False

    def validate(self) -> None:
        """验证配置有效性。

        Raises:
            ConfigError: 配置参数无效
        """
        if self.k is not None and self.k < 1:
            raise ConfigError(f"k 必须 >= 1，当前值: {self.k}")
        if self.l is not None and self.l < 1:
            raise ConfigError(f"l 必须 >= 1，当前值: {self.l}")
        if self.n is not None and self.n < 1:
            raise ConfigError(f"n 必须 >= 1，当前值: {self.n}")
        if self.bkz_block_size is not None and self.bkz_block_size < 2:
            raise ConfigError(f"bkz_block_size 必须 >= 2，当前值: {self.bkz_block_size}")
        if self.mp_dps is not None and self.mp_dps < 15:
            raise ConfigError(f"mp_dps 必须 >= 15，当前值: {self.mp_dps}")

    def to_dict(self) -> dict[str, Any]:
        """转换为字典格式（兼容旧 API）。"""
        return {
            "params_name": self.params_name,
            "k": self.k,
            "l": self.l,
            "n": self.n,
            "no_bkz": self.no_bkz,
            "bkz_block_size": self.bkz_block_size,
            "bkz_max_loops": self.bkz_max_loops,
            "bkz_threads": self.bkz_threads,
            "mp_dps": self.mp_dps,
            "seed": self.seed,
            "verbose": self.verbose,
            "log_level": self.log_level,
            "cert_path": self.cert_path,
            "use_toy_params": self.use_toy_params,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FPLLConfig:
        """从字典创建配置实例。

        Args:
            data: 配置字典

        Returns:
            FPLLConfig 实例
        """
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)
