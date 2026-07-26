"""AttackConfig 数据类 — 格攻击配置的唯一定义。

从 src/api.py 和 src/common/config.py 迁移至 domain 层，提供类型安全的访问方式。
合并 FPLLConfig 字段，作为项目配置的唯一来源。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .exceptions import ConfigError


@dataclass
class AttackConfig:
    """格攻击配置参数（合并 FPLLConfig）。

    Attributes:
        params_name: ML-DSA 参数集名称（"toy", "ML-DSA-44", ...）
        k: 矩阵行维度（可选，覆盖参数集默认值）
        l: 矩阵列维度（可选）
        n: 多项式维度（可选）
        no_bkz: 禁用 BKZ，仅使用 LLL
        bkz_block_size: BKZ 块大小（≥ 2）
        bkz_max_loops: BKZ 最大循环次数
        bkz_threads: BKZ 并行线程数
        bkz_auto_abort: BKZ 自动中止
        lll_delta: LLL δ 参数（0.25 ~ 1.0）
        seed: 随机种子
        use_slack: 启用 slack 向量
        d: Power2Round 低位比特数
        cert_path: 证书文件路径
        toy_params: 使用 toy 参数
        output_dir: 输出目录
        auto_precision: 启用自动精度分层
        mp_dps: mpmath 精度（小数位数）
        float_type: 浮点类型（"mpfr" / "mpf"）
        precision: 浮点精度位数
        verbose: 启用详细输出
        log_level: 日志级别
    """
    # 基本参数
    params_name: str = "toy"
    k: Optional[int] = None
    l: Optional[int] = None
    n: Optional[int] = None

    # LLL / BKZ
    no_bkz: bool = False
    bkz_block_size: Optional[int] = None
    bkz_max_loops: Optional[int] = None
    bkz_threads: Optional[int] = None
    bkz_auto_abort: bool = False
    lll_delta: float = 0.79

    # 随机种子
    seed: Optional[int] = None

    # Power2Round
    use_slack: bool = False
    d: Optional[int] = None

    # 证书模式
    cert_path: Optional[str] = None
    toy_params: bool = False

    # 输出
    output_dir: Optional[str] = None

    # 精度控制
    auto_precision: bool = True
    mp_dps: Optional[int] = None
    float_type: Optional[str] = None
    precision: Optional[int] = None

    # 日志
    verbose: bool = False
    log_level: Optional[str] = None

    def validate(self) -> None:
        """验证配置有效性。

        Raises:
            ConfigError: 配置参数无效
        """
        if self.k is not None and self.k < 1:
            raise ConfigError(f"k must be ≥ 1, got {self.k}")
        if self.l is not None and self.l < 1:
            raise ConfigError(f"l must be ≥ 1, got {self.l}")
        if self.n is not None and self.n < 1:
            raise ConfigError(f"n must be ≥ 1, got {self.n}")
        if self.bkz_block_size is not None and self.bkz_block_size < 2:
            raise ConfigError(f"bkz_block_size must be ≥ 2, got {self.bkz_block_size}")
        if not (0.25 <= self.lll_delta <= 1.0):
            raise ConfigError(f"lll_delta must be in [0.25, 1.0], got {self.lll_delta}")
        if self.mp_dps is not None and self.mp_dps < 1:
            raise ConfigError(f"mp_dps must be ≥ 1, got {self.mp_dps}")

    def to_dict(self) -> dict[str, Any]:
        """转为 dict，跳过 None 值。"""
        return {k: v for k, v in self.__dict__.items() if v is not None}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AttackConfig:
        """从字典创建配置实例。

        Args:
            data: 配置字典

        Returns:
            AttackConfig 实例
        """
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


__all__ = ["AttackConfig"]
