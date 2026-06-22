"""
Protocol Adapter — FIPS 204 Power2Round 编解码器。

FIPS 204 §4.1 (Power2Round)
============================
对 r ∈ Z_q，分解为高位 r1 和低位 r0：

    r1 = ⌊r / 2^d⌋           (高位，存入公钥)
    r0 = r - r1 · 2^d        (原始低位，非负，范围 [0, 2^d))
    r0c = r0                  若 r0 ≤ 2^{d-1}
    r0c = r0 - 2^d            若 r0 > 2^{d-1}   (居中，范围 [-2^{d-1}, 2^{d-1}))

还原：r = r1 · 2^d + r0  (mod q)   ← 用原始 r0
误差约束：|r0c| ≤ 2^{d-1}           ← 用居中 r0c

ML-DSA 参数中 d 的取值：
  - ML-DSA-44: d = 10
  - ML-DSA-65: d = 13
  - ML-DSA-87: d = 13
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

# FIPS 204 标准 d 值
D_BY_PARAMS = {
    "ML-DSA-44": 10,
    "ML-DSA-65": 13,
    "ML-DSA-87": 13,
    # toy 参数用较小的 d 做测试
    "toy": 10,
    "easy": 10,
}


def power2round_encode(r: np.ndarray, d: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Power2Round 编码：将 r 分解为 (r1, r0_raw, r0_centered)。

    Args:
        r: 原始多项式系数向量，值域 [0, q)
        d: 低位比特数

    Returns:
        (r1, r0_raw, r0_centered):
          - r1: 高位，存入公钥
          - r0_raw: 原始低位 [0, 2^d)，用于精确还原
          - r0_centered: 居中低位 [-2^{d-1}, 2^{d-1}]，用于误差约束
    """
    two_d = 1 << d
    half = 1 << (d - 1)

    r = r.astype(np.int64)
    r1 = r >> d
    r0_raw = r - r1 * two_d  # [0, 2^d)

    r0_centered = r0_raw.copy()
    r0_centered[r0_raw > half] -= two_d  # [-2^{d-1}, 2^{d-1}]

    return r1, r0_raw, r0_centered


def power2round_decode(r1: np.ndarray, d: int, r0_raw: np.ndarray | None = None) -> np.ndarray:
    """Power2Round 还原。

    若提供 r0_raw: r = r1 * 2^d + r0_raw  (精确还原)
    若不提供:      r_approx = r1 * 2^d     (高位近似，误差 ≤ 2^{d-1})

    Args:
        r1: 高位系数向量
        d: 低位比特数
        r0_raw: 原始低位（可选）

    Returns:
        还原值
    """
    r_recon = (r1.astype(np.int64) << d)
    if r0_raw is not None:
        r_recon = r_recon + r0_raw.astype(np.int64)
    return r_recon


def get_error_bound(d: int) -> int:
    """返回 t0 的误差上界：2^(d-1)。

    满足 |t0_centered| ≤ 2^(d-1)。
    """
    return 1 << (d - 1)


class ProtocolAdapter:
    """FIPS 204 协议适配器。

    处理公钥中 t1（高位）到完整 t 的转换，以及误差项管理。
    """

    def __init__(self, d: int, q: int = 8380417):
        self.d = d
        self.q = q
        self.error_bound = get_error_bound(d)
        logger.info(f"ProtocolAdapter: d={d}, error_bound=±{self.error_bound} (2^{d-1})")

    def encode(self, t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """完整 t → (t1, t0_raw, t0_centered)。模拟公钥压缩过程。"""
        return power2round_encode(t, self.d)

    def decode(self, t1: np.ndarray, t0_raw: np.ndarray | None = None) -> np.ndarray:
        """t1 → t_reconstructed。若提供 t0_raw 则精确还原。"""
        return power2round_decode(t1, self.d, t0_raw)

    def get_slack_range(self) -> tuple[int, int]:
        """返回误差项 t0 的取值范围。

        Returns:
            (lo, hi): t0 ∈ [lo, hi]，用于格基构造中松弛变量的约束。
        """
        half = 1 << (self.d - 1)
        return (-half, half)

    def describe(self) -> str:
        """返回适配器参数描述。"""
        lo, hi = self.get_slack_range()
        return (
            f"ProtocolAdapter(d={self.d}, q={self.q}, "
            f"error_bound=±{self.error_bound}, "
            f"t0 ∈ [{lo}, {hi}], "
            f"t1 精度 = {self.q.bit_length() - self.d} bit)"
        )
