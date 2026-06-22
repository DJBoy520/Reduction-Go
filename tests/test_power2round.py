#!/usr/bin/env python3
"""Power2Round 往返测试 — 验证编码/解码正确性。"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
from src.protocol_adapter import (
    ProtocolAdapter, power2round_encode, power2round_decode,
    get_error_bound, D_BY_PARAMS
)
from src.keygen import keygen
from src.params import get_params


def test_roundtrip():
    """测试: encode → decode(r1, r0_raw) 应该精确还原 r。"""
    print("=" * 60)
    print("测试 1: Power2Round 往返一致性")
    print("=" * 60)

    q = 8380417
    rng = np.random.default_rng(42)

    for d in [10, 13]:
        adapter = ProtocolAdapter(d=d, q=q)
        print(f"\n  d={d}, {adapter.describe()}")

        for _ in range(1000):
            r = rng.integers(0, q, size=30)
            r1, r0_raw, r0_c = adapter.encode(r)
            r_recon = adapter.decode(r1, t0_raw=r0_raw)

            assert np.array_equal(r, r_recon), \
                f"往返失败! d={d}, r={r[:3]}, r1={r1[:3]}, r0_raw={r0_raw[:3]}"

        print("    ✓ 1000 组往返一致 (encode → decode 精确还原)")


def test_error_bound():
    """测试: t0_centered 严格在误差范围内。"""
    print("\n" + "=" * 60)
    print("测试 2: t0 误差范围验证")
    print("=" * 60)

    q = 8380417
    rng = np.random.default_rng(123)

    for d in [10, 13]:
        adapter = ProtocolAdapter(d=d, q=q)
        lo, hi = adapter.get_slack_range()

        r = rng.integers(0, q, size=10000)
        _, r0_raw, r0_c = adapter.encode(r)

        assert np.all(r0_raw >= 0), f"r0_raw 下溢: min={r0_raw.min()}"
        assert np.all(r0_raw < (1 << d)), f"r0_raw 上溢: max={r0_raw.max()}"
        assert np.all(r0_c >= lo), f"r0_c 下溢: min={r0_c.min()}, expected >= {lo}"
        assert np.all(r0_c <= hi), f"r0_c 上溢: max={r0_c.max()}, expected <= {hi}"

        print(f"  d={d}: r0_raw ∈ [0, {1<<d}), r0_c ∈ [{r0_c.min()}, {r0_c.max()}] ⊂ [{lo}, {hi}] ✓")


def test_high_bits_only():
    """测试: 只用 t1 还原时，误差在预期范围内。"""
    print("\n" + "=" * 60)
    print("测试 3: 仅高位还原误差分析")
    print("=" * 60)

    q = 8380417
    rng = np.random.default_rng(999)

    for d in [10, 13]:
        adapter = ProtocolAdapter(d=d, q=q)

        r = rng.integers(0, q, size=10000)
        r1, r0_raw, r0_c = adapter.encode(r)

        # 只用 t1 还原（高位近似）
        r_approx = adapter.decode(r1)
        # 误差 = r - r_approx = r0_raw (范围 [0, 2^d))
        error = (r - r_approx).astype(np.int64)
        # 居中: 若 error > 2^{d-1}，则 error -= 2^d
        error_c = error.copy()
        two_d = 1 << d
        error_c[error > (1 << (d - 1))] -= two_d

        print(f"  d={d}: 还原误差 ∈ [{error_c.min()}, {error_c.max()}], "
              f"|误差| ≤ {adapter.error_bound} ✓")
        assert np.all(np.abs(error_c) <= adapter.error_bound)


def test_with_keygen():
    """测试: 用真实密钥生成验证 Power2Round。"""
    print("\n" + "=" * 60)
    print("测试 4: 真实密钥生成 + Power2Round")
    print("=" * 60)

    p = get_params("easy")
    p["k"], p["l"], p["n"] = 2, 2, 30
    seed = (12345).to_bytes(8, "big")
    rho, s1, s2, t, A = keygen("easy", seed=seed, params=p)

    d = 10
    adapter = ProtocolAdapter(d=d, q=p["q"])

    print(f"\n  t shape: {t.shape}, t 范围: [{t.min()}, {t.max()}]")

    t1, t0_raw, t0_c = adapter.encode(t)
    print(f"  t1 范围: [{t1.min()}, {t1.max()}]")
    print(f"  t0_raw 范围: [{t0_raw.min()}, {t0_raw.max()}]")
    print(f"  t0_c 范围: [{t0_c.min()}, {t0_c.max()}] (误差上界 ±{adapter.error_bound})")

    # 精确还原
    t_recon = adapter.decode(t1, t0_raw=t0_raw) % p["q"]
    assert np.array_equal(t % p["q"], t_recon % p["q"]), "精确还原失败!"
    print(f"  ✓ t = t1·2^d + t0_raw (mod q) 精确还原一致")

    # 方程验证
    from src.poly_math import mat_vec_mul, vec_add_mod
    lhs = vec_add_mod(mat_vec_mul(A, s1, p["q"]), s2, p["q"])
    assert np.array_equal(lhs % p["q"], t % p["q"]), "方程不成立!"
    print(f"  ✓ A·s1 + s2 ≡ t (mod q) 方程成立")

    # 关键指标
    t0_norm = float(np.linalg.norm(t0_c.flatten()))
    s_norm = float(np.sqrt(np.sum(s1**2) + np.sum(s2**2)))
    print(f"\n  秘密范数 ||(s1,s2)|| = {s_norm:.2f}")
    print(f"  误差范数 ||t0_c||    = {t0_norm:.2f}")
    print(f"  误差/秘密 比值       = {t0_norm/s_norm:.2f}x")
    print(f"  → 格攻击需要同时找到范数 ~{s_norm:.0f} 的秘密和吸收 ~{t0_norm:.0f} 的误差")


def test_slack_lattice_preview():
    """预览: 带松弛变量的格基维度。"""
    print("\n" + "=" * 60)
    print("测试 5: 带松弛格基维度预览")
    print("=" * 60)

    for name in ["easy", "medium", "hard", "extreme"]:
        p = get_params(name)
        k, l, n = p["k"], p["l"], p["n"]
        d = D_BY_PARAMS.get(name, 10)

        ln = l * n
        kn = k * n

        dim_kannan = ln + kn + 1
        dim_slack = ln + kn + 2  # +1 嵌入, +1 松弛

        print(f"  {name:10s}: k={k} l={l} n={n} d={d}  "
              f"Kannan={dim_kannan}  Slack={dim_slack}")


if __name__ == "__main__":
    test_roundtrip()
    test_error_bound()
    test_high_bits_only()
    test_with_keygen()
    test_slack_lattice_preview()
    print("\n" + "=" * 60)
    print("全部测试通过 ✓")
    print("=" * 60)
