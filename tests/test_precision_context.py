#!/usr/bin/env python3
"""
P0-1 验证：PrecisionContext.handle_precision_failure 单元测试。

覆盖场景：
  1. max_retry=0 → 不允许升级 → 抛出 ReductionFailedError
  2. 允许重试 (mode=AUTO, max_retry>=1) → 升级成功 → 返回新 dps，retry_count 递增
  3. 连续失败：第一次升级成功，第二次达到上限后抛出 ReductionFailedError
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np

from src.lattice.base.exceptions import (
    PrecisionFailureError, ReductionFailedError,
)
from src.lattice.base.precision import PrecisionContext
from src.lattice.base.precision_constants import (
    MP_DPS_LOW, MP_DPS_HIGH,
    PRECISION_MODE_AUTO, PRECISION_MODE_LOW,
)

errors = []
passed = []


def check(name, fn):
    try:
        fn()
        print(f"  ✓ {name}")
        passed.append(name)
    except Exception as e:
        print(f"  ✗ {name}: {e}")
        errors.append((name, e))


def expect_error(name, fn, error_type):
    try:
        fn()
        print(f"  ✗ {name}: 未抛出异常，期望 {error_type.__name__}")
        errors.append((name, Exception(f"Expected {error_type.__name__}")))
    except error_type as e:
        print(f"  ✓ {name}: {error_type.__name__}: {e}")
        passed.append(name)
    except Exception as e:
        print(f"  ✗ {name}: 抛出 {type(e).__name__}，期望 {error_type.__name__}: {e}")
        errors.append((name, e))


print("=" * 60)
print("P0-1: PrecisionContext.handle_precision_failure 测试")
print("=" * 60)


# ══════════════════════════════════════════════════════════════
# 场景 1：max_retry=0, mode=AUTO → 不允许升级 → 应抛出 ReductionFailedError
# ══════════════════════════════════════════════════════════════
print("\n[1] max_retry=0 → 不允许升级")

def test_no_retry_raises():
    basis = np.eye(5, dtype=np.float64)
    ctx = PrecisionContext(
        original_basis=basis, dim=5,
        initial_dps=MP_DPS_LOW, mode=PRECISION_MODE_AUTO,
        max_retry=0,
    )
    exc = PrecisionFailureError("NaN detected", ctx.current_dps)
    ctx.handle_precision_failure(exc)  # 应该抛出

expect_error(
    "max_retry=0 抛出 ReductionFailedError",
    test_no_retry_raises,
    ReductionFailedError,
)


# ══════════════════════════════════════════════════════════════
# 场景 1b：max_retry=0, mode=LOW (非AUTO) → 同样不允许升级
# ══════════════════════════════════════════════════════════════
print("\n[1b] mode=LOW (非AUTO) → 不允许升级")

def test_low_mode_raises():
    basis = np.eye(5, dtype=np.float64)
    ctx = PrecisionContext(
        original_basis=basis, dim=5,
        initial_dps=MP_DPS_LOW, mode=PRECISION_MODE_LOW,
        max_retry=5,  # 即使 max_retry>0，mode 不是 AUTO 也不允许
    )
    exc = PrecisionFailureError("stuck", ctx.current_dps)
    ctx.handle_precision_failure(exc)

expect_error(
    "mode=LOW 抛出 ReductionFailedError",
    test_low_mode_raises,
    ReductionFailedError,
)


# ══════════════════════════════════════════════════════════════
# 场景 2：允许重试 → 升级成功 → 返回新 dps，retry_count 递增
# ══════════════════════════════════════════════════════════════
print("\n[2] 允许重试 → 升级成功")

def test_upgrade_success():
    basis = np.eye(5, dtype=np.float64)
    ctx = PrecisionContext(
        original_basis=basis, dim=5,
        initial_dps=MP_DPS_LOW, mode=PRECISION_MODE_AUTO,
        max_retry=2,
    )
    assert ctx.retry_count == 0, f"初始 retry_count 应为 0，实际为 {ctx.retry_count}"
    assert ctx.current_dps == MP_DPS_LOW, f"初始 dps 应为 {MP_DPS_LOW}"

    exc = PrecisionFailureError("GSO negative", ctx.current_dps)
    new_dps = ctx.handle_precision_failure(exc)

    assert new_dps == MP_DPS_HIGH, f"升级后 dps 应为 {MP_DPS_HIGH}，实际为 {new_dps}"
    assert ctx.current_dps == MP_DPS_HIGH, f"ctx.current_dps 应为 {MP_DPS_HIGH}"
    assert ctx.retry_count == 1, f"retry_count 应为 1，实际为 {ctx.retry_count}"

check("允许重试时升级成功，返回 MP_DPS_HIGH，retry_count=1", test_upgrade_success)


# ══════════════════════════════════════════════════════════════
# 场景 3：连续失败 — 第一次升级成功，第二次达到上限抛异常
# ══════════════════════════════════════════════════════════════
print("\n[3] 连续失败 → 第一次成功，第二次抛异常")

def test_consecutive_failure():
    basis = np.eye(5, dtype=np.float64)
    ctx = PrecisionContext(
        original_basis=basis, dim=5,
        initial_dps=MP_DPS_LOW, mode=PRECISION_MODE_AUTO,
        max_retry=1,
    )
    # 第一次：应该成功升级
    exc1 = PrecisionFailureError("first failure", ctx.current_dps)
    new_dps = ctx.handle_precision_failure(exc1)
    assert new_dps == MP_DPS_HIGH
    assert ctx.retry_count == 1

    # 第二次：已达上限，应该抛出
    exc2 = PrecisionFailureError("second failure", ctx.current_dps)
    ctx.handle_precision_failure(exc2)

expect_error(
    "连续失败：第二次抛出 ReductionFailedError",
    test_consecutive_failure,
    ReductionFailedError,
)


# ══════════════════════════════════════════════════════════════
# 场景 4：异常链保留 — ReductionFailedError.__cause__ 是 PrecisionFailureError
# ══════════════════════════════════════════════════════════════
print("\n[4] 异常链保留验证")

def test_exception_chain():
    basis = np.eye(5, dtype=np.float64)
    ctx = PrecisionContext(
        original_basis=basis, dim=5,
        initial_dps=MP_DPS_LOW, mode=PRECISION_MODE_AUTO,
        max_retry=0,
    )
    original_exc = PrecisionFailureError("test reason", ctx.current_dps)
    try:
        ctx.handle_precision_failure(original_exc)
        raise AssertionError("应该抛出异常")
    except ReductionFailedError as e:
        assert e.__cause__ is original_exc, (
            f"ReductionFailedError.__cause__ 应为原始 PrecisionFailureError，"
            f"实际为 {type(e.__cause__)}"
        )

check("ReductionFailedError.__cause__ 保留原始异常", test_exception_chain)


# ══════════════════════════════════════════════════════════════
# 场景 5：PrecisionFailureError.reason 正确传递
# ══════════════════════════════════════════════════════════════
print("\n[5] reason 传递验证")

def test_reason_in_message():
    basis = np.eye(5, dtype=np.float64)
    ctx = PrecisionContext(
        original_basis=basis, dim=5,
        initial_dps=MP_DPS_LOW, mode=PRECISION_MODE_AUTO,
        max_retry=0,
    )
    original_exc = PrecisionFailureError("specific failure reason", ctx.current_dps)
    try:
        ctx.handle_precision_failure(original_exc)
    except ReductionFailedError as e:
        msg = str(e)
        assert "specific failure reason" in msg, f"错误信息应包含 reason，实际: {msg}"
        assert str(ctx.dim) in msg, f"错误信息应包含 dim，实际: {msg}"

check("错误信息包含 reason 和 dim", test_reason_in_message)


# ══════════════════════════════════════════════════════════════
# 结果汇总
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(f"结果: {len(passed)} 通过, {len(errors)} 失败")
print("=" * 60)

if errors:
    print("\n失败项:")
    for name, e in errors:
        print(f"  ✗ {name}: {e}")
    sys.exit(1)
else:
    print("\n✅ 全部通过！handle_precision_failure 方法实现正确。")
    sys.exit(0)
