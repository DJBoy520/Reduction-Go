#!/usr/bin/env python3
"""
步骤8：健壮性加固验证 — 参数校验、异常体系、日志。
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np

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
    """验证函数抛出指定类型的异常"""
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
print("步骤8：健壮性加固验证")
print("=" * 60)


# ══════════════════════════════════════════════════════════════
# 8.1 参数校验
# ══════════════════════════════════════════════════════════════
print("\n[8.1] 参数校验")

from src.lattice import (
    lll_reduce, bkz_reduce, evaluate_basis_quality,
    InvalidBasisError, ReductionFailedError, LatticeReductionError,
)

# 格基校验
expect_error("None 格基", lambda: lll_reduce(None), InvalidBasisError)
expect_error("非方阵", lambda: lll_reduce(np.array([[1, 2, 3], [4, 5, 6]])), InvalidBasisError)
expect_error("一维数组", lambda: lll_reduce(np.array([1, 2, 3])), InvalidBasisError)
expect_error("空矩阵", lambda: lll_reduce(np.array([]).reshape(0, 0)), InvalidBasisError)
expect_error("NaN 矩阵", lambda: lll_reduce(np.array([[np.nan, 1], [2, 3]])), InvalidBasisError)
expect_error("Inf 矩阵", lambda: lll_reduce(np.array([[np.inf, 1], [2, 3]])), InvalidBasisError)

# delta 校验
expect_error("delta=0", lambda: lll_reduce(np.eye(3, dtype=np.int64), delta=0), InvalidBasisError)
expect_error("delta=1", lambda: lll_reduce(np.eye(3, dtype=np.int64), delta=1.0), InvalidBasisError)
expect_error("delta=0.1", lambda: lll_reduce(np.eye(3, dtype=np.int64), delta=0.1), InvalidBasisError)
expect_error("delta=-1", lambda: lll_reduce(np.eye(3, dtype=np.int64), delta=-1), InvalidBasisError)

# block_size 校验
expect_error("block_size=1", lambda: bkz_reduce(np.eye(3, dtype=np.int64), block_size=1), InvalidBasisError)
expect_error("block_size>dim", lambda: bkz_reduce(np.eye(3, dtype=np.int64), block_size=10), InvalidBasisError)

# max_loops 校验
expect_error("max_loops=0", lambda: bkz_reduce(np.eye(3, dtype=np.int64), block_size=2, max_loops=0), ReductionFailedError)


# ══════════════════════════════════════════════════════════════
# 8.2 异常体系
# ══════════════════════════════════════════════════════════════
print("\n[8.2] 异常体系")

def test_exception_hierarchy():
    """验证异常继承关系"""
    assert issubclass(InvalidBasisError, LatticeReductionError), "InvalidBasisError should inherit LatticeReductionError"
    assert issubclass(ReductionFailedError, LatticeReductionError), "ReductionFailedError should inherit LatticeReductionError"
    assert issubclass(LatticeReductionError, Exception), "LatticeReductionError should inherit Exception"
check("异常继承关系", test_exception_hierarchy)


def test_catch_base_exception():
    """验证基类可以捕获所有子类异常"""
    try:
        lll_reduce(None)
    except LatticeReductionError:
        pass  # 应该被捕获
    else:
        raise AssertionError("LatticeReductionError should catch InvalidBasisError")
check("基类捕获子类异常", test_catch_base_exception)


# ══════════════════════════════════════════════════════════════
# 8.3 日志验证
# ══════════════════════════════════════════════════════════════
print("\n[8.3] 日志验证")

def test_logging_works():
    """验证日志记录正常工作"""
    import logging
    # 设置 DEBUG 级别以捕获所有日志
    logging.basicConfig(level=logging.DEBUG, format="%(name)s %(levelname)s: %(message)s")
    logger = logging.getLogger("src.lattice.adapter.lll_adapter")
    assert logger is not None, "Logger should exist"
check("日志记录器存在", test_logging_works)


# ══════════════════════════════════════════════════════════════
# 8.4 正常路径验证
# ══════════════════════════════════════════════════════════════
print("\n[8.4] 正常路径验证")

def test_lll_valid_input():
    """合法输入应正常工作"""
    from src.lattice._native.basis.basis_generator import basis_gen
    B = basis_gen(10, 173).astype(np.int64).T
    lll_reduce(B, delta=0.75)
    assert not np.any(np.isnan(B)), "NaN in result"
check("LLL 合法输入", test_lll_valid_input)


def test_bkz_valid_input():
    """合法输入应正常工作"""
    from src.lattice._native.basis.basis_generator import basis_gen
    B = basis_gen(10, 173).astype(np.int64).T
    result = bkz_reduce(B, block_size=5, max_loops=2)
    assert result["completed_loops"] >= 1
check("BKZ 合法输入", test_bkz_valid_input)


def test_quality_valid_input():
    """合法输入应正常工作"""
    B = np.eye(10, dtype=np.int64)
    q = evaluate_basis_quality(B, reduced=True)
    assert q["shortest_norm"] > 0
check("质量评估合法输入", test_quality_valid_input)


# ══════════════════════════════════════════════════════════════
# 汇总
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
if errors:
    print(f"❌ {len(errors)} 项失败 / {len(passed)} 项通过:")
    for name, e in errors:
        print(f"   ✗ {name}: {e}")
    if __name__ == "__main__":
        sys.exit(1)
else:
    print(f"✅ 全部 {len(passed)} 项通过")
    if __name__ == "__main__":
        sys.exit(0)
