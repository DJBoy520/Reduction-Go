#!/usr/bin/env python3
"""适配层冒烟验证 — 验证 lll_reduce / bkz_reduce / evaluate_basis_quality 接口正常。"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np

errors = []

def check(name, fn):
    try:
        fn()
        print(f"  ✓ {name}")
    except Exception as e:
        print(f"  ✗ {name}: {e}")
        errors.append((name, e))

print("=" * 50)
print("适配层冒烟验证")
print("=" * 50)

# ── 模块导入 ──
print("\n[1] 模块导入")
check("lattice_reduction 导入", lambda: __import__("src.lattice_reduction"))
check("lll_reduce 导入", lambda: __import__("src.lattice_reduction", fromlist=["lll_reduce"]))
check("bkz_reduce 导入", lambda: __import__("src.lattice_reduction", fromlist=["bkz_reduce"]))
check("evaluate_basis_quality 导入", lambda: __import__("src.lattice_reduction", fromlist=["evaluate_basis_quality"]))

# ── LLL 约减 ──
print("\n[2] LLL 约减（适配层）")
def test_lll_adapter():
    from src.lattice_reduction import lll_reduce
    from src.lattice_reduction._native.basis.basis_generator import basis_gen
    basis = basis_gen(10, 173).astype(np.int64)
    B = basis.T.copy()  # 转为行向量
    original = B.copy()
    lll_reduce(B, delta=0.75)
    # 约减后应该不同（除非已经约减）
    assert B.shape == original.shape, f"Shape changed: {B.shape} vs {original.shape}"
    # 约减后范数应该更小或相等
    orig_norms = np.linalg.norm(original, axis=1)
    new_norms = np.linalg.norm(B, axis=1)
    assert new_norms[0] <= orig_norms[0] * 2, "First vector norm should not explode"
check("LLL 约减 (dim=10)", test_lll_adapter)

# ── BKZ 约减 ──
print("\n[3] BKZ 约减（适配层）")
def test_bkz_adapter():
    from src.lattice_reduction import bkz_reduce
    from src.lattice_reduction._native.basis.basis_generator import basis_gen
    basis = basis_gen(10, 173).astype(np.int64)
    B = basis.T.copy()  # 转为行向量
    original = B.copy()
    result = bkz_reduce(B, block_size=5, max_loops=2, enum_algo="1")
    assert isinstance(result, dict), f"Expected dict, got {type(result)}"
    assert "completed_loops" in result, "Missing completed_loops"
    assert "shortest_norms" in result, "Missing shortest_norms"
    assert result["completed_loops"] >= 1, "Should complete at least 1 loop"
    assert B.shape == original.shape, f"Shape changed: {B.shape}"
check("BKZ 约减 (dim=10, block=5)", test_bkz_adapter)

# ── 质量评估 ──
print("\n[4] 格基质量评估（适配层）")
def test_quality_adapter():
    from src.lattice_reduction import evaluate_basis_quality, lll_reduce
    from src.lattice_reduction._native.basis.basis_generator import basis_gen
    basis = basis_gen(10, 173).astype(np.int64)
    B = basis.T.copy()
    lll_reduce(B, delta=0.75)
    result = evaluate_basis_quality(B, reduced=True)
    assert isinstance(result, dict), f"Expected dict, got {type(result)}"
    assert "shortest_norm" in result, "Missing shortest_norm"
    assert "root_hermite_factor" in result, "Missing root_hermite_factor"
    assert "hermite_factor" in result, "Missing hermite_factor"
    assert "orthogonality_defect" in result, "Missing orthogonality_defect"
    assert result["shortest_norm"] > 0, f"Shortest norm should be > 0: {result['shortest_norm']}"
check("格基质量评估", test_quality_adapter)

# ── 一致性测试：适配层 vs 原生层 ──
print("\n[5] 一致性测试：适配层 LLL 结果应与原生层一致")
def test_consistency():
    from src.lattice_reduction import lll_reduce
    from src.lattice_reduction._native.basis.basis_generator import basis_gen
    from src.lattice_reduction._native.lll.L3fp import l3fp
    from src.lattice_reduction.adapter.common_adapter import to_column_basis, to_row_basis
    basis = basis_gen(10, 173).astype(np.int64)
    B_adapter = basis.T.copy()  # 行向量，适配层用
    B_native = basis.copy()      # 列向量，原生层用

    # 适配层
    lll_reduce(B_adapter, delta=0.75)

    # 原生层
    reduced_native, _, _ = l3fp(B_native, Lovasz_cond_param=0.75)

    # 转为同格式比较（都转为行向量）
    adapter_row = B_adapter
    native_row = reduced_native.T

    # 约减后最短范数应该一致（容差内）
    adapter_norms = np.sort(np.linalg.norm(adapter_row, axis=1))
    native_norms = np.sort(np.linalg.norm(native_row, axis=1))
    diff = np.max(np.abs(adapter_norms - native_norms))
    assert diff < 1e-6, f"Norms differ by {diff}: adapter={adapter_norms[:3]} vs native={native_norms[:3]}"
check("LLL 一致性", test_consistency)

# ── 汇总 ──
print("\n" + "=" * 50)
if errors:
    print(f"❌ {len(errors)} 项失败:")
    for name, e in errors:
        print(f"   - {name}: {e}")
    sys.exit(1)
else:
    print("✅ 全部通过")
    sys.exit(0)
