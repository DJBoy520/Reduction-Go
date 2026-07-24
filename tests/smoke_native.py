#!/usr/bin/env python3
"""原生层冒烟验证脚本 — 检查所有核心模块可导入且基础功能正常。"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

errors = []

def check(name, fn):
    try:
        fn()
        print(f"  ✓ {name}")
    except Exception as e:
        print(f"  ✗ {name}: {e}")
        errors.append((name, e))

print("=" * 50)
print("原生层冒烟验证")
print("=" * 50)

# ── 模块导入测试 ──
print("\n[1] 模块导入")
check("basis.basis_generator", lambda: __import__("src.lattice._native.basis.basis_generator", fromlist=["basis_generator"]))
check("gso.gsofp_se", lambda: __import__("src.lattice._native.gso.gsofp_se", fromlist=["gso_step"]))
check("gso.initializer", lambda: __import__("src.lattice._native.gso.initializer", fromlist=["initialize"]))
check("gso.delete_zero", lambda: __import__("src.lattice._native.gso.delete_zero", fromlist=["delete_zero_vector"]))
check("lll.L3fp_params", lambda: __import__("src.lattice._native.lll.L3fp_params", fromlist=["LOVASZ_CONDITION_PARAM"]))
check("lll.reducer", lambda: __import__("src.lattice._native.lll.reducer", fromlist=["size_reduction_loop"]))
check("lll.L3fp", lambda: __import__("src.lattice._native.lll.L3fp", fromlist=["l3fp"]))
check("lll.L3fp_deep_insertion", lambda: __import__("src.lattice._native.lll.L3fp_deep_insertion", fromlist=["l3fp_deep_insert"]))
check("bkz.bkz_params", lambda: __import__("src.lattice._native.bkz.bkz_params", fromlist=["DELTA"]))
check("bkz.bkz_schnorr_euchner", lambda: __import__("src.lattice._native.bkz.bkz_schnorr_euchner", fromlist=["bkz_se"]))
check("bkz.bkz_schnorr_euchner_progress_check", lambda: __import__("src.lattice._native.bkz.bkz_schnorr_euchner_progress_check", fromlist=["bkz_se_pc"]))
check("enumeration.enum_se", lambda: __import__("src.lattice._native.enumeration.enum_schnorr_euchner", fromlist=["enum_se_solver"]))
check("enumeration.enum_se_og", lambda: __import__("src.lattice._native.enumeration.enum_schnorr_euchner_og", fromlist=["enum_se_og_solver"]))
check("enumeration.enum_sh", lambda: __import__("src.lattice._native.enumeration.enum_schnorr_horner", fromlist=["enum_sh_solver"]))
check("quality.basis_quality_characteristics", lambda: __import__("src.lattice._native.quality.basis_quality_characteristics", fromlist=["compute_column_norms"]))
check("quality.basis_quality_evaluation", lambda: __import__("src.lattice._native.quality.basis_quality_evaluation", fromlist=["compute_basis_quality_characteristics"]))

# ── 功能性测试 ──
print("\n[2] 基础 LLL 约减")
def test_lll():
    import numpy as np
    from src.lattice._native.basis.basis_generator import basis_gen
    from src.lattice._native.lll.L3fp import l3fp
    basis = basis_gen(10, 173)
    reduced, gsc, gs_norms = l3fp(basis.copy())
    assert reduced.shape == (10, 10), f"Shape mismatch: {reduced.shape}"
    assert gs_norms.shape == (10,), f"Norms shape: {gs_norms.shape}"
check("LLL 约减 (dim=10)", test_lll)

print("\n[3] 基础 BKZ 约减")
def test_bkz():
    import numpy as np
    from src.lattice._native.basis.basis_generator import basis_gen
    from src.lattice._native.bkz.bkz_schnorr_euchner_progress_check import bkz_se_pc
    basis = basis_gen(10, 173)
    reduced, gsc, gs_norms = bkz_se_pc(basis.copy(), 5, "1")
    assert reduced.shape == (10, 10), f"Shape mismatch: {reduced.shape}"
check("BKZ 约减 (dim=10, block=5)", test_bkz)

print("\n[4] 格基质量评估")
def test_quality():
    import numpy as np
    from src.lattice._native.basis.basis_generator import basis_gen
    from src.lattice._native.lll.L3fp import l3fp
    from src.lattice._native.quality.basis_quality_evaluation import compute_basis_quality_characteristics
    basis = basis_gen(10, 173)
    reduced, _, _ = l3fp(basis.copy())
    shortest, rhf, od = compute_basis_quality_characteristics(reduced, reduced=True)
    assert shortest > 0, f"Shortest norm should be > 0: {shortest}"
    # RHF 可能 < 1 对约减良好的基，只要 > 0 即可
    assert rhf > 0, f"RHF should be > 0: {rhf}"
check("格基质量评估", test_quality)

print("\n[5] SVP 枚举器")
def test_enum():
    import numpy as np
    from src.lattice._native.basis.basis_generator import basis_gen
    from src.lattice._native.lll.L3fp import l3fp
    from src.lattice._native.gso.gsofp_se import gso_step
    from src.lattice._native.enumeration import ENUM_ALGORITHMS
    basis = basis_gen(10, 173)
    reduced, gsc, gs_norms = l3fp(basis.copy())
    solver = ENUM_ALGORITHMS["2"]
    block = reduced[:, :5]
    gs_block = gs_norms[:5]
    gsc_block = gsc[:5, :5]
    result = solver(block, gs_block, gsc_block)
    # 结果应该是 tuple (search_radius, u) 或类似结构
    assert result is not None, "ENUM returned None"
check("SVP 枚举器", test_enum)

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
