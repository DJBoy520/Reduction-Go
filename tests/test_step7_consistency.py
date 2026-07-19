#!/usr/bin/env python3
"""
步骤7：功能一致性与全流程验证。

构造多组测试矩阵，验证迁移后的算法结果正确性。
覆盖：低/中/高维度、稀疏/稠密、小整数/大整数场景。
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import time

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


print("=" * 60)
print("步骤7：功能一致性与全流程验证")
print("=" * 60)


# ══════════════════════════════════════════════════════════════
# 7.1 单算法一致性验证（LLL）
# ══════════════════════════════════════════════════════════════
print("\n[7.1] LLL 约减一致性验证")

def test_lll_low_dim():
    """低维度 (dim=5): 精确验证约减后向量范数"""
    from src.lattice._native.basis.basis_generator import basis_gen
    from src.lattice._native.lll.L3fp import l3fp
    from src.lattice.adapter.common_adapter import to_column_basis, to_row_basis
    from src.lattice import lll_reduce
    basis = basis_gen(5, 50)
    B_adapter = basis.T.copy()
    B_native = basis.copy()
    lll_reduce(B_adapter, delta=0.75)
    reduced_native, _, _ = l3fp(B_native, Lovasz_cond_param=0.75)
    adapter_norms = np.sort(np.linalg.norm(B_adapter, axis=1))
    native_norms = np.sort(np.linalg.norm(reduced_native.T, axis=1))
    diff = np.max(np.abs(adapter_norms - native_norms))
    assert diff < 1e-6, f"Norms differ by {diff}"
check("LLL 低维度一致性 (dim=5)", test_lll_low_dim)


def test_lll_medium_dim():
    """中维度 (dim=20): 验证约减效果"""
    from src.lattice._native.basis.basis_generator import basis_gen
    from src.lattice import lll_reduce
    from src.lattice._native.quality.basis_quality_characteristics import compute_root_hermite_factor, compute_lattice_volume_log
    basis = basis_gen(20, 173)
    B = basis.T.copy()
    original_norms = np.sort(np.linalg.norm(B, axis=1))
    lll_reduce(B, delta=0.75)
    reduced_norms = np.sort(np.linalg.norm(B, axis=1))
    # LLL 约减后第一个向量应该更短或相等
    assert reduced_norms[0] <= original_norms[0] * 1.1, \
        f"LLL should not increase first vector: {reduced_norms[0]:.1f} vs {original_norms[0]:.1f}"
    # RHF 应该合理
    B_col = B.T
    log_vol = compute_lattice_volume_log(B_col)
    rhf = compute_root_hermite_factor(reduced_norms[0], log_vol, 20)
    assert 0.5 < rhf < 2.0, f"RHF should be reasonable: {rhf}"
check("LLL 中维度约减效果 (dim=20)", test_lll_medium_dim)


def test_lll_high_dim():
    """高维度 (dim=50): 验证约减不崩溃"""
    from src.lattice._native.basis.basis_generator import basis_gen
    from src.lattice import lll_reduce
    basis = basis_gen(50, 173)
    B = basis.T.copy()
    t0 = time.time()
    lll_reduce(B, delta=0.75)
    elapsed = time.time() - t0
    assert B.shape == (50, 50), f"Shape changed: {B.shape}"
    assert not np.any(np.isnan(B)), "NaN in result"
    assert not np.any(np.isinf(B)), "Inf in result"
    print(f"    (耗时 {elapsed:.2f}s)")
check("LLL 高维度不崩溃 (dim=50)", test_lll_high_dim)


# ══════════════════════════════════════════════════════════════
# 7.2 BKZ 一致性验证
# ══════════════════════════════════════════════════════════════
print("\n[7.2] BKZ 约减一致性验证")

def test_bkz_low_dim():
    """低维度 BKZ: 验证约减效果"""
    from src.lattice._native.basis.basis_generator import basis_gen
    from src.lattice import bkz_reduce
    basis = basis_gen(10, 173)
    B = basis.T.copy()
    original_norms = np.sort(np.linalg.norm(B, axis=1))
    result = bkz_reduce(B, block_size=5, max_loops=2, enum_algo="1")
    reduced_norms = np.sort(np.linalg.norm(B, axis=1))
    assert reduced_norms[0] <= original_norms[0] * 1.5, \
        f"BKZ should not explode first vector: {reduced_norms[0]:.1f} vs {original_norms[0]:.1f}"
    assert result["completed_loops"] >= 1, "Should complete at least 1 loop"
check("BKZ 低维度约减效果 (dim=10)", test_bkz_low_dim)


def test_bkz_improves_over_lll():
    """BKZ 应该比 LLL 更好（或相等）"""
    from src.lattice._native.basis.basis_generator import basis_gen
    from src.lattice import lll_reduce, bkz_reduce
    basis = basis_gen(15, 173)
    B_lll = basis.T.copy()
    B_bkz = basis.T.copy()
    lll_reduce(B_lll, delta=0.75)
    bkz_reduce(B_bkz, block_size=8, max_loops=2, enum_algo="1")
    lll_shortest = np.min(np.linalg.norm(B_lll, axis=1))
    bkz_shortest = np.min(np.linalg.norm(B_bkz, axis=1))
    # BKZ 应该找到更短或相等的向量
    assert bkz_shortest <= lll_shortest * 1.01, \
        f"BKZ should be better than LLL: {bkz_shortest:.1f} vs {lll_shortest:.1f}"
check("BKZ 改善优于 LLL (dim=15)", test_bkz_improves_over_lll)


# ══════════════════════════════════════════════════════════════
# 7.3 质量评估验证
# ══════════════════════════════════════════════════════════════
print("\n[7.3] 格基质量评估验证")

def test_quality_metrics():
    """验证质量评估指标的合理性"""
    from src.lattice._native.basis.basis_generator import basis_gen
    from src.lattice import lll_reduce, evaluate_basis_quality
    basis = basis_gen(15, 173)
    B = basis.T.copy()
    lll_reduce(B, delta=0.75)
    q = evaluate_basis_quality(B, reduced=True)
    assert q["shortest_norm"] > 0, "Shortest norm should be > 0"
    assert q["hermite_factor"] > 0, "Hermite factor should be > 0"
    assert q["orthogonality_defect"] > 0, "OD should be > 0"
    assert q["log_volume"] != 0, "Log volume should be non-zero"
    assert len(q["column_norms"]) == 15, f"Expected 15 norms, got {len(q['column_norms'])}"
check("质量评估指标合理性", test_quality_metrics)


def test_quality_identity():
    """单位矩阵的质量评估"""
    from src.lattice import evaluate_basis_quality
    B = np.eye(10, dtype=np.int64)
    q = evaluate_basis_quality(B, reduced=True)
    assert abs(q["shortest_norm"] - 1.0) < 1e-6, f"Identity shortest norm: {q['shortest_norm']}"
    assert abs(q["hermite_factor"] - 1.0) < 1e-6, f"Identity HF: {q['hermite_factor']}"
check("单位矩阵质量评估", test_quality_identity)


# ══════════════════════════════════════════════════════════════
# 7.4 边界与异常验证
# ══════════════════════════════════════════════════════════════
print("\n[7.4] 边界与异常验证")

def test_already_reduced():
    """已约减的基不应被破坏"""
    from src.lattice import lll_reduce
    B = np.eye(10, dtype=np.int64)
    original = B.copy()
    lll_reduce(B, delta=0.75)
    diff = np.max(np.abs(B - original))
    assert diff < 1e-6, f"Identity basis should stay identity: diff={diff}"
check("已约减基不被破坏 (identity)", test_already_reduced)


def test_singular_like_matrix():
    """奇异矩阵（行列式为0）应能处理"""
    from src.lattice import lll_reduce
    from src.lattice.adapter.common_adapter import LatticeReductionError
    B = np.array([[1, 2], [2, 4]], dtype=np.int64)  # 行线性相关
    try:
        lll_reduce(B, delta=0.75)
        assert not np.any(np.isnan(B)), "NaN in result"
    except (np.linalg.LinAlgError, ZeroDivisionError, ValueError, LatticeReductionError, FloatingPointError):
        pass  # 预期可能抛异常
check("奇异矩阵处理", test_singular_like_matrix)


def test_large_entries():
    """大整数矩阵（模拟密码学场景）"""
    from src.lattice import lll_reduce
    rng = np.random.default_rng(42)
    B = rng.integers(0, 8380417, size=(8, 8)).astype(np.int64)
    lll_reduce(B, delta=0.75)
    assert not np.any(np.isnan(B)), "NaN in result"
    assert not np.any(np.isinf(B)), "Inf in result"
check("大整数矩阵 (q=8380417, dim=8)", test_large_entries)


def test_sparse_matrix():
    """稀疏矩阵"""
    from src.lattice import lll_reduce
    B = np.zeros((6, 6), dtype=np.int64)
    for i in range(6):
        B[i, i] = 1
        B[i, (i + 1) % 6] = 100
    lll_reduce(B, delta=0.75)
    assert not np.any(np.isnan(B)), "NaN in result"
check("稀疏矩阵 (dim=6)", test_sparse_matrix)


# ══════════════════════════════════════════════════════════════
# 7.5 全流程验证（模拟攻击）
# ══════════════════════════════════════════════════════════════
print("\n[7.5] 全流程验证（模拟格攻击）")

def test_full_attack_toy():
    """toy 参数全流程攻击"""
    from src.lattice_attack import build_lattice_basis, run_attack, classify_results, verify_basis
    from src.poly_math import mat_vec_mul, vec_add_mod
    k, l, n = 2, 2, 5
    q = 8380417
    A = np.random.randint(0, 100, (k, l, n))
    s1 = np.random.randint(0, 3, (l, n))
    s2 = np.random.randint(0, 3, (k, n))
    t = vec_add_mod(mat_vec_mul(A, s1, q), s2, q) % q
    ok = verify_basis(A, t, q, s1, s2)
    assert ok.passed, f"Basis verification failed: {ok.error}"
    result = run_attack(A, t, q, s1, s2, no_bkz=True)
    perfect = sum(1 for c in result["candidates"] if c.get("perfect"))
    assert perfect >= 1, f"Should find perfect recovery, found {perfect}"
    print(f"    (LLL={result['lll_time']:.2f}s, 候选={len(result['candidates'])}, 完美={perfect})")
check("toy 参数全流程攻击", test_full_attack_toy)


def test_full_attack_with_bkz():
    """含 BKZ 的全流程攻击"""
    from src.lattice_attack import build_lattice_basis, run_attack, classify_results, verify_basis
    from src.poly_math import mat_vec_mul, vec_add_mod
    k, l, n = 2, 2, 5
    q = 8380417
    A = np.random.randint(0, 100, (k, l, n))
    s1 = np.random.randint(0, 3, (l, n))
    s2 = np.random.randint(0, 3, (k, n))
    t = vec_add_mod(mat_vec_mul(A, s1, q), s2, q) % q
    result = run_attack(A, t, q, s1, s2, no_bkz=False, bkz_block_size=5, bkz_max_loops=2)
    perfect = sum(1 for c in result["candidates"] if c.get("perfect"))
    assert perfect >= 1, f"Should find perfect recovery with BKZ, found {perfect}"
    print(f"    (LLL={result['lll_time']:.2f}s, BKZ={result['bkz_time']:.2f}s, 完美={perfect})")
check("含 BKZ 全流程攻击", test_full_attack_with_bkz)


def test_classify_results():
    """验证分类逻辑"""
    from src.lattice_attack import classify_results
    result = {
        "candidates": [
            {"eq_holds": True, "perfect": True, "s1_perfect": True, "ratio": 1.0, "cand_norm": 5.0},
            {"eq_holds": True, "perfect": False, "s1_perfect": True, "ratio": 1.1, "cand_norm": 6.0},
            {"eq_holds": True, "perfect": False, "s1_perfect": False, "ratio": 1.15, "cand_norm": 7.0},
            {"eq_holds": True, "perfect": False, "s1_perfect": False, "ratio": 2.0, "cand_norm": 20.0},
            {"eq_holds": False, "perfect": False, "s1_perfect": False, "ratio": 5.0, "cand_norm": 50.0},
        ]
    }
    classified = classify_results(result)
    verdicts = [c["verdict"] for c in classified]
    assert verdicts[0] == "完美恢复私钥", f"Expected 完美恢复私钥, got {verdicts[0]}"
    assert verdicts[1] == "s1 完美恢复 (s2 不匹配)", f"Expected s1完美, got {verdicts[1]}"
    assert verdicts[2] == "攻击成功，找到替代短向量", f"Expected 替代, got {verdicts[2]}"
    assert verdicts[3] == "满足方程但向量过长", f"Expected 过长, got {verdicts[3]}"
    assert verdicts[4] == "无效解", f"Expected 无效, got {verdicts[4]}"
check("分类逻辑验证", test_classify_results)


# ══════════════════════════════════════════════════════════════
# 汇总
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
if errors:
    print(f"❌ {len(errors)} 项失败 / {len(passed)} 项通过:")
    for name, e in errors:
        print(f"   ✗ {name}: {e}")
    for name in passed:
        print(f"   ✓ {name}")
    if __name__ == "__main__":
        sys.exit(1)
else:
    print(f"✅ 全部 {len(passed)} 项通过")
    if __name__ == "__main__":
        sys.exit(0)
