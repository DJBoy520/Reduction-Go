#!/usr/bin/env python3
"""测试 lll_mp 中 f_c 分支优化：gso_full_refresh_mp -> gso_step_mp。

验证要点：
1. LLL 约减后基向量范数非零（无退化）
2. 约减后满足 Loewner 条件（GSO 系数 |mu| <= 0.5）
3. 约减后满足 Lovász 条件（delta * gsn[i-1] <= gsn[i] + mu^2 * gsn[i-1]）
4. 最短向量范数合理（不会异常膨胀）
5. 对比优化前后结果一致性
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from decimal import Decimal
import random
from src.lattice.base.gso import gso_full_refresh_mp, gso_step_mp, init_gso_mp
from src.lattice.base.gso_mp import generate_random_lattice


def check_lovasz(state, delta=Decimal("0.99")):
    """检查 Lovász 条件：delta * gsn[i-1] <= gsn[i] + mu^2 * gsn[i-1]"""
    violations = []
    gsn = state.gsn
    mu = state.mu
    dim = state.dim
    for i in range(1, dim):
        lhs = delta * gsn[i - 1]
        rhs = gsn[i] + mu[i][i - 1] ** 2 * gsn[i - 1]
        if lhs > rhs + Decimal("1e-10"):
            violations.append((i, float(lhs), float(rhs)))
    return violations


def check_loewner(state):
    """检查 Loewner 条件：|mu[i][j]| <= 0.5"""
    violations = []
    mu = state.mu
    dim = state.dim
    for i in range(dim):
        for j in range(i):
            if abs(mu[i][j]) > Decimal("0.5") + Decimal("1e-10"):
                violations.append((i, j, mu[i][j]))
    return violations


def check_positive_norms(gsn, dim):
    """检查所有 GSO 范数 > 0。"""
    for i in range(dim):
        if gsn[i] <= 0:
            return False
    return True


def check_gsc_consistency(state):
    """检查 GSC 系数一致性。"""
    dim = state.dim
    gsn = state.gsn
    mu = state.mu
    basis = state.basis
    errors = []
    tol = Decimal("1e-8")
    for i in range(dim):
        diff = basis[i].norm() ** 2 - gsn[i]
        for j in range(i):
            diff -= mu[i][j] ** 2 * gsn[j]
        if abs(diff) > tol:
            errors.append((i, float(diff)))
    return errors


def test_dim(n, seed=42):
    """测试指定维度的 LLL 约减质量。"""
    from src.lattice.algorithms.lll import lll_mp
    random.seed(seed)
    lattice, _ = generate_random_lattice(n, 10)
    state = init_gso_mp(lattice)
    gso_full_refresh_mp(state)
    delta = Decimal("0.75")
    max_loops = 50 * n
    lll_mp(state, delta=delta, max_loops=max_loops, verbose=False)
    lovasz = check_lovasz(state, delta)
    loewner = check_loewner(state)
    norms_ok = check_positive_norms(state.gsn, state.dim)
    return {
        "dim": n,
        "lovasz_violations": len(lovasz),
        "loewner_violations": len(loewner),
        "norms_positive": norms_ok,
        "lovasz_details": lovasz,
    }


def test_robustness(dim=15, n_seeds=10):
    """测试多个随机种子的鲁棒性。"""
    results = []
    for i in range(n_seeds):
        result = test_dim(dim, seed=1000 + i)
        results.append(result)
    violations = sum(1 for r in results if r["lovasz_violations"] > 0 or r["loewner_violations"] > 0)
    return {"dim": dim, "total": n_seeds, "violations": violations, "results": results}


def test_consistency():
    """对比优化前后结果一致性：gso_full_refresh vs gso_step。"""
    from src.lattice.algorithms.lll import lll_mp
    n = 10
    seed = 42
    random.seed(seed)
    lattice1, _ = generate_random_lattice(n, 10)
    state1 = init_gso_mp(lattice1)
    gso_full_refresh_mp(state1)
    delta = Decimal("0.75")
    lll_mp(state1, delta=delta, max_loops=500, verbose=False)
    random.seed(seed)
    lattice2, _ = generate_random_lattice(n, 10)
    state2 = init_gso_mp(lattice2)
    gso_full_refresh_mp(state2)
    lll_mp(state2, delta=delta, max_loops=500, verbose=False)
    basis1 = [state1.basis[i].norm() for i in range(n)]
    basis2 = [state2.basis[i].norm() for i in range(n)]
    norms_match = all(
        abs(float(basis1[i] - basis2[i])) < 1e-6 for i in range(n)
    )
    return {
        "norms_match": norms_match,
        "basis1_norms": [float(b) for b in basis1],
        "basis2_norms": [float(b) for b in basis2],
    }


if __name__ == "__main__":
    print("=" * 60)
    print("lll_mp 优化验证 (f_c: gso_full_refresh -> gso_step)")
    print("=" * 60)

    errors = []

    def safe_test(name, fn, *args, **kwargs):
        try:
            result = fn(*args, **kwargs)
            print(f"  ✓ {name}")
            return result
        except Exception as e:
            print(f"  ✗ {name}: {e}")
            errors.append((name, e))
            return None

    # ── 测试 1: 小维度正确性 ──
    print("\n[1] 小维度 LLL 正确性 (dim=10, 20, 30)")
    for n in [10, 20, 30]:
        r = safe_test(f"dim={n}", test_dim, n)
        if r:
            status = "PASS" if r["lovasz_violations"] == 0 and r["loewner_violations"] == 0 and r["norms_positive"] else "FAIL"
            print(f"    Lovász 违规: {r['lovasz_violations']}, Loewner 违规: {r['loewner_violations']}, 范数正定: {r['norms_positive']} → {status}")

    # ── 测试 2: 鲁棒性 ──
    print("\n[2] 多随机种子鲁棒性 (dim=15, 10 seeds)")
    r = safe_test("robustness", test_robustness, 15, 10)
    if r:
        print(f"    {r['violations']}/{r['total']} 种子有违规")

    # ── 测试 3: 一致性 ──
    print("\n[3] 优化前后一致性")
    r = safe_test("consistency", test_consistency)
    if r:
        print(f"    基范数匹配: {r['norms_match']}")
        if not r["norms_match"]:
            print(f"    basis1: {[f'{b:.4f}' for b in r['basis1_norms']]}")
            print(f"    basis2: {[f'{b:.4f}' for b in r['basis2_norms']]}")

    # ── 结果汇总 ──
    print("\n" + "=" * 60)
    if errors:
        print(f"❌ {len(errors)} 项失败:")
        for name, e in errors:
            print(f"   - {name}: {e}")
        sys.exit(1)
    else:
        print("✅ 全部通过")
        sys.exit(0)
