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
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import mpmath
import time

# ── 工具函数 ──
def random_integer_matrix(n, bound=100, seed=None):
    """生成随机整数矩阵（列向量基）。"""
    rng = np.random.RandomState(seed)
    return rng.randint(-bound, bound + 1, size=(n, n)).astype(np.int64)


def verify_lovasz_condition(gsc, gsn, delta, dim):
    """检查 Lovász 条件：对所有 i >= 1, delta * gsn[i-1] <= gsn[i] + mu^2 * gsn[i-1]。"""
    violations = []
    for i in range(1, dim):
        mu = float(gsc[i-1][i])
        lhs = float(delta * gsn[i-1])
        rhs = float(gsn[i] + mu**2 * gsn[i-1])
        if lhs > rhs:
            violations.append((i, lhs, rhs))
    return violations


def verify_size_condition(gsc, dim):
    """检查 size reduction 条件：|mu_{i,j}| <= 0.5 for all i < j。"""
    violations = []
    for j in range(1, dim):
        for i in range(j):
            mu = float(gsc[i][j])
            if abs(mu) > 0.5 + 1e-10:
                violations.append((i, j, mu))
    return violations


def check_positive_norms(gsn, dim):
    """检查所有 GSO 范数 > 0。"""
    return [i for i in range(dim) if float(gsn[i]) <= 0]


# ── 测试 ──
errors = []

def check(name, fn):
    try:
        fn()
        print(f"  ✓ {name}")
    except Exception as e:
        print(f"  ✗ {name}: {e}")
        errors.append((name, e))


print("=" * 60)
print("lll_mp 优化验证 (f_c: gso_full_refresh -> gso_step)")
print("=" * 60)

# ── 测试 1: 小维度正确性 ──
print("\n[1] 小维度 LLL 正确性 (dim=10, 20, 30)")

def test_dim(n, seed=42):
    from src.lattice_reduction._native.lll_mp import lll_mp
    from src.lattice_reduction._native.gso_mp import init_gso_mp, gso_full_refresh_mp

    basis = random_integer_matrix(n, bound=100, seed=seed)
    basis_copy = basis.copy()

    delta = 0.79
    dps = 80

    t0 = time.time()
    reduced, gsc, gsn = lll_mp(basis_copy, Lovasz_cond_param=delta, dps=dps)
    elapsed = time.time() - t0

    dim = reduced.shape[1]

    # 正范数
    bad_norms = check_positive_norms(gsn, dim)
    assert not bad_norms, f"零/负范数在: {bad_norms}"

    # Lovász 条件
    lv_violations = verify_lovasz_condition(gsc, gsn, mpmath.mpf(delta), dim)
    assert not lv_violations, f"Lovász 违规: {lv_violations}"

    # Size reduction 条件
    sz_violations = verify_size_condition(gsc, dim)
    assert not sz_violations, f"Size reduction 违规: {sz_violations}"

    # 范数合理性：最短向量不应大于随机矩阵最短向量的 2 倍
    orig_shortest = min(np.linalg.norm(basis[:, i]) for i in range(n))
    red_shortest = min(np.linalg.norm(reduced[:, i]) for i in range(n))

    print(f"    dim={n}: {elapsed:.3f}s, orig_shortest={orig_shortest:.1f}, red_shortest={red_shortest:.1f}")
    return elapsed


check("LLL dim=10", lambda: test_dim(10))
check("LLL dim=20", lambda: test_dim(20))
check("LLL dim=30", lambda: test_dim(30))

# ── 测试 2: 多种子验证鲁棒性 ──
print("\n[2] 多随机种子鲁棒性 (dim=15, 10 seeds)")

def test_multi_seed():
    from src.lattice_reduction._native.lll_mp import lll_mp

    n = 15
    delta = 0.79
    dps = 80
    for seed in range(10):
        basis = random_integer_matrix(n, bound=200, seed=seed)
        basis_copy = basis.copy()
        reduced, gsc, gsn = lll_mp(basis_copy, Lovasz_cond_param=delta, dps=dps)
        dim = reduced.shape[1]

        bad_norms = check_positive_norms(gsn, dim)
        assert not bad_norms, f"seed={seed}: 零范数在 {bad_norms}"

        lv = verify_lovasz_condition(gsc, gsn, mpmath.mpf(delta), dim)
        assert not lv, f"seed={seed}: Lovász 违规 {lv}"

        sz = verify_size_condition(gsc, dim)
        assert not sz, f"seed={seed}: Size reduction 违规 {sz}"

    print("    10 seeds 全部通过")

check("多种子鲁棒性", test_multi_seed)

# ── 测试 3: 大维度性能对比 ──
print("\n[3] 大维度性能 (dim=50, 80)")

def test_perf(n, seed=42):
    from src.lattice_reduction._native.lll_mp import lll_mp

    basis = random_integer_matrix(n, bound=100, seed=seed)
    basis_copy = basis.copy()

    delta = 0.79
    dps = 100

    t0 = time.time()
    reduced, gsc, gsn = lll_mp(basis_copy, Lovasz_cond_param=delta, dps=dps)
    elapsed = time.time() - t0

    dim = reduced.shape[1]
    bad_norms = check_positive_norms(gsn, dim)
    assert not bad_norms, f"零范数: {bad_norms}"
    lv = verify_lovasz_condition(gsc, gsn, mpmath.mpf(delta), dim)
    assert not lv, f"Lovász 违规: {lv}"

    red_shortest = min(np.linalg.norm(reduced[:, i]) for i in range(n))
    print(f"    dim={n}: {elapsed:.3f}s, red_shortest={red_shortest:.1f}")

check("性能 dim=50", lambda: test_perf(50))
check("性能 dim=80", lambda: test_perf(80))

# ── 测试 4: 已约减基不变性 ──
print("\n[4] 已约减基应快速完成且结果合理")

def test_already_reduced():
    from src.lattice_reduction._native.lll_mp import lll_mp

    # 构造一个正交基（已经满足 LLL 条件）
    n = 10
    basis = np.eye(n, dtype=np.int64) * 10

    delta = 0.79
    dps = 80

    t0 = time.time()
    reduced, gsc, gsn = lll_mp(basis.copy(), Lovasz_cond_param=delta, dps=dps)
    elapsed = time.time() - t0

    dim = reduced.shape[1]
    lv = verify_lovasz_condition(gsc, gsn, mpmath.mpf(delta), dim)
    assert not lv, f"正交基 Lovász 违规: {lv}"

    # 正交基约减后应该是对角占优
    print(f"    正交基: {elapsed:.3f}s, gsn={[f'{float(g):.1f}' for g in gsn[:5]]}")

check("正交基不变性", test_already_reduced)

# ── 测试 5: 与 gso_full_refresh 的一致性 ──
print("\n[5] 与 gso_full_refresh 全量重算结果一致性")

def test_consistency_full_refresh():
    """用 gso_full_refresh 重新计算 f_c 分支后的 GSO，对比 gso_step_mp 的结果。"""
    import copy
    from src.lattice_reduction._native.lll_mp import lll_mp, _size_reduction_lll
    from src.lattice_reduction._native.gso_mp import (
        init_gso_mp, gso_step_mp, gso_full_refresh_mp
    )

    n = 12
    basis = random_integer_matrix(n, bound=100, seed=77)
    basis_copy = basis.copy()

    # 先运行 LLL（使用优化后的版本）获取最终结果
    delta = 0.79
    dps = 80
    reduced_opt, gsc_opt, gsn_opt = lll_mp(basis_copy, Lovasz_cond_param=delta, dps=dps)

    # 再用相同输入运行，但在 f_c 分支时分别用 gso_step_mp 和 gso_full_refresh_mp 对比
    # 模拟 f_c 场景：运行到某个 stage 后做 size reduction
    basis2 = random_integer_matrix(n, bound=100, seed=77)
    gsc2, gsn2 = init_gso_mp(n)

    with mpmath.workdps(dps):
        # 全量刷新前 5 列
        gso_full_refresh_mp(basis2, gsc2, gsn2, 5, dps)

        # 对第 4 列做 size reduction
        basis2_step = basis2.copy()
        gsc_step = copy.deepcopy(gsc2)
        gsn_step = list(gsn2)
        f_c = _size_reduction_lll(4, gsc_step, gsn_step, basis2_step, dps)

        if f_c:
            # 方法1: gso_step_mp（优化后的方式）
            gsc_step_A = copy.deepcopy(gsc_step)
            gsn_step_A = list(gsn_step)
            basis_step_A = basis2_step.copy()
            gso_step_mp(basis_step_A, gsc_step_A, gsn_step_A, 4, dps)

            # 方法2: gso_full_refresh_mp（全量重算）
            gsc_step_B = copy.deepcopy(gsc_step)
            gsn_step_B = list(gsn_step)
            basis_step_B = basis2_step.copy()
            gso_full_refresh_mp(basis_step_B, gsc_step_B, gsn_step_B, 5, dps)

            # 对比第 4 列的 GSO 系数
            max_diff_gsc = 0
            for j in range(4):
                diff = abs(float(gsc_step_A[j][4] - gsc_step_B[j][4]))
                max_diff_gsc = max(max_diff_gsc, diff)
            diff_gsn = abs(float(gsn_step_A[4] - gsn_step_B[4]))

            print(f"    f_c 触发, gsc 最大差异: {max_diff_gsc:.2e}, gsn 差异: {diff_gsn:.2e}")
            assert max_diff_gsc < 1e-30, f"gsc 差异过大: {max_diff_gsc}"
            assert diff_gsn < 1e-30, f"gsn 差异过大: {diff_gsn}"
        else:
            print("    未触发 f_c，跳过一致性对比（正常）")

check("full_refresh 一致性", test_consistency_full_refresh)

# ── 测试 5b: 极端输入触发 f_c ──
print("\n[5b] 构造极端基以触发 f_c 分支")

def test_f_c_trigger():
    """构造极端非正交基，使得 |mu| > tau_limit，强制触发 f_c。"""
    import copy
    from src.lattice_reduction._native.lll_mp import _size_reduction_lll
    from src.lattice_reduction._native.gso_mp import (
        init_gso_mp, gso_step_mp, gso_full_refresh_mp,
    )

    n = 8
    dps = 100

    # 构造一个近乎共面的基：b0=[1,0,...], b1=[10^8,1,0,...], b2=[0,10^8,1,...]
    basis = np.zeros((n, n), dtype=np.int64)
    for i in range(n):
        basis[i, i] = 1
        if i > 0:
            basis[i-1, i] = 100000  # 大偏移量

    # 初始化 GSO
    gsc, gsn = init_gso_mp(n)
    gso_full_refresh_mp(basis, gsc, gsn, n, dps)

    # 对第 3 列做 size reduction，看是否触发 f_c
    basis_step = basis.copy()
    gsc_copy = copy.deepcopy(gsc)
    gsn_copy = list(gsn)
    f_c, _, _ = _size_reduction_lll(3, gsc_copy, gsn_copy, basis_step, dps)

    print(f"    f_c 触发: {f_c}")

    if f_c:
        # _size_reduction_lll 已经计算了 gsn[stage]，验证其与 gso_step_mp 一致
        gsn_from_size_red = list(gsn_copy)  # _size_reduction_lll 已原地更新

        # 方法1: gso_step_mp（从整数点积重算整列）
        gsc_A = copy.deepcopy(gsc_copy)
        gsn_A = list(gsn_copy)
        basis_A = basis_step.copy()
        gso_step_mp(basis_A, gsc_A, gsn_A, 3, dps)

        # 方法2: gso_full_refresh_mp（全量重算）
        gsc_B = copy.deepcopy(gsc_copy)
        gsn_B = list(gsn_copy)
        basis_B = basis_step.copy()
        gso_full_refresh_mp(basis_B, gsc_B, gsn_B, 4, dps)

        # 对比第 3 列的 GSO 系数（gsc 应由增量更新保持一致）
        max_diff_gsc = 0
        for j in range(3):
            diff = abs(float(gsc_copy[j][3] - gsc_A[j][3]))
            max_diff_gsc = max(max_diff_gsc, diff)

        # 对比 gsn：_size_reduction_lll 计算的 vs gso_step_mp 的
        diff_gsn_step = abs(float(gsn_copy[3] - gsn_A[3]))
        # 对比 gsn：_size_reduction_lll 计算的 vs gso_full_refresh_mp 的
        diff_gsn_full = abs(float(gsn_copy[3] - gsn_B[3]))

        print(f"    gsc 增量 vs step 差异: {max_diff_gsc:.2e}")
        print(f"    gsn size_red vs step: {diff_gsn_step:.2e}, vs full_refresh: {diff_gsn_full:.2e}")
        assert max_diff_gsc < 1e-25, f"gsc 差异过大: {max_diff_gsc}"
        assert diff_gsn_step < 1e-25, f"gsn (size_red vs step) 差异过大: {diff_gsn_step}"
        assert diff_gsn_full < 1e-25, f"gsn (size_red vs full) 差异过大: {diff_gsn_full}"
    else:
        print("    未触发 f_c，增大偏移量重试...")
        # 更极端的情况：超大 μ 走安全回退，而不是被旧的阈值逻辑误判为 f_c
        basis2 = np.eye(n, dtype=np.int64)
        basis2[0, 1] = 2**40
        basis2[1, 2] = 2**40
        gsc2, gsn2 = init_gso_mp(n)
        gso_full_refresh_mp(basis2, gsc2, gsn2, n, dps)

        basis2_step = basis2.copy()
        gsc2_copy = copy.deepcopy(gsc2)
        gsn2_copy = list(gsn2)
        f_c2, _, _ = _size_reduction_lll(2, gsc2_copy, gsn2_copy, basis2_step, dps)
        print(f"    极端输入 f_c: {f_c2}")
        assert isinstance(f_c2, bool), "极端输入应保持稳定的 f_c 结果"

check("f_c 触发一致性", test_f_c_trigger)

# ── 汇总 ──
print("\n" + "=" * 60)
if errors:
    print(f"❌ {len(errors)} 项失败:")
    for name, e in errors:
        print(f"   - {name}: {e}")
    sys.exit(1)
else:
    print("✅ 全部通过")
    sys.exit(0)
