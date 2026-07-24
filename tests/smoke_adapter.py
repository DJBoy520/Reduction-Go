#!/usr/bin/env python3
"""冒烟测试：验证 lattice 迁移后的模块可达性。"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def smoke_import():
    """测试所有核心模块可导入。"""
    from src.lattice import lll_reduce, bkz_reduce, evaluate_basis_quality
    from src.lattice.adapter import LatticeReducer, LatticeReductionError, LatticeParams
    from src.lattice.algorithms.lll import lll_mp
    from src.lattice.algorithms.bkz import bkz_mp
    from src.lattice.base.gso import init_gso_mp, gso_full_refresh_mp, gso_step_mp
    from src.lattice.base.precision_constants import DEFAULT_PRECISION_BITS
    from src.lattice.base.exceptions import LatticeReductionError, InvalidBasisError, ReductionFailedError, PrecisionFailureError
    from src.lattice.base.lll_state import LLLState
    print("✅ 模块导入：全部成功")
    return True


def smoke_lll_reduce():
    """测试 lll_reduce 实际执行。"""
    from src.lattice import lll_reduce
    import gmpy2
    n, m = 4, 4
    b = [[int(gmpy2.mpz_random(gmpy2.random_state(0), 100)) for _ in range(m)] for _ in range(n)]
    r = lll_reduce(b, precision_bits=128)
    print(f"✅ lll_reduce：{r.status}")
    return True


def smoke_bkz_reduce():
    """测试 bkz_reduce 实际执行。"""
    from src.lattice import bkz_reduce
    import gmpy2
    n, m = 4, 4
    b = [[int(gmpy2.mpz_random(gmpy2.random_state(0), 100)) for _ in range(m)] for _ in range(n)]
    r = bkz_reduce(b, block_size=2, precision_bits=128)
    print(f"✅ bkz_reduce：{r.status}")
    return True


def smoke_evaluate():
    """测试 evaluate_basis_quality 实际计算。"""
    from src.lattice import evaluate_basis_quality
    n, m = 3, 3
    b = [[2, 1, 0], [1, 3, 1], [0, 1, 4]]
    q = evaluate_basis_quality(b, compute_gso=True)
    print(f"✅ evaluate_basis_quality：quality_score={q['quality_score']:.4f}")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("lattice 模块冒烟测试")
    print("=" * 60)

    tests = [
        ("模块导入", smoke_import),
        ("LLL 约减", smoke_lll_reduce),
        ("BKZ 约减", smoke_bkz_reduce),
        ("质量评估", smoke_evaluate),
    ]

    ok, fail = 0, 0
    for name, fn in tests:
        try:
            if fn():
                ok += 1
        except Exception as e:
            print(f"❌ {name}: {e}")
            fail += 1

    print(f"\n{'='*60}")
    print(f"结果：{ok} 通过 / {fail} 失败")
    sys.exit(0 if fail == 0 else 1)
