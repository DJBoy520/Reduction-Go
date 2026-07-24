#!/usr/bin/env python3
"""阶段一：全链路性能埋点统计，定位实际瓶颈。"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np

def profile_dim(dim, label):
    """在指定维度下 profiling 各环节耗时。"""
    from src.lattice._native.basis.basis_generator import basis_gen
    from src.lattice._native.lll.L3fp import l3fp
    from src.lattice.algorithms.bkz import bkz
    from src.lattice._native.gso.gsofp_se import gso_step
    from src.lattice._native.quality.basis_quality_evaluation import compute_basis_quality_characteristics

    print(f"\n{'='*60}")
    print(f"  [{label}] dim={dim}")
    print(f"{'='*60}")

    basis = basis_gen(dim, 173)

    # --- LLL ---
    t0 = time.time()
    B_lll, gsc, gs_norms = l3fp(basis.copy())
    t_lll = time.time() - t0
    print(f"  LLL:        {t_lll:.4f}s")

    # --- GSO 单独 ---
    t0 = time.time()
    B_col = basis.copy().astype(np.float64)
    width = B_col.shape[1]
    gs_sq = np.zeros(width, dtype=np.float64)
    gsc_mat = np.zeros((width, width), dtype=np.float64)
    for stage in range(width):
        gs_sq[:stage+1], gsc_mat[:, :stage+1] = gso_step(B_col[:, :stage+1], gsc_mat, gs_sq, stage)
    t_gso = time.time() - t0
    print(f"  GSO (full): {t_gso:.4f}s")

    # --- BKZ ---
    if dim <= 30:
        block_size = min(dim // 2, 10)
        t0 = time.time()
        B_bkz, _, _ = bkz(basis.copy(), block_size, "1")
        t_bkz = time.time() - t0
        print(f"  BKZ (b={block_size}): {t_bkz:.4f}s")
    else:
        print(f"  BKZ:        skipped (dim too large for quick profile)")

    # --- 质量评估 ---
    t0 = time.time()
    compute_basis_quality_characteristics(B_lll, reduced=True)
    t_quality = time.time() - t0
    print(f"  Quality:    {t_quality:.4f}s")

    # --- LLL 迭代次数估算 ---
    print(f"  LLL rows processed: ~{dim} (1 pass through basis)")

    return {"dim": dim, "lll": t_lll, "gso": t_gso, "quality": t_quality}


print("=" * 60)
print("  阶段一：性能瓶颈定位")
print("=" * 60)

results = []
for dim, label in [(10, "低维度"), (20, "中维度"), (50, "高维度")]:
    results.append(profile_dim(dim, label))

print(f"\n{'='*60}")
print("  耗时趋势分析")
print(f"{'='*60}")
for r in results:
    print(f"  dim={r['dim']:3d}  LLL={r['lll']:.4f}s  GSO={r['gso']:.4f}s  Quality={r['quality']:.4f}s")
