# lattice_reduction 模块使用说明

## 概述

`lattice_reduction` 是纯 Python 格基约减算法模块，替代原 fpylll/fplll C++ 依赖。
基于 [LatticeReductionAlgorithms](https://github.com/vttresearch/LatticeReductionAlgorithms)（GPL v2）。

## 快速开始

```python
import numpy as np
from src.lattice_reduction import lll_reduce, bkz_reduce, evaluate_basis_quality

# 创建一个随机格基（行向量格式）
B = np.random.randint(0, 100, size=(10, 10)).astype(np.int64)

# LLL 约减（原地修改）
lll_reduce(B, delta=0.75)

# 或 BKZ 约减
bkz_reduce(B, block_size=5, max_loops=8)

# 评估约减质量
quality = evaluate_basis_quality(B, reduced=True)
print(f"最短向量范数: {quality['shortest_norm']:.2f}")
print(f"Hermite 因子: {quality['hermite_factor']:.4f}")
print(f"正交缺陷: {quality['orthogonality_defect']:.4f}")
```

## API 参考

### `lll_reduce(B, delta=0.999, float_type="mpfr", precision=200, method="proved")`

LLL 约减，原地修改行向量基 B。

**参数：**
- `B`: numpy int64 二维方阵（行向量基），原地修改
- `delta`: LLL δ 参数，范围 (0.25, 1.0)，默认 0.999
- `float_type`, `precision`, `method`: 保留兼容性参数，当前不生效

**返回：** None（原地修改 B）

**异常：**
- `InvalidBasisError`: 格基格式非法（None、非方阵、NaN/Inf 等）
- `ReductionFailedError`: 约减过程失败

### `bkz_reduce(B, block_size=20, max_loops=8, enum_algo="1", auto_abort=False, float_type="mpfr", precision=200)`

BKZ 约减，原地修改行向量基 B。

**参数：**
- `B`: numpy int64 二维方阵（行向量基），原地修改
- `block_size`: BKZ 块大小，必须 >= 2 且 <= 格维度
- `max_loops`: 最大循环次数，默认 8
- `enum_algo`: 枚举算法 ("1"=SE-OG, "2"=SE, "3"=SH)
- `auto_abort`: 连续无改善时提前终止
- `float_type`, `precision`: 保留兼容性参数

**返回：** dict
- `completed_loops`: 实际完成的循环次数
- `shortest_norms`: 每轮最短向量范数列表

**异常：**
- `InvalidBasisError`: 格基格式或参数非法
- `ReductionFailedError`: 约减过程失败

### `evaluate_basis_quality(B, reduced=True)`

评估格基质量。

**参数：**
- `B`: numpy int64 二维方阵（行向量基）
- `reduced`: 是否已约减（影响列范数排序）

**返回：** dict
- `shortest_norm`: 最短列范数
- `root_hermite_factor`: 根 Hermite 因子
- `hermite_factor`: Hermite 因子
- `orthogonality_defect`: 正交缺陷
- `log_volume`: 格体积（对数）
- `column_norms`: 所有列范数

### `basis_to_numpy(B)`

将任意格式的格基转为 numpy int64 数组。

### 异常类

- `LatticeReductionError`: 模块统一异常基类
- `InvalidBasisError(LatticeReductionError)`: 格基格式或内容非法
- `ReductionFailedError(LatticeReductionError)`: 约减过程失败

## 与 fpylll 的差异

| 特性 | fpylll | lattice_reduction |
|------|--------|-------------------|
| 底层实现 | C++ (fplll) | 纯 Python |
| 依赖 | GMP/MPFR/C编译环境 | numpy |
| 格基格式 | IntegerMatrix (行向量) | numpy array (行向量) |
| 原地修改 | ✓ | ✓ |
| 浮点精度 | 可配置 (double/mpfr) | 固定 float64 |
| 进度回调 | 不支持 | 支持（通过 progress 模块） |

## 架构说明

```
src/lattice_reduction/
├── __init__.py          # 统一入口（仅导出适配层接口）
├── _native/             # 上游原生源码（不直接使用）
│   ├── basis/           # 格基生成器
│   ├── gso/             # Gram-Schmidt 正交化
│   ├── lll/             # LLL 算法
│   ├── bkz/             # BKZ 算法
│   ├── enumeration/     # SVP 枚举器
│   └── quality/         # 质量评估
├── adapter/             # 适配层（对齐 fpylll 接口）
│   ├── common_adapter.py
│   ├── lll_adapter.py
│   ├── bkz_adapter.py
│   └── quality_adapter.py
└── docs/
```

修改算法核心逻辑 → 直接修改 `_native/` 层对应源码。
新增接口 → 在 `adapter/` 层扩展封装。
