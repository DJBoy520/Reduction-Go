# lattice_reduction 模块架构文档

## 模块定位

`lattice_reduction` 是 Reduction-Go 项目的格基约减算法引擎，承载全部格约减、格问题求解、格质量评估能力。
替代原 fpylll/fplll C++ 依赖，实现纯 Python 格约减能力。

## 目录结构

```
src/lattice_reduction/
├── __init__.py              # 对外统一入口，仅导出适配后的接口
├── _native/                 # 原生源码层：上游 LatticeReductionAlgorithms 源码
│   ├── basis/               # 格基数据结构、基础行变换与操作
│   ├── gso/                 # Gram-Schmidt 正交化实现
│   ├── lll/                 # LLL 算法变种实现
│   ├── bkz/                 # BKZ 算法变种实现
│   ├── enumeration/         # SVP 枚举器核心实现
│   ├── quality/             # 格基质量评估：正交缺陷、Hermite 因子等
│   ├── vector_problems/     # 向量问题：CVP、Babai 等（预留）
│   └── internal_utils/      # 格算法专用工具
├── adapter/                 # 封装适配层：对接现有项目接口
│   ├── lll_adapter.py       # LLL 系列接口适配
│   ├── bkz_adapter.py       # BKZ 系列接口适配
│   ├── quality_adapter.py   # 质量评估接口适配
│   └── common_adapter.py    # 通用格式转换、异常映射
└── docs/                    # 模块内架构说明、接口文档
    └── ARCHITECTURE.md      # 本文件
```

## 分层职责

### 原生源码层 (`_native/`)

存放上游 LatticeReductionAlgorithms 项目的全部功能性算法源码。

**修改规则：**
- ✅ 允许：修正导入路径以适配新目录结构
- ✅ 允许：修复明显的 bug
- ❌ 禁止：修改算法核心逻辑
- ❌ 禁止：删除版权声明

### 封装适配层 (`adapter/`)

100% 对齐现有项目的接口规范，做入参/出参/异常兼容。

**职责：**
- 入参格式转换（fpylll IntegerMatrix ↔ numpy array）
- 函数签名对齐（上游接口 → 现有项目接口）
- 出参结构对齐
- 异常类型映射
- 浮点精度控制桥接

### 统一入口 (`__init__.py`)

仅导出适配层的类与函数，禁止直接暴露 `_native` 层内部。

## 上游项目信息

- **来源：** https://github.com/vttresearch/LatticeReductionAlgorithms
- **协议：** GPL v2（与项目兼容）
- **依赖：** numpy, tqdm（核心）

## 当前项目 fpylll 使用范围

仅 `src/lattice_attack.py` 导入 fpylll：
```python
from fpylll import IntegerMatrix, LLL, BKZ, FPLLL
```

使用场景：
1. `build_lattice_basis()` → 创建 `IntegerMatrix`
2. `LLL.reduction(B, ...)` → LLL 约减
3. `BKZ.reduction(B, Param(...))` → BKZ 约减
4. `FPLLL.set_precision()` → 设置浮点精度

## 适配层接口清单

适配层需提供以下接口（与原 fpylll 调用完全兼容）：

| 接口 | 原调用 | 适配后调用 |
|------|--------|------------|
| 格基构建 | `IntegerMatrix(dim, dim)` + 赋值 | `build_lattice_basis(A, t, q)` → numpy array |
| LLL 约减 | `LLL.reduction(B, delta=0.999, float_type="mpfr")` | `lll_reduce(B, delta=0.999, float_type="mpfr")` |
| BKZ 约减 | `BKZ.reduction(B, Param(block_size=20))` | `bkz_reduce(B, block_size=20, max_loops=8)` |
| 精度控制 | `FPLLL.set_precision(200)` | 内部处理（上游无此概念） |
| 格基读取 | `int(B[i, j])` | numpy array 直接索引 |
