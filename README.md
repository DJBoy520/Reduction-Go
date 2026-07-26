# Reduction-Go — ML-DSA 格攻击闭环测试环境

基于 Kannan 嵌入法的 ML-DSA (FIPS 204) 格攻击实现。  
从公钥 (A, t) 出发，构造 LWE 格基，通过 LLL/BKZ 约减恢复私钥 (s1, s2)。

**纯 Python 实现**，无 C/C++ 依赖，无需编译环境，开箱即用。

## 项目结构

```
Reduction-Go/
├── main.py                     # CLI 入口
├── manage.sh                   # 后台管理脚本 (start/stop/status)
├── requirements.txt            # Python 依赖
├── src/
│   ├── api.py                  # 统一 API 层（AttackConfig / run_attack）
│   ├── lattice_attack.py       # 格攻击核心（Kannan 嵌入 + LLL/BKZ + 候选提取）
│   ├── poly_math.py            # 多项式运算（negacyclic 卷积，自动 NTT 切换）
│   ├── progress.py             # 进度显示（LLL/BKZ 实时状态 + ETA）
│   ├── domain/                 # 领域层（零外部依赖，唯一定义源）
│   │   ├── params.py           # ML-DSA 参数集 + 精度常量 + LLL 参数
│   │   ├── exceptions.py       # 统一异常体系（FPLLError 基类）
│   │   ├── config.py           # AttackConfig 数据类
│   │   └── result.py           # VerifyResult 数据类
│   ├── common/                 # 通用基础模块（从 domain 重导出）
│   │   ├── logger.py           # 日志配置（控制台 + 文件双输出）
│   │   └── utils.py            # Timer、格式化等工具函数
│   ├── crypto/                 # 密码学基础
│   │   └── ntt.py              # 数论变换 (NTT)
│   ├── protocol/               # FIPS 204 协议层
│   │   ├── keygen.py           # 密钥生成（ExpandA + CBD，FIPS 204 §4.2）
│   │   ├── pubkey.py           # DER 公钥编解码
│   │   ├── spki.py             # SubjectPublicKeyInfo 编解码
│   │   ├── power2round.py      # Power2Round 编解码（FIPS 204 §4.1）
│   │   ├── cert_generator.py   # 测试证书生成
│   │   ├── cert_parser.py      # X.509 证书解析（支持 PEM/DER）
│   │   └── der_utils.py        # DER 编码工具
│   ├── lattice/                # 格基约减算法（纯 Python + mpmath）
│   │   ├── adapter.py          # 适配层（统一接口 + 精度管理）
│   │   ├── algorithms/         # 核心算法
│   │   │   ├── lll.py          # LLL 约减（mpmath 高精度）
│   │   │   ├── bkz.py          # BKZ 约减（Schnorr-Euchner 1994）
│   │   │   └── deep_insert.py  # 深插入策略
│   │   └── base/               # 基础设施
│   │       ├── gso.py          # Gram-Schmidt 正交化（纯 mpmath）
│   │       ├── enumeration/    # SVP 枚举器（三种变体，统一接口）
│   │       └── precision.py    # 精度管理（维度分层 + 自动升级）
│   └── utils/                  # （已清空，保留目录兼容）
├── tests/                      # 测试
├── certs/                      # 证书存放目录
└── logs/                       # 运行日志
```

### 依赖流（单向，无循环）

```
Domain (底)  →  Common / Lattice / Protocol  →  lattice_attack  →  API  →  CLI
```

`domain/` 层零外部业务依赖，仅使用 Python 标准库 + dataclasses。

## 依赖

```bash
pip install -r requirements.txt
```

- **numpy** — 矩阵运算
- **mpmath** — 多精度浮点运算（格约减核心精度引擎）
- **asn1crypto** — X.509 证书 ASN.1 解析

无 C/C++ 依赖，无需 GMP/MPFR，纯 Python 环境即可运行。

## 快速开始

```bash
# 最小参数集，快速验证
python3 main.py toy --no-bkz --n 10

# 标准 toy 测试
python3 main.py toy --no-bkz

# 带 BKZ 的完整攻击
python3 main.py toy --bkz-block-size 5 --bkz-max-loops 2
```

## 参数集

| 名称 | k | l | n | η | 格维度 | BKZ block | BKZ loops | 精度 (dps) | 用途 |
|------|---|---|---|---|--------|-----------|-----------|------------|------|
| `toy` (easy) | 2 | 2 | 16 | 2 | 65 | 10 | 2 | 80 | 秒级验证 |
| `medium` | 3 | 3 | 32 | 3 | 225 | 15 | 3 | 80 | 快速测试 |
| `hard` | 4 | 4 | 64 | 4 | 513 | 20 | 5 | 200 | 中等规模 |
| `extreme` | 6 | 6 | 128 | 4 | 1537 | 25 | 8 | 200 | 大规模测试 |
| `ML-DSA-44` | 4 | 4 | 256 | 2 | 2049 | 25 | 8 | 200 | FIPS 204 标准 |
| `ML-DSA-65` | 6 | 6 | 256 | 4 | 3073 | 30 | 10 | 200 | FIPS 204 标准 |
| `ML-DSA-87` | 8 | 8 | 256 | 2 | 4097 | 40 | 15 | 200 | FIPS 204 标准 |

> `toy` 是 `easy` 的别名。通过 `--k / --l / --n` 可覆盖任意参数集的维度。
> 
> `--n 10` 时格维度 = 2×10 + 2×10 + 1 = **41**，可用于秒级冒烟测试。

格维度 = `k*n + l*n + 1`（Kannan 嵌入，+1 为权重维度）。

## CLI 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `params` | `toy` | 参数集名称（toy/easy/medium/hard/extreme/ML-DSA-44/65/87） |
| `--no-bkz` | off | 跳过 BKZ，只跑 LLL |
| `--bkz-block-size N` | 配置值 | BKZ 块大小（越大约减越好但越慢） |
| `--bkz-max-loops N` | 配置值 | BKZ 最大循环数 |
| `--bkz-auto-abort` | off | BKZ 连续无改善时提前终止 |
| `--k / --l / --n` | 配置值 | 覆盖矩阵维度 |
| `--lll-delta` | 0.79 | LLL δ 参数，范围 (0.25, 1.0) |
| `--seed N` | 随机 | 随机种子（便于复现） |
| `--mp-dps N` | 自动 | 强制指定 mpmath 精度位数（覆盖自动分层） |
| `--no-auto-precision` | off | 关闭自适应精度升级，失败直接报错 |
| `--slack` | off | 启用 Power2Round 合并误差模式 |
| `--d N` | 13 | Power2Round 的 d 参数 |
| `--cert PATH` | 无 | 证书文件路径（PEM/DER），直接从证书提取公钥攻击 |
| `--toy-params` | off | 用 toy 参数集解析证书（k=l=2, n=30） |
| `--verbose` | off | DEBUG 级别日志 |
| `--log-level` | INFO | 自定义日志级别（DEBUG/INFO/WARNING/ERROR/CRITICAL） |

优先级：**CLI > 配置文件 > 默认值**。

---

## 使用示例

### 示例 1：最小参数，秒级验证

```bash
python3 main.py toy --no-bkz --n 10 --seed 42
```

- 格维度：2×10 + 2×10 + 1 = **41**
- 用途：开发调试、快速验证代码改动

### 示例 2：标准 toy 参数，LLL 约减

```bash
python3 main.py toy --no-bkz --verbose
```

- 格维度：**65**（k=2, l=2, n=16）
- 用途：日常测试，LLL 单独即可在小维度完美恢复私钥

### 示例 3：toy 参数 + BKZ 完整攻击

```bash
python3 main.py toy --bkz-block-size 5 --bkz-max-loops 2 --seed 42
```

- 格维度：**65**，BKZ block=5
- 用途：验证 BKZ 流程完整性

### 示例 4：medium 参数

```bash
python3 main.py medium --verbose
```

- 格维度：**225**（k=3, l=3, n=32）
- 用途：中等规模验证

### 示例 5：hard 参数，大维度挑战

```bash
python3 main.py hard --bkz-auto-abort --seed 42
```

- 格维度：**513**（k=4, l=4, n=64），BKZ block=20
- 用途：大维度性能测试

### 示例 6：自定义维度

```bash
python3 main.py toy --k 3 --l 3 --n 20 --no-bkz --seed 42
```

- 格维度：3×20 + 3×20 + 1 = **121**
- 用途：灵活调整问题规模

### 示例 7：从 X.509 证书攻击

```bash
# 先生成测试证书
python3 tests/gen_test_cert.py certs/test.pem --seed 42 --n 30

# 从证书提取公钥攻击
python3 main.py --cert certs/test.pem --toy-params --no-bkz

# 带 BKZ 的证书攻击
python3 main.py --cert certs/test.pem --toy-params --bkz-block-size 8 --bkz-auto-abort
```

### 示例 8：复现特定结果

```bash
python3 main.py toy --seed 12345 --no-bkz --verbose
```

固定随机种子，每次运行产生相同的密钥和格基，便于调试和对比。

### 示例 9：后台运行 + 查看日志

```bash
# 后台启动
bash manage.sh start toy --no-bkz --verbose

# 查看状态
bash manage.sh status

# 查看实时日志
tail -f logs/attack.log

# 停止
bash manage.sh stop
```

---

## 参数选择指南

### 根据目的选择

| 目的 | 推荐命令 |
|------|----------|
| 开发调试 | `python3 main.py toy --no-bkz --n 10` |
| 快速验证 | `python3 main.py toy --no-bkz` |
| 完整流程测试 | `python3 main.py toy --bkz-block-size 5 --bkz-max-loops 2` |
| 算法质量评估 | `python3 main.py medium --verbose` |
| 性能压测 | `python3 main.py hard --bkz-auto-abort` |
| 证书攻击 | `python3 main.py --cert xxx.pem --toy-params` |

### 格维度与相对耗时的关系

格约减耗时与维度呈超线性关系，以最小参数（dim=41）为基准 1：

| 格维度 | LLL 相对耗时 | BKZ 相对耗时 | 推荐场景 |
|--------|-------------|-------------|----------|
| ~40 | 1 | — | 单元测试、秒级验证 |
| ~65 | ~10 | ~100 | 日常开发（toy 默认） |
| ~225 | ~500 | ~5,000 | 功能验证（medium） |
| ~500 | ~5,000 | ~50,000 | 中等规模（hard） |
| ~1500 | ~50,000 | ~500,000 | 大规模（extreme） |
| ~2000+ | ~300,000 | 不推荐 | 需要耐心（ML-DSA-44） |

> 实际耗时取决于硬件性能。耗时与格维度呈超线性关系（约 O(n²·³)）。

### BKZ block_size 选择

- **block_size=2**: 等价于 LLL，最快
- **block_size=5-10**: 小规模测试，改善明显
- **block_size=15-25**: 中大规模，显著提升约减质量
- **block_size=30+**: 大规模，耗时急剧增长

建议：先用 `--no-bkz` 跑 LLL 看基线效果，再逐步增大 block_size。

### 自适应精度

项目内置维度分层精度管理：
- dim < 150：dps=80（快速模式）
- dim ≥ 250：dps=200（高精度模式）
- 精度不足时自动升级并重试（`--no-auto-precision` 可关闭）

可通过 `--mp-dps N` 强制指定精度。

---

## 输出说明

运行结束后输出三层验证结果：

```
候选统计:
  完美恢复:       1      ← 私钥完全匹配 ✓
  s1 完美恢复:    0      ← s1 匹配但 s2 不匹配
  替代短向量:     0      ← 范数接近但不完全匹配
  满足方程但过长: 0      ← 方程成立但范数太大
  无效解:         0      ← 方程不成立
```

三层验证逻辑：
1. **方程验证**：A·s1' + s2' ≡ t (mod q)
2. **范数比较**：候选范数 vs 真实私钥范数
3. **精确匹配**：逐元素比较 s1' == s1 且 s2' == s2

日志输出到 `logs/attack.log`。

## 测试

```bash
# 全部测试
python3 -m pytest tests/ -v

# 单项测试
python3 -m pytest tests/test_power2round.py -v      # Power2Round 往返测试
python3 tests/smoke_adapter.py                       # 格约减适配层冒烟
python3 tests/test_step7_consistency.py              # 功能一致性验证
python3 tests/test_step8_robustness.py               # 健壮性加固验证
```

## 技术细节

### Kannan 嵌入

将 LWE 方程 A·s1 + s2 ≡ t (mod q) 转化为格中短向量问题：

```
格基 B = [ I_{ln}   -A_flat^T   0 ]
         [ 0        q·I_{kn}    0 ]
         [ 0        t^T         1 ]
```

格中包含向量 v = (s1, s2, 1)，其范数即为私钥范数。

### 格约减算法

- **LLL** (Lenstra–Lenstra–Lovász)：多项式时间格基约减，保证找到近似最短向量
  - 实现：Schnorr-Euchner 1994，纯 mpmath 高精度
  - Size Reduction + Lovász 条件检查 + GSO 增量更新
- **BKZ** (Block Korkine–Zolotarev)：分块约减，block_size 越大质量越好但越慢
  - 实现：Schnorr-Euchner 1994，纯 mpmath
  - LLL 预约减 → 遍历 block → SVP 枚举（局部方阵切片） → Deep Insertion
  - DELTA = 0.999（严格 Lovász 条件，更强约减力度）
- **SVP 枚举器**（三种变体，统一接口 `(basis_block, gs_norms, gs_coeffs) → (proj_len, coeff_vec)`）：
  - Schnorr-Euchner 1991 (ceil-bound) — 默认
  - Schnorr-Euchner 1994 (controlled stepping)
  - Schnorr-Hörner 1995

### GSO 数学不变量

- **Size Reduction**：只修改 GSO 系数 μ，正交范数 B* 严格不变
- **列交换**：格体积守恒 — B'_s · B'_{s+1} = B_s · B_{s+1}
- **点积**：快速路径（np.int64 原生 dot）+ 安全降级（溢出时 fallback 到 Python 大整数）

### 精度管理

- 基向量：int64（精确整数）
- GSO 系数/范数：mpmath.mpf（任意精度）
- 点积：Python int（object dtype，无溢出）
- 自适应精度：维度分层 + 失败检测 + 自动升级重试

### Power2Round 模式

当公钥仅含高位 t1（FIPS 204 标准格式）时：
- t = t1·2^d + t0，其中 t0 为低位误差
- 令 s2' = s2 - t0，用 t_recon = t1·2^d 攻击
- s2' 仍相对较小，标准 Kannan 嵌入可恢复

启用方式：`--slack --d 13`

## License

本项目代码为内部研发使用。
格约减算法模块基于 [LatticeReductionAlgorithms](https://github.com/vttresearch/LatticeReductionAlgorithms)（GPL v2）。
