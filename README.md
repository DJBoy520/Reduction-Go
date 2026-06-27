# Reduction-Go — ML-DSA 格攻击闭环测试环境

基于 Kannan 嵌入法的 ML-DSA (FIPS 204) 格攻击实现。  
从公钥 (A, t) 出发，构造 LWE 格基，通过 LLL/BKZ 约减恢复私钥 (s1, s2)。

**纯 Python 实现**，无 C/C++ 依赖，无需编译环境，开箱即用。

## 项目结构

```
Reduction-Go/
├── main.py                     # CLI 入口
├── manage.sh                   # 后台管理脚本 (start/stop/status)
├── src/
│   ├── api.py                  # 统一 API 层（AttackConfig / run_attack）
│   ├── lattice_attack.py       # 格攻击核心（Kannan 嵌入 + LLL/BKZ）
│   ├── poly_math.py            # 多项式运算（negacyclic 卷积）
│   ├── protocol_adapter.py     # FIPS 204 Power2Round 编解码
│   ├── progress.py             # 进度显示（LLL/BKZ 实时状态）
│   ├── crypto/                 # 密码学基础
│   │   └── ntt.py              # 数论变换 (NTT)
│   ├── keys/                   # 密钥与证书
│   │   ├── keygen.py           # 密钥生成（expand_a, A 矩阵展开）
│   │   ├── pubkey.py           # DER 公钥编解码
│   │   ├── spki.py             # SubjectPublicKeyInfo 编解码
│   │   ├── cert_generator.py   # 测试证书生成
│   │   ├── cert_parser.py      # X.509 证书解析
│   │   └── der_utils.py        # DER 编码工具
│   ├── utils/                  # 通用工具
│   │   ├── params.py           # 参数集配置
│   │   └── logger.py           # 日志配置
│   └── lattice_reduction/      # 格基约减算法（纯 Python）
│       ├── adapter/            # 适配层（对齐 fpylll 接口）
│       └── _native/            # 上游 LatticeReductionAlgorithms 源码
├── tests/                      # 测试
├── certs/                      # 证书存放目录
└── logs/                       # 运行日志
```

## 依赖

```bash
pip install numpy tqdm asn1crypto
```

- **numpy** — 矩阵运算
- **tqdm** — 进度条
- **asn1crypto** — X.509 证书 ASN.1 解析

无 C/C++ 依赖，无需 GMP/MPFR，纯 Python 环境即可运行。

## 快速开始

```bash
# 最小参数集，快速验证（约 0.1 秒）
python3 main.py toy --no-bkz --n 10

# 标准 toy 测试（约 30-60 秒）
python3 main.py toy --no-bkz

# 带 BKZ 的完整攻击
python3 main.py toy --bkz-block-size 5 --bkz-max-loops 2
```

## 参数集

| 名称 | k | l | n | 格维度 | BKZ block | 浮点精度 | 预计耗时 | 用途 |
|------|---|---|---|--------|-----------|----------|----------|------|
| `toy` (easy) | 2 | 2 | 50 | 201 | 8 | double | ~30s LLL | 快速测试 |
| `medium` | 3 | 3 | 80 | 481 | 15 | mpfr/200 | ~数分钟 | 中等规模 |
| `hard` | 4 | 4 | 120 | 961 | 20 | mpfr/200 | ~数十分钟 | 大规模 |
| `extreme` | 5 | 5 | 200 | 2001 | 25 | mpfr/200 | ~数小时 | 极限测试 |
| `ML-DSA-44` | 4 | 4 | 256 | 2049 | 25 | mpfr/200 | FIPS 204 标准 | 真实参数 |
| `ML-DSA-65` | 6 | 6 | 256 | 3073 | 30 | mpfr/200 | FIPS 204 标准 | 真实参数 |
| `ML-DSA-87` | 8 | 8 | 256 | 4097 | 35 | mpfr/200 | FIPS 204 标准 | 真实参数 |

格维度 = `k*n + l*n + 1`（Kannan 嵌入，+1 为权重维度）。

## CLI 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `params` | `toy` | 参数集名称 |
| `--no-bkz` | off | 跳过 BKZ，只跑 LLL |
| `--bkz-block-size N` | 配置值 | BKZ 块大小（越大约减越好但越慢） |
| `--bkz-max-loops N` | 配置值 | BKZ 最大循环数 |
| `--bkz-auto-abort` | off | BKZ 连续无改善时提前终止 |
| `--k / --l / --n` | 配置值 | 覆盖矩阵维度 |
| `--lll-delta` | 0.999 | LLL 约减质量参数 (0.25, 1.0) |
| `--seed N` | 随机 | 随机种子（便于复现） |
| `--float-type` | 配置值 | `mpfr` / `double` / `long double` |
| `--precision N` | 配置值 | MPFR 精度 (bit) |
| `--slack` | off | 启用 Power2Round 合并误差模式 |
| `--d N` | 13 | Power2Round 的 d 参数 |
| `--cert PATH` | 无 | 证书文件路径，直接从证书提取公钥攻击 |
| `--toy-params` | off | 用 toy 参数集解析证书 |
| `--verbose` | off | DEBUG 级别日志 |
| `--log-level` | INFO | 自定义日志级别 |

优先级：**CLI > 配置文件 > 默认值**。

---

## 使用示例

### 示例 1：最小参数，秒级验证

```bash
python3 main.py toy --no-bkz --n 10 --k 2 --l 2 --seed 42
```

- 格维度：2×10 + 2×10 + 1 = **41**
- 耗时：**< 0.1 秒**
- 用途：开发调试、快速验证代码改动

### 示例 2：标准 toy 参数，LLL 约减

```bash
python3 main.py toy --no-bkz --verbose
```

- 格维度：**201**
- 耗时：**约 30-60 秒**
- 用途：日常测试，LLL 单独即可在小维度完美恢复私钥

### 示例 3：toy 参数 + BKZ 完整攻击

```bash
python3 main.py toy --bkz-block-size 5 --bkz-max-loops 2 --seed 42
```

- 格维度：**201**，BKZ block=5
- 耗时：**约 1-2 分钟**
- 用途：验证 BKZ 流程完整性

### 示例 4：medium 参数，MPFR 高精度

```bash
python3 main.py medium --verbose
```

- 格维度：**481**，MPFR 200-bit
- 耗时：**约数分钟**
- 用途：中等规模验证

### 示例 5：hard 参数，大维度挑战

```bash
python3 main.py hard --bkz-auto-abort --seed 42
```

- 格维度：**961**，BKZ block=20
- 耗时：**约数十分钟**
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

### 格维度与耗时的关系

格约减是纯 Python 实现，耗时与格维度呈超线性关系：

| 格维度 | LLL 耗时 | BKZ 耗时 | 推荐场景 |
|--------|----------|----------|----------|
| ~40 | < 0.1s | ~1s | 单元测试 |
| ~200 | ~30s | ~1-2min | 日常开发 |
| ~500 | ~数分钟 | ~数十分钟 | 功能验证 |
| ~1000 | ~数十分钟 | ~数小时 | 性能测试 |
| ~2000+ | ~数小时 | 不推荐 | 需要耐心 |

### BKZ block_size 选择

- **block_size=2**: 等价于 LLL，最快
- **block_size=5-10**: 小规模测试，改善明显
- **block_size=15-25**: 中大规模，显著提升约减质量
- **block_size=30+**: 大规模，耗时急剧增长

建议：先用 `--no-bkz` 跑 LLL 看基线效果，再逐步增大 block_size。

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
python3 tests/smoke_native.py                        # 格约减原生层冒烟
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
- **BKZ** (Block Korkine–Zolotarev)：分块约减，block_size 越大质量越好但越慢
- **SVP 枚举器**：Schnorr-Euchner / Schnorr-Hörner 枚举策略

### Power2Round 模式

当公钥仅含高位 t1（FIPS 204 标准格式）时：
- t = t1·2^d + t0，其中 t0 为低位误差
- 令 s2' = s2 - t0，用 t_recon = t1·2^d 攻击
- s2' 仍相对较小，标准 Kannan 嵌入可恢复

启用方式：`--slack --d 13`

## License

本项目代码为内部研发使用。
格约减算法模块基于 [LatticeReductionAlgorithms](https://github.com/vttresearch/LatticeReductionAlgorithms)（GPL v2）。
