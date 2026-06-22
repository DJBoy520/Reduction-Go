# Reduction-Go — ML-DSA 格攻击闭环测试环境

基于 Kannan 嵌入法的 ML-DSA (FIPS 204) 格攻击实现。  
从公钥 (A, t) 出发，构造 LWE 格基，通过 LLL/BKZ 约减恢复私钥 (s1, s2)。

## 项目结构

```
Reduction-Go/
├── main.py                  # 主入口，CLI 参数解析 + 五步流程
├── manage.sh                # 后台管理脚本 (start/stop/status)
├── src/
│   ├── __init__.py
│   ├── params.py            # 参数集配置 (easy/medium/hard/extreme)
│   ├── keygen.py            # 密钥生成 (expand_a, A 矩阵展开)
│   ├── pubkey.py            # DER 公钥编解码
│   ├── lattice_attack.py    # 格攻击核心 (Kannan 嵌入 + LLL/BKZ)
│   ├── poly_math.py         # 多项式运算 (矩阵向量乘, 模加)
│   ├── progress.py          # tqdm 进度条 (LLL 动画 + BKZ 循环)
│   ├── protocol_adapter.py  # FIPS 204 Power2Round 编解码
│   ├── cert_parser.py       # X.509 证书解析 (PEM/DER → 公钥)
│   ├── cert_generator.py    # 测试证书生成工具
│   ├── spki.py              # SubjectPublicKeyInfo 编解码
│   └── logger.py            # 日志配置
├── tests/
│   ├── gen_test_cert.py     # 测试证书生成脚本
│   └── test_power2round.py  # Power2Round 往返测试
├── certs/                   # 证书存放目录
├── logs/                    # 运行日志
└── .venv/                   # Python 虚拟环境
```

## 依赖

```bash
pip install numpy fpylll tqdm asn1crypto
```

- **numpy** — 矩阵运算
- **fpylll** — LLL/BKZ 格基约减 (依赖 GMP、MPFR)
- **tqdm** — 进度条
- **asn1crypto** — X.509 证书 ASN.1 解析

## 快速开始

```bash
# 最简单的运行 (easy 参数集, double 精度)
python3 main.py toy

# 详细输出
python3 main.py toy --verbose

# 只跑 LLL, 跳过 BKZ
python3 main.py toy --no-bkz
```

## 参数集

| 名称 | k | l | n | 格维度 | BKZ block | 浮点精度 | 用途 |
|------|---|---|---|--------|-----------|----------|------|
| `easy` (toy) | 2 | 2 | 50 | 201 | 8 | double / 53bit | 快速测试 |
| `medium` | 3 | 3 | 80 | 481 | 15 | mpfr / 200bit | 中等规模 |
| `hard` | 4 | 4 | 120 | 961 | 20 | mpfr / 200bit | 大规模 |
| `extreme` | 5 | 5 | 200 | 2001 | 25 | mpfr / 200bit | 极限测试 |

格维度 = `k*n + l*n + 1` (Kannan 嵌入，+1 为权重维度)。

## CLI 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `params` | `toy` | 参数集名称 (easy/medium/hard/extreme) |
| `--verbose` | off | DEBUG 级别日志 |
| `--no-bkz` | off | 跳过 BKZ，只跑 LLL |
| `--bkz-block-size` | 配置值 | BKZ 块大小 |
| `--bkz-max-loops` | 配置值 | BKZ 最大循环数 |
| `--bkz-auto-abort` | off | BKZ 无改善时提前终止 |
| `--k / --l / --n` | 配置值 | 覆盖矩阵维度 |
| `--lll-delta` | 0.999 | LLL 约减质量参数 |
| `--seed` | 随机 | 随机种子 (便于复现) |
| `--float-type` | 配置值 | 浮点类型: `mpfr` / `double` / `long double` |
| `--precision` | 配置值 | MPFR 精度 (bit)，仅 mpfr 模式生效 |
| `--cert` | 无 | 证书文件路径 (PEM/DER)，直接从证书提取公钥进行攻击 |
| `--toy-params` | off | 使用 toy 参数集 (k=l=2, n=30) 解析证书 |

命令行参数优先级：**CLI > 配置文件 > 默认值**。

## 使用示例

### 基础测试 (easy + double)

```bash
python3 main.py toy --verbose
```

- 参数：k=2, l=2, n=50, 格维度 201
- 浮点：IEEE 754 双精度 (53 bit)
- LLL 耗时约 30s，能完美恢复私钥

### 只跑 LLL，跳过 BKZ

```bash
python3 main.py toy --no-bkz
```

### 指定随机种子复现结果

```bash
python3 main.py toy --seed 12345 --verbose
```

### MPFR 高精度

```bash
python3 main.py medium --float-type mpfr --precision 200 --verbose
```

### 从 X.509 证书攻击

```bash
# 生成测试证书
python3 tests/gen_test_cert.py certs/test.pem --seed 42 --n 30

# 从证书攻击 (toy 参数，只跑 LLL)
python3 main.py --cert certs/test.pem --toy-params --no-bkz

# 从证书攻击 (带 BKZ)
python3 main.py --cert certs/test.pem --toy-params --bkz-block-size 8 --bkz-auto-abort
```

### 后台运行 + 查看状态

```bash
# 后台启动
bash manage.sh start toy --verbose

# 查看状态
bash manage.sh status

# 停止
bash manage.sh stop
```

日志输出到 `logs/attack.log`，标准输出到 `logs/stdout.log`。

## ⚠️ 已知问题：BKZ Babai 死循环

### 现象

fpylll 的 BKZ 实现在某些格基上会触发：

```
terminate called after throwing an instance of 'std::runtime_error'
  what():  infinite loop in babai
```

随后进程收到 SIGSEGV 崩溃。

### 原因

BKZ 内部的 Babai 最近平面算法在格基质量差或维度较高时可能不收敛，导致无限循环。这是 **fpylll 库的底层 C++ 问题**，不是我们的代码 bug。

### 规避方法

**方法 1：跳过 BKZ（推荐）**

```bash
python3 main.py toy --no-bkz
```

对 easy 参数集，LLL 单独就能完美恢复私钥。

**方法 2：减小 BKZ block_size**

```bash
python3 main.py toy --bkz-block-size 4
```

**方法 3：使用 `--bkz-auto-abort`**

```bash
python3 main.py toy --bkz-auto-abort
```

**方法 4：换浮点精度**

```bash
python3 main.py toy --float-type mpfr --precision 100
```

**方法 5：缩小问题规模**

```bash
python3 main.py toy --k 2 --l 2 --n 30
```

### 建议策略

```
优先级: --no-bkz > --bkz-auto-abort > 减小 block_size > 换浮点精度 > 缩小 n
```

## 测试

```bash
# Power2Round 编解码往返测试
cd tests && python3 test_power2round.py
```

5 项测试覆盖：往返一致性、误差范围、仅高位还原、真实密钥验证、松弛格维度预览。

## 输出说明

运行结束后在 `logs/summary.txt` 生成摘要：

```
候选统计:
  完美恢复:       1      ← 私钥完全匹配
  替代短向量:     0      ← 范数接近但不完全匹配
  满足方程但过长: 0      ← 方程成立但范数太大
  无效解:         0      ← 方程不成立
```

三层验证逻辑：
1. **方程验证**：A·s1' + s2' ≡ t (mod q)
2. **范数比较**：候选范数 vs 真实私钥范数
3. **精确匹配**：逐元素比较 s1' == s1 且 s2' == s2
