"""
进度显示模块 — LLL/BKZ 实时状态反馈。

LLL: l3fp() 是阻塞 Python 调用，无迭代回调，
     用后台线程每秒刷新 elapsed time。
BKZ: 拆成 max_loops 次 bkz_se_pc() 调用，
     每轮结束 Python 拿回控制权，更新进度条+最短范数。

输出策略:
  - TTY 环境 (终端直接跑): 用 tqdm 动态进度条
  - 非 TTY (重定向/管道): 用 print + \r 手动刷新，兼容日志文件
"""

import sys
import threading
import time

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


def _is_tty() -> bool:
    """检测输出是否为真实终端。"""
    return sys.stderr.isatty() if hasattr(sys.stderr, 'isatty') else False


def _fmt_time(seconds: float) -> str:
    """格式化秒数。"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s"


def _fmt_norm(norm: float) -> str:
    """格式化范数。"""
    if norm < 1000:
        return f"{norm:.1f}"
    elif norm < 1e6:
        return f"{norm/1000:.1f}k"
    elif norm < 1e9:
        return f"{norm/1e6:.1f}M"
    else:
        return f"{norm/1e6:.0f}M"


# ── 经验耗时估算 ────────────────────────────────────────────────────────────

def estimate_lll_time(dim: int) -> tuple[float, float]:
    """估算 LLL 耗时范围（秒），基于实际性能数据。

    fpylll wrapper: ~dim^2.5 * 0.001s
    native JIT: ~dim^2.5 * 0.01s (小维度)
    """
    base = (dim / 200) ** 2.5
    # fpylll 可用时更快，native 在大维度时显著变慢
    try:
        import fpylll  # noqa
        lo = max(1, base * 5)
        hi = lo * 5
    except ImportError:
        lo = max(1, base * 10)
        hi = lo * 10
    return lo, hi


def estimate_bkz_time(dim: int, block_size: int, max_loops: int) -> tuple[float, float]:
    """估算 BKZ 耗时范围（秒）。"""
    base = (dim / 200) ** 2.5 * (block_size / 10) ** 1.5
    try:
        import fpylll  # noqa
        lo = max(1, base * 3 * max_loops)
        hi = lo * 3
    except ImportError:
        lo = max(1, base * 5 * max_loops)
        hi = lo * 4
    return lo, hi


def print_estimate(dim: int, block_size: int, max_loops: int,
                   float_type: str, precision: int):
    """打印参数和耗时估算。"""
    lll_lo, lll_hi = estimate_lll_time(dim)
    bkz_lo, bkz_hi = estimate_bkz_time(dim, block_size, max_loops)

    print(f"\n{'='*50}")
    print(f"  维度 = {dim}")
    print(f"  BlockSize = {block_size}")
    print(f"  Precision = {float_type}/{precision}bit" if float_type == "mpfr"
          else f"  FloatType = {float_type}")
    print(f"  预计 LLL: {_fmt_time(lll_lo)} ~ {_fmt_time(lll_hi)}")
    print(f"  预计 BKZ: {_fmt_time(bkz_lo)} ~ {_fmt_time(bkz_hi)}")
    print(f"{'='*50}\n")


# ── LLL 进度 ────────────────────────────────────────────────────────────────

class LLLProgress:
    """LLL 约减进度 — 后台线程每秒刷新 elapsed time。

    l3fp() 是单次阻塞 Python 调用，
    Python 拿不到控制权，只能用后台线程显示计时。

    输出方式:
      - TTY: tqdm 进度条 (total=None, 纯计时模式)
      - 非 TTY: \r 行首刷新 + 纯文本
    """

    def __init__(self, dim: int, float_type: str, precision: int):
        self.dim = dim
        self.float_type = float_type
        self.precision = precision
        self.t0 = None
        self._stop = threading.Event()
        self._thread = None
        self._pbar = None
        self._use_tqdm = HAS_TQDM and _is_tty()

    def start(self):
        self.t0 = time.time()

        if self._use_tqdm:
            self._pbar = tqdm(
                total=None,
                desc="LLL",
                bar_format="{desc} | {elapsed}",
                leave=True,
                file=sys.stderr,
            )
            self._thread = threading.Thread(target=self._update_loop_tqdm, daemon=True)
            self._thread.start()
        else:
            # 非 TTY: 用 \r 手动刷新
            sys.stdout.write("\nLLL 运行中... \n")
            sys.stdout.flush()
            self._thread = threading.Thread(target=self._update_loop_plain, daemon=True)
            self._thread.start()

    def _update_loop_tqdm(self):
        """tqdm 模式: 每秒更新描述。"""
        while not self._stop.is_set():
            elapsed = time.time() - self.t0
            self._pbar.set_description(f"LLL | elapsed={_fmt_time(elapsed)}")
            self._pbar.refresh()
            self._stop.wait(1.0)

    def _update_loop_plain(self):
        """纯文本模式: 用 \r 在行首刷新。"""
        while not self._stop.is_set():
            elapsed = time.time() - self.t0
            msg = f"\rLLL | elapsed={_fmt_time(elapsed)}"
            sys.stdout.write(msg)
            sys.stdout.flush()
            self._stop.wait(1.0)

    def finish(self) -> float:
        elapsed = time.time() - self.t0
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

        if self._pbar is not None:
            self._pbar.set_description(f"LLL ✓ {_fmt_time(elapsed)}")
            self._pbar.refresh()
            self._pbar.close()
        else:
            # 清除当前行，输出最终状态
            sys.stdout.write("\r" + " " * 60 + "\r")
            print(f"LLL ✓ {_fmt_time(elapsed)}")
        return elapsed


# ── BKZ 进度 ────────────────────────────────────────────────────────────────

class BKZProgress:
    """BKZ 约减进度 — 循环驱动，每轮更新进度条。

    把 BKZ 拆成 max_loops 次单轮 bkz_se_pc() 调用，
    每轮结束后 Python 拿回控制权更新显示。

    输出方式:
      - TTY: tqdm 进度条 (有 total, 显示 x/max_loops)
      - 非 TTY: 每轮一行 print (不刷行, 保留历史)
    """

    def __init__(self, dim: int, block_size: int, max_loops: int,
                 float_type: str, precision: int):
        self.dim = dim
        self.block_size = block_size
        self.max_loops = max_loops
        self.float_type = float_type
        self.precision = precision
        self.t0 = None
        self.loop = 0
        self._pbar = None
        self._use_tqdm = HAS_TQDM and _is_tty()

    def start(self):
        self.t0 = time.time()
        self.loop = 0

        if self._use_tqdm:
            self._pbar = tqdm(
                total=self.max_loops,
                desc="BKZ",
                bar_format="{desc} | {n_fmt}/{total_fmt} |{bar}| {elapsed} | {postfix}",
                leave=True,
                file=sys.stderr,
            )
        else:
            # 非 TTY: 每轮一行输出
            print(f"\nBKZ 开始 (block_size={self.block_size}, max_loops={self.max_loops})")

    def update(self, loop: int, shortest_norm: float = None,
               real_norm: float = None):
        """每轮 BKZ 结束后调用。"""
        self.loop = loop
        elapsed = time.time() - self.t0

        # 构建 postfix
        parts = [f"elapsed={_fmt_time(elapsed)}"]
        if shortest_norm is not None:
            parts.append(f"|r*|={_fmt_norm(shortest_norm)}")
        if real_norm is not None and real_norm > 0:
            ratio = shortest_norm / real_norm
            parts.append(f"ratio={ratio:.2f}x")
        postfix = " | ".join(parts)

        if self._pbar is not None:
            self._pbar.update(1)
            self._pbar.set_postfix_str(postfix)
            self._pbar.refresh()
        else:
            # 非 TTY: 每轮一行
            print(f"  BKZ Loop {loop}/{self.max_loops} | {postfix}")

    def finish(self) -> float:
        elapsed = time.time() - self.t0
        if self._pbar is not None:
            self._pbar.set_description(f"BKZ ✓ {_fmt_time(elapsed)}")
            self._pbar.refresh()
            self._pbar.close()
        else:
            print(f"BKZ ✓ {_fmt_time(elapsed)}")
        return elapsed
