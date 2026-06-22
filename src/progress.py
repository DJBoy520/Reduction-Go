"""
进度条模块 — LLL/BKZ 动态进度显示。

fpylll 的 LLL.reduction() 是阻塞调用，无法获取内部迭代回调，
所以 LLL 用计时+状态动画显示。BKZ 有 callback 接口，可以实时显示循环进度。
"""

import sys
import threading
import time

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


def _fmt_time(seconds: float) -> str:
    """格式化秒数为人类可读字符串。"""
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


class LLLProgress:
    """LLL 约减进度显示。

    fpylll 的 LLL.reduction() 是单次阻塞调用，没有迭代回调，
    所以用后台线程做状态动画 + 计时。
    """

    def __init__(self, dim: int, float_type: str, precision: int):
        self.dim = dim
        self.float_type = float_type
        self.precision = precision
        self.t0 = None
        self._stop = threading.Event()
        self._thread = None
        self._pbar = None

    def start(self):
        self.t0 = time.time()
        if not HAS_TQDM:
            return

        desc = f"LLL {self.dim}×{self.dim}"
        if self.float_type == "mpfr":
            desc += f" mpfr/{self.precision}bit"
        else:
            desc += f" {self.float_type}"

        self._pbar = tqdm(
            total=None,
            desc=desc,
            bar_format="{desc}: {elapsed}",
            leave=True,
            file=sys.stderr,
        )
        self._thread = threading.Thread(target=self._update_loop, daemon=True)
        self._thread.start()

    def _update_loop(self):
        dots = 0
        while not self._stop.is_set():
            elapsed = time.time() - self.t0
            spin = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"[dots % 10]
            self._pbar.set_description(
                f"LLL {self.dim}×{self.dim} {spin} {_fmt_time(elapsed)}"
            )
            self._pbar.update(0)
            dots += 1
            self._stop.wait(0.5)

    def finish(self) -> float:
        elapsed = time.time() - self.t0
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        if self._pbar:
            self._pbar.set_description(
                f"LLL {self.dim}×{self.dim} ✓ {_fmt_time(elapsed)}"
            )
            self._pbar.close()
        return elapsed


class BKZProgress:
    """BKZ 约减进度显示。

    fpylll BKZ 有 callback 接口，每次循环都会调用，
    可以实时显示循环数、当前最短向量范数、耗时。
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

    def start(self):
        self.t0 = time.time()
        if not HAS_TQDM:
            return

        desc = f"BKZ b={self.block_size}"
        if self.float_type == "mpfr":
            desc += f" mpfr/{self.precision}bit"

        self._pbar = tqdm(
            total=self.max_loops,
            desc=desc,
            bar_format="{desc}: {n_fmt}/{total_fmt} |{bar}| {elapsed}",
            leave=True,
            file=sys.stderr,
        )

    def __call__(self, loop: int, bkz):
        self.loop = loop
        elapsed = time.time() - self.t0

        # 从 BKZ 状态获取最短向量范数
        min_norm = None
        try:
            r = [bkz.M.get_r(i, i) for i in range(bkz.M.d)]
            non_zero = [x for x in r if x != 0]
            if non_zero:
                min_norm = min(abs(x) ** 0.5 for x in non_zero)
        except Exception:
            pass

        if self._pbar:
            self._pbar.update(1)
            postfix = _fmt_time(elapsed)
            if min_norm is not None:
                postfix += f" |r*|≈{_fmt_norm(min_norm)}"
            self._pbar.set_postfix_str(postfix)
        elif HAS_TQDM is False:
            # 回退到 logger
            import logging
            logger = logging.getLogger(__name__)
            norm_str = f" |r*|≈{_fmt_norm(min_norm)}" if min_norm else ""
            logger.info(f"    BKZ {loop}/{self.max_loops}: "
                        f"耗时 {_fmt_time(elapsed)}{norm_str}")

    def finish(self) -> float:
        elapsed = time.time() - self.t0
        if self._pbar:
            desc = f"BKZ b={self.block_size}"
            if self.float_type == "mpfr":
                desc += f" mpfr/{self.precision}bit"
            self._pbar.set_description(f"{desc} ✓")
            self._pbar.set_postfix_str(
                f"{self.loop} loops, {_fmt_time(elapsed)}"
            )
            self._pbar.close()
        return elapsed
