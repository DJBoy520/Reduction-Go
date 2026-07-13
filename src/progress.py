"""进度显示 — LLL/BKZ 实时状态 + 动态 ETA。

无外部依赖：通过 stderr 单行刷新（\\r）显示进度，
后台守护线程以固定间隔刷新显示，避免阻塞主计算线程。
"""

import logging
import sys
import threading
import time

logger = logging.getLogger(__name__)


def _is_tty():
    return sys.stderr.isatty() if hasattr(sys.stderr, "isatty") else False


def _fmt_time(seconds):
    """将秒数格式化为 HH:MM:SS 或 MM:SS。"""
    if seconds is None or seconds < 0:
        return "??:??"
    seconds = int(seconds)
    if seconds < 0:
        return "??:??"
    h, remainder = divmod(seconds, 3600)
    m, s = divmod(remainder, 60)
    if h > 0:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:d}:{s:02d}"


def print_estimate(dim, block_size, beta, *args, **kwargs):
    """打印预估时间（不使用 tqdm），一次性输出。
    *args/**kwargs 兼容调用方传入 float_type, precision, dps 等额外参数。
    """
    if dim <= 0 or block_size <= 0:
        print(f"[预估] dim={dim}, block_size={block_size}, β={beta}")
        return
    # 简单经验公式
    est = 0.1 * dim ** 2 * (block_size / dim) ** 4
    if est < 1:
        print(f"[预估] dim={dim}, block_size={block_size}, β={beta} → < 1s")
    elif est < 60:
        print(f"[预估] dim={dim}, block_size={block_size}, β={beta} → ~{est:.0f}s")
    elif est < 3600:
        print(f"[预估] dim={dim}, block_size={block_size}, β={beta} → ~{est / 60:.1f}min")
    else:
        print(f"[预估] dim={dim}, block_size={block_size}, β={beta} → ~{est / 3600:.1f}h")


class _ProgressBase:
    """进度追踪器基类：后台线程 + 单行刷新。"""

    def __init__(self, dim, dps=None):
        self.dim = dim
        self.dps = dps
        self._start_time = None
        self._stop_event = threading.Event()
        self._refresh_thread = None
        self._last_line = ""

    def start(self):
        self._start_time = time.monotonic()
        self._stop_event.clear()
        if _is_tty():
            self._refresh_thread = threading.Thread(
                target=self._refresh_loop, daemon=True
            )
            self._refresh_thread.start()

    def _refresh_loop(self):
        """后台线程：每 0.3 秒刷新一次进度行。"""
        while not self._stop_event.is_set():
            try:
                line = self._build_line()
                if line != self._last_line:
                    sys.stderr.write(f"\r{line}")
                    sys.stderr.flush()
                    self._last_line = line
            except Exception:
                pass
            self._stop_event.wait(timeout=0.3)

    def _build_line(self):
        """子类实现：返回当前状态行文本。"""
        raise NotImplementedError

    def update(self, stats):
        """子类实现：更新统计信息。返回 True 继续，False 停止。"""
        raise NotImplementedError

    def finish(self):
        elapsed = time.monotonic() - self._start_time if self._start_time else 0.0
        self._stop_event.set()
        if self._refresh_thread and self._refresh_thread.is_alive():
            self._refresh_thread.join(timeout=1.0)
        # 清除进度行，输出完成信息
        if _is_tty():
            sys.stderr.write(f"\r{'':80}\r")
        return elapsed

    def _fmt_elapsed(self):
        if self._start_time is None:
            return "0:00"
        return _fmt_time(time.monotonic() - self._start_time)


class LLLProgress(_ProgressBase):
    """LLL 约减实时进度追踪。"""

    def __init__(self, dim, dps=None):
        super().__init__(dim, dps)
        self.max_stage_reached = 0
        self.iterations = 0
        self.swap_count = 0
        self.size_count = 0
        self.gso_count = 0

    def _build_line(self):
        elapsed = self._fmt_elapsed()
        progress = self.max_stage_reached / self.dim if self.dim > 0 else 0.0
        pct = progress * 100

        # ETA
        eta = None
        if progress > 0:
            remaining = (time.monotonic() - self._start_time) * (1.0 - progress) / progress
            eta = _fmt_time(remaining)

        parts = [
            f"[LLL] dim={self.dim}",
            f"{pct:5.1f}%",
            f"stage={self.max_stage_reached}/{self.dim}",
            f"iter={self.iterations}",
            f"swap={self.swap_count}",
        ]
        if self.dps is not None:
            parts.append(f"dps={self.dps}")
        if eta:
            parts.append(f"ETA={eta}")
        parts.append(f"elapsed={elapsed}")
        return "  ".join(parts)

    def update_stats(self, stage, iterations, swap_count, size_count, gso_count):
        """适配 lll_mp 的位置参数调用，转换为字典格式后交给 update()。"""
        stats = {
            "max_stage_reached": stage,
            "iterations": iterations,
            "swap_count": swap_count,
            "size_count": size_count,
            "gso_count": gso_count,
        }
        return self.update(stats)

    def update(self, stats):
        self.max_stage_reached = max(self.max_stage_reached, stats.get("max_stage_reached", 0))
        self.iterations = stats.get("iterations", self.iterations)
        self.swap_count = stats.get("swap_count", self.swap_count)
        self.size_count = stats.get("size_count", self.size_count)
        self.gso_count = stats.get("gso_count", self.gso_count)
        return True

    def finish(self):
        elapsed = super().finish()
        if _is_tty():
            summary = (
                f"LLL 完成: dim={self.dim}, "
                f"iter={self.iterations}, swap={self.swap_count}, "
                f"elapsed={_fmt_time(elapsed)}"
            )
            print(summary, file=sys.stderr)
        else:
            print(f"LLL 完成: {_fmt_time(elapsed)}")
        return elapsed


class BKZProgress(_ProgressBase):
    """BKZ 约减实时进度追踪。"""

    def __init__(self, dim, block_size, dps=None):
        super().__init__(dim, dps)
        self.block_size = block_size
        self.z = 0
        self.m = 0
        self.shortest_norm = None
        self.total_steps = max(1, dim - block_size + 1)

    def _build_line(self):
        elapsed = self._fmt_elapsed()
        progress = self.z / self.total_steps if self.total_steps > 0 else 0.0
        pct = progress * 100

        # ETA
        eta = None
        if progress > 0:
            remaining = (time.monotonic() - self._start_time) * (1.0 - progress) / progress
            eta = _fmt_time(remaining)

        norm_str = f"shortest={self.shortest_norm:.6e}" if self.shortest_norm is not None else "shortest=?"
        parts = [
            f"[BKZ] β={self.block_size}",
            f"{pct:5.1f}%",
            f"z={self.z}/{self.total_steps}",
            norm_str,
        ]
        if self.dps is not None:
            parts.append(f"dps={self.dps}")
        if eta:
            parts.append(f"ETA={eta}")
        parts.append(f"elapsed={elapsed}")
        return "  ".join(parts)

    def update(self, stats):
        self.z = max(self.z, stats.get("z", 0))
        self.m = max(self.m, stats.get("m", 0))
        shortest = stats.get("shortest_norm")
        if shortest is not None:
            if self.shortest_norm is None or shortest < self.shortest_norm:
                self.shortest_norm = shortest
        return True

    def finish(self):
        elapsed = super().finish()
        if _is_tty():
            norm_str = f", shortest={self.shortest_norm:.6e}" if self.shortest_norm is not None else ""
            summary = (
                f"BKZ 完成: β={self.block_size}, "
                f"steps={self.z}/{self.total_steps}{norm_str}, "
                f"elapsed={_fmt_time(elapsed)}"
            )
            print(summary, file=sys.stderr)
        else:
            print(f"BKZ 完成: {_fmt_time(elapsed)}")
        return elapsed
