"""事件驱动的进度系统。

将原有的 print 直接输出替换为事件驱动模型：
- 定义进度事件数据结构
- CLIProgressEmitter：终端单行刷新输出
- WebProgressEmitter：预留 Web 接口（仅接口定义）
- EventDrivenProgress：桥接旧接口到新事件

保留 print_estimate / LLLProgress / BKZProgress 的外观，
内部改为通过事件驱动回调。
"""

import logging
import sys
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── 事件类型 ──

class EventType:
    """进度事件类型常量"""
    ALGORITHM_START = "algorithm_start"
    ALGORITHM_PROGRESS = "algorithm_progress"
    ALGORITHM_COMPLETE = "algorithm_complete"
    ALGORITHM_FAILED = "algorithm_failed"
    PHASE_START = "phase_start"
    PHASE_COMPLETE = "phase_complete"
    ETA_UPDATE = "eta_update"
    SUMMARY = "summary"


# ── 事件数据类 ──

@dataclass
class ProgressEvent:
    """进度事件基类"""
    event_type: str
    algorithm: str
    timestamp: float = field(default_factory=time.time)
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AlgorithmStartEvent(ProgressEvent):
    """算法开始事件"""
    dim: int = 0
    block_size: int = 0
    max_loops: int = 0

    def __post_init__(self):
        self.event_type = EventType.ALGORITHM_START


@dataclass
class AlgorithmProgressEvent(ProgressEvent):
    """算法进度更新事件"""
    loop: int = 0
    max_loops: int = 0
    rho_ratio: float = 0.0
    best_rho: float = 0.0
    elapsed: float = 0.0
    eta_seconds: float = 0.0
    status_msg: str = ""

    def __post_init__(self):
        self.event_type = EventType.ALGORITHM_PROGRESS


@dataclass
class AlgorithmCompleteEvent(ProgressEvent):
    """算法完成事件"""
    loops_done: int = 0
    rho_ratio: float = 0.0
    elapsed: float = 0.0
    summary: str = ""

    def __post_init__(self):
        self.event_type = EventType.ALGORITHM_COMPLETE


@dataclass
class AlgorithmFailedEvent(ProgressEvent):
    """算法失败事件"""
    reason: str = ""
    elapsed: float = 0.0

    def __post_init__(self):
        self.event_type = EventType.ALGORITHM_FAILED


@dataclass
class SummaryEvent(ProgressEvent):
    """最终汇总事件"""
    summary_text: str = ""
    total_loops: int = 0
    final_rho: float = 0.0

    def __post_init__(self):
        self.event_type = EventType.SUMMARY


# ── 抽象发射器 ──

class ProgressEmitter(ABC):
    """进度发射器抽象基类。子类只需实现 emit 方法。"""

    @abstractmethod
    def emit(self, event: ProgressEvent) -> None:
        """发射一个进度事件"""

    def on_complete(self) -> None:
        """算法完成时的清理（可选重写）"""

    def on_failed(self) -> None:
        """算法失败时的清理（可选重写）"""


class WebProgressEmitter(ProgressEmitter):
    """Web 进度发射器 — 预留接口。

    使用有界环形队列（maxlen=1000）存储事件，队列满时自动丢弃最旧事件，
    杜绝内存无限增长。所有对 _events 的读写操作通过线程锁保护。

    未来实现时只需重写 emit() 方法，
    无需修改核心算法或 adapter 代码。

    用法示例（未来）：
        emitter = WebProgressEmitter(ws_connection)
        engine.run(progress_callback=emitter.emit)
    """

    def __init__(self, ws=None):
        self._ws = ws
        self._events: deque = deque(maxlen=1000)
        self._lock = threading.Lock()

    def emit(self, event: ProgressEvent) -> None:
        """存储事件到内存列表，未来可通过 WebSocket 推送"""
        with self._lock:
            self._events.append(event)
        logger.debug("WebProgress emit: %s %s", event.event_type, event.algorithm)

    def get_events(self) -> List[ProgressEvent]:
        """返回当前事件列表的副本（线程安全）。"""
        with self._lock:
            return list(self._events)

    def clear_events(self) -> None:
        """清空事件队列（线程安全）。"""
        with self._lock:
            self._events.clear()


# ── CLI 进度发射器 ──

class CLIProgressEmitter(ProgressEmitter):
    """终端进度发射器 — 单行刷新 + 完成汇总。

    兼容原有 LLLProgress / BKZProgress 的输出格式，
    通过事件驱动触发刷新。
    """

    def __init__(self, output_stream=None):
        self._out = output_stream or sys.stderr
        self._is_tty = (
            hasattr(self._out, "isatty") and self._out.isatty()
        )
        self._last_len = 0
        self._lock = threading.Lock()

    # ── 内部格式化 ──

    @staticmethod
    def _fmt_time(seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.1f}s"
        if seconds < 3600:
            m = int(seconds // 60)
            s = seconds - m * 60
            return f"{m}m {s:.0f}s"
        h = int(seconds // 3600)
        m = int((seconds - h * 3600) // 60)
        return f"{h}h {m:02d}m"

    @staticmethod
    def _fmt_num(x: float, width: int = 6) -> str:
        return f"{x:>{width}.4f}"

    @staticmethod
    def _fmt_eta(seconds: float) -> str:
        if seconds < 0:
            return "ETA ..."
        return f"ETA {CLIProgressEmitter._fmt_time(seconds)}"

    def _write_line(self, line: str) -> None:
        with self._lock:
            if self._is_tty:
                pad = max(self._last_len - len(line), 0)
                self._out.write(f"\r{line}{' ' * pad}")
                self._out.flush()
                self._last_len = len(line)
            else:
                self._out.write(line + "\n")
                self._out.flush()

    def _clear_line(self) -> None:
        with self._lock:
            if self._is_tty and self._last_len > 0:
                self._out.write(f"\r{' ' * self._last_len}\r")
                self._out.flush()
                self._last_len = 0

    # ── emit 实现 ──

    def emit(self, event: ProgressEvent) -> None:
        etype = event.event_type

        if etype == EventType.ALGORITHM_START:
            self._on_start(event)
        elif etype == EventType.ALGORITHM_PROGRESS:
            self._on_progress(event)
        elif etype == EventType.ALGORITHM_COMPLETE:
            self._on_complete(event)
        elif etype == EventType.ALGORITHM_FAILED:
            self._on_failed(event)
        elif etype == EventType.SUMMARY:
            self._on_summary(event)

    def _on_start(self, event: ProgressEvent) -> None:
        if isinstance(event, AlgorithmStartEvent):
            if event.block_size > 0:
                tag = f"[BKZ-{event.block_size}]"
            else:
                tag = "[LLL]"
            self._write_line(f"{tag} dim={event.dim} max_loops={event.max_loops} ...")

    def _on_progress(self, event: ProgressEvent) -> None:
        if not isinstance(event, AlgorithmProgressEvent):
            return
        algo = event.algorithm.upper()
        if event.block_size > 0 if hasattr(event, "block_size") else False:
            tag = f"[BKZ-{event.block_size}]"
        else:
            tag = f"[{algo}]"

        parts = [
            tag,
            f"loop {event.loop}/{event.max_loops}",
            f"rho={self._fmt_num(event.rho_ratio)}",
            self._fmt_eta(event.eta_seconds),
        ]
        if event.status_msg:
            parts.append(event.status_msg)
        self._write_line(" ".join(parts))

    def _on_complete(self, event: ProgressEvent) -> None:
        self._clear_line()
        if isinstance(event, AlgorithmCompleteEvent):
            self._out.write(f"{event.summary}\n")
            self._out.flush()

    def _on_failed(self, event: ProgressEvent) -> None:
        self._clear_line()
        if isinstance(event, AlgorithmFailedEvent):
            self._out.write(f"{event.algorithm.upper()} 失败: {event.reason}\n")
            self._out.flush()

    def _on_summary(self, event: ProgressEvent) -> None:
        self._clear_line()
        if isinstance(event, SummaryEvent) and event.summary_text:
            self._out.write(f"{event.summary_text}\n")
            self._out.flush()

    def on_complete(self) -> None:
        self._clear_line()

    def on_failed(self) -> None:
        self._clear_line()


# ── 事件驱动进度控制器 ──

class EventDrivenProgress:
    """事件驱动进度控制器。

    提供与原有 LLLProgress / BKZProgress 兼容的接口，
    内部通过发射事件到 emitter 实现输出。
    """

    def __init__(
        self,
        algorithm: str,
        dim: int,
        base_matrix,
        loop: int = 0,
        max_loops: int = 0,
        block_size: int = 0,
        emitter: Optional[ProgressEmitter] = None,
    ):
        self._algorithm = algorithm
        self._dim = dim
        self._base_matrix = base_matrix
        self._loop = loop
        self._max_loops = max_loops
        self._block_size = block_size
        self._emitter = emitter
        self._start_time = time.time()
        self._last_rho = 0.0
        self._best_rho = float("inf")
        self._active = False
        self._done = False

    # ── 兼容 LLLProgress / BKZProgress 的属性 ──

    @property
    def loop(self) -> int:
        return self._loop

    @loop.setter
    def loop(self, v: int):
        self._loop = v

    @property
    def rho_ratio(self) -> float:
        return self._last_rho

    @rho_ratio.setter
    def rho_ratio(self, v: float):
        self._last_rho = v
        if v < self._best_rho:
            self._best_rho = v

    @property
    def best_rho(self) -> float:
        return self._best_rho

    @property
    def elapsed(self) -> float:
        return time.time() - self._start_time

    # ── 控制方法 ──

    def activate(self) -> None:
        """激活进度显示"""
        self._active = True
        if self._emitter is None:
            return
        event = AlgorithmStartEvent(
            event_type=EventType.ALGORITHM_START,
            algorithm=self._algorithm,
            dim=self._dim,
            block_size=self._block_size,
            max_loops=self._max_loops,
        )
        self._emitter.emit(event)

    def loop_done(self) -> None:
        """完成一个循环，刷新进度"""
        if not self._active or self._emitter is None:
            return
        elapsed = self.elapsed
        if self._loop > 0 and elapsed > 0:
            rate = elapsed / self._loop
            eta = rate * (self._max_loops - self._loop)
        else:
            eta = 0
        event = AlgorithmProgressEvent(
            event_type=EventType.ALGORITHM_PROGRESS,
            algorithm=self._algorithm,
            loop=self._loop,
            max_loops=self._max_loops,
            rho_ratio=self._last_rho,
            best_rho=self._best_rho,
            elapsed=elapsed,
            eta_seconds=eta,
        )
        self._emitter.emit(event)

    def done(self) -> float:
        """标记完成，发射完成事件，返回耗时"""
        if self._done:
            return self.elapsed
        self._done = True
        elapsed = self.elapsed

        summary = self._build_summary(elapsed)

        if self._emitter is not None:
            event = AlgorithmCompleteEvent(
                event_type=EventType.ALGORITHM_COMPLETE,
                algorithm=self._algorithm,
                loops_done=self._loop,
                rho_ratio=self._last_rho,
                elapsed=elapsed,
                summary=summary,
            )
            self._emitter.emit(event)
            self._emitter.on_complete()

        return elapsed

    def aborted(self, reason: str) -> float:
        """标记中止，发射失败事件，返回耗时"""
        if self._done:
            return self.elapsed
        self._done = True
        elapsed = self.elapsed

        if self._emitter is not None:
            event = AlgorithmFailedEvent(
                event_type=EventType.ALGORITHM_FAILED,
                algorithm=self._algorithm,
                reason=reason,
                elapsed=elapsed,
            )
            self._emitter.emit(event)
            self._emitter.on_failed()

        return elapsed

    def _build_summary(self, elapsed: float) -> str:
        if self._block_size > 0:
            prefix = f"BKZ-{self._block_size}"
        else:
            prefix = self._algorithm.upper()
        return (
            f"{prefix} 完成: {self._loop} loops, "
            f"rho={self._last_rho:.4f} (best={self._best_rho:.4f}), "
            f"elapsed={CLIProgressEmitter._fmt_time(elapsed)}"
        )


# ── print_estimate 兼容层 ──

def print_estimate(
    dim: int,
    rho: float,
    loop: int,
    max_loops: int,
    rho_ratio: float,
    best_rho: float,
    elapsed: float,
    eta_seconds: float,
    status_msg: str = "",
    emitter: Optional[ProgressEmitter] = None,
) -> None:
    """打印单次估计（兼容原 progress.py 的 print_estimate）。

    如果传入 emitter，则通过事件驱动输出；否则回退到直接打印。
    """
    if emitter is not None:
        event = AlgorithmProgressEvent(
            event_type=EventType.ALGORITHM_PROGRESS,
            algorithm="lll",
            loop=loop,
            max_loops=max_loops,
            rho_ratio=rho_ratio,
            best_rho=best_rho,
            elapsed=elapsed,
            eta_seconds=eta_seconds,
            status_msg=status_msg,
        )
        emitter.emit(event)
        return

    # 回退：直接打印（兼容旧路径）
    fmt = CLIProgressEmitter._fmt_time
    _fmt_num = CLIProgressEmitter._fmt_num
    _fmt_eta = CLIProgressEmitter._fmt_eta
    parts = [
        f"[LLL]",
        f"loop {loop}/{max_loops}",
        f"rho={_fmt_num(rho_ratio)}",
        _fmt_eta(eta_seconds),
    ]
    if status_msg:
        parts.append(status_msg)
    line = " ".join(parts)
    print(line, file=sys.stderr, flush=True)


# ── 兼容类：LLLProgress / BKZProgress ──

class LLLProgress:
    """LLL 进度上下文管理器（兼容原有接口）。

    内部通过 EventDrivenProgress 实现输出。
    emitter 可选：传入时启用进度显示，为 None 时静默运行。
    """

    def __init__(self, dim: int, base_matrix, loop: int = 0,
                 max_loops: int = 0, emitter: Optional[ProgressEmitter] = None):
        self._impl = EventDrivenProgress(
            algorithm="lll",
            dim=dim,
            base_matrix=base_matrix,
            loop=loop,
            max_loops=max_loops,
            emitter=emitter,
        )

    @property
    def loop(self) -> int:
        return self._impl.loop

    @loop.setter
    def loop(self, v: int):
        self._impl.loop = v

    @property
    def rho_ratio(self) -> float:
        return self._impl.rho_ratio

    @rho_ratio.setter
    def rho_ratio(self, v: float):
        self._impl.rho_ratio = v

    @property
    def best_rho(self) -> float:
        return self._impl.best_rho

    @property
    def elapsed(self) -> float:
        return self._impl.elapsed

    def loop_done(self) -> None:
        self._impl.loop_done()

    def done(self) -> float:
        return self._impl.done()

    def aborted(self, reason: str) -> float:
        return self._impl.aborted(reason)

    def __enter__(self):
        self._impl.activate()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self._impl.aborted(str(exc_val))
        else:
            self._impl.done()
        return False


class BKZProgress:
    """BKZ 进度上下文管理器（兼容原有接口）。

    内部通过 EventDrivenProgress 实现输出。
    emitter 可选：传入时启用进度显示，为 None 时静默运行。
    """

    def __init__(
        self,
        block_size: int,
        dim: int,
        base_matrix,
        loop: int = 0,
        max_loops: int = 0,
        emitter: Optional[ProgressEmitter] = None,
    ):
        self._impl = EventDrivenProgress(
            algorithm="bkz",
            dim=dim,
            base_matrix=base_matrix,
            loop=loop,
            max_loops=max_loops,
            block_size=block_size,
            emitter=emitter,
        )

    @property
    def loop(self) -> int:
        return self._impl.loop

    @loop.setter
    def loop(self, v: int):
        self._impl.loop = v

    @property
    def rho_ratio(self) -> float:
        return self._impl.rho_ratio

    @rho_ratio.setter
    def rho_ratio(self, v: float):
        self._impl.rho_ratio = v

    @property
    def best_rho(self) -> float:
        return self._impl.best_rho

    @property
    def elapsed(self) -> float:
        return self._impl.elapsed

    def loop_done(self) -> None:
        self._impl.loop_done()

    def done(self) -> float:
        return self._impl.done()

    def aborted(self, reason: str) -> float:
        return self._impl.aborted(reason)

    def __enter__(self):
        self._impl.activate()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self._impl.aborted(str(exc_val))
        else:
            self._impl.done()
        return False


# ── 工具函数 ──

def fmt_time(seconds: float) -> str:
    """格式化时间（公开接口，兼容外部调用）"""
    return CLIProgressEmitter._fmt_time(seconds)
