"""FPLL 通用工具函数。

无外部依赖的纯工具函数。
"""

from __future__ import annotations

import time
from typing import Any


def format_duration(seconds: float) -> str:
    """格式化持续时间为人类可读字符串。

    Args:
        seconds: 秒数

    Returns:
        格式化字符串，如 "1m 23s" 或 "45ms"
    """
    if seconds < 1.0:
        return f"{seconds * 1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.0f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


class Timer:
    """简单计时器，用于性能测量。

    用法:
        with Timer("操作名") as t:
            do_something()
        print(t.elapsed)
    """

    def __init__(self, name: str = "") -> None:
        self.name = name
        self.start: float = 0.0
        self.end: float = 0.0
        self.elapsed: float = 0.0

    def __enter__(self) -> Timer:
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        self.end = time.perf_counter()
        self.elapsed = self.end - self.start

    def __str__(self) -> str:
        if self.name:
            return f"{self.name}: {format_duration(self.elapsed)}"
        return format_duration(self.elapsed)


def ensure_list(value: Any) -> list[Any]:
    """确保值是列表类型。

    Args:
        value: 任意值

    Returns:
        列表形式的值
    """
    if isinstance(value, list):
        return value
    if isinstance(value, (tuple, set, frozenset)):
        return list(value)
    return [value]


def flatten_dict(d: dict[str, Any], parent_key: str = "", sep: str = ".") -> dict[str, Any]:
    """展平嵌套字典。

    Args:
        d: 嵌套字典
        parent_key: 父键前缀
        sep: 键分隔符

    Returns:
        展平后的字典

    Example:
        >>> flatten_dict({"a": {"b": 1, "c": 2}})
        {"a.b": 1, "a.c": 2}
    """
    items: list[tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
