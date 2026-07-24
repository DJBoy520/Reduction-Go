"""FPLL 日志工厂。

统一日志配置，支持控制台 + 文件双输出。
从原 src/utils/logger.py 迁移，保持功能一致。
"""

import logging
import os
from pathlib import Path


def setup_logging(log_file: str = "attack.log", console_level: int = logging.INFO) -> None:
    """配置根 logger：控制台 + 文件双输出。

    Args:
        log_file: 日志文件名（写入项目根目录 logs/）
        console_level: 控制台输出级别
    """
    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                handler.setLevel(console_level)
        return

    root.setLevel(logging.DEBUG)

    # 控制台输出
    console = logging.StreamHandler()
    console.setLevel(console_level)
    console.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(console)

    # 文件输出 — 写到项目根目录 logs/
    project_root = _find_project_root()
    log_dir = project_root / "logs"
    log_dir.mkdir(exist_ok=True)

    file_handler = logging.FileHandler(
        log_dir / log_file, mode="a", encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    root.addHandler(file_handler)


def _find_project_root() -> Path:
    """向上查找项目根目录（包含 manage.sh 或 .git）。"""
    current = Path(__file__).resolve().parent
    for _ in range(10):  # 最多向上查找10层
        if (current / "manage.sh").exists() or (current / ".git").exists():
            return current
        parent = current.parent
        if parent == current:
            break
        current = parent
    # 兜底：相对于当前文件的两级目录
    return Path(__file__).resolve().parent.parent.parent
