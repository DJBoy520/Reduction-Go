"""日志配置。"""

import logging
import os


def setup_logging(log_file: str = "attack.log", console_level: int = logging.INFO):
    """配置根 logger：控制台 + 文件双输出。"""
    root = logging.getLogger()
    if root.handlers:
        for h in root.handlers:
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
                h.setLevel(console_level)
        return

    root.setLevel(logging.DEBUG)

    # 控制台
    console = logging.StreamHandler()
    console.setLevel(console_level)
    console.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(console)

    # 文件 — 写到项目根目录 logs/
    project_root = os.path.join(os.path.dirname(__file__), '..', '..')
    log_dir = os.path.join(project_root, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    fh = logging.FileHandler(os.path.join(log_dir, log_file), mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    root.addHandler(fh)
