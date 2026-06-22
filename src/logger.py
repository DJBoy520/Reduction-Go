"""
Unified logging configuration for the FPLL project.

- Console: INFO level, human-readable format.
- File:    DEBUG level, full detail with timestamp/module/line.
"""

import logging
import os


def setup_logging(log_file: str = "attack.log", console_level: int = logging.INFO):
    """Configure root logger with console + file handlers."""
    root = logging.getLogger()
    if root.handlers:
        # Update console level if already configured
        for h in root.handlers:
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
                h.setLevel(console_level)
        return

    root.setLevel(logging.DEBUG)

    # Console handler
    console = logging.StreamHandler()
    console.setLevel(console_level)
    console.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(console)

    # File handler — DEBUG
    log_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
    log_path = os.path.join(log_dir, log_file)
    os.makedirs(log_dir, exist_ok=True)
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    root.addHandler(fh)
