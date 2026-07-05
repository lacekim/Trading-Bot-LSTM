"""Logging helpers for the modular V4 bot."""

from __future__ import annotations

import logging
from pathlib import Path


def build_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    Path("logs").mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.FileHandler("logs/v4.log")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        logger.addHandler(handler)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter("%(levelname)s %(name)s %(message)s"))
        logger.addHandler(stream_handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger
