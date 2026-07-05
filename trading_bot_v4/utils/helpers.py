"""Helper utilities for the modular V4 trading bot."""

from __future__ import annotations

from pathlib import Path


def ensure_directory(path: str | Path):
    Path(path).mkdir(parents=True, exist_ok=True)
    return Path(path)
