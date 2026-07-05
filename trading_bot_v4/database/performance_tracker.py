"""Simple performance tracking utility for V4 that writes basic trade summaries to CSV."""

from __future__ import annotations

import csv
from pathlib import Path


class PerformanceTracker:
    def __init__(self, path: str | Path = "data/v4_performance.csv"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with self.path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["timestamp", "symbol", "profit", "capital"])
                writer.writeheader()

    def log_trade(self, timestamp, symbol, profit, capital):
        with self.path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["timestamp", "symbol", "profit", "capital"])
            writer.writerow({"timestamp": timestamp, "symbol": symbol, "profit": profit, "capital": capital})
